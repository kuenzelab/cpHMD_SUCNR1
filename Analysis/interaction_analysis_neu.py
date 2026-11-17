#!/usr/bin/env python3
"""
Interaction partners and their stability for selected residues (MDAnalysis).

For every residue you specify (CHAIN:RESNUM) the script finds ALL other residues
(of --partners, default "protein") that it contacts, classifies the interaction
type and quantifies how stable each interaction is. Several conditions with
several replicas each are supported; results are combined per condition.

Input: samples table (CSV/TSV) with columns  condition, topology, trajectory
    condition,topology,trajectory
    WT,WT_rep1/system.pdb,WT_rep1/md.xtc
    WT,WT_rep2/system.pdb,WT_rep2/md.xtc
    MutA,MutA_rep1/system.pdb,MutA_rep1/md.xtc

Example:
    python interaction_analysis.py samples.csv -r A:12 B:52 \
        --cutoff 4.5 --min-fraction 0.2 --out inter

Interaction categories (all geometric, heavy-atom based, per frame):
  contact      any atom pair of the two residues closer than --cutoff (4.5 A)
  hbond        N/O ... O pair (N-N excluded) closer than --hbond-cutoff (3.5 A).
               Heavy-atom distance PROXY only: no angle criterion and no
               donor/acceptor assignment. Use MDAnalysis' HydrogenBondAnalysis
               for strict H-bonds.
  salt         Lys NZ / Arg NE,NH1,NH2  <->  Asp OD1,OD2 / Glu OE1,OE2 closer
               than --salt-cutoff (4.0 A)
  hydrophobic  side-chain carbons of A,V,L,I,M,F,W,P,Y pairs closer than
               --hydrophobic-cutoff (4.5 A); both residues must be from this set
Constant-pH (GROMACS CpHMD) residues ASPT/GLUT/LYST/ARGT/HSPT are treated as
ASP/GLU/LYS/ARG/HIS for the type classification and are included as partners.
Their hydrogens (possibly dummy atoms) are never used. The protonation state per
frame is NOT considered: salt-bridge / H-bond classes are geometric, so e.g. a
protonated Asp near a Lys can be counted as 'salt' (see notes in the docs).
Residues covalently near the query (|resid difference| <= --exclude-neighbors,
same chain; default 1) are ignored, because they are always in contact.

Stability metrics (per interaction = query residue + partner + category):
  fraction        frames with the interaction / frames analysed (0-1)
  events          number of continuous episodes (breaks <= --gap frames are
                  bridged; default 0 = every missing frame ends an episode)
  longest         longest continuous episode (ns, or frames if the trajectory has
                  no timestep information)
  mean_event      mean episode length (episode span incl. bridged gaps)

Replica combination: fraction = frame-weighted mean (total frames with
interaction / total frames); SEM = std between replica fractions / sqrt(n_replicas)
(replicas where the interaction never occurs count as 0). 'longest' is reported
as mean over replicas and as maximum over replicas.
"""
import argparse
import csv
import sys
import warnings
from collections import OrderedDict, defaultdict

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance

warnings.filterwarnings("ignore", module="MDAnalysis")

CATEGORIES = ["contact", "hbond", "salt", "hydrophobic"]

# GROMACS constant-pH (CpHMD) titratable residue names -> standard residue names.
# Edit here if your force-field uses other names.
TITRATABLE = {"ASPT": "ASP", "GLUT": "GLU", "LYST": "LYS", "ARGT": "ARG",
              "HSPT": "HIS"}
TIT_NAMES = " ".join(TITRATABLE)
# MDAnalysis' "protein" keyword does NOT know these names, so add them explicitly
DEFAULT_PARTNERS = f"protein or resname {TIT_NAMES}"
# Hydrogens of titratable residues may be dummy atoms (present in the trajectory with
# coordinates but not necessarily "real" at that frame's protonation state), so they
# are NEVER used for contacts. Other hydrogens only count with --include-h.
FILTER_HEAVY = "not name H*"
FILTER_WITH_H = f"not (resname {TIT_NAMES} and name H*)"
HYDROPHOBIC_RES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "TYR"}
POS_ATOMS = {"LYS": {"NZ"}, "ARG": {"NE", "NH1", "NH2"}}
NEG_ATOMS = {"ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def parse_residue_spec(spec):
    s = spec.strip()
    if ":" in s:
        chain, num = s.split(":", 1)
    else:
        chain, num = s[0], s[1:]
    return chain, int(num)


def read_samples(path):
    with open(path, newline="") as fh:
        sample = fh.read(2048)
        fh.seek(0)
        delim = "\t" if sample.count("\t") > sample.count(",") else ","
        samples = OrderedDict()
        for r in csv.DictReader(fh, delimiter=delim):
            samples.setdefault(r["condition"].strip(), []).append(
                (r["topology"].strip(), r["trajectory"].strip()))
    return samples


def select_residue(u, chain, resnum, atom_filter):
    sel = f"(chainID {chain} or segid {chain}) and resid {resnum}"
    ag = u.select_atoms(sel)
    if len(ag) == 0:
        raise ValueError(f"residue {chain}:{resnum} not found ('{sel}')")
    if len(ag.residues) > 1:
        raise ValueError(f"residue {chain}:{resnum} is ambiguous "
                         f"({len(ag.residues)} residues match '{sel}')")
    if atom_filter:
        ag = ag.select_atoms(atom_filter)
        if len(ag) == 0:
            raise ValueError(f"residue {chain}:{resnum} has no atoms after filter")
    return ag


def chain_label(residue):
    try:
        c = str(residue.atoms[0].chainID).strip()
        if c:
            return c
    except Exception:
        pass
    return str(residue.segid).strip()


def atom_flags(ag):
    """Per-atom boolean flags used to classify interaction types."""
    names = np.asarray(ag.names, dtype=str)
    resn = np.array([TITRATABLE.get(r, r) for r in np.asarray(ag.resnames, dtype=str)])
    first = np.array([n[:1] for n in names])
    f = {
        "N": first == "N",
        "O": first == "O",
        "pos": np.array([n in POS_ATOMS.get(r, ()) for n, r in zip(names, resn)]),
        "neg": np.array([n in NEG_ATOMS.get(r, ()) for n, r in zip(names, resn)]),
        "hphob": np.array([(r in HYDROPHOBIC_RES) and n.startswith("C")
                           and n not in ("C", "CA") for n, r in zip(names, resn)]),
    }
    return f


def run_stats(frames, gap):
    """frames: sorted array of frame indices. Returns (n_events, longest, span_sum)."""
    f = np.asarray(frames)
    breaks = np.where(np.diff(f) > gap + 1)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(f) - 1]
    spans = f[ends] - f[starts] + 1
    return len(spans), int(spans.max()), int(spans.sum())


# --------------------------------------------------------------------------- #
# per-replica analysis
# --------------------------------------------------------------------------- #
def analyse_replica(top, traj, queries, partner_sel, cut, atom_filter, excl, gap,
                    start, stop, step):
    u = mda.Universe(top, traj)

    q_groups = [select_residue(u, c, n, atom_filter) for c, n in queries]
    q_idx = np.concatenate([g.indices for g in q_groups])
    q_atoms = u.atoms[q_idx]
    q2k = np.concatenate([np.full(len(g), k) for k, g in enumerate(q_groups)])

    partners = u.select_atoms(partner_sel)
    if atom_filter:
        partners = partners.select_atoms(atom_filter)
    if len(partners) == 0:
        raise ValueError(f"{top}: partner selection '{partner_sel}' is empty")
    n_tit_all = len(u.select_atoms(f"resname {TIT_NAMES}").residues)
    n_tit_sel = len(partners.select_atoms(f"resname {TIT_NAMES}").residues)
    if n_tit_all and n_tit_sel < n_tit_all:
        print(f"WARNING: {top}: only {n_tit_sel} of {n_tit_all} titratable residues "
              f"are in the partner selection", file=sys.stderr)

    qf, pf = atom_flags(q_atoms), atom_flags(partners)
    q_seg, q_resid = q_atoms.segindices, q_atoms.resids
    p_seg, p_resid, p_residx = partners.segindices, partners.resids, partners.resindices
    n_res = u.atoms.n_residues + 1
    maxcut = max(cut.values())

    hits = {cat: defaultdict(list) for cat in CATEGORIES}
    times = []
    nf = 0
    for ts in u.trajectory[start:stop:step]:
        times.append(ts.time)
        box = ts.dimensions
        if box is None or np.allclose(box[:3], 0):
            box = None
        pairs, d = capped_distance(q_atoms.positions, partners.positions, maxcut,
                                   box=box, return_distances=True)
        if len(pairs):
            qi, pi = pairs[:, 0], pairs[:, 1]
            valid = np.ones(len(qi), dtype=bool)
            if excl >= 0:
                valid &= ~((q_seg[qi] == p_seg[pi]) &
                           (np.abs(q_resid[qi] - p_resid[pi]) <= excl))
            masks = {
                "contact": d <= cut["contact"],
                "hbond": (((qf["N"][qi] & pf["O"][pi]) | (qf["O"][qi] & pf["N"][pi])
                           | (qf["O"][qi] & pf["O"][pi])) & (d <= cut["hbond"])),
                "salt": (((qf["pos"][qi] & pf["neg"][pi]) | (qf["neg"][qi] & pf["pos"][pi]))
                         & (d <= cut["salt"])),
                "hydrophobic": qf["hphob"][qi] & pf["hphob"][pi] & (d <= cut["hydrophobic"]),
            }
            for cat, m in masks.items():
                sel = valid & m
                if sel.any():
                    keys = np.unique(q2k[qi[sel]].astype(np.int64) * n_res + p_residx[pi[sel]])
                    for key in keys:
                        hits[cat][int(key)].append(nf)
        nf += 1

    if len(times) > 1 and np.median(np.diff(times)) > 0:
        dt_ns = float(np.median(np.diff(times))) / 1000.0
    else:
        dt_ns = np.nan

    recs, pinfo = {}, {}
    for cat in CATEGORIES:
        for key, frames in hits[cat].items():
            k, ridx = divmod(key, n_res)
            n_ev, longest, span = run_stats(frames, gap)
            recs[(k, ridx, cat)] = (len(frames), n_ev, longest, span)
            if ridx not in pinfo:
                r = u.residues[ridx]
                pinfo[ridx] = (chain_label(r), str(r.resname), int(r.resid))
    return {"nf": nf, "dt_ns": dt_ns, "recs": recs, "pinfo": pinfo}


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def plot_query(path, title, conditions, labels, frac, longest, unit, cat, annotate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = max(6, 0.75 * len(labels) + 3)
    h = max(5, 1.0 * len(conditions) + 4)
    fig, axes = plt.subplots(2, 1, figsize=(w, h), sharex=True)
    panels = [
        (axes[0], frac, "viridis", 0.0, 1.0, "Contact fraction" if cat == "contact"
         else f"{cat} fraction", "{:.2f}"),
        (axes[1], longest, "magma", 0.0, max(np.nanmax(longest), 1e-9),
         f"Mean longest continuous episode ({unit})",
         "{:.2f}" if np.nanmax(longest) < 1 else "{:.1f}"),
    ]
    for ax, mat, cmap, vmin, vmax, lab, fmt in panels:
        cm = plt.get_cmap(cmap).copy()
        im = ax.imshow(mat, cmap=cm, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels(conditions)
        ax.set_ylabel("Condition")
        fig.colorbar(im, ax=ax).set_label(lab)
        if annotate:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    v = mat[i, j]
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                            color="white" if v < 0.6 * vmax else "black")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=90)
    axes[1].set_xlabel("Interaction partner")
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples")
    ap.add_argument("-r", "--residues", nargs="+", required=True,
                    help="Query residues as CHAIN:RESNUM")
    ap.add_argument("-p", "--partners", default=DEFAULT_PARTNERS,
                    help=f'MDAnalysis selection of possible partners (default "{DEFAULT_PARTNERS}"); '
                         'NOTE: plain "protein" misses ASPT/GLUT/LYST/ARGT/HSPT. '
                         'e.g. "protein or resname LIG" or "chainID D"')
    ap.add_argument("-c", "--cutoff", type=float, default=4.5,
                    help="General contact cutoff in A (default 4.5)")
    ap.add_argument("--hbond-cutoff", type=float, default=3.5)
    ap.add_argument("--salt-cutoff", type=float, default=4.0)
    ap.add_argument("--hydrophobic-cutoff", type=float, default=4.5)
    ap.add_argument("--exclude-neighbors", type=int, default=1,
                    help="Ignore partners with |resid diff| <= N in the same chain "
                         "(default 1; 0 = only the residue itself; -1 = none)")
    ap.add_argument("--gap", type=int, default=0,
                    help="Bridge breaks of up to this many frames inside an episode")
    ap.add_argument("--min-fraction", type=float, default=0.1,
                    help="Report/plot only interactions with condition-mean fraction "
                         ">= this value (default 0.1)")
    ap.add_argument("--plot-category", choices=CATEGORIES, default="contact",
                    help="Category shown in the heatmaps (default contact)")
    ap.add_argument("--include-h", action="store_true",
                    help="Include hydrogens in the contact definition (hydrogens of "
                         "titratable residues are always excluded)")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--out", default="inter")
    ap.add_argument("--no-annot", action="store_true")
    args = ap.parse_args()

    samples = read_samples(args.samples)
    queries = [parse_residue_spec(r) for r in args.residues]
    qlabels = [f"{c}:{n}" for c, n in queries]
    atom_filter = FILTER_WITH_H if args.include_h else FILTER_HEAVY
    cut = {"contact": args.cutoff, "hbond": args.hbond_cutoff,
           "salt": args.salt_cutoff, "hydrophobic": args.hydrophobic_cutoff}
    conditions = list(samples)

    # ---- analyse every replica ------------------------------------------- #
    results = OrderedDict((c, []) for c in conditions)
    for cond, reps in samples.items():
        for k, (top, traj) in enumerate(reps, start=1):
            print(f"[{cond}] replica {k}: {traj}", file=sys.stderr)
            res = analyse_replica(top, traj, queries, args.partners, cut, atom_filter,
                                  args.exclude_neighbors, args.gap,
                                  args.start, args.stop, args.step)
            res["traj"] = traj
            results[cond].append(res)

    use_ns = all(not np.isnan(r["dt_ns"]) for rs in results.values() for r in rs)
    unit = "ns" if use_ns else "frames"
    conv = (lambda frames, r: frames * r["dt_ns"]) if use_ns else (lambda frames, r: float(frames))

    # global partner key = (chain, resid); keep a resname for labels
    plabel = {}
    for rs in results.values():
        for r in rs:
            for chain, resname, resid in r["pinfo"].values():
                plabel.setdefault((chain, resid), f"{chain}:{resname}{resid}")

    def rec_by_pkey(r):
        out = {}
        for (k, ridx, cat), v in r["recs"].items():
            chain, _, resid = r["pinfo"][ridx]
            out[(k, (chain, resid), cat)] = v
        return out

    # ---- per-replica CSV -------------------------------------------------- #
    with open(f"{args.out}_per_replica.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "replica", "trajectory", "query", "partner", "category",
                    "frames_with_interaction", "n_frames", "fraction", "n_events",
                    f"longest_event_{unit}", f"mean_event_{unit}"])
        for cond, rs in results.items():
            for rep, r in enumerate(rs, start=1):
                for (k, pk, cat), (cnt, nev, lng, span) in sorted(
                        rec_by_pkey(r).items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
                    w.writerow([cond, rep, r["traj"], qlabels[k], plabel[pk], cat, cnt,
                                r["nf"], cnt / r["nf"], nev, conv(lng, r),
                                conv(span / nev, r)])

    # ---- aggregate per condition ------------------------------------------ #
    agg = {}  # (cond, k, pk, cat) -> dict
    for cond, rs in results.items():
        per_rep = [rec_by_pkey(r) for r in rs]
        keys = set().union(*[d.keys() for d in per_rep])
        nfs = np.array([r["nf"] for r in rs], float)
        for key in keys:
            cnt = np.array([d.get(key, (0, 0, 0, 0))[0] for d in per_rep], float)
            nev = np.array([d.get(key, (0, 0, 0, 0))[1] for d in per_rep], float)
            lng = np.array([conv(d.get(key, (0, 0, 0, 0))[2], r) for d, r in zip(per_rep, rs)])
            span = np.array([conv(d.get(key, (0, 0, 0, 0))[3], r) for d, r in zip(per_rep, rs)])
            frac = cnt / nfs
            n_rep = len(rs)
            agg[(cond,) + key] = {
                "fraction": cnt.sum() / nfs.sum(),
                "sem": frac.std(ddof=1) / np.sqrt(n_rep) if n_rep > 1 else np.nan,
                "n_obs": int((cnt > 0).sum()), "n_rep": n_rep,
                "events": int(nev.sum()),
                "mean_longest": lng.mean(), "max_longest": lng.max(),
                "mean_event": span.sum() / nev.sum() if nev.sum() else np.nan,
            }

    with open(f"{args.out}_per_condition.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "query", "partner", "category", "fraction_mean",
                    "fraction_sem", "replicas_observed", "n_replicas", "events_total",
                    f"mean_event_{unit}", f"longest_event_mean_{unit}",
                    f"longest_event_max_{unit}"])
        for (cond, k, pk, cat), a in sorted(
                agg.items(), key=lambda x: (conditions.index(x[0][0]), x[0][1],
                                            x[0][3], -x[1]["fraction"])):
            if a["fraction"] < args.min_fraction:
                continue
            w.writerow([cond, qlabels[k], plabel[pk], cat, a["fraction"], a["sem"],
                        a["n_obs"], a["n_rep"], a["events"], a["mean_event"],
                        a["mean_longest"], a["max_longest"]])

    # ---- heatmaps: one figure per query residue --------------------------- #
    cat = args.plot_category
    written = []
    for k, ql in enumerate(qlabels):
        pks = sorted({key[2] for key, a in agg.items()
                      if key[1] == k and key[3] == cat and a["fraction"] >= args.min_fraction},
                     key=lambda x: (x[0], x[1]))
        if not pks:
            print(f"NOTE: no '{cat}' partners >= {args.min_fraction} for {ql}; no plot",
                  file=sys.stderr)
            continue
        frac = np.zeros((len(conditions), len(pks)))
        lng = np.zeros_like(frac)
        for i, cond in enumerate(conditions):
            for j, pk in enumerate(pks):
                a = agg.get((cond, k, pk, cat))
                if a:
                    frac[i, j], lng[i, j] = a["fraction"], a["mean_longest"]
        path = f"{args.out}_{ql.replace(':', '')}_{cat}_heatmap.png"
        plot_query(path, f"{ql}: {cat} partners (cutoff {cut[cat]} Å)", conditions,
                   [plabel[pk] for pk in pks], frac, lng, unit, cat, not args.no_annot)
        written.append(path)

    print("Wrote: " + ", ".join([f"{args.out}_per_replica.csv",
                                 f"{args.out}_per_condition.csv"] + written))


if __name__ == "__main__":
    main()
