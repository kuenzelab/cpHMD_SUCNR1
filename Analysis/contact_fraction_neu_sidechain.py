#!/usr/bin/env python3
"""
Contact fraction (0-1) of selected residues with a target chain selection,
per condition (several replicas each), with heatmap. Uses MDAnalysis.

Input: samples table (CSV/TSV) with columns  condition, topology, trajectory
    condition,topology,trajectory
    WT,WT_rep1/system.pdb,WT_rep1/md.xtc
    WT,WT_rep2/system.pdb,WT_rep2/md.xtc
    MutA,MutA_rep1/system.pdb,MutA_rep1/md.xtc

Example:
    python contact_fraction.py samples.csv -r A:12 A:23 B:52 D:51 \
        --target "chainID D" --cutoff 4.5 --out contacts

Definition: a residue is "in contact" in a frame if ANY of its (heavy) atoms is
closer than --cutoff (Angstrom) to ANY (heavy) atom of the target selection.
Periodic boundary conditions are handled with the minimum-image convention using
the box of each frame (no unwrapping/centering of the trajectory is needed for
atom-atom distances).

Replica combination: contact fraction = (contact frames) / (frames) per replica;
per condition the mean is frame-weighted over replicas (= total contact frames /
total frames). The spread between replicas (sample std of the replica fractions)
is reported as well, together with the standard error of the mean (SEM)
between replicas: SEM = std / sqrt(n_replicas). Frames within a trajectory are
autocorrelated, so the replicas (not the frames) are the independent samples.
SEM needs >= 2 replicas per condition (otherwise it is written as NaN).
"""
import argparse
import csv
import sys
from collections import OrderedDict

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance


# GROMACS constant-pH titratable residues: their hydrogens may be dummy atoms that
# still carry coordinates, so they are never used for contacts.
TIT_NAMES = "ASPT GLUT LYST ARGT HSPT"
FILTER_HEAVY = "not name H*"
FILTER_WITH_H = f"not (resname {TIT_NAMES} and name H*)"


# --------------------------------------------------------------------------- #
def parse_residue_spec(spec):
    s = spec.strip()
    if ":" in s:
        chain, num = s.split(":", 1)
    else:  # allow "A12"
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
    """Return AtomGroup of the residue (chainID or segid match), filtered atoms."""
    sel = f"(chainID {chain} or segid {chain}) and resid {resnum} and not (name N or name C or name O)"
    ag = u.select_atoms(sel)
    if len(ag) == 0:
        raise ValueError(f"residue {chain}:{resnum} not found ('{sel}')")
    if len(ag.residues) > 1:
        raise ValueError(f"residue {chain}:{resnum} is ambiguous "
                         f"({len(ag.residues)} residues match '{sel}')")
    if atom_filter:
        ag = ag.select_atoms(atom_filter)
        if len(ag) == 0:
            raise ValueError(f"residue {chain}:{resnum} has no atoms after "
                             f"filter '{atom_filter}'")
    return ag


def contacts_for_replica(top, traj, residues, target_sel, cutoff, atom_filter,
                         start, stop, step):
    """Return (n_frames, contact_counts[array len(residues)])."""
    u = mda.Universe(top, traj)

    target = u.select_atoms(target_sel)
    if len(target) == 0:
        raise ValueError(f"{top}: target selection '{target_sel}' is empty")
    if atom_filter:
        target = target.select_atoms(atom_filter)

    res_groups = [select_residue(u, c, n, atom_filter) for c, n in residues]

    # warn if a residue is part of the target (would trivially be in contact)
    tgt_ids = set(target.indices)
    for (c, n), ag in zip(residues, res_groups):
        if tgt_ids.intersection(ag.indices):
            print(f"WARNING: {top}: residue {c}:{n} is part of the target "
                  f"selection - its contacts include itself/neighbours",
                  file=sys.stderr)

    # one combined atom list; map each atom -> residue index
    atom2res = np.concatenate([np.full(len(g), k) for k, g in enumerate(res_groups)])
    all_atoms = u.atoms[np.concatenate([g.indices for g in res_groups])]

    counts = np.zeros(len(residues), dtype=np.int64)
    n_frames = 0
    for ts in u.trajectory[start:stop:step]:
        box = ts.dimensions
        if box is None or np.allclose(box[:3], 0):
            box = None
        pairs = capped_distance(all_atoms.positions, target.positions, cutoff,
                                box=box, return_distances=False)
        if len(pairs):
            hit = np.unique(atom2res[pairs[:, 0]])
            counts[hit] += 1
        n_frames += 1
    return n_frames, counts


# --------------------------------------------------------------------------- #
def plot_heatmap(mat, conditions, res_labels, out_png, title, cmap="viridis",
                 annotate=True, sem=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = max(6, 0.7 * len(res_labels) + 3)
    h = max(3, 0.6 * len(conditions) + 2)
    fig, ax = plt.subplots(figsize=(w, h))
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("lightgrey")
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cm, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(res_labels)))
    ax.set_xticklabels(res_labels, rotation=90)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions)
    ax.set_xlabel("Residue")
    ax.set_ylabel("Condition")
    ax.set_title(title)
    fig.colorbar(im, ax=ax).set_label("Contact fraction")
    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if not np.isnan(v):
                    txt = f"{v:.2f}"
                    if sem is not None and not np.isnan(sem[i, j]):
                        txt += f"\n±{sem[i, j]:.2f}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                            color="white" if v < 0.6 else "black")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", help="CSV/TSV with columns: condition,topology,trajectory")
    ap.add_argument("-r", "--residues", nargs="+", required=True,
                    help="Residues as CHAIN:RESNUM, e.g. A:12 B:52")
    ap.add_argument("-t", "--target", required=True,
                    help='MDAnalysis selection for the target, same for all residues, '
                         'e.g. "chainID D" or "segid D" or "chainID D and resid 40-60"')
    ap.add_argument("-c", "--cutoff", type=float, default=4.5,
                    help="Contact distance cutoff in Angstrom (default 4.5)")
    ap.add_argument("--include-h", action="store_true",
                    help="Include hydrogens (default: heavy atoms only; hydrogens of "
                         "constant-pH residues ASPT/GLUT/LYST/ARGT/HSPT are always excluded)")
    ap.add_argument("--start", type=int, default=None, help="First frame index")
    ap.add_argument("--stop", type=int, default=None, help="Stop frame index (exclusive)")
    ap.add_argument("--step", type=int, default=None, help="Frame stride")
    ap.add_argument("--out", default="contacts", help="Output prefix")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--no-annot", action="store_true")
    ap.add_argument("--annot-sem", action="store_true",
                    help="Annotate heatmap cells as mean ± SEM")
    args = ap.parse_args()

    samples = read_samples(args.samples)
    residues = [parse_residue_spec(r) for r in args.residues]
    res_labels = [f"{c}:{n}" for c, n in residues]
    atom_filter = FILTER_WITH_H if args.include_h else FILTER_HEAVY
    conditions = list(samples)

    rep_rows = []  # per-replica results
    cond_counts = {c: np.zeros(len(residues), dtype=np.int64) for c in conditions}
    cond_frames = {c: 0 for c in conditions}
    cond_repfrac = {c: [] for c in conditions}

    for cond, reps in samples.items():
        for k, (top, traj) in enumerate(reps, start=1):
            print(f"[{cond}] replica {k}: {traj}", file=sys.stderr)
            nf, cnt = contacts_for_replica(top, traj, residues, args.target,
                                           args.cutoff, atom_filter,
                                           args.start, args.stop, args.step)
            frac = cnt / nf
            cond_counts[cond] += cnt
            cond_frames[cond] += nf
            cond_repfrac[cond].append(frac)
            for j, lab in enumerate(res_labels):
                rep_rows.append([cond, k, traj, lab, int(cnt[j]), nf, frac[j]])

    # per-replica CSV
    with open(f"{args.out}_per_replica.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "replica", "trajectory", "residue",
                    "contact_frames", "n_frames", "contact_fraction"])
        w.writerows(rep_rows)

    # per-condition: frame-weighted mean + std between replicas
    mean = np.zeros((len(conditions), len(residues)))
    sd = np.full_like(mean, np.nan)
    sem = np.full_like(mean, np.nan)
    for i, cond in enumerate(conditions):
        mean[i] = cond_counts[cond] / cond_frames[cond]
        fr = np.array(cond_repfrac[cond])
        if len(fr) > 1:
            sd[i] = fr.std(axis=0, ddof=1)
            sem[i] = sd[i] / np.sqrt(len(fr))

    with open(f"{args.out}_per_condition.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "residue", "contact_fraction_mean",
                    "std_between_replicas", "sem_between_replicas",
                    "n_replicas", "n_frames_total"])
        for i, cond in enumerate(conditions):
            for j, lab in enumerate(res_labels):
                w.writerow([cond, lab, mean[i, j], sd[i, j], sem[i, j],
                            len(samples[cond]), cond_frames[cond]])

    for name, mat in (("", mean), ("_sem", sem)):
        with open(f"{args.out}_matrix{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["condition"] + res_labels)
            for i, cond in enumerate(conditions):
                w.writerow([cond] + list(mat[i]))

    png = f"{args.out}_heatmap.png"
    plot_heatmap(mean, conditions, res_labels, png,
                 f"Contact fraction with '{args.target}' (cutoff {args.cutoff} Å)",
                 cmap=args.cmap, annotate=not args.no_annot,
                 sem=sem if args.annot_sem else None)
    print(f"Wrote: {args.out}_per_replica.csv, {args.out}_per_condition.csv, "
          f"{args.out}_matrix.csv, {args.out}_matrix_sem.csv, {png}")


if __name__ == "__main__":
    main()
