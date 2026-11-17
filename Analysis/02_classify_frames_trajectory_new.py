#!/usr/bin/env python3

import os
import re
import csv
from collections import Counter, defaultdict

import yaml
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
from MDAnalysis.coordinates.XTC import XTCWriter


# ============================================================
# Configuration
# ============================================================

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

TOPOLOGY = cfg["input"]["topology"]
TRAJECTORY = cfg["input"]["trajectory"]
LAMBDA_DIR = cfg["input"]["lambda_directory"]

EQUIL_PS = cfg["analysis"]["equilibration_ns"] * 1000.0
INTERVAL_PS = cfg["analysis"]["frame_interval_ps"]

LIGAND_SELECTION = cfg["ligand"]["selection"]

CONTACT_CUTOFF = (
    cfg["ligand"]["contact_cutoff_nm"] * 10.0
)

BINDING_SITE = set(
    int(x) for x in cfg["binding_site_residues"]
)

BINARY_STATES = cfg["binary_states"]
HIS_STATES = {
    int(k): v
    for k, v in cfg["histidine_states"].items()
}

HYDROGEN_RULES = cfg["hydrogens"]

OUTPUT = cfg["output"]["directory"]

LAMBDA_MAPPING = "lambda_mapping.tsv"


# ============================================================
# Directories
# ============================================================

LAMBDA_DIR = os.path.abspath(LAMBDA_DIR)

REPORT_DIR = os.path.join(
    OUTPUT,
    "reports"
)

GROUP_DIR = os.path.join(
    OUTPUT,
    "groups"
)

REJECT_DIR = os.path.join(
    OUTPUT,
    "rejected"
)

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(GROUP_DIR, exist_ok=True)
os.makedirs(REJECT_DIR, exist_ok=True)


# ============================================================
# Read mapping
# ============================================================

def read_mapping(filename):

    residues = defaultdict(list)

    with open(filename) as f:

        header = f.readline().strip().split()

        for line in f:

            line = line.strip()

            if not line:
                continue

            fields = line.split()

            if len(fields) < 5:
                continue

            resname = fields[0]
            state = int(fields[1])
            resid = int(fields[2])
            chain = fields[3]
            coordinate_file = fields[4]

            residues[resid].append({
                "resname": resname,
                "state": state,
                "chain": chain,
                "file": coordinate_file,
            })

    return residues


residue_mapping = read_mapping(
    LAMBDA_MAPPING
)
print(residue_mapping)


# ============================================================
# Print mapping
# ============================================================

print()
print("CpHMD residue mapping")
print("=====================")

for resid in sorted(residue_mapping):

    entries = residue_mapping[resid]

    print(
        f"{resid:4d} "
        f"{entries[0]['resname']:5s} "
        f"{len(entries)} lambda coordinate(s)"
    )


# ============================================================
# Load Universe
# ============================================================

print()
print("Loading trajectory...")

u = mda.Universe(
    "../equil_out.pdb",
    TRAJECTORY
)
print("\nHydrogen atoms in titratable residues:")
print("---------------------------------------")

for resid in sorted(residue_mapping):

    atoms = u.select_atoms(
        f"resid {resid}"
    )

    hydrogen_names = [
        atom.name
        for atom in atoms
        if atom.name.upper().startswith("H")
    ]

    print(
        f"{resid:4d} "
        f"{residue_mapping[resid][0]['resname']:5s} "
        f"{' '.join(hydrogen_names)}"
    )

protein = u.select_atoms(
    cfg["ligand"]["selection"]
    if False
    else "protein"
)

ligand = u.select_atoms(
    LIGAND_SELECTION
)

print(
    f"Protein atoms: {protein.n_atoms}"
)

print(
    f"Ligand atoms : {ligand.n_atoms}"
)


# ============================================================
# Read XVG
# ============================================================

def read_xvg(filename):

    times = []
    values = []

    with open(filename) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith("@") or line.startswith("#"):
                continue

            fields = line.split()

            try:
                nums = [float(x) for x in fields]
            except ValueError:
                continue

            if len(nums) < 2:
                continue

            times.append(nums[0])
            values.append(nums[1])

    return np.asarray(times), np.asarray(values)


# ============================================================
# Load lambda trajectories
# ============================================================

lambda_data = {}

for resid, entries in residue_mapping.items():

    lambda_data[resid] = []

    for entry in sorted(
        entries,
        key=lambda x: x["state"]
    ):

        filename = os.path.join(
            LAMBDA_DIR,
            entry["file"]
        )

        if not os.path.isfile(filename):

            raise FileNotFoundError(
                f"Missing lambda file: {filename}"
            )

        times, values = read_xvg(filename)

        lambda_data[resid].append({
            "state": entry["state"],
            "resname": entry["resname"],
            "times": times,
            "values": values,
        })


# ============================================================
# Verify lambda time axes
# ============================================================

reference_times = None

for resid in lambda_data:

    for coord in lambda_data[resid]:

        times = coord["times"]

        if reference_times is None:
            reference_times = times

        elif len(times) != len(reference_times):

            raise RuntimeError(
                "Lambda trajectories have different "
                "numbers of frames."
            )

        elif not np.allclose(
            times,
            reference_times
        ):

            raise RuntimeError(
                "Lambda trajectories have different "
                "time axes."
            )


lambda_times = reference_times

print()
print(
    f"Lambda trajectory: "
    f"{len(lambda_times)} frames"
)

print(
    f"Lambda time range: "
    f"{lambda_times[0]:.1f} - "
    f"{lambda_times[-1]:.1f} ps"
)


# ============================================================
# State assignment
# ============================================================

def nearest_lambda_index(time_ps):

    idx = np.searchsorted(
        lambda_times,
        time_ps
    )

    if idx == 0:
        return 0

    if idx >= len(lambda_times):
        return len(lambda_times) - 1

    before = idx - 1
    after = idx

    if abs(
        lambda_times[before] - time_ps
    ) <= abs(
        lambda_times[after] - time_ps
    ):
        return before

    return after


def state_from_lambda(resid, lambda_index):
    """
    Determine the protonation state of one titratable residue.

    Binary titratable residues:
        lambda < 0.5  -> protonated
        lambda >= 0.5 -> deprotonated

    Histidines:
        three lambda coordinates are compared;
        the largest lambda determines the state.
    """

    coords = lambda_data[resid]

    # Original CpHMD residue name from the mapping
    original_resname = residue_mapping[resid][0]["resname"]
    base = original_resname[:3]

    # --------------------------------------------------------
    # Histidine
    # --------------------------------------------------------

    if len(coords) == 3:

        values = np.array([
            c["values"][lambda_index]
            for c in coords
        ])

        max_index = int(np.argmax(values))

        state_number = max_index + 1

        if state_number not in HIS_STATES:
            raise RuntimeError(
                f"Invalid histidine state {state_number} "
                f"for residue {resid}"
            )

        state_name = HIS_STATES[state_number]

        return state_name, values

    # --------------------------------------------------------
    # Ordinary binary titratable residue
    # --------------------------------------------------------

    elif len(coords) == 1:

        value = coords[0]["values"][lambda_index]

        if base not in BINARY_STATES:
            raise RuntimeError(
                f"No binary-state definition for "
                f"residue type '{base}' "
                f"(residue {resid}, "
                f"original name {original_resname})"
            )

        if value > 0.5:

            state = BINARY_STATES[base]["deprotonated"]

        else:

            state = BINARY_STATES[base]["protonated"]

        return state, np.array([value])

    # --------------------------------------------------------
    # Unexpected number of coordinates
    # --------------------------------------------------------

    else:

        raise RuntimeError(
            f"Unexpected number of lambda coordinates "
            f"for residue {resid}: {len(coords)}"
        )


# ============================================================
# Determine dominant state
# ============================================================

print()
print("Determining dominant states...")
print(
    "Only post-equilibration frames are used."
)

dominant_counts = {}

for resid in sorted(residue_mapping):

    if resid in BINDING_SITE:
        continue

    states = []

    for i, time_ps in enumerate(
        lambda_times
    ):

        if time_ps < EQUIL_PS:
            continue

        state, values = state_from_lambda(
            resid,
            i
        )

        states.append(state)

    counts = Counter(states)

    dominant_state, dominant_count = (
        counts.most_common(1)[0]
    )

    dominant_counts[resid] = {
        "state": dominant_state,
        "counts": counts,
        "total": len(states),
    }


# ============================================================
# Write dominant-state report
# ============================================================

dominant_file = os.path.join(
    REPORT_DIR,
    "dominant_states.csv"
)

with open(
    dominant_file,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "resid",
        "original_resname",
        "dominant_state",
        "n_frames",
        "fraction",
        "state_counts",
    ])

    for resid in sorted(
        dominant_counts
    ):

        info = dominant_counts[resid]

        original = residue_mapping[
            resid
        ][0]["resname"]

        counts = info["counts"]

        writer.writerow([
            resid,
            original,
            info["state"],
            info["total"],
            counts[
                info["state"]
            ] / info["total"],
            ";".join(
                f"{state}:{n}"
                for state, n
                in sorted(counts.items())
            ),
        ])

print(
    "Dominant-state report:",
    dominant_file
)


# ============================================================
# Group topology and trajectory output
# ============================================================
def atom_should_be_removed(atom, states):
    resid = int(atom.resid)

    if resid not in states:
        return False

    original_resname = residue_mapping[resid][0]["resname"]
    base = original_resname[:3].upper()
    atom_name = atom.name.strip().upper()

    if base == "HSP":
        rules = HYDROGEN_RULES["HIS"][states[resid]]

    elif base in BINARY_STATES:
        state = states[resid]

        if state == BINARY_STATES[base]["protonated"]:
            state_type = "protonated"
        elif state == BINARY_STATES[base]["deprotonated"]:
            state_type = "deprotonated"
        else:
            raise RuntimeError(
                f"Invalid state {state} for {base}{resid}"
            )

        rules = HYDROGEN_RULES[base][state_type]

    else:
        return False

    return atom_name in {
        name.strip().upper()
        for name in rules.get("remove", [])
    }

def get_kept_atom_indices(universe, states):
    """Keep protein, ligand and POPC; remove only state-specific H atoms."""

    candidates = universe.select_atoms(
        "segid A or resname LIG or resname POPC"
    )

    keep = []

    for atom in candidates:
        if atom_should_be_removed(atom, states):
            continue

        keep.append(atom.index)

    return np.asarray(keep, dtype=int)


def write_group_topology(universe, states, atom_indices, filename):
    """Write one topology PDB for a protonation-state group."""
    atoms = universe.atoms[atom_indices]

    with open(filename, "w") as f:
        # Optional box information
        if universe.dimensions is not None:
            lx, ly, lz, alpha, beta, gamma = universe.dimensions
            if lx > 0 and ly > 0 and lz > 0:
                f.write(
                    f"CRYST1{lx:9.3f}{ly:9.3f}{lz:9.3f}"
                    f"{alpha:7.2f}{beta:7.2f}{gamma:7.2f}"
                    f" P 1           1\n"
                )

        for serial, atom in enumerate(atoms, 1):
            resid = int(atom.resid)
            resname = states.get(resid, atom.resname)

            chain = getattr(atom, "chainID", "") or getattr(
                atom, "segid", ""
            )
            chain = chain[:1]

            x, y, z = atom.position
            element = getattr(atom, "element", "") or ""

            f.write(
                f"ATOM  {serial:5d} "
                f"{atom.name:<4s} "
                f"{resname:>4s}"
                f"{chain:1s}"
                f"{resid:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"{1.00:6.2f}{0.00:6.2f}"
                f"          {element:>2s}\n"
            )

        f.write("END\n")


def report_removed_hydrogens(universe, states, group):

    removed = []

    for atom in universe.select_atoms("protein"):

        resid = int(atom.resid)

        if resid in states and atom_should_be_removed(atom, states):
            removed.append(
                (
                    resid,
                    atom.resname,
                    atom.name,
                    states[resid],
                )
            )

    if removed:
        print(f"\nHydrogens removed for group {group}:")

        for resid, resname, atomname, state in removed:
            print(
                f"  {resname}{resid:4d} "
                f"{atomname:<5s} -> state {state}"
            )

def close_group_writers(group_writers):
    for group, info in group_writers.items():
        info["writer"].close()
        print(
            f"Closed trajectory: {info['trajectory']} "
            f"({info['n_frames']} frames)"
        )


# ============================================================
# Contact analysis
# ============================================================

def contact_analysis():

    # Remove hydrogens from contact analysis
    protein_heavy = protein.select_atoms(
        "not element H"
    )

    ligand_heavy = ligand.select_atoms(
        "not element H"
    )

    distances = distance_array(
        protein_heavy.positions,
        ligand_heavy.positions,
        box=u.dimensions
    )

    contact_indices = np.argwhere(
        distances <= CONTACT_CUTOFF
    )

    residues = set()

    for pidx, lidx in contact_indices:

        atom = protein_heavy[pidx]

        residues.add(
            f"{atom.resname}{atom.resid}"
        )

    return (
        len(contact_indices) > 0,
        len(contact_indices),
        sorted(residues),
    )


# ============================================================
# Frame classification
# ============================================================

frame_report = os.path.join(
    REPORT_DIR,
    "frame_classification.csv"
)

binding_distribution = defaultdict(Counter)
report_rows = []
last_time = None

# One fixed topology + XTC trajectory per protonation-state group.
group_writers = {}
group_frame_counter = Counter()

print()
print("Classifying trajectory frames")
print("=============================")

try:
    for ts in u.trajectory:

        time_ps = ts.time

        if time_ps < EQUIL_PS:
            continue

        if (
            last_time is not None
            and time_ps - last_time < INTERVAL_PS
        ):
            continue

        last_time = time_ps

        lambda_idx = nearest_lambda_index(time_ps)

        # --------------------------------------------------------
        # Ligand contact
        # --------------------------------------------------------

        bound, n_contacts, contact_residues = contact_analysis()

        if not bound:
            report_rows.append([
                ts.frame,
                time_ps,
                "no_contact",
                n_contacts,
                ";".join(contact_residues),
                "",
            ])
            continue

        # --------------------------------------------------------
        # Determine states
        # --------------------------------------------------------

        states = {}

        for resid in sorted(residue_mapping):
            instantaneous_state, values = state_from_lambda(
                resid,
                lambda_idx
            )

            if resid in BINDING_SITE:
                states[resid] = instantaneous_state
                binding_distribution[resid][instantaneous_state] += 1
            else:
                states[resid] = dominant_counts[resid]["state"]

        # --------------------------------------------------------
        # Group key
        # --------------------------------------------------------

        group_parts = []

        for resid in sorted(BINDING_SITE):
            if resid not in residue_mapping:
                continue

            group_parts.append(
                f"{residue_mapping[resid][0]['resname'][:3]}"
                f"{resid}-{states[resid]}"
            )

        group_key = "__".join(group_parts)

        safe_group = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            group_key
        )

        group_dir = os.path.join(GROUP_DIR, safe_group)
        os.makedirs(group_dir, exist_ok=True)

        # --------------------------------------------------------
        # Create fixed topology and XTC writer for new group
        # --------------------------------------------------------

        if safe_group not in group_writers:

            topology_file = os.path.join(
                group_dir,
                "topology.pdb"
            )
            trajectory_file = os.path.join(
                group_dir,
                "trajectory.xtc"
            )

            atom_indices = get_kept_atom_indices(
                u,
                states
            )

            if len(atom_indices) == 0:
                raise RuntimeError(
                    f"No atoms left for group {group_key}"
                )

            write_group_topology(
                u,
                states,
                atom_indices,
                topology_file
            )

            report_removed_hydrogens(
                u,
                states,
                group_key
            )

            writer = XTCWriter(
                trajectory_file,
                n_atoms=len(atom_indices),
                precision=3
            )

            group_writers[safe_group] = {
                "writer": writer,
                "indices": atom_indices,
                "trajectory": trajectory_file,
                "topology": topology_file,
                "n_frames": 0,
            }

            print(
                f"Created group trajectory: {safe_group}"
            )

        # --------------------------------------------------------
        # Write current coordinates to group trajectory
        # --------------------------------------------------------

        info = group_writers[safe_group]

        info["writer"].write(
            u.atoms[info["indices"]]
        )
        info["n_frames"] += 1

        group_frame_counter[safe_group] += 1

        # --------------------------------------------------------
        # Report
        # --------------------------------------------------------

        report_rows.append([
            ts.frame,
            time_ps,
            "bound",
            n_contacts,
            ";".join(contact_residues),
            group_key,
        ])

        print(
            f"{time_ps/1000:10.3f} ns | "
            f"{n_contacts:4d} contacts | "
            f"{group_key}"
        )

finally:
    close_group_writers(group_writers)

# ============================================================
# Write frame report
# ============================================================

with open(
    frame_report,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "trajectory_frame",
        "time_ps",
        "ligand_status",
        "n_contacts",
        "contact_residues",
        "group",
    ])

    writer.writerows(
        report_rows
    )


# ============================================================
# Binding-site distribution
# ============================================================

binding_report = os.path.join(
    REPORT_DIR,
    "binding_site_state_distribution.csv"
)

with open(
    binding_report,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "resid",
        "state",
        "n_frames",
        "fraction",
    ])

    for resid in sorted(
        binding_distribution
    ):

        counts = binding_distribution[
            resid
        ]

        total = sum(
            counts.values()
        )

        for state, count in sorted(
            counts.items()
        ):

            writer.writerow([
                resid,
                state,
                count,
                count / total,
            ])


print()
print("==========================================")
print("Classification finished")
print("==========================================")
print(
    "Frame report:",
    frame_report
)
print(
    "Dominant states:",
    dominant_file
)
print(
    "Binding-site distribution:",
    binding_report
)
print()
print(
    "Groups:",
    len(group_frame_counter)
)

for group, n in sorted(
    group_frame_counter.items()
):

    print(
        f"{n:6d} frames  {group}"
    )