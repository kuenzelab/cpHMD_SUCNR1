import numpy as np
import matplotlib.pyplot as plt
import re
import logging

logging.basicConfig(
    format="%(asctime)s %(message)s",
    level=logging.INFO,
)

# --- SETTINGS ---
LAMBDA_THRESHOLD = 0.8  # protonated if lambda > threshold
LAMBDA_THRESHOLD_deporotonated = 0.2

# --- PARSE FILE ---
def parse_xvg(filename):
    legends = []
    data = []

    with open(filename) as f:
        for line in f:
            # collect legend entries
            if line.startswith("@ s"):
                match = re.search(r'legend "(.*?)"', line)
                if match:
                    legends.append(match.group(1))

            # skip comments/meta
            elif line.startswith(("#", "@")):
                continue

            # numeric data
            else:
                data.append([float(x) for x in line.split()])

    data = np.array(data)
    time = data[:, 0]
    lambdas = data[:, 1:]

    return time, lambdas, legends


# --- PROTONATION STATE LOGIC ---
def lambda_to_state(lambda_values, residue_name):
    """
    Convert lambda values into discrete protonation states.
    """

    # Histidine (3 lambda coordinates per residue)
    if "HSPT" in residue_name:
        # Expect triplets later → handled separately
        return lambda_values

    # Standard titratable residues (1 lambda)
    return lambda_values


def process_histidines(lambdas, legends):
    """
    Convert histidine triplets into states:
    HID / HIE / HIP
    """
    states = {}
    i = 0

    while i < len(legends):
        if "HSPT" in legends[i]:
            # take 3 lambda coordinates
            triplet = lambdas[:, i:i+3]

            # assign state based on largest lambda
            state = np.argmax(triplet, axis=1)
            np.savetxt("Titration_group_His_{}_State.txt", state)
            logging.info(np.bincount(state).argmax())
            values, counts = np.unique(state[10000:], return_counts=True)
            percents = counts/sum(counts)
            logging.info("Index {}: states: {}, counts: {}, percents: {}".format(i, values, counts, percents))
            # map: 0=HIP, 1=HID, 2=HIE
            states[i] = state
            i += 3
        else:
            i += 1

    return states


# --- MAIN ---
filename = "cph2.xvg"

time, lambdas, legends = parse_xvg(filename)

# standard residues
states = []
labels = []

i = 0
while i < len(legends):
    name = legends[i]

    if "HSPT" in name:
        # histidine handled separately
        i += 3
        continue

    state = lambda_to_state(lambdas[:, i], name)
    dp = 0
    p = 0
    for frame in state[10000:]:
        if frame < 0.2:
            dp += 1
        elif frame > 0.8:
            p += 1
        percentsdp = dp/state[10000:].size
        percentsp = p/state[10000:].size
    np.savetxt("Titration_group_{}_State.txt".format(i), states)
    #values, counts = np.unique(state, return_counts=True)
    #percents = counts/sum(counts)
    logging.info("Index {}: states: 2, counts: {}, {}, percents: {}, {}".format(i, dp, p, percentsdp, percentsp))
    states.append(state)
    labels.append(name)

    i += 1

states = np.array(states)

# histidine states
his_states = process_histidines(lambdas, legends)


# --- PLOTTING ---
for i, state in enumerate(states):# --- PLOTTING ---
    plt.figure(figsize=(12, 3))
    plt.ylim(-0.1,1.1)
    plt.xlim(0,time[-1])
    plt.hlines(0.8, 0, time[-1], colors="green", label="deprotonated")
    plt.hlines(0.2, 0, time[-1], colors="orange", label="protonated")
    plt.plot(time, state)
    plt.xlabel("Time (ps)")
    plt.ylabel("Protonation state")
    label_here=labels[i]
    plt.title("{}".format(label_here))
    plt.tight_layout()
    plt.legend()
    plt.savefig("Titration_group_{}_State.svg".format(i))
    plt.close()


# plot histidines
#offset = len(states)
for idx, state in his_states.items():
    plt.figure(figsize=(12, 3))
    plt.hlines(0, 0, time[-1], colors="green", label="HSP")
    plt.hlines(1, 0, time[-1], colors="orange", label="HSD")
    plt.hlines(2, 0, time[-1], colors="pink", label="HSE")
    plt.ylim(-0.3,2.3)
    plt.xlim(0,time[-1])
    plt.plot(time, state, linestyle='--')
    plt.xlabel("Time (ps)")
    plt.ylabel("Protonation state")
    plt.title("Histidine {}".format(idx))
    plt.tight_layout()
    plt.legend()
    plt.savefig("His_{}_State.svg".format(idx))
    plt.close()
    #offset += 1

#plt.xlabel("Time (ps)")
#plt.ylabel("Protonation state (offset for clarity)")
#plt.title("Protonation States over Time")
#plt.tight_layout()

#plt.show()
