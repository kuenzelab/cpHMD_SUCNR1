#!/usr/bin/env bash

set -euo pipefail

TPR="${1:-md.tpr}"
EDR="${2:-md.edr}"

OUTDIR="output/lambda"

mkdir -p "$OUTDIR"

echo "=============================================="
echo "Extracting CpHMD lambda coordinates"
echo "=============================================="
echo "TPR : $TPR"
echo "EDR : $EDR"
echo

gmx cphmd \
    -s "$TPR" \
    -e "$EDR" \
    -o "$OUTDIR/cphmd" \
    -coordinate \
    -nodvdl \
    -novelocity \
    -numplot 1

echo
echo "Lambda files:"
ls -lh "$OUTDIR"