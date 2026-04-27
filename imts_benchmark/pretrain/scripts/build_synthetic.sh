#!/usr/bin/env bash
# Generate the two synthetic Arrow IPC files used by the pretraining
# pipeline. Defaults match plan §5 (N=100k each, jitter + drop_prob set
# so the files contain real irregularity even before our augmentations).
#
# Override sizes / paths via env vars:
#   N_CHRONOS=20000 N_KERNEL=20000 ./build_synthetic.sh
#   PYTHON_BIN=/home/ubuntu/envs/mamba/bin/python ./build_synthetic.sh
#   OUT_DIR=/some/other/path ./build_synthetic.sh

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/envs/mamba/bin/python}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PRETRAIN_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
SYNTH_DIR="$( cd "$PRETRAIN_DIR/../synthetic_data" && pwd )"
OUT_DIR="${OUT_DIR:-$SYNTH_DIR}"

N_CHRONOS="${N_CHRONOS:-100000}"
N_KERNEL="${N_KERNEL:-100000}"
N_JOBS="${N_JOBS:--1}"

echo "[build_synthetic] python : $PYTHON_BIN"
echo "[build_synthetic] script : $SYNTH_DIR"
echo "[build_synthetic] out    : $OUT_DIR"
echo "[build_synthetic] N_CHRONOS=$N_CHRONOS  N_KERNEL=$N_KERNEL  N_JOBS=$N_JOBS"

mkdir -p "$OUT_DIR"

CHRONOS_OUT="$OUT_DIR/chronos2-synth.arrow"
KERNEL_OUT="$OUT_DIR/kernelsynth-irregular.arrow"

# ---- chronos2_synth ------------------------------------------------------
# jitter-std=600s ~10 minute jitter on hourly nominal step.
# drop-prob=0.20 leaves ~80% of points per variate (irregular but dense
# enough that the per-variate Mamba sees enough context).
echo "==== chronos2_synth ===="
"$PYTHON_BIN" "$SYNTH_DIR/chronos2_synth.py" \
    -N "$N_CHRONOS" \
    --length 1024 \
    --max-variates 5 \
    --p-univariate 0.3 \
    --jitter-std 600 \
    --drop-prob 0.20 \
    --batch-size 500 \
    --n-jobs "$N_JOBS" \
    -o "$CHRONOS_OUT"

# ---- kernelsynth_irregular ----------------------------------------------
# Same irregularity recipe.
echo "==== kernelsynth_irregular ===="
"$PYTHON_BIN" "$SYNTH_DIR/kernelsynth_irregular.py" \
    -N "$N_KERNEL" \
    -J 5 \
    --jitter-std 600 \
    --drop-prob 0.20 \
    --batch-size 1000 \
    --n-jobs "$N_JOBS" \
    -o "$KERNEL_OUT"

echo "[build_synthetic] done. files:"
ls -lh "$CHRONOS_OUT" "$KERNEL_OUT"
