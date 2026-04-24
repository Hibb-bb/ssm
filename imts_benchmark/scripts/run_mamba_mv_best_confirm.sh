#!/usr/bin/env bash
# Multi-seed confirmation of the winning config from
# run_mamba_mv_phase3_local_sweep.sh.
#
# Winning config (test_mse=0.013, R^2=0.70 at seed=1):
#   regime=multisin_high_irreg, dt_mode=replace, lr=2e-3, train_batch_size=64
#
# Re-runs that config across 5 seeds (1..5) so we can report mean ± std.

set -u
set -o pipefail

# ---------- environment ----------------------------------------------------
MAMBA_ENV=/home/ubuntu/envs/mamba
PYTHON=${MAMBA_ENV}/bin/python

REPO_DIR=/home/ubuntu/hongyu/ssm
CODE_DIR=${REPO_DIR}
DATA_ROOT=${REPO_DIR}/data_correct_async
LOG_BASE=${REPO_DIR}/output/mamba_mv_best_confirm

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_MODE=offline
export WANDB_SILENT=true

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: python not found at ${PYTHON}" >&2
  exit 1
fi

# ---------- fixed (winning) config ----------------------------------------
REGIME=multisin_high_irreg
DT_MODE=replace
LR=2e-3
BS=64
SEEDS=(1 2 3 4 5)

mkdir -p "${LOG_BASE}"
SUMMARY_CSV=${LOG_BASE}/confirm_summary.csv
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "dt_mode,lr,train_batch_size,seed,regime,exit_code,wall_sec,output_dir" > "${SUMMARY_CSV}"
fi

# ---------- run loop -------------------------------------------------------
cd "${CODE_DIR}"
SWEEP_START=$(date +%s)
TOTAL=${#SEEDS[@]}
RUN_IDX=0

for SEED in "${SEEDS[@]}"; do
  RUN_IDX=$(( RUN_IDX + 1 ))
  TAG="dt-${DT_MODE}_lr-${LR}_bs-${BS}_seed-${SEED}"
  OUTPUT_DIR=${LOG_BASE}/${REGIME}/${TAG}
  mkdir -p "${OUTPUT_DIR}"
  RUN_LOG=${OUTPUT_DIR}/run.log

  echo "=========================================="
  echo "[${RUN_IDX}/${TOTAL}] regime=${REGIME} dt_mode=${DT_MODE} lr=${LR} bs=${BS} seed=${SEED}"
  echo "  output_dir: ${OUTPUT_DIR}"
  echo "  start:      $(date)"

  CMD=(
    "${PYTHON}" -m imts_benchmark.mamba_mv.train_mv
    --regime "${REGIME}"
    --dt_mode "${DT_MODE}"
    --seed "${SEED}"
    --data_root "${DATA_ROOT}"
    --output_dir "${OUTPUT_DIR}"
    --lr "${LR}"
    --train_batch_size "${BS}"
    --use_wandb
    --wandb_project "mamba_mv_best_confirm"
    --wandb_run_name "${TAG}"
  )

  export WANDB_DIR=${OUTPUT_DIR}

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "  DRY_RUN: ${CMD[*]}"
    continue
  fi

  RUN_START=$(date +%s)
  "${CMD[@]}" 2>&1 | tee "${RUN_LOG}"
  EXIT_CODE=${PIPESTATUS[0]}
  RUN_END=$(date +%s)
  WALL=$(( RUN_END - RUN_START ))

  echo "  exit_code:  ${EXIT_CODE}"
  echo "  wall:       ${WALL}s ($(( WALL / 60 ))m $(( WALL % 60 ))s)"
  echo "${DT_MODE},${LR},${BS},${SEED},${REGIME},${EXIT_CODE},${WALL},${OUTPUT_DIR}" >> "${SUMMARY_CSV}"
done

SWEEP_END=$(date +%s)
TOTAL_WALL=$(( SWEEP_END - SWEEP_START ))
echo ""
echo "=========================================="
echo "Confirm done. Total wall: ${TOTAL_WALL}s ($(( TOTAL_WALL / 60 ))m)."
echo "Summary CSV: ${SUMMARY_CSV}"
