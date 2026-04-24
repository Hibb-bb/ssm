#!/usr/bin/env bash
# Local (no-Slurm) hyperparameter sweep for Mamba-MV, Phase 3 (async dense).
#
# Adapted from run_mamba_mv_phase3.sbatch for a single H100 box.
# Sweep: regime=multisin_high_irreg, seed=1, dt_mode in {learned, replace},
#        lr in {1e-4, 5e-4, 2e-3}, train_batch_size in {64, 128, 256}.
# Total: 2 * 3 * 3 = 18 sequential runs.
#
# Usage:
#   bash imts_benchmark/scripts/run_mamba_mv_phase3_local_sweep.sh
#   SKIP_DATA_GEN=1 bash imts_benchmark/scripts/run_mamba_mv_phase3_local_sweep.sh
#   DRY_RUN=1      bash imts_benchmark/scripts/run_mamba_mv_phase3_local_sweep.sh

set -u  # do NOT set -e: a single failed run should not kill the whole sweep.
set -o pipefail

# ---------- environment ----------------------------------------------------
MAMBA_ENV=/home/ubuntu/envs/mamba
PYTHON=${MAMBA_ENV}/bin/python

REPO_DIR=/home/ubuntu/hongyu/ssm
CODE_DIR=${REPO_DIR}                                    # train_mv is run as `-m imts_benchmark.mamba_mv.train_mv`
DATA_ROOT=${REPO_DIR}/data_correct_async
LOG_BASE=${REPO_DIR}/output/mamba_mv_sweep

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# train_mv.py registers a LearningRateMonitor callback unconditionally, which
# requires the trainer to have a logger. We satisfy that with an OFFLINE wandb
# logger (no account / no network needed). Per-run dirs land under each run's
# output_dir/wandb/ and can be sync'd later with `wandb sync` if desired.
export WANDB_MODE=offline
export WANDB_SILENT=true

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: python not found at ${PYTHON}" >&2
  exit 1
fi

# ---------- data generation (run once) -------------------------------------
if [[ "${SKIP_DATA_GEN:-0}" == "0" ]]; then
  if [[ -d "${DATA_ROOT}/multisin_high_irreg" && -f "${DATA_ROOT}/multisin_stats.json" ]]; then
    echo "[data] ${DATA_ROOT} already populated, skipping generation."
  else
    echo "[data] generating async multisin dataset under ${DATA_ROOT}"
    cd "${REPO_DIR}"
    "${PYTHON}" generate_multivariate_sinusodial_data_async.py \
      --output_root "${DATA_ROOT}"
  fi
else
  echo "[data] SKIP_DATA_GEN=1, skipping generation."
fi

# ---------- sweep grid -----------------------------------------------------
REGIME=multisin_high_irreg
SEED=1
DT_MODES=("learned" "replace")
LRS=("1e-4" "5e-4" "2e-3")
BATCH_SIZES=(64 128 256)

mkdir -p "${LOG_BASE}"
SUMMARY_CSV=${LOG_BASE}/sweep_summary.csv
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "dt_mode,lr,train_batch_size,seed,regime,exit_code,wall_sec,output_dir" > "${SUMMARY_CSV}"
fi

# ---------- run loop -------------------------------------------------------
cd "${CODE_DIR}"
SWEEP_START=$(date +%s)
RUN_IDX=0
TOTAL=$(( ${#DT_MODES[@]} * ${#LRS[@]} * ${#BATCH_SIZES[@]} ))

for DT_MODE in "${DT_MODES[@]}"; do
  for LR in "${LRS[@]}"; do
    for BS in "${BATCH_SIZES[@]}"; do
      RUN_IDX=$(( RUN_IDX + 1 ))
      TAG="dt-${DT_MODE}_lr-${LR}_bs-${BS}_seed-${SEED}"
      OUTPUT_DIR=${LOG_BASE}/${REGIME}/${TAG}
      mkdir -p "${OUTPUT_DIR}"
      RUN_LOG=${OUTPUT_DIR}/run.log

      echo "=========================================="
      echo "[${RUN_IDX}/${TOTAL}] regime=${REGIME} dt_mode=${DT_MODE} lr=${LR} bs=${BS} seed=${SEED}"
      echo "  output_dir: ${OUTPUT_DIR}"
      echo "  log:        ${RUN_LOG}"
      echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
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
        --wandb_project "mamba_mv_local_sweep"
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
  done
done

SWEEP_END=$(date +%s)
TOTAL_WALL=$(( SWEEP_END - SWEEP_START ))
echo ""
echo "=========================================="
echo "Sweep done. Total wall: ${TOTAL_WALL}s ($(( TOTAL_WALL / 60 ))m)."
echo "Summary CSV: ${SUMMARY_CSV}"
