#!/usr/bin/env bash
# Local (no-Slurm) WINNER confirmation for Mamba-MV Phase 2 (sync, dense) data.
#
# Mirrors colleague's imts_benchmark/scripts/run_mamba_mv_winner_phase3.sbatch
# but adapted for:
#   * sync data (data_correct/ from generate_multivariate_sinusodial_data.py)
#   * single local H100 (no sbatch / no array)
#   * online wandb in the shared TSKing project
#
# Confirms HPO winners across 3 dt_modes × 5 seeds = 15 sequential runs.
# Reads per-dt_mode winner HPs from
#   ${REPO_DIR}/output/mamba_mv_hpo_phase2_sync/winners_by_mode.json
# (same JSON schema as colleague's winners_by_mode.json).
#
# Run-name convention: hpo_p2_sync_winner_${TAG}_seed${SEED}, distinct from
#   * colleague's hpo_p3_winner_*  (Phase 3, async)
#   * colleague's hpo_p42_winner_* (Phase 4-2, gap_random)
#
# Usage:
#   export WANDB_API_KEY=...   # required, fail-fast otherwise
#   bash imts_benchmark/scripts/run_mamba_mv_winner_phase2_local.sh
#   DRY_RUN=1 bash imts_benchmark/scripts/run_mamba_mv_winner_phase2_local.sh

set -u
set -o pipefail

# ---------- environment ----------------------------------------------------
MAMBA_ENV=/home/ubuntu/envs/mamba
PYTHON=${MAMBA_ENV}/bin/python

REPO_DIR=/home/ubuntu/hongyu/ssm
CODE_DIR=${REPO_DIR}
DATA_ROOT=${REPO_DIR}/data_correct                       # sync data
LOG_BASE=${REPO_DIR}/output/mamba_mv_winner_phase2_sync
WINNERS_JSON=${REPO_DIR}/output/mamba_mv_hpo_phase2_sync/winners_by_mode.json

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ---------- preflight: env, data, winners JSON, wandb key -----------------
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: python not found at ${PYTHON}" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}/multisin_high_irreg" ]]; then
  echo "ERROR: ${DATA_ROOT}/multisin_high_irreg missing." >&2
  echo "       Run: ${PYTHON} ${REPO_DIR}/generate_multivariate_sinusodial_data.py --output_root ${DATA_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${WINNERS_JSON}" ]]; then
  echo "ERROR: ${WINNERS_JSON} not found." >&2
  exit 1
fi

# wandb online: enforce a clean API key just like colleague's sbatch.
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is not set. Run: export WANDB_API_KEY=... before this script." >&2
  exit 1
fi
_KEY_LEN=${#WANDB_API_KEY}
_KEY_CLEAN_LEN=$(printf '%s' "${WANDB_API_KEY}" | tr -cd 'A-Za-z0-9_' | wc -c)
if [[ "${_KEY_LEN}" -ne "${_KEY_CLEAN_LEN}" ]] || [[ "${_KEY_LEN}" -lt 30 ]]; then
  echo "ERROR: WANDB_API_KEY appears malformed (len=${_KEY_LEN}, clean-len=${_KEY_CLEAN_LEN})." >&2
  exit 1
fi
export WANDB_MODE=online
unset WANDB_SILENT 2>/dev/null || true

# ---------- sweep grid -----------------------------------------------------
REGIME=multisin_high_irreg
DT_MODES=("learned" "replace" "concat")
SEEDS=(1 2 3 4 5)

mkdir -p "${LOG_BASE}"
SUMMARY_CSV=${LOG_BASE}/winner_summary.csv
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "dt_mode,lr,train_batch_size,seed,regime,exit_code,wall_sec,output_dir" > "${SUMMARY_CSV}"
fi

# ---------- run loop -------------------------------------------------------
cd "${CODE_DIR}"
SWEEP_START=$(date +%s)
TOTAL=$(( ${#DT_MODES[@]} * ${#SEEDS[@]} ))
RUN_IDX=0

for DT_MODE in "${DT_MODES[@]}"; do
  # Pull winner HPs for this dt_mode (matches colleague's parsing convention).
  LR=$("${PYTHON}" -c "import json; print(json.load(open('${WINNERS_JSON}'))['${DT_MODE}']['lr'])")
  BS=$("${PYTHON}" -c "import json; print(json.load(open('${WINNERS_JSON}'))['${DT_MODE}']['train_batch_size'])")

  for SEED in "${SEEDS[@]}"; do
    RUN_IDX=$(( RUN_IDX + 1 ))
    TAG="dt-${DT_MODE}_lr-${LR}_bs-${BS}"
    OUTPUT_DIR=${LOG_BASE}/${REGIME}/${TAG}/seed${SEED}
    mkdir -p "${OUTPUT_DIR}"
    RUN_LOG=${OUTPUT_DIR}/run.log
    RUN_NAME="hpo_p2_sync_winner_${TAG}_seed${SEED}"

    echo "=========================================="
    echo "[${RUN_IDX}/${TOTAL}] regime=${REGIME} dt_mode=${DT_MODE} lr=${LR} bs=${BS} seed=${SEED}"
    echo "  output_dir: ${OUTPUT_DIR}"
    echo "  run_name:   ${RUN_NAME}"
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
      --wandb_project "TSKing"
      --wandb_run_name "${RUN_NAME}"
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

SWEEP_END=$(date +%s)
TOTAL_WALL=$(( SWEEP_END - SWEEP_START ))
echo ""
echo "=========================================="
echo "Winner confirm done. Total wall: ${TOTAL_WALL}s ($(( TOTAL_WALL / 60 ))m)."
echo "Summary CSV: ${SUMMARY_CSV}"
