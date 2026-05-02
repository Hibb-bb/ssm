#!/usr/bin/env bash
# Local (no-Slurm) port of imts_benchmark/scripts/run_mamba_mv_matched_imm_delta_x86.sbatch.
# Trains Mamba-MV on TIME-IMM datasets under the IMM-TSF "Without Textual Data"
# baseline protocol, so each number is directly comparable to paper Tables 3-11
# 'Uni' rows.
#
# Differences vs colleague's Delta sbatch:
#   * sequential bash (no Slurm / no --array)
#   * env path  : /home/ubuntu/envs/mamba
#   * code dir  : /home/ubuntu/hongyu/ssm
#   * data root : /home/ubuntu/hongyu/ssm/time_imm_data  (committed by colleague,
#                 produced by data_pipeline/import_timeimm.py)
#   * log base  : /home/ubuntu/hongyu/ssm/output/mamba_mv_matched_imm
#   * online wandb (TSKing project), key from env (fail-fast if missing)
#
# Matched hyperparameters (from _imm_tsf_repo/main.py + main_all.py):
#   lr=1e-3, weight_decay=0.01, train_batch_size=8, accumulate_grad_batches=1,
#   max_epochs=1000, patience=3, num_warmup_steps=0, num_training_steps=1e9,
#   precision=32-true (AMP off).
#   dt_mode=concat (PhysioNet HPO winner; architectural choice, not an HP).
#
# Sweep grid (defaults, override via env):
#   DATASETS = (imm_gdelt imm_repohealth imm_fnspid imm_clustertrace
#               imm_studentlife imm_ilinet imm_cesnet imm_epa_air)
#   DT_MODE  = concat                            # single dt_mode
#   SEED     = 1                                 # single seed (matches paper)
# = 8 * 1 * 1 = 8 sequential runs.
#
# Override examples:
#   DATASETS="imm_fnspid"       bash ...run_mamba_mv_matched_imm_local.sh
#   DT_MODE=learned SEED=2      bash ...run_mamba_mv_matched_imm_local.sh
#   DRY_RUN=1                   bash ...run_mamba_mv_matched_imm_local.sh
#   MAX_EPOCHS=5                bash ...run_mamba_mv_matched_imm_local.sh   # smoke
#
# To run detached (recommended for the full 8-dataset sweep, ~hours):
#   screen -dmS mv_matched bash -lc 'export WANDB_API_KEY=...; \
#       bash /home/ubuntu/hongyu/ssm/imts_benchmark/scripts/hongyu_lambda_script/run_mamba_mv_matched_imm_local.sh'

set -u
set -o pipefail

# ---------- environment ----------------------------------------------------
MAMBA_ENV=${MAMBA_ENV:-/home/ubuntu/envs/mamba}
PYTHON=${MAMBA_ENV}/bin/python

REPO_DIR=/home/ubuntu/hongyu/ssm
CODE_DIR=${REPO_DIR}
DATA_ROOT=${REPO_DIR}/time_imm_data
LOG_BASE=${REPO_DIR}/output/mamba_mv_matched_imm

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ---------- preflight ------------------------------------------------------
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: python not found at ${PYTHON}" >&2
  exit 1
fi

DEFAULT_DATASETS=(
  imm_gdelt
  imm_repohealth
  imm_fnspid
  imm_clustertrace
  imm_studentlife
  imm_ilinet
  imm_cesnet
  imm_epa_air
)

read -ra DATASETS <<< "${DATASETS:-${DEFAULT_DATASETS[*]}}"
DT_MODE=${DT_MODE:-concat}
SEED=${SEED:-1}

for d in "${DATASETS[@]}"; do
  if [[ ! -f "${DATA_ROOT}/${d}/norm_stats.json" ]]; then
    echo "ERROR: ${DATA_ROOT}/${d}/norm_stats.json missing." >&2
    echo "       Run data_pipeline/import_timeimm.py first, or pull latest mamba-sinusoidal-multivariate." >&2
    exit 1
  fi
done

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

mkdir -p "${LOG_BASE}"
SUMMARY_CSV=${LOG_BASE}/matched_imm_summary.csv
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "dataset,dt_mode,seed,exit_code,wall_sec,output_dir" > "${SUMMARY_CSV}"
fi

# ---------- run loop -------------------------------------------------------
cd "${CODE_DIR}"
SWEEP_START=$(date +%s)
TOTAL=${#DATASETS[@]}
RUN_IDX=0

for DS in "${DATASETS[@]}"; do
  RUN_IDX=$(( RUN_IDX + 1 ))
  OUTPUT_DIR=${LOG_BASE}/${DS}/${DT_MODE}/seed${SEED}
  mkdir -p "${OUTPUT_DIR}"
  RUN_LOG=${OUTPUT_DIR}/run.log
  RUN_NAME="matched_mv_local_${DS}_${DT_MODE}_seed${SEED}"

  echo "=========================================="
  echo "[${RUN_IDX}/${TOTAL}] ds=${DS} dt_mode=${DT_MODE} seed=${SEED}"
  echo "  protocol:   IMM-TSF matched (lr=1e-3 bs=8 patience=3 fp32, no scheduler)"
  echo "  output_dir: ${OUTPUT_DIR}"
  echo "  run_name:   ${RUN_NAME}"
  echo "  start:      $(date)"

  # All non-trivial flags pinned to _imm_tsf_repo/main_all.py + main.py defaults
  # so the resulting test MSE is directly comparable to paper Tables 3-11
  # 'Without Textual Data' rows.
  CMD=(
    "${PYTHON}" -m imts_benchmark.mamba_mv.train_mv
    --data_root "${DATA_ROOT}"
    --regime "${DS}"
    --auto_meta
    --dt_mode "${DT_MODE}"
    --grid_K 256
    --lr 1e-3
    --weight_decay 0.01
    --train_batch_size 8
    --accumulate_grad_batches 1
    --max_epochs 1000
    --patience 3
    --num_warmup_steps 0
    --num_training_steps 1000000000
    --precision 32-true
    --seed "${SEED}"
    --output_dir "${OUTPUT_DIR}"
    --use_wandb
    --wandb_project "TSKing"
    --wandb_run_name "${RUN_NAME}"
  )
  # Optional epoch cap for smoke testing (e.g. MAX_EPOCHS=5).
  if [[ -n "${MAX_EPOCHS:-}" ]]; then
    CMD+=(--max_epochs "${MAX_EPOCHS}")
  fi

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
  echo "${DS},${DT_MODE},${SEED},${EXIT_CODE},${WALL},${OUTPUT_DIR}" >> "${SUMMARY_CSV}"
done

SWEEP_END=$(date +%s)
TOTAL_WALL=$(( SWEEP_END - SWEEP_START ))
echo ""
echo "=========================================="
echo "Matched-IMM sweep done. Total wall: ${TOTAL_WALL}s ($(( TOTAL_WALL / 60 ))m)."
echo "Summary CSV: ${SUMMARY_CSV}"
