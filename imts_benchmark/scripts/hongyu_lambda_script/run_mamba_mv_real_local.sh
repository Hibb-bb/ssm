#!/usr/bin/env bash
# Local (no-Slurm) port of imts_benchmark/scripts/run_mamba_mv_real.sbatch.
# Trains Mamba-MV on real T-PatchGNN datasets (PhysioNet/Activity/USHCN).
#
# Differences vs colleague's sbatch:
#   * sequential bash (no Slurm / no array)
#   * env path  : /home/ubuntu/envs/mamba
#   * code dir  : /home/ubuntu/hongyu/ssm
#   * data root : /home/ubuntu/hongyu/ssm/tpatchgnn_data  (committed by colleague)
#   * log base  : /home/ubuntu/hongyu/ssm/output/mamba_mv_real
#   * online wandb (TSKing project), key from env (fail-fast if missing)
#
# Sweep grid (defaults, override via env):
#   DATASETS = (physionet activity ushcn)
#   DT_MODES = (replace)                        # winning dt_mode from sync/async HPO
#   SEEDS    = (1 2 3 4 5)
# = 3 * 1 * 5 = 15 sequential runs.
#
# To override (e.g. smoke test):
#   DATASETS="physionet" DT_MODES="replace" SEEDS="1" \
#     bash imts_benchmark/scripts/run_mamba_mv_real_local.sh
#
# Per-dataset grid_K (Nyquist-style: K = next_pow2(2 * p95(max_obs/var))):
#   physionet -> 256, activity -> 128, ushcn -> 128.
#
# Per-dataset (train_batch_size, accumulate_grad_batches) chosen to keep
# effective_batch = bs * accum = 128 (matches fair_defaults), while fitting on
# a single H100 (80 GB). PhysioNet's 41 vars at grid_K=256 OOMs at bs=128
# (smoke-test, 2026-04-25), so we drop bs and bump accum proportionally:
#   physionet -> bs=16, accum=8   (eff 128)
#   activity  -> bs=64, accum=2   (eff 128)
#   ushcn     -> bs=128, accum=1  (eff 128)
#
# Usage:
#   export WANDB_API_KEY=...
#   bash imts_benchmark/scripts/run_mamba_mv_real_local.sh
#   DRY_RUN=1     bash imts_benchmark/scripts/run_mamba_mv_real_local.sh
#   MAX_EPOCHS=2  bash imts_benchmark/scripts/run_mamba_mv_real_local.sh   # smoke

set -u
set -o pipefail

# ---------- environment ----------------------------------------------------
MAMBA_ENV=/home/ubuntu/envs/mamba
PYTHON=${MAMBA_ENV}/bin/python

REPO_DIR=/home/ubuntu/hongyu/ssm
CODE_DIR=${REPO_DIR}
DATA_ROOT=${REPO_DIR}/tpatchgnn_data
LOG_BASE=${REPO_DIR}/output/mamba_mv_real

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ---------- preflight ------------------------------------------------------
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: python not found at ${PYTHON}" >&2
  exit 1
fi
for d in physionet activity ushcn; do
  if [[ ! -f "${DATA_ROOT}/${d}/norm_stats.json" ]]; then
    echo "ERROR: ${DATA_ROOT}/${d}/norm_stats.json missing." >&2
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

# ---------- sweep grid (overridable via env) -------------------------------
read -ra DATASETS <<< "${DATASETS:-physionet activity ushcn}"
read -ra DT_MODES <<< "${DT_MODES:-replace}"
read -ra SEEDS    <<< "${SEEDS:-1 2 3 4 5}"

mkdir -p "${LOG_BASE}"
SUMMARY_CSV=${LOG_BASE}/real_summary.csv
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "dataset,dt_mode,seed,grid_K,exit_code,wall_sec,output_dir" > "${SUMMARY_CSV}"
fi

# ---------- run loop -------------------------------------------------------
cd "${CODE_DIR}"
SWEEP_START=$(date +%s)
TOTAL=$(( ${#DATASETS[@]} * ${#DT_MODES[@]} * ${#SEEDS[@]} ))
RUN_IDX=0

for DS in "${DATASETS[@]}"; do
  case ${DS} in
    physionet) GRID_K=256; BS=16;  ACCUM=8 ;;
    activity)  GRID_K=128; BS=64;  ACCUM=2 ;;
    ushcn)     GRID_K=128; BS=128; ACCUM=1 ;;
    *)         GRID_K=128; BS=128; ACCUM=1 ;;
  esac

  for DT_MODE in "${DT_MODES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      RUN_IDX=$(( RUN_IDX + 1 ))
      OUTPUT_DIR=${LOG_BASE}/${DS}/${DT_MODE}/seed${SEED}
      mkdir -p "${OUTPUT_DIR}"
      RUN_LOG=${OUTPUT_DIR}/run.log
      RUN_NAME="real_mv_local_${DS}_${DT_MODE}_seed${SEED}"

      echo "=========================================="
      echo "[${RUN_IDX}/${TOTAL}] ds=${DS} dt_mode=${DT_MODE} seed=${SEED} grid_K=${GRID_K} bs=${BS} accum=${ACCUM} (eff_bs=$(( BS * ACCUM )))"
      echo "  output_dir: ${OUTPUT_DIR}"
      echo "  run_name:   ${RUN_NAME}"
      echo "  start:      $(date)"

      CMD=(
        "${PYTHON}" -m imts_benchmark.mamba_mv.train_mv
        --data_root "${DATA_ROOT}"
        --regime "${DS}"
        --auto_meta
        --dt_mode "${DT_MODE}"
        --grid_K "${GRID_K}"
        --seed "${SEED}"
        --output_dir "${OUTPUT_DIR}"
        --train_batch_size "${BS}"
        --accumulate_grad_batches "${ACCUM}"
        --use_wandb
        --wandb_project "TSKing"
        --wandb_run_name "${RUN_NAME}"
      )
      # Optional epoch cap for smoke testing (e.g. MAX_EPOCHS=2).
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
      echo "${DS},${DT_MODE},${SEED},${GRID_K},${EXIT_CODE},${WALL},${OUTPUT_DIR}" >> "${SUMMARY_CSV}"
    done
  done
done

SWEEP_END=$(date +%s)
TOTAL_WALL=$(( SWEEP_END - SWEEP_START ))
echo ""
echo "=========================================="
echo "Real-data sweep done. Total wall: ${TOTAL_WALL}s ($(( TOTAL_WALL / 60 ))m)."
echo "Summary CSV: ${SUMMARY_CSV}"
