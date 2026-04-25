#!/usr/bin/env bash
# Local (no-Slurm) port of run_mamba_mv_hpo_replace_real.sbatch.
# HPO wave 1: Mamba-MV dt_mode=replace on real T-PatchGNN datasets, seed=1.
#
# Diff vs colleague's sbatch:
#   * sequential bash (no Slurm array)
#   * env path  : /home/ubuntu/envs/mamba
#   * code dir  : /home/ubuntu/hongyu/ssm
#   * data root : /home/ubuntu/hongyu/ssm/tpatchgnn_data
#   * log base  : /home/ubuntu/hongyu/ssm/output/mamba_mv_hpo_real_replace
#   * datasets  : activity, ushcn (skip physionet by user request 2026-04-25)
#
# Grid (matches colleague):
#   datasets : activity, ushcn  (auto_meta)
#   lr       : 1e-4, 5e-4, 2e-3
#   bs       : 64, 128, 256
#   seed     : 1
#   patience : 10  (fail-fast; bad configs die early)
# Total: 2 * 3 * 3 = 18 runs.
#
# Usage:
#   export WANDB_API_KEY=...
#   bash imts_benchmark/scripts/run_mamba_mv_hpo_replace_real_local.sh
#   DRY_RUN=1 bash imts_benchmark/scripts/run_mamba_mv_hpo_replace_real_local.sh
#   DATASETS="ushcn" LRS="2e-3" BATCH_SIZES="64" \
#     bash imts_benchmark/scripts/run_mamba_mv_hpo_replace_real_local.sh   # smoke

set -u
set -o pipefail

MAMBA_ENV=/home/ubuntu/envs/mamba
PYTHON=${MAMBA_ENV}/bin/python

REPO_DIR=/home/ubuntu/hongyu/ssm
CODE_DIR=${REPO_DIR}
DATA_ROOT=${REPO_DIR}/tpatchgnn_data
LOG_BASE=${REPO_DIR}/output/mamba_mv_hpo_real_replace

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: python not found at ${PYTHON}" >&2; exit 1
fi
for d in activity ushcn; do
  if [[ ! -f "${DATA_ROOT}/${d}/norm_stats.json" ]]; then
    echo "ERROR: ${DATA_ROOT}/${d}/norm_stats.json missing." >&2; exit 1
  fi
done

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is not set." >&2; exit 1
fi
_KEY_LEN=${#WANDB_API_KEY}
_KEY_CLEAN_LEN=$(printf '%s' "${WANDB_API_KEY}" | tr -cd 'A-Za-z0-9_' | wc -c)
if [[ "${_KEY_LEN}" -ne "${_KEY_CLEAN_LEN}" ]] || [[ "${_KEY_LEN}" -lt 30 ]]; then
  echo "ERROR: WANDB_API_KEY appears malformed (len=${_KEY_LEN})." >&2; exit 1
fi
export WANDB_MODE=online
unset WANDB_SILENT 2>/dev/null || true

read -ra DATASETS    <<< "${DATASETS:-activity ushcn}"
read -ra LRS         <<< "${LRS:-1e-4 5e-4 2e-3}"
read -ra BATCH_SIZES <<< "${BATCH_SIZES:-64 128 256}"
DT_MODE=${DT_MODE:-replace}
SEED=${SEED:-1}
PATIENCE=${PATIENCE:-10}

mkdir -p "${LOG_BASE}"
SUMMARY_CSV=${LOG_BASE}/hpo_summary.csv
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "dataset,dt_mode,lr,bs,seed,grid_K,patience,exit_code,wall_sec,output_dir" > "${SUMMARY_CSV}"
fi

cd "${CODE_DIR}"
SWEEP_START=$(date +%s)
TOTAL=$(( ${#DATASETS[@]} * ${#LRS[@]} * ${#BATCH_SIZES[@]} ))
RUN_IDX=0

for DS in "${DATASETS[@]}"; do
  case ${DS} in
    physionet) GRID_K=256 ;;
    activity)  GRID_K=128 ;;
    ushcn)     GRID_K=128 ;;
    *)         GRID_K=128 ;;
  esac

  for LR in "${LRS[@]}"; do
    for BS in "${BATCH_SIZES[@]}"; do
      RUN_IDX=$(( RUN_IDX + 1 ))
      TAG=lr-${LR}_bs-${BS}
      OUTPUT_DIR=${LOG_BASE}/${DS}/${TAG}/seed${SEED}
      mkdir -p "${OUTPUT_DIR}"
      RUN_LOG=${OUTPUT_DIR}/run.log
      RUN_NAME="hpo_real_replace_${DS}_${TAG}_seed${SEED}"

      echo "=========================================="
      echo "[${RUN_IDX}/${TOTAL}] ds=${DS} lr=${LR} bs=${BS} seed=${SEED} grid_K=${GRID_K} patience=${PATIENCE}"
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
        --lr "${LR}"
        --train_batch_size "${BS}"
        --patience "${PATIENCE}"
        --seed "${SEED}"
        --output_dir "${OUTPUT_DIR}"
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

      echo "  exit_code: ${EXIT_CODE}  wall: ${WALL}s"
      echo "${DS},${DT_MODE},${LR},${BS},${SEED},${GRID_K},${PATIENCE},${EXIT_CODE},${WALL},${OUTPUT_DIR}" \
        >> "${SUMMARY_CSV}"
    done
  done
done

SWEEP_END=$(date +%s)
TOTAL_WALL=$(( SWEEP_END - SWEEP_START ))
echo ""
echo "=========================================="
echo "HPO sweep done. Total wall: ${TOTAL_WALL}s ($(( TOTAL_WALL / 60 ))m)."
echo "Summary CSV: ${SUMMARY_CSV}"
echo "Next: python -m imts_benchmark.eval.aggregate_hpo_real --log_base ${LOG_BASE}"
