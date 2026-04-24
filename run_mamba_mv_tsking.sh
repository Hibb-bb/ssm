#!/usr/bin/env bash
# Sequential train_mv runs (Weights & Biases online) over a seed range.
#
# Default: seeds 1..10 (same regime / dt_mode each time). Data is generated once.
#
# Usage:
#   export WANDB_API_KEY="..."
#   bash run_mamba_mv_tsking.sh
#
#   SEED_START=1 SEED_END=3 bash run_mamba_mv_tsking.sh
#   REGIME=multisin_med_irreg DT_MODE=learned bash run_mamba_mv_tsking.sh
#   SKIP_DATA_GEN=1 bash run_mamba_mv_tsking.sh
#   USE_SRUN=1 bash run_mamba_mv_tsking.sh
#   DRY_RUN=1 bash run_mamba_mv_tsking.sh

if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

set -uo pipefail

# ---------- paths (this repo = hibb/ssm) ------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${REPO_DIR}"

MAMBA_ENV="${MAMBA_ENV:-/home/ubuntu/envs/mamba}"
PYTHON="${MAMBA_ENV}/bin/python"

DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/data_correct_async}"

REGIME="${REGIME:-multisin_high_irreg}"
DT_MODE="${DT_MODE:-replace}"
SEED_START="${SEED_START:-1}"
SEED_END="${SEED_END:-10}"

OUTPUT_BASE="${OUTPUT_BASE:-${REPO_DIR}/output/mamba_mv_tsking}"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export WANDB_MODE=online
unset WANDB_SILENT 2>/dev/null || true

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: python not found at ${PYTHON}" >&2
  exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is not set. Export it before running, e.g.:" >&2
  echo "  export WANDB_API_KEY=..." >&2
  exit 1
fi

# ---------- data generation (async multisin; all regimes in one pass) -------
if [[ "${SKIP_DATA_GEN:-0}" == "0" ]]; then
  if [[ -d "${DATA_ROOT}/${REGIME}" && -f "${DATA_ROOT}/multisin_stats.json" ]]; then
    echo "[data] ${DATA_ROOT} already has ${REGIME} + multisin_stats.json, skipping generation."
  else
    echo "[data] generating async multisin dataset under ${DATA_ROOT}"
    cd "${REPO_DIR}"
    "${PYTHON}" generate_multivariate_sinusodial_data_async.py \
      --output_root "${DATA_ROOT}"
  fi
else
  echo "[data] SKIP_DATA_GEN=1, skipping generation."
fi

if [[ "${USE_SRUN:-0}" == "1" ]]; then
  LAUNCH=(srun)
else
  LAUNCH=()
fi

cd "${CODE_DIR}"

echo "=========================================="
echo "seeds:       ${SEED_START}..${SEED_END} ($((${SEED_END} - ${SEED_START} + 1)) runs)"
echo "regime=${REGIME} dt_mode=${DT_MODE}"
echo "data_root:   ${DATA_ROOT}"
echo "output_base: ${OUTPUT_BASE}"
echo "wandb:       mode=${WANDB_MODE} project=TSKing"
echo "launcher:    ${LAUNCH[*]:-(direct)}"
echo "sweep start: $(date)"
echo "=========================================="

RUN_IDX=0
TOTAL=$((${SEED_END} - ${SEED_START} + 1))

for SEED in $(seq "${SEED_START}" "${SEED_END}"); do
  RUN_IDX=$((RUN_IDX + 1))
  TAG="${REGIME}_${DT_MODE}_seed${SEED}"
  OUTPUT_DIR="${OUTPUT_BASE}/${TAG}"
  mkdir -p "${OUTPUT_DIR}"
  export WANDB_DIR="${OUTPUT_DIR}"

  CMD=(
    "${PYTHON}" -m imts_benchmark.mamba_mv.train_mv
    --regime "${REGIME}"
    --dt_mode "${DT_MODE}"
    --seed "${SEED}"
    --data_root "${DATA_ROOT}"
    --output_dir "${OUTPUT_DIR}"
    --use_wandb
    --wandb_project TSKing
    --wandb_run_name "${TAG}"
  )

  echo ""
  echo "------------------------------------------"
  echo "[${RUN_IDX}/${TOTAL}] seed=${SEED}  run=${TAG}"
  echo "  output_dir: ${OUTPUT_DIR}"
  echo "  start:      $(date)"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "  DRY_RUN: ${LAUNCH[*]} ${CMD[*]}"
    continue
  fi

  "${LAUNCH[@]}" "${CMD[@]}"
  echo "  end:        $(date)"
done

echo ""
echo "=========================================="
echo "All seeds ${SEED_START}-${SEED_END} finished. sweep end: $(date)"
