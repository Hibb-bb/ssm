#!/usr/bin/env bash
# Local (no-Slurm) confirm runner for the IMM HPO winners.
# Lambda-side analog of run_mamba_mv_confirm_imm_delta_x86.sbatch.
#
# Pipeline per dataset:
#   1. If winners_by_dt_mode.json missing: run aggregate_hpo_winners_imm
#      against output/mamba_mv_hpo_imm/<DS>/.
#   2. Read winner config for each requested dt_mode (default: learned only).
#   3. Run N seeds (default 3) sequentially with the winner's
#      (lr, phys_bs, accum), patience=10, grid_K=256.
#
# Defaults (override via env):
#   DATASETS  = (imm_gdelt imm_repohealth imm_fnspid imm_clustertrace
#                imm_studentlife imm_ilinet imm_cesnet imm_epa_air)
#   DT_MODES  = (learned)         # match the HPO sweep we just ran
#   SEEDS     = (1 2 3)           # 3-seed average per user spec
#   EFF_BS_OVERRIDE = (use per-dataset preset matching the HPO sweep)
#
# Usage:
#   export WANDB_API_KEY=...
#
#   # Default: 8 datasets * 1 dt_mode * 3 seeds = 24 runs, sequential
#   bash imts_benchmark/scripts/hongyu_lambda_script/run_mamba_mv_confirm_imm_local.sh
#
#   # One dataset
#   DATASETS=imm_fnspid \
#     bash imts_benchmark/scripts/hongyu_lambda_script/run_mamba_mv_confirm_imm_local.sh
#
#   # Detached
#   screen -dmS mv_confirm bash -lc 'export WANDB_API_KEY=...; \
#       bash /home/ubuntu/hongyu/ssm/imts_benchmark/scripts/hongyu_lambda_script/run_mamba_mv_confirm_imm_local.sh'

set -u
set -o pipefail

# ---------- environment ----------------------------------------------------
MAMBA_ENV=${MAMBA_ENV:-/home/ubuntu/envs/mamba}
PYTHON=${MAMBA_ENV}/bin/python

REPO_DIR=/home/ubuntu/hongyu/ssm
# Override D_MODEL/D_HIDDEN to shrink params (e.g. 64 -> ~259K params).
D_MODEL=${D_MODEL:-384}
D_HIDDEN=${D_HIDDEN:-384}
# LOG_SUFFIX is appended to LOG_BASE/HPO_BASE so distinct configs don't collide.
LOG_SUFFIX=${LOG_SUFFIX:-}
CODE_DIR=${REPO_DIR}
DATA_ROOT=${REPO_DIR}/time_imm_data
HPO_BASE=${REPO_DIR}/output/mamba_mv_hpo_imm${LOG_SUFFIX}
LOG_BASE=${REPO_DIR}/output/mamba_mv_confirm_imm${LOG_SUFFIX}

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
read -ra DT_MODES <<< "${DT_MODES:-learned}"
read -ra SEEDS    <<< "${SEEDS:-1 2 3}"

# Per-dataset eff_bs grid (matches HPO sweep). Used by aggregator.
declare -A EFF_BS_FOR
EFF_BS_FOR[imm_gdelt]='64 128 256'
EFF_BS_FOR[imm_repohealth]='64 128 256'
EFF_BS_FOR[imm_fnspid]='64 128 256'
EFF_BS_FOR[imm_clustertrace]='32 64 128'
EFF_BS_FOR[imm_studentlife]='64 128 256'
EFF_BS_FOR[imm_ilinet]='8 16 32'
EFF_BS_FOR[imm_cesnet]='8 16 32'
EFF_BS_FOR[imm_epa_air]='32 64 128'

for d in "${DATASETS[@]}"; do
  if [[ ! -f "${DATA_ROOT}/${d}/norm_stats.json" ]]; then
    echo "ERROR: ${DATA_ROOT}/${d}/norm_stats.json missing." >&2
    exit 1
  fi
  if [[ ! -d "${HPO_BASE}/${d}" ]]; then
    echo "ERROR: HPO sweep dir missing: ${HPO_BASE}/${d}" >&2
    echo "       Run the HPO sweep first (run_mamba_mv_hpo_imm_local.sh)." >&2
    exit 1
  fi
done

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is not set." >&2
  exit 1
fi
_KEY_LEN=${#WANDB_API_KEY}
_KEY_CLEAN_LEN=$(printf '%s' "${WANDB_API_KEY}" | tr -cd 'A-Za-z0-9_' | wc -c)
if [[ "${_KEY_LEN}" -ne "${_KEY_CLEAN_LEN}" ]] || [[ "${_KEY_LEN}" -lt 30 ]]; then
  echo "ERROR: WANDB_API_KEY appears malformed." >&2
  exit 1
fi
export WANDB_MODE=online

mkdir -p "${LOG_BASE}"
SUMMARY_CSV=${LOG_BASE}/confirm_imm_summary.csv
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "dataset,dt_mode,lr,eff_bs,phys_bs,accum,seed,exit_code,wall_sec,output_dir" > "${SUMMARY_CSV}"
fi

# ---------- per-dataset loop ----------------------------------------------
cd "${CODE_DIR}"
SWEEP_START=$(date +%s)
GRID_K=${GRID_K:-256}
PATIENCE=${PATIENCE:-10}

for DS in "${DATASETS[@]}"; do
  EFF_BS_STR="${EFF_BS_OVERRIDE:-${EFF_BS_FOR[$DS]:-64 128 256}}"
  read -ra EFF_BS_ARR <<< "${EFF_BS_STR}"
  WINNERS_JSON=${HPO_BASE}/${DS}/winners_by_dt_mode.json

  echo "=========================================="
  echo "Dataset: ${DS}"

  # --- step 1: aggregator (idempotent, re-runs cheap) -----------------------
  echo "  [aggregator] eff_bs=${EFF_BS_ARR[*]}"
  "${PYTHON}" -m imts_benchmark.eval.aggregate_hpo_winners_imm \
    --hpo_log_dir "${HPO_BASE}/${DS}" \
    --eff_bs ${EFF_BS_ARR[*]} \
    --seed 1
  AGG_EXIT=$?
  if [[ "${AGG_EXIT}" -eq 1 ]]; then
    echo "  [aggregator] ERROR exit_code=${AGG_EXIT}; skipping ${DS}" >&2
    continue
  fi
  if [[ ! -f "${WINNERS_JSON}" ]]; then
    echo "  [aggregator] winners JSON not produced; skipping ${DS}" >&2
    continue
  fi

  # --- step 2: per-dt_mode winner -> per-seed run --------------------------
  for DT_MODE in "${DT_MODES[@]}"; do
    PARSED=$("${PYTHON}" - "${WINNERS_JSON}" "${DT_MODE}" <<'PY'
import json, sys
try:
    winners = json.load(open(sys.argv[1]))
    cfg = winners.get(sys.argv[2])
    if cfg is None:
        sys.exit(1)
    print(cfg["lr"], cfg["phys_bs"], cfg["accum"], cfg["eff_bs"])
except Exception as e:
    sys.stderr.write(f"parse error: {e}\n")
    sys.exit(1)
PY
)
    PARSE_EXIT=$?
    if [[ "${PARSE_EXIT}" -ne 0 ]] || [[ -z "${PARSED}" ]]; then
      echo "  [confirm] ERROR: no winner for dt_mode=${DT_MODE} in ${WINNERS_JSON}; skipping" >&2
      continue
    fi
    read LR PHYS_BS ACCUM EFF_BS <<< "${PARSED}"

    echo "  [confirm] dt=${DT_MODE} winner lr=${LR} phys_bs=${PHYS_BS} accum=${ACCUM} eff_bs=${EFF_BS}"

    for SEED in "${SEEDS[@]}"; do
      VARIANT_DIR="${DT_MODE}_lr-${LR}_bs-${PHYS_BS}_accum-${ACCUM}"
      OUTPUT_DIR=${LOG_BASE}/${DS}/${VARIANT_DIR}/seed${SEED}
      mkdir -p "${OUTPUT_DIR}"
      RUN_LOG=${OUTPUT_DIR}/run.log
      RUN_NAME="confirm_mv_local_${DS}_${DT_MODE}_lr-${LR}_bs-${PHYS_BS}_accum-${ACCUM}_seed${SEED}"

      echo "  ---"
      echo "  [seed ${SEED}/${SEEDS[-1]}] output_dir=${OUTPUT_DIR}"
      echo "  start: $(date)"

      CMD=(
        "${PYTHON}" -m imts_benchmark.mamba_mv.train_mv
        --data_root "${DATA_ROOT}"
        --regime "${DS}"
        --auto_meta
        --dt_mode "${DT_MODE}"
        --d_model "${D_MODEL}"
        --d_hidden "${D_HIDDEN}"
        --grid_K "${GRID_K}"
        --lr "${LR}"
        --train_batch_size "${PHYS_BS}"
        --accumulate_grad_batches "${ACCUM}"
        --patience "${PATIENCE}"
        --seed "${SEED}"
        --output_dir "${OUTPUT_DIR}"
        --use_wandb
        --wandb_project "TSKing"
        --wandb_run_name "${RUN_NAME}"
      )
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

      echo "  exit_code=${EXIT_CODE}  wall=${WALL}s ($(( WALL / 60 ))m)"
      echo "${DS},${DT_MODE},${LR},${EFF_BS},${PHYS_BS},${ACCUM},${SEED},${EXIT_CODE},${WALL},${OUTPUT_DIR}" >> "${SUMMARY_CSV}"
    done
  done
done

SWEEP_END=$(date +%s)
TOTAL_WALL=$(( SWEEP_END - SWEEP_START ))
echo ""
echo "=========================================="
echo "Confirm sweep done. Total wall: ${TOTAL_WALL}s ($(( TOTAL_WALL / 60 ))m)."
echo "Summary CSV: ${SUMMARY_CSV}"
echo ""
echo "Next: aggregate per-dataset 3-seed mean/std from ${LOG_BASE}/<DS>/<variant>/seed*/test_metrics.csv"
