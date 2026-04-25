#!/usr/bin/env bash
# Combined local pipeline: HPO -> aggregate -> 5-seed confirm.
# Local port of:
#   imts_benchmark/scripts/run_mamba_mv_hpo_replace_real.sbatch
#   imts_benchmark/eval/aggregate_hpo_real.py
#   imts_benchmark/scripts/run_mamba_mv_hpo_replace_confirm.sbatch
#
# Patience policy (decided 2026-04-25):
#   HPO     : 10  (matches colleague; fail-fast for ranking)
#   CONFIRM : 25  (halfway between colleague's 50 and HPO's 10; user trade-off
#                  for shorter wall while reducing risk of dying at the 25%-of-
#                  cosine-schedule plateau colleague flagged in fair_defaults.py)
#
# Datasets   : activity, ushcn      (PhysioNet skipped per user 2026-04-25)
# dt_mode    : replace
# HPO grid   : 3 lr * 3 bs * 1 seed = 9 runs/dataset; 18 total.
# Confirm    : winner(lr,bs) * 5 seeds = 5 runs/dataset; 10 total.
#
# Phase skip flags (re-running individual phases):
#   SKIP_HPO=1      bash run_mamba_mv_real_pipeline_local.sh
#   SKIP_AGG=1      bash run_mamba_mv_real_pipeline_local.sh
#   SKIP_CONFIRM=1  bash run_mamba_mv_real_pipeline_local.sh
#
# Usage:
#   export WANDB_API_KEY=...
#   bash imts_benchmark/scripts/run_mamba_mv_real_pipeline_local.sh
#   DRY_RUN=1 bash imts_benchmark/scripts/run_mamba_mv_real_pipeline_local.sh

set -u
set -o pipefail

# ---------- env ------------------------------------------------------------
MAMBA_ENV=/home/ubuntu/envs/mamba
PYTHON=${MAMBA_ENV}/bin/python

REPO_DIR=/home/ubuntu/hongyu/ssm
CODE_DIR=${REPO_DIR}
DATA_ROOT=${REPO_DIR}/tpatchgnn_data
HPO_LOG_BASE=${REPO_DIR}/output/mamba_mv_hpo_real_replace
CONFIRM_LOG_BASE=${REPO_DIR}/output/mamba_mv_hpo_real_replace_confirm

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

if [[ ! -x "${PYTHON}" ]]; then echo "ERROR: python not found at ${PYTHON}" >&2; exit 1; fi
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

# ---------- shared config --------------------------------------------------
read -ra DATASETS    <<< "${DATASETS:-activity ushcn}"
read -ra LRS         <<< "${LRS:-1e-4 5e-4 2e-3}"
read -ra BATCH_SIZES <<< "${BATCH_SIZES:-64 128 256}"
read -ra SEEDS       <<< "${SEEDS:-1 2 3 4 5}"
DT_MODE=${DT_MODE:-replace}
HPO_PATIENCE=${HPO_PATIENCE:-10}
CONFIRM_PATIENCE=${CONFIRM_PATIENCE:-25}

# Memory-constrained datasets: fix per-step batch size and sweep accumulation
# instead of true bs. Effective batch = BATCH_FIXED * accum.
# Example for PhysioNet:
#   BATCH_FIXED=16 ACCUM_GRID="4 8 16" DATASETS=physionet
# Cell tag becomes effbs-<bs*accum> instead of bs-<bs>; aggregator + confirm
# follow suit (see --cell_tag_format=effbs).
BATCH_FIXED=${BATCH_FIXED:-}
read -ra ACCUM_GRID <<< "${ACCUM_GRID:-1}"
if [[ -n "${BATCH_FIXED}" ]]; then
  CELL_TAG_FORMAT=effbs
  # Override BATCH_SIZES so it carries the EFFECTIVE batch sizes used by the
  # aggregator. Each effbs = BATCH_FIXED * accum.
  EFF_BS=()
  for A in "${ACCUM_GRID[@]}"; do EFF_BS+=("$((BATCH_FIXED * A))"); done
  BATCH_SIZES=("${EFF_BS[@]}")
else
  CELL_TAG_FORMAT=bs
fi
echo "[config] cell_tag_format=${CELL_TAG_FORMAT}  batch_fixed=${BATCH_FIXED:-N/A}"
echo "[config] BATCH_SIZES=(${BATCH_SIZES[*]})  ACCUM_GRID=(${ACCUM_GRID[*]})"

for d in "${DATASETS[@]}"; do
  if [[ ! -f "${DATA_ROOT}/${d}/norm_stats.json" ]]; then
    echo "ERROR: ${DATA_ROOT}/${d}/norm_stats.json missing." >&2; exit 1
  fi
done

grid_K_for() {
  case $1 in
    physionet) echo 256 ;;
    activity)  echo 128 ;;
    ushcn)     echo 128 ;;
    *)         echo 128 ;;
  esac
}

train_one() {
  local DS=$1 LR=$2 BS=$3 ACCUM=$4 SEED=$5 PATIENCE=$6 OUT=$7 RUN_NAME=$8
  local GRID_K
  GRID_K=$(grid_K_for "${DS}")
  mkdir -p "${OUT}"
  local CMD=(
    "${PYTHON}" -m imts_benchmark.mamba_mv.train_mv
    --data_root "${DATA_ROOT}"
    --regime "${DS}"
    --auto_meta
    --dt_mode "${DT_MODE}"
    --grid_K "${GRID_K}"
    --lr "${LR}"
    --train_batch_size "${BS}"
    --accumulate_grad_batches "${ACCUM}"
    --patience "${PATIENCE}"
    --seed "${SEED}"
    --output_dir "${OUT}"
    --use_wandb
    --wandb_project "TSKing"
    --wandb_run_name "${RUN_NAME}"
  )
  echo "  cmd: ${CMD[*]}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then echo "  DRY_RUN"; return 0; fi
  export WANDB_DIR=${OUT}
  local START END WALL EC
  START=$(date +%s)
  "${CMD[@]}" 2>&1 | tee "${OUT}/run.log"
  EC=${PIPESTATUS[0]}
  END=$(date +%s)
  WALL=$(( END - START ))
  echo "  exit_code: ${EC}  wall: ${WALL}s"
  echo "${EC},${WALL}" > "${OUT}/.run_meta"
  return ${EC}
}

cd "${CODE_DIR}"
PIPELINE_START=$(date +%s)

# ---------- PHASE 1: HPO ---------------------------------------------------
HPO_SUMMARY=${HPO_LOG_BASE}/hpo_summary.csv
if [[ "${SKIP_HPO:-0}" != "1" ]]; then
  echo "##########################################"
  echo "# PHASE 1/3 : HPO  (patience=${HPO_PATIENCE})"
  echo "##########################################"
  mkdir -p "${HPO_LOG_BASE}"
  if [[ ! -f "${HPO_SUMMARY}" ]]; then
    echo "dataset,dt_mode,lr,bs,accum,effbs,seed,patience,exit_code,wall_sec,output_dir" > "${HPO_SUMMARY}"
  fi
  TOTAL=$(( ${#DATASETS[@]} * ${#LRS[@]} * ${#BATCH_SIZES[@]} ))
  IDX=0
  for DS in "${DATASETS[@]}"; do
    for LR in "${LRS[@]}"; do
      for BS_OR_EFFBS in "${BATCH_SIZES[@]}"; do
        IDX=$(( IDX + 1 ))
        if [[ "${CELL_TAG_FORMAT}" == "effbs" ]]; then
          BS=${BATCH_FIXED}
          ACCUM=$(( BS_OR_EFFBS / BATCH_FIXED ))
          EFFBS=${BS_OR_EFFBS}
          TAG=lr-${LR}_effbs-${EFFBS}
        else
          BS=${BS_OR_EFFBS}
          ACCUM=1
          EFFBS=${BS}
          TAG=lr-${LR}_bs-${BS}
        fi
        OUT=${HPO_LOG_BASE}/${DS}/dt-${DT_MODE}_${TAG}/seed1
        RN="mamba_mv_hpo_real_${DT_MODE}_${DS}_${TAG}_seed1"
        echo ""
        echo "[HPO ${IDX}/${TOTAL}] ds=${DS} lr=${LR} bs=${BS} accum=${ACCUM} effbs=${EFFBS}"
        train_one "${DS}" "${LR}" "${BS}" "${ACCUM}" 1 "${HPO_PATIENCE}" "${OUT}" "${RN}" || true
        if [[ -f "${OUT}/.run_meta" ]]; then
          IFS=, read -r EC WALL < "${OUT}/.run_meta"
          echo "${DS},${DT_MODE},${LR},${BS},${ACCUM},${EFFBS},1,${HPO_PATIENCE},${EC},${WALL},${OUT}" >> "${HPO_SUMMARY}"
        fi
      done
    done
  done
else
  echo "[SKIP_HPO=1] skipping phase 1"
fi

# ---------- PHASE 2: aggregate --------------------------------------------
WINNERS_JSON=${HPO_LOG_BASE}/winners_by_dataset.json
if [[ "${SKIP_AGG:-0}" != "1" ]]; then
  echo ""
  echo "##########################################"
  echo "# PHASE 2/3 : aggregate -> winners JSON"
  echo "##########################################"
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    AGG_EXTRA=()
    if [[ "${CELL_TAG_FORMAT}" == "effbs" ]]; then
      AGG_EXTRA+=(--cell_tag_format effbs --fixed_batch_size "${BATCH_FIXED}")
    fi
    "${PYTHON}" -m imts_benchmark.eval.aggregate_hpo_real \
      --hpo_log_dir "${HPO_LOG_BASE}" \
      --datasets "${DATASETS[@]}" \
      --dt_mode "${DT_MODE}" \
      --lrs "${LRS[@]}" \
      --batch_sizes "${BATCH_SIZES[@]}" \
      --seed 1 \
      "${AGG_EXTRA[@]}"
    if [[ ! -f "${WINNERS_JSON}" ]]; then
      echo "ERROR: winners JSON not written: ${WINNERS_JSON}" >&2; exit 1
    fi
    echo "winners:"; cat "${WINNERS_JSON}"
  else
    echo "  DRY_RUN: would call aggregate_hpo_real"
  fi
else
  echo "[SKIP_AGG=1] skipping phase 2"
fi

# ---------- PHASE 3: 5-seed confirm ---------------------------------------
CONFIRM_SUMMARY=${CONFIRM_LOG_BASE}/confirm_summary.csv
if [[ "${SKIP_CONFIRM:-0}" != "1" ]]; then
  echo ""
  echo "##########################################"
  echo "# PHASE 3/3 : 5-seed confirm  (patience=${CONFIRM_PATIENCE})"
  echo "##########################################"
  if [[ "${DRY_RUN:-0}" != "1" && ! -f "${WINNERS_JSON}" ]]; then
    echo "ERROR: ${WINNERS_JSON} not found; cannot run confirm." >&2; exit 1
  fi
  mkdir -p "${CONFIRM_LOG_BASE}"
  if [[ ! -f "${CONFIRM_SUMMARY}" ]]; then
    echo "dataset,dt_mode,lr,bs,accum,effbs,seed,patience,exit_code,wall_sec,output_dir" > "${CONFIRM_SUMMARY}"
  fi
  # Skip any dataset whose HPO failed completely (not present in winners JSON).
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    AVAILABLE_DS=$("${PYTHON}" -c "import json; print(' '.join(json.load(open('${WINNERS_JSON}')).keys()))")
    SKIPPED_DS=()
    KEPT_DS=()
    for DS in "${DATASETS[@]}"; do
      if [[ " ${AVAILABLE_DS} " == *" ${DS} "* ]]; then
        KEPT_DS+=("${DS}")
      else
        SKIPPED_DS+=("${DS}")
      fi
    done
    if [[ ${#SKIPPED_DS[@]} -gt 0 ]]; then
      echo "WARNING: skipping confirm for datasets with no HPO winner: ${SKIPPED_DS[*]}"
      echo "         (all HPO cells crashed / produced no val_mse; see ${HPO_LOG_BASE}/hpo_ranked.csv)"
      for DS in "${SKIPPED_DS[@]}"; do
        echo "${DS},${DT_MODE},NA,NA,NA,NA,NA,NA,NA,NA,SKIPPED_NO_WINNER" >> "${CONFIRM_SUMMARY}"
      done
    fi
    if [[ ${#KEPT_DS[@]} -eq 0 ]]; then
      echo "ERROR: no datasets have a winner; aborting confirm phase." >&2
      exit 1
    fi
    CONFIRM_DATASETS=("${KEPT_DS[@]}")
  else
    CONFIRM_DATASETS=("${DATASETS[@]}")
  fi

  TOTAL=$(( ${#CONFIRM_DATASETS[@]} * ${#SEEDS[@]} ))
  IDX=0
  for DS in "${CONFIRM_DATASETS[@]}"; do
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      LR="(dryrun)"; BS="(dryrun)"; ACCUM=1; EFFBS="(dryrun)"
    else
      LR=$("${PYTHON}" -c "import json; print(json.load(open('${WINNERS_JSON}'))['${DS}']['lr'])")
      BS=$("${PYTHON}" -c "import json; print(json.load(open('${WINNERS_JSON}'))['${DS}']['train_batch_size'])")
      ACCUM=$("${PYTHON}" -c "import json; d=json.load(open('${WINNERS_JSON}'))['${DS}']; print(d.get('accumulate_grad_batches', 1))")
      EFFBS=$((BS * ACCUM))
    fi
    for SEED in "${SEEDS[@]}"; do
      IDX=$(( IDX + 1 ))
      if [[ "${CELL_TAG_FORMAT}" == "effbs" ]]; then
        TAG=lr-${LR}_effbs-${EFFBS}
      else
        TAG=lr-${LR}_bs-${BS}
      fi
      OUT=${CONFIRM_LOG_BASE}/${DS}/dt-${DT_MODE}_${TAG}/seed${SEED}
      RN="mamba_mv_confirm_real_${DT_MODE}_${DS}_${TAG}_seed${SEED}"
      echo ""
      echo "[CONFIRM ${IDX}/${TOTAL}] ds=${DS} lr=${LR} bs=${BS} accum=${ACCUM} effbs=${EFFBS} seed=${SEED}"
      train_one "${DS}" "${LR}" "${BS}" "${ACCUM}" "${SEED}" "${CONFIRM_PATIENCE}" "${OUT}" "${RN}" || true
      if [[ -f "${OUT}/.run_meta" ]]; then
        IFS=, read -r EC WALL < "${OUT}/.run_meta"
        echo "${DS},${DT_MODE},${LR},${BS},${ACCUM},${EFFBS},${SEED},${CONFIRM_PATIENCE},${EC},${WALL},${OUT}" >> "${CONFIRM_SUMMARY}"
      fi
    done
  done
else
  echo "[SKIP_CONFIRM=1] skipping phase 3"
fi

PIPELINE_END=$(date +%s)
WALL=$(( PIPELINE_END - PIPELINE_START ))
echo ""
echo "##########################################"
echo "# Pipeline done. Wall: ${WALL}s ($(( WALL / 3600 ))h $(( (WALL % 3600) / 60 ))m)"
echo "# HPO summary    : ${HPO_SUMMARY}"
echo "# Winners JSON   : ${WINNERS_JSON}"
echo "# Confirm summary: ${CONFIRM_SUMMARY}"
echo "##########################################"
