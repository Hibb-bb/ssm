#!/usr/bin/env bash
# Local (no-Slurm) port of imts_benchmark/scripts/run_mamba_mv_hpo_imm_delta_x86.sbatch.
# Mamba-MV HPO sweep on TIME-IMM datasets, sequential bash (no array).
#
# Grid per dataset (27 cells):
#   dt_mode ∈ {replace, learned, concat}    (3)
#   lr      ∈ {1e-4, 5e-4, 2e-3}            (3)
#   eff_bs  ∈ <per-dataset grid>            (3)
# = 3 * 3 * 3 = 27 cells.
#
# Per-dataset eff_bs grid (mirrors submit_imm_delta.sh):
#   imm_gdelt        : 64  128 256
#   imm_repohealth   : 64  128 256
#   imm_fnspid       : 64  128 256
#   imm_studentlife  : 64  128 256
#   imm_clustertrace : 32  64  128
#   imm_epa_air      : 32  64  128
#   imm_ilinet       : 8   16  32      (tiny train ~85)
#   imm_cesnet       : 8   16  32      (tiny train ~23)
#
# eff_bs -> (phys_bs, accumulate_grad_batches) policy:
#   8   -> (8,  1)      32  -> (32, 1)      128 -> (32, 4)
#   16  -> (16, 1)      64  -> (16, 4)      256 -> (32, 8)
#
# Each cell: seed=1, patience=10, grid_K=256.  The aggregator picks the cell
# with the best val/mse (run separately).
#
# Defaults (override via env):
#   DATASETS = (imm_gdelt imm_repohealth imm_fnspid imm_clustertrace
#               imm_studentlife imm_ilinet imm_cesnet imm_epa_air)
#   DT_MODES = (replace learned concat)
#   LRS      = (1e-4 5e-4 2e-3)
#   EFF_BS_OVERRIDE  = (use per-dataset preset)
#   CELL_INDEX       = (run a single cell, 1..27)
#   CELLS_FROM/CELLS_TO = (run a range, e.g. 1..9 = all dt_mode=replace)
#
# Total wall-clock @ 1xH100, all 8 datasets:
#   ~15-90 min/cell (depends on dataset size + early stop) * 27 cells * 8 datasets
#   = roughly 50-300 GPU-hours.  Run on detached screen, not foreground.
#
# Usage:
#   export WANDB_API_KEY=...
#
#   # Smoke (one cell, one dataset)
#   DATASETS=imm_fnspid CELL_INDEX=1 MAX_EPOCHS=2 \
#     bash imts_benchmark/scripts/hongyu_lambda_script/run_mamba_mv_hpo_imm_local.sh
#
#   # All 27 cells for one dataset
#   DATASETS=imm_fnspid \
#     bash imts_benchmark/scripts/hongyu_lambda_script/run_mamba_mv_hpo_imm_local.sh
#
#   # Full 8-dataset sweep, detached
#   screen -dmS mv_hpo bash -lc 'export WANDB_API_KEY=...; \
#       bash /home/ubuntu/hongyu/ssm/imts_benchmark/scripts/hongyu_lambda_script/run_mamba_mv_hpo_imm_local.sh'
#
#   # Resume from cell 13 in the current dataset (e.g. after a crash)
#   DATASETS=imm_fnspid CELLS_FROM=13 \
#     bash imts_benchmark/scripts/hongyu_lambda_script/run_mamba_mv_hpo_imm_local.sh

set -u
set -o pipefail

# ---------- environment ----------------------------------------------------
MAMBA_ENV=${MAMBA_ENV:-/home/ubuntu/envs/mamba}
PYTHON=${MAMBA_ENV}/bin/python

REPO_DIR=/home/ubuntu/hongyu/ssm
CODE_DIR=${REPO_DIR}
DATA_ROOT=${REPO_DIR}/time_imm_data
# Override D_MODEL/D_HIDDEN to shrink params (e.g. 64 -> ~259K params).
# Default 384 matches the original full-size sweep.
D_MODEL=${D_MODEL:-384}
D_HIDDEN=${D_HIDDEN:-384}
# LOG_SUFFIX is appended to LOG_BASE so distinct configs don't collide.
LOG_SUFFIX=${LOG_SUFFIX:-}
LOG_BASE=${REPO_DIR}/output/mamba_mv_hpo_imm${LOG_SUFFIX}

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
read -ra DT_MODES <<< "${DT_MODES:-replace learned concat}"
read -ra LRS      <<< "${LRS:-1e-4 5e-4 2e-3}"

# Per-dataset eff_bs grid (matches submit_imm_delta.sh).
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
SUMMARY_CSV=${LOG_BASE}/hpo_imm_summary.csv
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "dataset,dt_mode,lr,eff_bs,phys_bs,accum,cell_idx,exit_code,wall_sec,output_dir" > "${SUMMARY_CSV}"
fi

# ---------- eff_bs -> (phys_bs, accum) -------------------------------------
ebs_to_phys_accum () {
  case $1 in
    8)   echo  "8 1" ;;
    16)  echo "16 1" ;;
    32)  echo "32 1" ;;
    64)  echo "16 4" ;;
    128) echo "32 4" ;;
    256) echo "32 8" ;;
    *)   echo "" ;;
  esac
}

# ---------- run loop -------------------------------------------------------
cd "${CODE_DIR}"
SWEEP_START=$(date +%s)
GRID_K=${GRID_K:-256}

for DS in "${DATASETS[@]}"; do
  EFF_BS_STR="${EFF_BS_OVERRIDE:-${EFF_BS_FOR[$DS]:-64 128 256}}"
  read -ra EFF_BS_ARR <<< "${EFF_BS_STR}"
  if [[ ${#EFF_BS_ARR[@]} -ne 3 ]]; then
    echo "ERROR: eff_bs grid for ${DS} must contain exactly 3 values, got: ${EFF_BS_ARR[*]}" >&2
    exit 1
  fi

  N_CELLS=$(( ${#DT_MODES[@]} * ${#LRS[@]} * ${#EFF_BS_ARR[@]} ))
  echo "=========================================="
  echo "Dataset: ${DS}  (${N_CELLS} cells, dt=${DT_MODES[*]} lr=${LRS[*]} ebs=${EFF_BS_ARR[*]})"

  CELL=0
  for DT_MODE in "${DT_MODES[@]}"; do
    for LR in "${LRS[@]}"; do
      for EBS in "${EFF_BS_ARR[@]}"; do
        CELL=$(( CELL + 1 ))

        # Cell selection (skip if out of range).
        if [[ -n "${CELL_INDEX:-}" ]] && [[ "${CELL}" -ne "${CELL_INDEX}" ]]; then continue; fi
        if [[ -n "${CELLS_FROM:-}" ]] && [[ "${CELL}" -lt "${CELLS_FROM}" ]]; then continue; fi
        if [[ -n "${CELLS_TO:-}"   ]] && [[ "${CELL}" -gt "${CELLS_TO}"   ]]; then continue; fi

        PA=$(ebs_to_phys_accum "${EBS}")
        if [[ -z "${PA}" ]]; then
          echo "ERROR: unknown eff_bs=${EBS}; extend ebs_to_phys_accum()" >&2
          continue
        fi
        PHYS_BS=$(echo "${PA}" | awk '{print $1}')
        ACCUM=$(  echo "${PA}" | awk '{print $2}')

        CELL_TAG="lr-${LR}_ebs-${EBS}"
        OUTPUT_DIR=${LOG_BASE}/${DS}/${DT_MODE}/${CELL_TAG}/seed1
        mkdir -p "${OUTPUT_DIR}"
        RUN_LOG=${OUTPUT_DIR}/run.log
        RUN_NAME="hpo_mv_local_${DS}_${DT_MODE}_lr-${LR}_ebs-${EBS}_seed1"

        echo "------------------------------------------"
        echo "[${DS} ${CELL}/${N_CELLS}] dt=${DT_MODE} lr=${LR} ebs=${EBS} (phys=${PHYS_BS} accum=${ACCUM}) grid_K=${GRID_K}"
        echo "  output_dir: ${OUTPUT_DIR}"
        echo "  run_name:   ${RUN_NAME}"
        echo "  start:      $(date)"

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
          --patience 10
          --seed 1
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

        echo "  exit_code:  ${EXIT_CODE}"
        echo "  wall:       ${WALL}s ($(( WALL / 60 ))m $(( WALL % 60 ))s)"
        echo "${DS},${DT_MODE},${LR},${EBS},${PHYS_BS},${ACCUM},${CELL},${EXIT_CODE},${WALL},${OUTPUT_DIR}" >> "${SUMMARY_CSV}"
      done
    done
  done
done

SWEEP_END=$(date +%s)
TOTAL_WALL=$(( SWEEP_END - SWEEP_START ))
echo ""
echo "=========================================="
echo "HPO IMM sweep done. Total wall: ${TOTAL_WALL}s ($(( TOTAL_WALL / 60 ))m)."
echo "Summary CSV: ${SUMMARY_CSV}"
echo ""
echo "Next step: aggregate winners. The Delta sbatch's aggregator is:"
echo "  imts_benchmark/scripts/run_aggregate_hpo_imm_delta_x86.sbatch"
echo "  -> calls imts_benchmark.eval.aggregate_hpo_winners_imm"
echo "Run that python module locally (no Slurm needed)."
