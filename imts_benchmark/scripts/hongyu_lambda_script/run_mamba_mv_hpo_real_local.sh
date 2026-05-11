#!/usr/bin/env bash
# ============================================================================
#  Local (non-Slurm) port of run_mamba_mv_hpo_{replace,learned,concat}_real.sbatch
#  Sweeps Mamba-MV across {activity, ushcn} × {replace, learned, concat}
#  × {lr=1e-4, 5e-4, 2e-3} × {bs=64, 128, 256}.  Seed=1, patience=10.
#
#  Output layout (consumed by aggregate_hpo_real.py):
#    <LOG_BASE>/<DS>/dt-<DT>_lr-<LR>_bs-<BS>/seed1/{checkpoints,test_metrics.csv,run.log}
#
#  Env-var overrides:
#    DATASETS  = "activity ushcn"        (default)
#    DT_MODES  = "replace learned concat"
#    LRS       = "1e-4 5e-4 2e-3"
#    BATCH_SIZES = "64 128 256"
#    D_MODEL   = 384  (override to 64 for ~259K params)
#    D_HIDDEN  = 384
#    LOG_SUFFIX = ""  (suffix appended to LOG_BASE; e.g. "_d64")
#    DRY_RUN   = "" (set =1 to print commands without launching)
#    CELL_INDEX, CELLS_FROM, CELLS_TO  = restrict to a subset of cells
# ============================================================================
set -e
set -u
set -o pipefail

# ---------- environment ----------------------------------------------------
MAMBA_ENV=${MAMBA_ENV:-/home/ubuntu/envs/mamba}
PYTHON=${MAMBA_ENV}/bin/python

REPO_DIR=/home/ubuntu/hongyu/ssm
CODE_DIR=${REPO_DIR}
DATA_ROOT=${REPO_DIR}/tpatchgnn_data
D_MODEL=${D_MODEL:-384}
D_HIDDEN=${D_HIDDEN:-384}
LOG_SUFFIX=${LOG_SUFFIX:-}
LOG_BASE=${REPO_DIR}/output/mamba_mv_hpo_real${LOG_SUFFIX}

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

if [[ -z "${WANDB_API_KEY:-}" ]] && [[ -f ~/.netrc ]]; then
    export WANDB_API_KEY=$(awk '/api.wandb.ai/{f=1} f && /password/{print $2; exit}' ~/.netrc)
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY not set." >&2; exit 1
fi
export WANDB_MODE=online

if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: python not found at ${PYTHON}" >&2; exit 1
fi

DEFAULT_DATASETS=(activity ushcn)
read -ra DATASETS    <<< "${DATASETS:-${DEFAULT_DATASETS[*]}}"
read -ra DT_MODES    <<< "${DT_MODES:-replace learned concat}"
read -ra LRS         <<< "${LRS:-1e-4 5e-4 2e-3}"
read -ra BATCH_SIZES <<< "${BATCH_SIZES:-64 128 256}"

for d in "${DATASETS[@]}"; do
    if [[ ! -d "${DATA_ROOT}/${d}" ]]; then
        echo "ERROR: dataset dir missing: ${DATA_ROOT}/${d}" >&2; exit 1
    fi
done

# ---------- run loop -------------------------------------------------------
cd "${CODE_DIR}"
SWEEP_START=$(date +%s)

N_TOTAL=$(( ${#DATASETS[@]} * ${#DT_MODES[@]} * ${#LRS[@]} * ${#BATCH_SIZES[@]} ))
echo "=========================================="
echo "Mamba-MV HPO sweep on REAL benchmark"
echo "  datasets : ${DATASETS[*]}"
echo "  dt_modes : ${DT_MODES[*]}"
echo "  lrs      : ${LRS[*]}"
echo "  bss      : ${BATCH_SIZES[*]}"
echo "  d_model  : ${D_MODEL}  (d_hidden=${D_HIDDEN})"
echo "  total    : ${N_TOTAL} cells"
echo "  output   : ${LOG_BASE}"
echo "=========================================="

CELL=0
for DS in "${DATASETS[@]}"; do
    case "${DS}" in
        physionet) GRID_K=256 ;;
        activity)  GRID_K=128 ;;
        ushcn)     GRID_K=128 ;;
        *)         GRID_K=128 ;;
    esac

    for DT_MODE in "${DT_MODES[@]}"; do
        for LR in "${LRS[@]}"; do
            for BS in "${BATCH_SIZES[@]}"; do
                CELL=$(( CELL + 1 ))

                if [[ -n "${CELL_INDEX:-}" ]] && [[ "${CELL}" -ne "${CELL_INDEX}" ]]; then continue; fi
                if [[ -n "${CELLS_FROM:-}" ]] && [[ "${CELL}" -lt "${CELLS_FROM}" ]]; then continue; fi
                if [[ -n "${CELLS_TO:-}"   ]] && [[ "${CELL}" -gt "${CELLS_TO}"   ]]; then continue; fi

                TAG="dt-${DT_MODE}_lr-${LR}_bs-${BS}"
                OUTPUT_DIR=${LOG_BASE}/${DS}/${TAG}/seed1
                mkdir -p "${OUTPUT_DIR}"
                RUN_LOG=${OUTPUT_DIR}/run.log
                RUN_NAME="hpo_real_local_${DS}_${DT_MODE}_lr-${LR}_bs-${BS}_seed1"

                echo "------------------------------------------"
                echo "[${CELL}/${N_TOTAL}] ds=${DS} dt=${DT_MODE} lr=${LR} bs=${BS} grid_K=${GRID_K}"
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
                    --train_batch_size "${BS}"
                    --patience 10
                    --seed 1
                    --output_dir "${OUTPUT_DIR}"
                    --use_wandb
                    --wandb_project "TSKing"
                    --wandb_run_name "${RUN_NAME}"
                )

                if [[ -n "${DRY_RUN:-}" ]]; then
                    echo "  DRY_RUN: ${CMD[*]}"
                    continue
                fi

                CELL_START=$(date +%s)
                "${CMD[@]}" > "${RUN_LOG}" 2>&1 || true
                CELL_END=$(date +%s)
                echo "  done in $((CELL_END - CELL_START))s   (log: ${RUN_LOG})"
            done
        done
    done
done

SWEEP_END=$(date +%s)
echo "=========================================="
echo "HPO real sweep done. Total wall: $((SWEEP_END - SWEEP_START))s"
echo "Aggregate winners: imts_benchmark/eval/aggregate_hpo_real.py"
echo "=========================================="
