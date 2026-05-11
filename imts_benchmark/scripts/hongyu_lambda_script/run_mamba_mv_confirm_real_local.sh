#!/usr/bin/env bash
# ============================================================================
#  Local (non-Slurm) port of run_mamba_mv_p10_{replace,learned,concat}_real_small.sbatch
#  Runs Mamba-MV at HPO winner config across SEEDS for {activity, ushcn} × {replace,learned,concat}.
#  Defaults to 3 seeds (vs colleague's 5).  patience=10.
#
#  Per-(DS, DT_MODE) winner is read from:
#    <HPO_BASE>/winners_replace.json, winners_learned.json, winners_concat.json
#  Each JSON: {"<DS>": {"lr": "<lr>", "train_batch_size": <bs>, ...}, ...}
#  (produced by aggregate_hpo_real.py — script invokes it if missing).
#
#  Output layout:
#    <LOG_BASE>/<DS>/<DT_MODE>_lr-<LR>_bs-<BS>/seed<N>/
#
#  Env-var overrides:
#    DATASETS  = "activity ushcn"
#    DT_MODES  = "replace learned concat"
#    SEEDS     = "1 2 3"   (default 3 seeds; colleague's was "1 2 3 4 5")
#    PATIENCE  = 10
#    D_MODEL/D_HIDDEN/LOG_SUFFIX  -- same semantics as HPO runner
# ============================================================================
set -e
set -u
set -o pipefail

MAMBA_ENV=${MAMBA_ENV:-/home/ubuntu/envs/mamba}
PYTHON=${MAMBA_ENV}/bin/python

REPO_DIR=/home/ubuntu/hongyu/ssm
CODE_DIR=${REPO_DIR}
DATA_ROOT=${REPO_DIR}/tpatchgnn_data
D_MODEL=${D_MODEL:-384}
D_HIDDEN=${D_HIDDEN:-384}
LOG_SUFFIX=${LOG_SUFFIX:-}
HPO_BASE=${REPO_DIR}/output/mamba_mv_hpo_real${LOG_SUFFIX}
LOG_BASE=${REPO_DIR}/output/mamba_mv_confirm_real${LOG_SUFFIX}
PATIENCE=${PATIENCE:-10}

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

if [[ -z "${WANDB_API_KEY:-}" ]] && [[ -f ~/.netrc ]]; then
    export WANDB_API_KEY=$(awk '/api.wandb.ai/{f=1} f && /password/{print $2; exit}' ~/.netrc)
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY not set." >&2; exit 1
fi
export WANDB_MODE=online

read -ra DATASETS <<< "${DATASETS:-activity ushcn}"
read -ra DT_MODES <<< "${DT_MODES:-replace learned concat}"
read -ra SEEDS    <<< "${SEEDS:-1 2 3}"

mkdir -p "${LOG_BASE}"
SUMMARY=${LOG_BASE}/confirm_real_summary.csv
echo "dataset,dt_mode,lr,bs,grid_K,seed,exit_code,wall_sec,output_dir" > "${SUMMARY}"

# ---------- aggregate winners per dt_mode (if not present) -----------------
for DT_MODE in "${DT_MODES[@]}"; do
    WIN_JSON=${HPO_BASE}/winners_${DT_MODE}.json
    if [[ ! -f "${WIN_JSON}" ]]; then
        echo "[winners] aggregating ${DT_MODE} from ${HPO_BASE}"
        "${PYTHON}" -m imts_benchmark.eval.aggregate_hpo_real \
            --hpo_log_dir "${HPO_BASE}" \
            --datasets "${DATASETS[@]}" \
            --dt_mode "${DT_MODE}" \
            --lrs 1e-4 5e-4 2e-3 \
            --batch_sizes 64 128 256 \
            --seed 1 || { echo "ERROR: aggregator failed for ${DT_MODE}" >&2; exit 1; }
        # aggregator writes winners_by_dataset.json next to ranked CSV;
        # rename to dt-specific name so all 3 dt_modes coexist.
        if [[ -f "${HPO_BASE}/winners_by_dataset.json" ]]; then
            mv "${HPO_BASE}/winners_by_dataset.json" "${WIN_JSON}"
        fi
    fi
    if [[ ! -f "${WIN_JSON}" ]]; then
        echo "ERROR: winners JSON still missing: ${WIN_JSON}" >&2; exit 1
    fi
done

# ---------- run loop -------------------------------------------------------
cd "${CODE_DIR}"
SWEEP_START=$(date +%s)

N_TOTAL=$(( ${#DATASETS[@]} * ${#DT_MODES[@]} * ${#SEEDS[@]} ))
echo "=========================================="
echo "Mamba-MV CONFIRM (3-seed) on REAL benchmark"
echo "  datasets : ${DATASETS[*]}"
echo "  dt_modes : ${DT_MODES[*]}"
echo "  seeds    : ${SEEDS[*]}"
echo "  d_model  : ${D_MODEL}  (d_hidden=${D_HIDDEN})"
echo "  total    : ${N_TOTAL} runs"
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
        WIN_JSON=${HPO_BASE}/winners_${DT_MODE}.json
        # Parse JSON for this DS via python (avoid jq dep).
        read -r LR BS < <("${PYTHON}" -c "
import json, sys
w = json.load(open('${WIN_JSON}'))
e = w.get('${DS}')
if e is None:
    print('NOLR NOBS'); sys.exit(0)
print(e['lr'], e['train_batch_size'])
")
        if [[ "${LR}" == "NOLR" ]]; then
            echo "WARN: no winner for ds=${DS} dt=${DT_MODE} in ${WIN_JSON} — skipping"
            continue
        fi

        for SEED in "${SEEDS[@]}"; do
            CELL=$(( CELL + 1 ))
            VARIANT_DIR="${DT_MODE}_lr-${LR}_bs-${BS}"
            OUTPUT_DIR=${LOG_BASE}/${DS}/${VARIANT_DIR}/seed${SEED}
            mkdir -p "${OUTPUT_DIR}"
            RUN_LOG=${OUTPUT_DIR}/run.log
            RUN_NAME="confirm_real_local_${DS}_${DT_MODE}_lr-${LR}_bs-${BS}_seed${SEED}"

            echo "------------------------------------------"
            echo "[${CELL}/${N_TOTAL}] ds=${DS} dt=${DT_MODE} lr=${LR} bs=${BS} grid_K=${GRID_K} seed=${SEED}"
            echo "  output_dir: ${OUTPUT_DIR}"

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
                --patience "${PATIENCE}"
                --seed "${SEED}"
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
            EXIT_CODE=$?
            CELL_END=$(date +%s)
            ELAPSED=$((CELL_END - CELL_START))
            echo "  done in ${ELAPSED}s exit=${EXIT_CODE}   (log: ${RUN_LOG})"
            echo "${DS},${DT_MODE},${LR},${BS},${GRID_K},${SEED},${EXIT_CODE},${ELAPSED},${OUTPUT_DIR}" >> "${SUMMARY}"
        done
    done
done

SWEEP_END=$(date +%s)
echo "=========================================="
echo "Confirm real done. Total wall: $((SWEEP_END - SWEEP_START))s"
echo "Summary CSV: ${SUMMARY}"
echo "Per-run test_metrics.csv at: ${LOG_BASE}/<DS>/<variant>/seed<N>/test_metrics.csv"
echo "=========================================="
