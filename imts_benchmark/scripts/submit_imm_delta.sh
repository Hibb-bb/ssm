#!/usr/bin/env bash
# Submit the full Mamba-MV HPO+confirm chain for one or more TIME-IMM datasets
# on Delta x86. Per dataset:
#   sweep (27 cells)  -> aggregator (CPU)  -> confirm (15 = 3 dt x 5 seeds)
#
# Dependencies (per dataset):
#   sweep   --dependency=afterok:<smoke>   (gate on smoke job, if SMOKE_ID set)
#   aggreg  --dependency=afterany:<sweep>  (partial-failure tolerant)
#   confirm --dependency=afterok:<aggreg>  (gate on winners json existing)
#
# Usage:
#   bash submit_imm_delta.sh                # all 8 public datasets
#   bash submit_imm_delta.sh imm_fnspid     # one dataset
#   bash submit_imm_delta.sh imm_fnspid imm_gdelt
#   SMOKE_ID=17965083 bash submit_imm_delta.sh
#   DRY_RUN=1 bash submit_imm_delta.sh

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS="${REPO_DIR}/imts_benchmark/scripts"

# Per-dataset eff_bs grid. Tiny train sets (CESNET ~23, ILINet ~85) use a
# smaller grid so we don't end up with 1-batch epochs.
declare -A EFF_BS_FOR
EFF_BS_FOR[imm_gdelt]='64 128 256'
EFF_BS_FOR[imm_repohealth]='64 128 256'
EFF_BS_FOR[imm_fnspid]='64 128 256'
EFF_BS_FOR[imm_clustertrace]='32 64 128'
EFF_BS_FOR[imm_studentlife]='64 128 256'
EFF_BS_FOR[imm_ilinet]='8 16 32'
EFF_BS_FOR[imm_cesnet]='8 16 32'
EFF_BS_FOR[imm_epa_air]='32 64 128'
EFF_BS_FOR[imm_mimic]='64 128 256'

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

if [[ $# -ge 1 ]]; then
    DATASETS=("$@")
else
    DATASETS=("${DEFAULT_DATASETS[@]}")
fi

submit () {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY: sbatch $*" >&2
        echo "1234567"
    else
        sbatch --parsable "$@"
    fi
}

for DS in "${DATASETS[@]}"; do
    EFF_BS="${EFF_BS_FOR[$DS]:-}"
    if [[ -z "${EFF_BS}" ]]; then
        echo "[imm-submit] WARNING: no EFF_BS_FOR[${DS}] preset; using default '64 128 256'"
        EFF_BS='64 128 256'
    fi

    DATA_REGIME_DIR="${REPO_DIR}/time_imm_data/${DS}"
    if [[ ! -f "${DATA_REGIME_DIR}/norm_stats.json" ]]; then
        echo "[imm-submit] ERROR: ${DATA_REGIME_DIR}/norm_stats.json missing; run data_pipeline/import_timeimm.py first" >&2
        continue
    fi

    DEP_SMOKE=""
    if [[ -n "${SMOKE_ID:-}" ]]; then
        DEP_SMOKE="--dependency=afterok:${SMOKE_ID}"
    fi

    echo "[imm-submit] === ${DS} (eff_bs=${EFF_BS}) ==="

    SWEEP_ID=$(submit ${DEP_SMOKE} \
        --export=ALL,DS_NAME=${DS},EFF_BS_OVERRIDE="${EFF_BS}" \
        --job-name=mv_hpo_${DS} \
        "${SCRIPTS}/run_mamba_mv_hpo_imm_delta_x86.sbatch")
    echo "[imm-submit]   sweep job_id=${SWEEP_ID} (27 tasks)"

    AGG_ID=$(submit --dependency=afterany:${SWEEP_ID} \
        --export=ALL,DS_NAME=${DS},EFF_BS_AGG="${EFF_BS}" \
        --job-name=mv_hpo_pick_${DS} \
        "${SCRIPTS}/run_aggregate_hpo_imm_delta_x86.sbatch")
    echo "[imm-submit]   aggregator job_id=${AGG_ID} (afterany sweep)"

    CONFIRM_ID=$(submit --dependency=afterok:${AGG_ID} \
        --export=ALL,DS_NAME=${DS} \
        --job-name=mv_cf_${DS} \
        "${SCRIPTS}/run_mamba_mv_confirm_imm_delta_x86.sbatch")
    echo "[imm-submit]   confirm job_id=${CONFIRM_ID} (15 tasks; afterok aggregator)"

    echo ""
done

echo "Done. Monitor with:"
echo "  squeue -u \$USER --noheader -o '%.10i %.18j %.8T %.5D %R'"
