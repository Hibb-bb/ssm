#!/usr/bin/env bash
# Submit Mamba-MV under the IMM-TSF baseline protocol for one or more TIME-IMM
# datasets (Delta x86, H200). One single-seed run per dataset, no HPO; matches
# _imm_tsf_repo/main_all.py exactly so numbers are directly comparable to paper
# Tables 3-11 'Without Textual Data' rows.
#
# Usage:
#   bash submit_imm_matched_delta.sh                      # all 8 public datasets
#   bash submit_imm_matched_delta.sh imm_fnspid           # one dataset
#   DT_MODE=learned bash submit_imm_matched_delta.sh ...  # override dt_mode (default concat)
#   SEED=2 bash submit_imm_matched_delta.sh ...           # override seed (default 1)
#   DRY_RUN=1 bash submit_imm_matched_delta.sh

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS="${REPO_DIR}/imts_benchmark/scripts"

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

DT_MODE="${DT_MODE:-concat}"
SEED="${SEED:-1}"

submit () {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY: sbatch $*" >&2
        echo "DRY-1234567"
    else
        sbatch --parsable "$@"
    fi
}

for DS in "${DATASETS[@]}"; do
    DATA_REGIME_DIR="${REPO_DIR}/time_imm_data/${DS}"
    if [[ ! -f "${DATA_REGIME_DIR}/norm_stats.json" ]]; then
        echo "[matched-submit] ERROR: ${DATA_REGIME_DIR}/norm_stats.json missing; run data_pipeline/import_timeimm.py first" >&2
        continue
    fi

    JID=$(submit \
        --export=ALL,DS_NAME=${DS},DT_MODE=${DT_MODE},SEED=${SEED} \
        --job-name=mv_matched_${DS} \
        "${SCRIPTS}/run_mamba_mv_matched_imm_delta_x86.sbatch")
    echo "[matched-submit] ${DS}  dt=${DT_MODE}  seed=${SEED}  job_id=${JID}"
done

echo ""
echo "Done. Monitor with:"
echo "  squeue -u \$USER --noheader -o '%.10i %.30j %.8T %.5D %R'"
