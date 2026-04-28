#!/usr/bin/env bash
# Submit the Mamba-MV HPO chain on PhysioNet (Delta GH):
#   sweep (27 cells)  -> aggregator (CPU)  -> confirm (15 = 3 dt × 5 seeds)
#
# Dependency model:
#   sweep   --dependency=afterok:<smoke>   (gate on the smoke job's success)
#   aggreg  --dependency=afterany:<sweep>  (run even if some cells failed —
#                                           the python script warns and only
#                                           bails if no usable cells per dt_mode)
#   confirm --dependency=afterok:<aggreg>  (run only if winners JSON was written
#                                           with all 3 dt_modes)
#
# Usage:
#   bash submit_hpo_physionet_delta.sh                # auto-detect smoke
#   SMOKE_ID=2197568 bash submit_hpo_physionet_delta.sh
#   DRY_RUN=1 bash submit_hpo_physionet_delta.sh
#
# If SMOKE_ID is empty/unset, the sweep is submitted without a dependency
# (use only after smoke has been verified PASS once).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS="${REPO_DIR}/imts_benchmark/scripts"

submit () {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY: sbatch $*" >&2
        echo "1234567"
    else
        sbatch --parsable "$@"
    fi
}

DEP_SMOKE=""
if [[ -n "${SMOKE_ID:-}" ]]; then
    DEP_SMOKE="--dependency=afterok:${SMOKE_ID}"
    echo "[hpo-submit] gating sweep on smoke job ${SMOKE_ID}"
else
    echo "[hpo-submit] WARNING: no SMOKE_ID provided; sweep will run without smoke gating."
fi

SWEEP_ID=$(submit ${DEP_SMOKE} "${SCRIPTS}/run_mamba_mv_hpo_physionet_delta.sbatch")
echo "[hpo-submit] sweep job_id=${SWEEP_ID}  (27 tasks)"

AGG_ID=$(submit --dependency=afterany:${SWEEP_ID} "${SCRIPTS}/run_aggregate_hpo_physionet_delta.sbatch")
echo "[hpo-submit] aggregator job_id=${AGG_ID}  (afterany sweep)"

CONFIRM_ID=$(submit --dependency=afterok:${AGG_ID} "${SCRIPTS}/run_mamba_mv_confirm_physionet_delta.sbatch")
echo "[hpo-submit] confirm job_id=${CONFIRM_ID}  (15 tasks; afterok aggregator)"

echo ""
echo "Submitted. Monitor:"
echo "  squeue -u \$USER -j ${SWEEP_ID},${AGG_ID},${CONFIRM_ID}"
