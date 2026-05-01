#!/usr/bin/env bash
# Submit the Mamba-MV HPO chain on MIMIC (Delta x86 / H200):
#   smoke (1) -> sweep (27) afterok:smoke -> aggregator afterany:sweep -> 5-seed confirm (15) afterok:aggregator
#
# Usage:
#   bash imts_benchmark/scripts/submit_hpo_mimic_delta.sh
#
# Env overrides:
#   DRY_RUN=1     print sbatch commands without submitting
#   SKIP_SMOKE=1  skip the smoke gate (use only after a smoke has passed once)

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
if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
    SMOKE_ID=$(submit "${SCRIPTS}/run_mimic_smoke_delta_x86.sbatch")
    echo "[mimic] smoke job_id=${SMOKE_ID}"
    DEP_SMOKE="--dependency=afterok:${SMOKE_ID}"
else
    echo "[mimic] SKIP_SMOKE=1 — submitting sweep without smoke gate."
fi

SWEEP_ID=$(submit ${DEP_SMOKE} "${SCRIPTS}/run_mamba_mv_hpo_mimic_delta_x86.sbatch")
echo "[mimic] sweep job_id=${SWEEP_ID}  (27 tasks)"

AGG_ID=$(submit --dependency=afterany:${SWEEP_ID} "${SCRIPTS}/run_aggregate_hpo_mimic_delta_x86.sbatch")
echo "[mimic] aggregator job_id=${AGG_ID}  (afterany sweep)"

CONFIRM_ID=$(submit --dependency=afterok:${AGG_ID} "${SCRIPTS}/run_mamba_mv_confirm_mimic_delta_x86.sbatch")
echo "[mimic] confirm job_id=${CONFIRM_ID}  (15 tasks; afterok aggregator)"

echo ""
echo "Submitted. Monitor:"
echo "  squeue -u \$USER"
if [[ -n "${SMOKE_ID:-}" ]]; then
    echo "  squeue -j ${SMOKE_ID},${SWEEP_ID},${AGG_ID},${CONFIRM_ID}"
else
    echo "  squeue -j ${SWEEP_ID},${AGG_ID},${CONFIRM_ID}"
fi
