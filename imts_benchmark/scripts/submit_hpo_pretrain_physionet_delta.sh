#!/usr/bin/env bash
# Submit the pretrain-architecture Mamba-MV chain on PhysioNet (Delta x86 H200):
#   smoke (1) -> sweep (27) afterok:smoke -> aggregator afterany:sweep -> 5-seed confirm (15) afterok:aggregator
#
# Usage:
#   bash imts_benchmark/scripts/submit_hpo_pretrain_physionet_delta.sh
#
# Env overrides:
#   DRY_RUN=1     print sbatch commands without submitting
#   SKIP_SMOKE=1  skip the smoke gate (use only after smoke has passed once)

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
    SMOKE_ID=$(submit "${SCRIPTS}/run_mamba_pretrain_smoke_physionet_delta_x86.sbatch")
    echo "[pretrain-phy] smoke job_id=${SMOKE_ID}"
    DEP_SMOKE="--dependency=afterok:${SMOKE_ID}"
else
    echo "[pretrain-phy] SKIP_SMOKE=1 — submitting sweep without smoke gate."
fi

SWEEP_ID=$(submit ${DEP_SMOKE} "${SCRIPTS}/run_mamba_pretrain_hpo_physionet_delta_x86.sbatch")
echo "[pretrain-phy] sweep job_id=${SWEEP_ID}  (27 tasks)"

AGG_ID=$(submit --dependency=afterany:${SWEEP_ID} "${SCRIPTS}/run_aggregate_pretrain_hpo_physionet_delta_x86.sbatch")
echo "[pretrain-phy] aggregator job_id=${AGG_ID}  (afterany sweep)"

CONFIRM_ID=$(submit --dependency=afterok:${AGG_ID} "${SCRIPTS}/run_mamba_pretrain_confirm_physionet_delta_x86.sbatch")
echo "[pretrain-phy] confirm job_id=${CONFIRM_ID}  (15 tasks; afterok aggregator)"

echo ""
echo "Submitted. Monitor:"
echo "  squeue -u \$USER"
if [[ -n "${SMOKE_ID:-}" ]]; then
    echo "  squeue -j ${SMOKE_ID},${SWEEP_ID},${AGG_ID},${CONFIRM_ID}"
else
    echo "  squeue -j ${SWEEP_ID},${AGG_ID},${CONFIRM_ID}"
fi
echo "Outputs at: /projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/{hpo_mamba_pretrain_phy,mamba_pretrain_p10}/"
