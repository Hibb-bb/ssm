#!/usr/bin/env bash
# Submit the UEA classification v2 pipeline on DeltaAI (GH200, aarch64):
#   smoke (1) -> HPO (72) afterok:smoke -> aggregator afterany:HPO -> final (24) afterok:aggregator
#
# Usage:
#   bash imts_benchmark/scripts/submit_uea_cls_pipeline_v2_delta.sh
#
# Env overrides:
#   DRY_RUN=1     print sbatch commands without submitting
#   SKIP_SMOKE=1  skip the smoke gate (use only after smoke has passed once)
#
# Filesystem note: /projects/bfrf/ is shared with Delta x86, so code, data, env,
# and output dirs carry across. The Delta x86 chain writes to the same output
# root — only run one chain at a time per cluster, or both will write to
# output/log/uea_cls_v2/hpo/ in parallel.

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
    SMOKE_ID=$(submit "${SCRIPTS}/run_uea_cls_smoke_v2_delta.sbatch")
    echo "[uea-v2-dai] smoke job_id=${SMOKE_ID}"
    DEP_SMOKE="--dependency=afterok:${SMOKE_ID}"
else
    echo "[uea-v2-dai] SKIP_SMOKE=1 — submitting HPO without smoke gate."
fi

HPO_ID=$(submit ${DEP_SMOKE} "${SCRIPTS}/run_uea_cls_hpo_v2_delta.sbatch")
echo "[uea-v2-dai] HPO job_id=${HPO_ID}  (72 tasks)"

AGG_ID=$(submit --dependency=afterany:${HPO_ID} "${SCRIPTS}/run_aggregate_uea_hpo_v2_delta.sbatch")
echo "[uea-v2-dai] aggregator job_id=${AGG_ID}  (afterany HPO)"

FINAL_ID=$(submit --dependency=afterok:${AGG_ID} "${SCRIPTS}/run_uea_cls_final_v2_delta.sbatch")
echo "[uea-v2-dai] final job_id=${FINAL_ID}  (24 tasks; afterok aggregator)"

echo ""
echo "Submitted. Monitor:"
echo "  squeue -u \$USER"
if [[ -n "${SMOKE_ID:-}" ]]; then
    echo "  squeue -j ${SMOKE_ID},${HPO_ID},${AGG_ID},${FINAL_ID}"
else
    echo "  squeue -j ${HPO_ID},${AGG_ID},${FINAL_ID}"
fi
echo "Outputs at: /projects/bfrf/seojininus/ssm/output/log/uea_cls_v2/"
