#!/usr/bin/env bash
# Submit the UEA classification v3 chain on Delta x86 H200:
#   HB HPO (48) + LSST HPO (81) -> aggregator afterany:both -> 18-cell final afterok:agg
#
# v3 vs v2 (HB+LSST only — CT/EP keep v2 winners):
#   - Adds dt_mode=concat (v2 had only learned/replace).
#   - Adds head_dropout sweep (v2 inherited Table 12 defaults verbatim).
#   - HB: extends LR DOWN (3e-5, 1e-4, 3e-4, 1e-3) since v2 winner was at LR=1e-4 (lower edge).
#   - LSST: shifts LR DOWN (1e-4, 3e-4, 1e-3) since every v2 cell at LR≥3e-3 collapsed.
#   - Output root: output/log/uea_cls_v3/  (preserves v2 ground-truth tree)
#
# Usage:
#   bash imts_benchmark/scripts/submit_uea_cls_pipeline_v3_delta_x86.sh
#
# Env overrides:
#   DRY_RUN=1     print sbatch commands without submitting

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

HB_ID=$(submit "${SCRIPTS}/run_uea_cls_hpo_v3_HB_delta_x86.sbatch")
echo "[uea-v3] HB HPO     job_id=${HB_ID}    (48 tasks)"

LSST_ID=$(submit "${SCRIPTS}/run_uea_cls_hpo_v3_LSST_delta_x86.sbatch")
echo "[uea-v3] LSST HPO   job_id=${LSST_ID}  (81 tasks)"

AGG_ID=$(submit --dependency=afterany:${HB_ID}:${LSST_ID} "${SCRIPTS}/run_aggregate_uea_hpo_v3_delta_x86.sbatch")
echo "[uea-v3] aggregator job_id=${AGG_ID}   (afterany HB+LSST)"

FINAL_ID=$(submit --dependency=afterok:${AGG_ID} "${SCRIPTS}/run_uea_cls_final_v3_delta_x86.sbatch")
echo "[uea-v3] final      job_id=${FINAL_ID} (18 tasks; afterok aggregator)"

echo ""
echo "Submitted. Monitor:"
echo "  squeue -u \$USER"
echo "  squeue -j ${HB_ID},${LSST_ID},${AGG_ID},${FINAL_ID}"
echo "Outputs at: /projects/bfrf/seojininus/ssm/output/log/uea_cls_v3/"
