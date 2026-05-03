#!/usr/bin/env bash
# Submit the UEA v3 concat ADD-ON chain on Delta x86 H200, gated on v3 HB+LSST
# completion (job 18007086 = v3 final array).
#
#   HPO (27) afterany:18007086 -> aggregator afterany:HPO -> final (9) afterok:agg
#
# Scope: BM + CT + EP × concat under the EXACT v2 HPO grid (no extra knobs).
# Lets the head-to-head table show all 3 dt_modes for all 5 datasets.
# Output root: output/log/uea_cls_v3_addon/  (separate from v2 and v3 trees).
#
# Usage:
#   bash imts_benchmark/scripts/submit_uea_cls_v3_concat_addon_delta_x86.sh
#
# Env overrides:
#   DRY_RUN=1            print sbatch commands without submitting
#   GATE_JOB=<jobid>     override the v3 completion gate (default: 18007086)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS="${REPO_DIR}/imts_benchmark/scripts"
GATE_JOB="${GATE_JOB:-18007086}"

submit () {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY: sbatch $*" >&2
        echo "1234567"
    else
        sbatch --parsable "$@"
    fi
}

HPO_ID=$(submit --dependency=afterany:${GATE_JOB} "${SCRIPTS}/run_uea_cls_hpo_v3_concat_addon_delta_x86.sbatch")
echo "[uea-v3-addon] HPO        job_id=${HPO_ID}    (27 tasks; afterany v3 final ${GATE_JOB})"

AGG_ID=$(submit --dependency=afterany:${HPO_ID} "${SCRIPTS}/run_aggregate_uea_hpo_v3_concat_addon_delta_x86.sbatch")
echo "[uea-v3-addon] aggregator job_id=${AGG_ID}   (afterany HPO)"

FINAL_ID=$(submit --dependency=afterok:${AGG_ID} "${SCRIPTS}/run_uea_cls_final_v3_concat_addon_delta_x86.sbatch")
echo "[uea-v3-addon] final      job_id=${FINAL_ID} (9 tasks; afterok aggregator)"

echo ""
echo "Submitted. Monitor:"
echo "  squeue -u \$USER"
echo "  squeue -j ${HPO_ID},${AGG_ID},${FINAL_ID}"
echo "Outputs at: /projects/bfrf/seojininus/ssm/output/log/uea_cls_v3_addon/"
