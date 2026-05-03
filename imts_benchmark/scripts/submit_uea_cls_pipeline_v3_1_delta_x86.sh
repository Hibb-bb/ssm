#!/usr/bin/env bash
# Submit v3.1 cleanup chain on Delta x86 H200, gated on the in-flight v3 final
# (18007086) AND the v3-addon final (18007096) — so we don't race the existing
# pipelines that are still writing into the v3 hpo / final trees.
#
#   concat HPO rerun (43) afterany:both -> aggregator afterok:HPO -> final (12) afterok:agg
#
# Cleans up two issues from v3:
#   1. train_cls.py CLI rejected --dt_mode concat (now patched). All v3 concat HPO
#      cells (16 HB + 27 LSST) failed; this rerun captures them.
#   2. v3 HPO picked HB lr=3e-5; HB final TIMEOUT'd at 2h walltime. v3.1 uses
#      patience=10 / max_epochs=200 uniformly across HPO and final so the picker
#      rewards within-budget convergence and the final actually finishes.
#
# Output dirs: writes into the existing output/log/uea_cls_v3/{hpo,final} trees
# so the merged-tree aggregator picks winners across all 3 dt_modes for HB/LSST.
# Old winner_configs.json is backed up to .v3_pre_v31 by the aggregator.
#
# Usage:
#   bash imts_benchmark/scripts/submit_uea_cls_pipeline_v3_1_delta_x86.sh
#
# Env overrides:
#   DRY_RUN=1                print sbatch commands without submitting
#   GATE_V3_FINAL=<jobid>    override v3 final gate (default: 18007086)
#   GATE_ADDON_FINAL=<jobid> override v3-addon final gate (default: 18007096)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS="${REPO_DIR}/imts_benchmark/scripts"
GATE_V3="${GATE_V3_FINAL:-18007086}"
GATE_ADDON="${GATE_ADDON_FINAL:-18007096}"

submit () {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY: sbatch $*" >&2
        echo "1234567"
    else
        sbatch --parsable "$@"
    fi
}

HPO_ID=$(submit --dependency=afterany:${GATE_V3}:${GATE_ADDON} "${SCRIPTS}/run_uea_cls_hpo_v3_1_concat_HB_LSST_delta_x86.sbatch")
echo "[uea-v31] HPO        job_id=${HPO_ID}    (43 tasks; afterany v3 final ${GATE_V3} + addon final ${GATE_ADDON})"

AGG_ID=$(submit --dependency=afterok:${HPO_ID} "${SCRIPTS}/run_aggregate_uea_hpo_v3_1_delta_x86.sbatch")
echo "[uea-v31] aggregator job_id=${AGG_ID}   (afterok HPO)"

FINAL_ID=$(submit --dependency=afterok:${AGG_ID} "${SCRIPTS}/run_uea_cls_final_v3_1_delta_x86.sbatch")
echo "[uea-v31] final      job_id=${FINAL_ID} (12 tasks; afterok aggregator)"

echo ""
echo "Submitted. Monitor:"
echo "  squeue -u \$USER"
echo "  squeue -j ${HPO_ID},${AGG_ID},${FINAL_ID}"
echo "Outputs at: /projects/bfrf/seojininus/ssm/output/log/uea_cls_v3/  (merged tree)"
