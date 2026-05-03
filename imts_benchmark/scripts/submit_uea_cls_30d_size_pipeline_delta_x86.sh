#!/usr/bin/env bash
# Submit the 30%-drop UEA size-ablation chain (BIG-vs-SMALL width axis under
# RoMAE Table 4 protocol):
#   smoke (1 cell, BM, size=small) -> final (90 cells) afterok:smoke
#
# Verifies the colleague's d=64 small preset adopts to UEA. The new diagnostic
# pipeline (test_predictions.json + CSVLogger metrics.csv) lands per cell, so
# eval/uea_confusion.py and eval/plot_uea_cls_curves.py both work on the
# resulting tree.
#
# Output root: output/log/uea_cls_pretrain_30d_size/
#   final/<size>/<dataset>/<dt_mode>/seed<S>/final/{summary.json,
#       test_predictions.json, csv/version_0/metrics.csv}
#
# Usage:
#   bash imts_benchmark/scripts/submit_uea_cls_30d_size_pipeline_delta_x86.sh
#
# Env overrides:
#   DRY_RUN=1     print sbatch commands without submitting
#   SKIP_SMOKE=1  skip the smoke gate

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
    SMOKE_ID=$(submit "${SCRIPTS}/run_uea_cls_pretrain_30d_size_smoke_delta_x86.sbatch")
    echo "[uea-30d-size] smoke   job_id=${SMOKE_ID}"
    DEP_SMOKE="--dependency=afterok:${SMOKE_ID}"
else
    echo "[uea-30d-size] SKIP_SMOKE=1 — submitting final without smoke gate."
fi

FINAL_ID=$(submit ${DEP_SMOKE} "${SCRIPTS}/run_uea_cls_pretrain_30d_size_delta_x86.sbatch")
echo "[uea-30d-size] final   job_id=${FINAL_ID} (45 tasks; 5 ds × 3 dt × 3 seeds, size=small only)"

echo ""
echo "Submitted. Monitor:"
echo "  squeue -u \$USER"
echo "Outputs at: /projects/bfrf/seojininus/ssm/output/log/uea_cls_pretrain_30d_size/"
echo ""
echo "After cells start landing:"
echo "  python -m imts_benchmark.eval.plot_uea_cls_curves \\"
echo "      --root output/log/uea_cls_pretrain_30d_size/final/small \\"
echo "      --out_dir output/log/uea_cls_pretrain_30d_size/curves_small"
echo "  python -m imts_benchmark.eval.uea_confusion \\"
echo "      --root output/log/uea_cls_pretrain_30d_size/final/small \\"
echo "      --out_dir output/log/uea_cls_pretrain_30d_size/confusion_small"
