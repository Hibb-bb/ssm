#!/usr/bin/env bash
# Submit the pretrain-architecture UEA classification chain at 70% drop, under
# the ContiFormer REPO fixed-HP protocol
# (microsoft/physiopro/docs/configs/contiformer_mask_classification.yml verbatim;
# paper text §C.2.1 disagrees with YAML — repo is source-of-truth):
#   smoke (1) -> final (45) afterok:smoke
#
# No HPO — protocol fixes lr=1e-3, bs=16 across all baselines.
# Matching this ensures the head-to-head is apples-to-apples; HPO would give
# us an unfair advantage (per the user's feedback memory on baseline-protocol
# matching).
#
# 5 datasets × 3 dt_modes × 3 seeds {27, 42, 1024} = 45 final cells.
#
# Output root: /projects/bfrf/seojininus/ssm/output/log/uea_cls_pretrain_70d_repo/
#
# Usage:
#   bash imts_benchmark/scripts/submit_uea_cls_pretrain_70d_pipeline_delta_x86.sh
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
    SMOKE_ID=$(submit "${SCRIPTS}/run_uea_cls_pretrain_70d_smoke_delta_x86.sbatch")
    echo "[uea-pretrain-70d] smoke   job_id=${SMOKE_ID}"
    DEP_SMOKE="--dependency=afterok:${SMOKE_ID}"
else
    echo "[uea-pretrain-70d] SKIP_SMOKE=1 — submitting final without smoke gate."
fi

FINAL_ID=$(submit ${DEP_SMOKE} "${SCRIPTS}/run_uea_cls_pretrain_70d_final_delta_x86.sbatch")
echo "[uea-pretrain-70d] final   job_id=${FINAL_ID} (45 tasks; 5 ds × 3 dt × 3 seeds)"

echo ""
echo "Submitted. Monitor:"
echo "  squeue -u \$USER"
if [[ -n "${SMOKE_ID:-}" ]]; then
    echo "  squeue -j ${SMOKE_ID},${FINAL_ID}"
else
    echo "  squeue -j ${FINAL_ID}"
fi
echo "Outputs at: /projects/bfrf/seojininus/ssm/output/log/uea_cls_pretrain_70d_repo/"
