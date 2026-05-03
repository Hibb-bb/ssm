#!/usr/bin/env bash
# Submit the pretrain-architecture UEA classification chain on Delta x86 H200.
#
#   smoke (1) -> HPO (135) afterok:smoke -> aggregator afterany:HPO -> final (45) afterok:agg
#
# Pretrain encoder (slot-anonymous variates, Moirai binary attention bias, no
# variate-ID embedding, shared readout head) + same HAN-style classification head
# as supervised. Uniform protocol across all 5 UEA datasets:
#   3 dt × 3 LR (1e-4, 3e-4, 1e-3) × 2 BS (16, 32) = 18 cells / dataset HPO
#   patience=10, max_epochs=200 across ALL cells (HPO + final)
# LR/BS ranges chosen to cover all 5 datasets' v2 winner sweet spots (BM/CT/EP/LSST
# winners at lr=3e-4; HB winner at lr=1e-4; BS=16 or 32 wins everywhere).
#
# Independent of the in-flight v3 / v3-addon supervised chains. Output root
# output/log/uea_cls_pretrain/ — does not collide with v2/v3 trees.
#
# Usage:
#   bash imts_benchmark/scripts/submit_uea_cls_pretrain_pipeline_delta_x86.sh
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
    SMOKE_ID=$(submit "${SCRIPTS}/run_uea_cls_pretrain_smoke_delta_x86.sbatch")
    echo "[uea-pretrain] smoke      job_id=${SMOKE_ID}"
    DEP_SMOKE="--dependency=afterok:${SMOKE_ID}"
else
    echo "[uea-pretrain] SKIP_SMOKE=1 — submitting HPO without smoke gate."
fi

HPO_ID=$(submit ${DEP_SMOKE} "${SCRIPTS}/run_uea_cls_pretrain_hpo_delta_x86.sbatch")
echo "[uea-pretrain] HPO        job_id=${HPO_ID}    (90 tasks; afterok smoke)"

AGG_ID=$(submit --dependency=afterany:${HPO_ID} "${SCRIPTS}/run_aggregate_uea_pretrain_hpo_delta_x86.sbatch")
echo "[uea-pretrain] aggregator job_id=${AGG_ID}   (afterany HPO)"

FINAL_ID=$(submit --dependency=afterok:${AGG_ID} "${SCRIPTS}/run_uea_cls_pretrain_final_delta_x86.sbatch")
echo "[uea-pretrain] final      job_id=${FINAL_ID} (45 tasks; afterok aggregator)"

echo ""
echo "Submitted. Monitor:"
echo "  squeue -u \$USER"
if [[ -n "${SMOKE_ID:-}" ]]; then
    echo "  squeue -j ${SMOKE_ID},${HPO_ID},${AGG_ID},${FINAL_ID}"
else
    echo "  squeue -j ${HPO_ID},${AGG_ID},${FINAL_ID}"
fi
echo "Outputs at: /projects/bfrf/seojininus/ssm/output/log/uea_cls_pretrain/"
