#!/usr/bin/env bash
# Submit PhysioNet on Delta GH: smoke first, then chain S5+RoMAE+Mamba-MV
# full arrays via afterok dependency on the smoke job. Total 5+5+15 = 25
# production array tasks, gated on a single ~30-60 min smoke.
#
# Usage:
#   bash imts_benchmark/scripts/submit_physionet_delta.sh
#
# Env overrides:
#   DRY_RUN=1   print sbatch commands without submitting
#   SKIP_SMOKE=1 submit the 3 full arrays directly (no gating; use only after
#               smoke has passed once)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS="${REPO_DIR}/imts_benchmark/scripts"

submit () {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY: sbatch $*" >&2
        echo "1234567"  # fake jobid on stdout only
    else
        sbatch --parsable "$@"
    fi
}

if [[ "${SKIP_SMOKE:-0}" == "1" ]]; then
    DEP=""
    echo "[submit] SKIP_SMOKE=1 — submitting 3 full arrays without gating."
else
    SMOKE_ID=$(submit "${SCRIPTS}/run_physionet_smoke_delta.sbatch")
    echo "[submit] smoke job_id=${SMOKE_ID}"
    DEP="--dependency=afterok:${SMOKE_ID}"
fi

S5_ID=$(submit ${DEP} "${SCRIPTS}/run_s5_p10_physionet_delta.sbatch")
echo "[submit] s5 array job_id=${S5_ID}  (5 tasks; afterok smoke)"

ROMAE_ID=$(submit ${DEP} "${SCRIPTS}/run_romae_p10_physionet_delta.sbatch")
echo "[submit] romae array job_id=${ROMAE_ID}  (5 tasks; afterok smoke)"

MAMBA_ID=$(submit ${DEP} "${SCRIPTS}/run_mamba_mv_p10_physionet_delta.sbatch")
echo "[submit] mamba_mv array job_id=${MAMBA_ID}  (15 tasks; afterok smoke)"

echo ""
echo "Submitted. Monitor:"
echo "  squeue -u \$USER"
echo "  squeue -j ${S5_ID},${ROMAE_ID},${MAMBA_ID}"
