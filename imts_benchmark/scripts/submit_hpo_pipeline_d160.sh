#!/usr/bin/env bash
# Submit the d_model=160 Mamba-MV HPO + aggregate + 5-seed confirmation pipeline.
# Scope: Phase 3 (async dense), multisin_high_irreg only.
#
# Dependency graph:
#   J1: HPO sweep d=160  (array 1-27, seed=1)
#   J2: aggregate        (afterok J1) — writes phase3_d160/winners_by_mode.json
#   J3: winner-confirm   (afterok J2, array 1-15 — 3 dt_modes x 5 seeds)
#
# Usage:
#   export WANDB_API_KEY="..."
#   bash submit_hpo_pipeline_d160.sh
#
# Add DRY_RUN=1 to print the sbatch commands without submitting.

set -euo pipefail

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set. Run: export WANDB_API_KEY=..." >&2
    exit 1
fi
_KEY_LEN=${#WANDB_API_KEY}
_KEY_CLEAN_LEN=$(printf '%s' "$WANDB_API_KEY" | tr -cd 'A-Za-z0-9_' | wc -c)
if [[ "$_KEY_LEN" -ne "$_KEY_CLEAN_LEN" ]] || [[ "$_KEY_LEN" -lt 30 ]]; then
    echo "ERROR: WANDB_API_KEY appears malformed (len=$_KEY_LEN, clean-len=$_KEY_CLEAN_LEN)." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

S_HPO=${SCRIPT_DIR}/run_mamba_mv_hpo_phase3_d160.sbatch
S_AGG=${SCRIPT_DIR}/run_aggregate_hpo_d160.sbatch
S_WIN=${SCRIPT_DIR}/run_mamba_mv_winner_phase3_d160.sbatch

for f in "$S_HPO" "$S_AGG" "$S_WIN"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

sb() {
    local tag=$1; shift
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "  DRY_RUN[${tag}]: sbatch $*" >&2
        echo "0"
        return
    fi
    local jid
    jid=$(sbatch --parsable "$@")
    echo "  ${tag} -> JobID=${jid}" >&2
    echo "$jid"
}

echo "=== Submitting d_model=160 HPO pipeline ==="
J1=$(sb "HPO P3 d160"      "$S_HPO")
J2=$(sb "aggregate d160"   --dependency=afterok:${J1} "$S_AGG")
J3=$(sb "winner P3 d160"   --dependency=afterok:${J2} "$S_WIN")

echo ""
echo "=== Submitted. Check with: squeue -u $USER ==="
echo ""
echo "Pipeline:"
echo "  J1 HPO sweep    (${J1})"
echo "  J2 aggregate    (${J2})  — afterok J1"
echo "  J3 5-seed winner (${J3})  — afterok J2"
