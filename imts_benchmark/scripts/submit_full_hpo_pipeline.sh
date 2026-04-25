#!/usr/bin/env bash
# Submit the full Mamba-MV HPO + confirmation + comparison pipeline.
#
# Dependency graph:
#   J1: HPO P3    (array 1-27)
#   J2: HPO P4-2  (array 1-27)
#   J3: aggregate (afterany J1, J2) — picks per-phase winners
#   J4: winner-confirm P3   (afterok J3, array 1-15 — 3 dt_modes × 5 seeds)
#   J5: winner-confirm P4-2 (afterok J3, array 1-15 — 3 dt_modes × 5 seeds)
#   J6: final comparison    (afterany J4, J5)
#
# Usage:
#   export WANDB_API_KEY="..."
#   bash submit_full_hpo_pipeline.sh
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
    echo "       Must be alphanumeric+underscore only; no whitespace/quotes/newlines." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

S_HPO_P3=${SCRIPT_DIR}/run_mamba_mv_hpo_phase3.sbatch
S_HPO_P42=${SCRIPT_DIR}/run_mamba_mv_hpo_phase4_2.sbatch
S_AGG=${SCRIPT_DIR}/run_aggregate_hpo.sbatch
S_WIN_P3=${SCRIPT_DIR}/run_mamba_mv_winner_phase3.sbatch
S_WIN_P42=${SCRIPT_DIR}/run_mamba_mv_winner_phase4_2.sbatch
S_FINAL=${SCRIPT_DIR}/run_final_comparison.sbatch

for f in "$S_HPO_P3" "$S_HPO_P42" "$S_AGG" "$S_WIN_P3" "$S_WIN_P42" "$S_FINAL"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

sb() {
    # submit + echo + return JobID
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

echo "=== Submitting HPO pipeline ==="
J1=$(sb "HPO P3"     "$S_HPO_P3")
J2=$(sb "HPO P4-2"   "$S_HPO_P42")
J3=$(sb "aggregate"  --dependency=afterany:${J1}:${J2} "$S_AGG")
J4=$(sb "winner P3"  --dependency=afterok:${J3}         "$S_WIN_P3")
J5=$(sb "winner P4-2" --dependency=afterok:${J3}        "$S_WIN_P42")
J6=$(sb "final cmp"  --dependency=afterany:${J4}:${J5}  "$S_FINAL")

echo ""
echo "=== Submitted. Check with: squeue -u $USER ==="
echo ""
echo "Pipeline (all jobs are dependency-linked, nothing to do until email arrives):"
echo "  J1 HPO P3       (${J1})"
echo "  J2 HPO P4-2     (${J2})"
echo "  J3 aggregate    (${J3})  — afterany J1,J2"
echo "  J4 winner P3    (${J4})  — afterok J3"
echo "  J5 winner P4-2  (${J5})  — afterok J3"
echo "  J6 final cmp    (${J6})  — afterany J4,J5"
