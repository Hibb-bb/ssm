#!/usr/bin/env bash
# Submit the Mamba-MV HPO + winner-confirm + aggregate pipeline for the real
# T-PatchGNN datasets (PhysioNet, Activity, USHCN) with dt_mode=replace.
#
# Dependency graph:
#   J1: HPO sweep         (array 1-27 = 3 ds × 3 lr × 3 bs, seed=1, dt=replace, patience=10)
#   J2: aggregate (HPO winners + interim summary)  -- afterany J1
#   J3: winner-confirm    (array 1-15 = 3 ds × 5 seeds, dt=replace, patience=50)  -- afterok J2
#   J4: final aggregate   (cross-model summary)                 -- afterany J3
#
# After J4, output/log/imts_benchmark_v2_real/aggregate/{summary_real.csv,long_real.csv}
# is up to date with all Mamba-MV runs. S5 and RoMAE rows merge in too if those
# jobs have completed by then (aggregate_results.py crawls the whole tree).
#
# Usage:
#   bash submit_pipeline_real.sh          # submit
#   DRY_RUN=1 bash submit_pipeline_real.sh   # print sbatch commands without submitting

set -euo pipefail

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

S_HPO=${SCRIPT_DIR}/run_mamba_mv_hpo_replace_real.sbatch
S_AGG=${SCRIPT_DIR}/run_aggregate_real.sbatch
S_CONFIRM=${SCRIPT_DIR}/run_mamba_mv_hpo_replace_confirm.sbatch

for f in "$S_HPO" "$S_AGG" "$S_CONFIRM"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

sb() {
    # submit + echo + return JobID; honor DRY_RUN
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

echo "=== Submitting Mamba-MV pipeline (real datasets, dt=replace) ==="
J1=$(sb "HPO sweep"        "$S_HPO")
J2=$(sb "aggregate winners" --dependency=afterany:${J1} "$S_AGG")
J3=$(sb "winner confirm"   --dependency=afterok:${J2}  "$S_CONFIRM")
J4=$(sb "final aggregate"  --dependency=afterany:${J3} "$S_AGG")

echo ""
echo "=== Submitted. Pipeline (all dependency-linked) ==="
echo "  J1 HPO sweep        (${J1})  array 1-27, seed=1, patience=10"
echo "  J2 aggregate#1      (${J2})  afterany J1 — picks per-ds winners, writes winners_by_dataset.json"
echo "  J3 winner confirm   (${J3})  afterok J2,  array 1-15, 5 seeds at winning (lr,bs)"
echo "  J4 aggregate#2      (${J4})  afterany J3 — cross-model summary CSV"
echo ""
echo "Monitor:  squeue -u \$USER -j ${J1},${J2},${J3},${J4}"
echo "Outputs:  /projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/"
