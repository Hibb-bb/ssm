#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=phase5_agg_v2
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/aggregate/slurm_%j.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/aggregate/slurm_%j.err
#SBATCH --time=0:15:00
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Auto-aggregate Phase-5 v2 results once 5 array jobs (6521087/8/9/90/94) finish.
# Submit with:
#   sbatch --dependency=afterany:6521087:6521088:6521089:6521090:6521094 \
#          imts_benchmark/scripts/run_aggregate_v2_after_rerun.sbatch
#
# Reads the new test_metrics.csv files (which now have both
# target-weighted [mse/mae] and T-PatchGNN-style [mse_tpg/mae_tpg]
# metrics) and produces:
#   aggregate/phase5_consolidated_v2.csv
#   aggregate/phase5_consolidated_v2.md
#   aggregate/phase5_bars_v2_{activity,ushcn,combined}.{pdf,png}

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk

cd "${CODE_DIR}"

echo "=========================================="
echo "Phase-5 v2 aggregator + bar charts"
echo "Time: $(date)"
echo ""

echo "--- Step 1: aggregate to CSV + markdown ---"
${MAMBA_ENV}/bin/python -m imts_benchmark.eval.aggregate_phase5_v2

echo ""
echo "--- Step 2: bar charts ---"
${MAMBA_ENV}/bin/python -m imts_benchmark.eval.plot_phase5_bars_v2

echo ""
echo "Done at $(date)"
