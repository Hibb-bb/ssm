#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=hpo_final
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/hpo_mamba_mv/slurm_final_%j.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/hpo_mamba_mv/slurm_final_%j.err
#SBATCH --time=0:15:00
#SBATCH --partition=short
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Final comparison: aggregate the 5-seed winner confirmation runs and
# write HPO_COMPARISON_*.md tables comparing against the existing
# S5 / RoMAE / pre-HPO Mamba-MV rows on multisin_high_irreg.

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
cd $CODE_DIR

echo "=========================================="
echo "Final comparison (winner vs baselines)"
echo "Time: $(date)"
${MAMBA_ENV}/bin/python -m imts_benchmark.eval.compare_winner_vs_baseline
echo ""
echo "Done at $(date)"
