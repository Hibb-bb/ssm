#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=hpo_agg_d160
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/hpo_mamba_mv/phase3_d160/slurm_agg_%j.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/hpo_mamba_mv/phase3_d160/slurm_agg_%j.err
#SBATCH --time=0:15:00
#SBATCH --partition=short
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Aggregate d=160 Phase 3 HPO ranked CSVs and pick per-dt_mode winners.
# Writes winners_by_mode.json consumed by run_mamba_mv_winner_phase3_d160.sbatch.

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
LOG_BASE=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/hpo_mamba_mv

cd $CODE_DIR

set -e

echo "=========================================="
echo "HPO aggregation — Phase 3 (d_model=160)"
echo "Time: $(date)"
${MAMBA_ENV}/bin/python -m imts_benchmark.eval.aggregate_hpo_pick_winner \
    --phase_log_dir ${LOG_BASE}/phase3_d160 \
    --regime multisin_high_irreg

echo ""
echo "Done at $(date)"
