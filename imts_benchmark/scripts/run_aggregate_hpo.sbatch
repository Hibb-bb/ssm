#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=hpo_agg
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/hpo_mamba_mv/slurm_agg_%j.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/hpo_mamba_mv/slurm_agg_%j.err
#SBATCH --time=0:15:00
#SBATCH --partition=short
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Aggregate HPO ranked CSVs and pick per-phase winners (Phase 3 + Phase 4-2).
# Fails (non-zero exit) if either phase has < 50% usable cells — safety gate
# so the afterok-dep confirmations don't run against garbage.

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
LOG_BASE=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/hpo_mamba_mv

cd $CODE_DIR

set -e

echo "=========================================="
echo "HPO aggregation — Phase 3"
echo "Time: $(date)"
${MAMBA_ENV}/bin/python -m imts_benchmark.eval.aggregate_hpo_pick_winner \
    --phase_log_dir ${LOG_BASE}/phase3 \
    --regime multisin_high_irreg

echo ""
echo "=========================================="
echo "HPO aggregation — Phase 4-2"
echo "Time: $(date)"
${MAMBA_ENV}/bin/python -m imts_benchmark.eval.aggregate_hpo_pick_winner \
    --phase_log_dir ${LOG_BASE}/phase4_2 \
    --regime multisin_high_irreg

echo ""
echo "Done at $(date)"
