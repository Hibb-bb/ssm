#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=mvcmp_agg
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v1/aggregate_%j.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v1/aggregate_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Aggregate results across all completed runs.  Submit after the Mamba-MV
# and RoMAE array jobs finish, e.g.
#   sbatch --dependency=afterany:<mamba_jobid>:<romae_jobid> run_aggregate.sbatch

module purge

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
LOG_BASE=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v1

cd $CODE_DIR

${MAMBA_ENV}/bin/python -m imts_benchmark.eval.aggregate_results \
    --results_root ${LOG_BASE} \
    --output_csv ${LOG_BASE}/summary.csv \
    --long_csv   ${LOG_BASE}/long.csv
