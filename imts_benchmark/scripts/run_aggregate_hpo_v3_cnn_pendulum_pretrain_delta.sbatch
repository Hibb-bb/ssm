#!/bin/bash
#SBATCH --account=bfrf-dtai-gh
#SBATCH --job-name=pt_pend_v3_pick
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gpus-per-node=1
#SBATCH --partition=ghx4-interactive
#SBATCH --time=00:15:00
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/pendulum/hpo_mamba_pretrain_v3_cnn/aggregate_%j.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/pendulum/hpo_mamba_pretrain_v3_cnn/aggregate_%j.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# v3 Pendulum aggregator: pick (lr, B) winner per dt_mode from the
# 27 v3-CNN HPO cells. Reuses v1's aggregator script since axes are identical
# (LR × B × dt_mode, same grid values).

set -euo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba
CODE_DIR=/u/seojininus/ssm
HPO_DIR=/projects/bfrf/seojininus/ssm/output/log/pendulum/hpo_mamba_pretrain_v3_cnn

source /sw/user/python/miniforge3-pytorch-2.5.0/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

cd "${CODE_DIR}"
echo "Aggregating Pendulum v3_cnn HPO winners from ${HPO_DIR}"
echo "Time: $(date)"

python -m imts_benchmark.eval.aggregate_hpo_winners_pendulum \
    --hpo_log_dir "${HPO_DIR}"

EXIT_CODE=$?
echo "aggregate exit code: ${EXIT_CODE}  $(date)"
exit ${EXIT_CODE}
