#!/bin/bash
#SBATCH --account=bfrf-dtai-gh
#SBATCH --job-name=pt_pend_v2_pick
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gpus-per-node=1
#SBATCH --partition=ghx4-interactive
#SBATCH --time=00:15:00
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/pendulum/hpo_mamba_pretrain_v2_reg/aggregate_%j.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/pendulum/hpo_mamba_pretrain_v2_reg/aggregate_%j.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# v2 Pendulum aggregator: pick (wd, dropout) winner per dt_mode, carry
# (lr, B) from v1 winners JSON. Submitted with afterany dependency on the
# v2 HPO sweep array; the v2 confirm depends on this in turn.

set -euo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba
CODE_DIR=/u/seojininus/ssm
HPO_DIR=/projects/bfrf/seojininus/ssm/output/log/pendulum/hpo_mamba_pretrain_v2_reg
V1_WINNERS=/projects/bfrf/seojininus/ssm/output/log/pendulum/hpo_mamba_pretrain/winners_by_dt_mode.json

source /sw/user/python/miniforge3-pytorch-2.5.0/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

cd "${CODE_DIR}"
echo "Aggregating Pendulum v2_reg HPO winners from ${HPO_DIR}"
echo "Time: $(date)"

python -m imts_benchmark.eval.aggregate_hpo_winners_pendulum_v2_reg \
    --hpo_log_dir "${HPO_DIR}" \
    --v1_winners_json "${V1_WINNERS}"

EXIT_CODE=$?
echo "aggregate exit code: ${EXIT_CODE}  $(date)"
exit ${EXIT_CODE}
