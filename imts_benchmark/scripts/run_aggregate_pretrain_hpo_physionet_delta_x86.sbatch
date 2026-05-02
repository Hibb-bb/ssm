#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8-interactive
#SBATCH --job-name=pre_pick_phy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/hpo_mamba_pretrain_phy/aggregate_%j.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/hpo_mamba_pretrain_phy/aggregate_%j.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# CPU-bound: pick val/MSE-best (lr, eff_bs) per dt_mode from the 27-cell pretrain
# HPO sweep, write winners_by_dt_mode.json under hpo_mamba_pretrain_phy/.

set -euo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
HPO_DIR=${CODE_DIR}/output/log/imts_benchmark_v2_real/hpo_mamba_pretrain_phy

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

cd "${CODE_DIR}"
echo "Aggregating pretrain HPO winners from ${HPO_DIR}"
echo "Time: $(date)"

python -m imts_benchmark.eval.aggregate_hpo_winners_real \
    --hpo_log_dir "${HPO_DIR}" \
    --regime physionet

EXIT_CODE=$?
echo "aggregate exit code: ${EXIT_CODE}  $(date)"
exit ${EXIT_CODE}
