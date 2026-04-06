#!/bin/bash
#SBATCH --account=p32626
#SBATCH --job-name=test_mamba_cuda
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/train/mamba/%x_%j.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/train/mamba/%x_%j.err
#SBATCH --time=0:15:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module purge
module load gcc/11.2.0

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
TEST_SCRIPT=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_model/mamba_forecaster/test_cuda_kernel.py

echo "========================================="
echo "Mamba CUDA Kernel Test"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Time: $(date)"
echo ""

srun ${MAMBA_ENV}/bin/python ${TEST_SCRIPT}

echo ""
echo "Exit code: $?"
echo "Done at $(date)"
