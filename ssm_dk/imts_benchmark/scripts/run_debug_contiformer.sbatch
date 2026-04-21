#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=contiformer_dbg
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v1/debug/contiformer_%j.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v1/debug/contiformer_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Quick 2-epoch GPU sanity for ContiFormer at full size BEFORE launching the
# 20-run array. This is critical because:
#   1. ODE-attention backward was never verified on CPU (timed out).
#   2. The L^2-in-projection memory is unknown at full B=128 / T~360.
# If this OOMs, we'll downstep batch_size in run_contiformer.sbatch before
# launching the array.

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
OUTPUT_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v1/debug/contiformer_run

mkdir -p $OUTPUT_DIR
cd $CODE_DIR

srun ${MAMBA_ENV}/bin/python -m imts_benchmark.contiformer_forecaster.train_contiformer \
    --regime sparse_dependent \
    --seed 1 \
    --max_epochs 2 \
    --output_dir $OUTPUT_DIR
