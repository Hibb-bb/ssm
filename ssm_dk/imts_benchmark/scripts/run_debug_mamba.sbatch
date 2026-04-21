#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=mamba_mv_dbg
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v1/debug/mamba_%j.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v1/debug/mamba_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Quick 2-epoch sanity run to verify GPU path before the 30-job array.

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
OUTPUT_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v1/debug/mamba_run

mkdir -p $OUTPUT_DIR
cd $CODE_DIR

${MAMBA_ENV}/bin/python -c "
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
print('CUDA selective_scan_fn OK')
" || { echo 'CUDA kernel missing, aborting'; exit 2; }

srun ${MAMBA_ENV}/bin/python -m imts_benchmark.mamba_mv.train_mv \
    --regime sparse_dependent \
    --dt_mode replace \
    --seed 1 \
    --max_epochs 2 \
    --output_dir $OUTPUT_DIR
