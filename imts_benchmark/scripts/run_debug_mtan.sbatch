#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=mtan_dbg
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/debug/mtan_%j.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/debug/mtan_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Quick 5-epoch GPU sanity for mTAN at full size before launching the array.
# v2: runs on data_correct / multisin_med_irreg (Phase 2 rebuilt data).

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
OUTPUT_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/debug/mtan_run

mkdir -p $OUTPUT_DIR
cd $CODE_DIR

srun ${MAMBA_ENV}/bin/python -m imts_benchmark.mtan_forecaster.train_mtan \
    --regime multisin_med_irreg \
    --seed 1 \
    --max_epochs 5 \
    --output_dir $OUTPUT_DIR
