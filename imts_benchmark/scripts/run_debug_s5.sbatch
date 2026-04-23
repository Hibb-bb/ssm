#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=s5_dbg
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/debug/s5_%j.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/debug/s5_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Quick 5-epoch GPU sanity for S5 at full size (d_model=384, n_layers=6, 7.79M
# params) on the Phase-2 multisin_med_irreg data. Uses s5-pytorch v0.2.1 (the
# PyTorch port by Kwaijtaal) as the SSM backbone.

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
OUTPUT_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/debug/s5_run

mkdir -p $OUTPUT_DIR
cd $CODE_DIR

${MAMBA_ENV}/bin/python -c "import s5; print('s5-pytorch import OK')" \
    || { echo 's5-pytorch missing, aborting'; exit 2; }

srun ${MAMBA_ENV}/bin/python -m imts_benchmark.s5_forecaster.train_s5 \
    --regime multisin_med_irreg \
    --seed 1 \
    --max_epochs 5 \
    --train_batch_size 32 \
    --accumulate_grad_batches 4 \
    --output_dir $OUTPUT_DIR
# S5 at B=128 OOMs during training (12.66 GiB allocation during backward
# through parallel scan). Effective batch = 32 * 4 = 128 preserves shared
# optimization regime. Parallel-scan memory scales with B*V*L*state_dim
# and the backward pass holds all intermediate states.
