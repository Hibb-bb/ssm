#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=contiformer_dbg
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/debug/contiformer_%j.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/debug/contiformer_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Quick 5-epoch GPU sanity for ContiFormer at full size BEFORE launching the
# 20-run array. This is critical because:
#   1. ODE-attention backward was never verified on CPU (timed out).
#   2. The L^2-in-projection memory is unknown at full B=128 / T~360.
# If this OOMs, we'll downstep batch_size in run_contiformer.sbatch before
# launching the array.
# v2: runs on data_correct / multisin_med_irreg; actfn_ode default now softplus.

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
OUTPUT_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/debug/contiformer_run

mkdir -p $OUTPUT_DIR
cd $CODE_DIR

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments reduces fragmentation on the
# long allocation chains produced by the ODE solver.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun ${MAMBA_ENV}/bin/python -m imts_benchmark.contiformer_forecaster.train_contiformer \
    --regime multisin_med_irreg \
    --seed 1 \
    --max_epochs 5 \
    --train_batch_size 32 \
    --accumulate_grad_batches 4 \
    --atol_ode 0.5 \
    --rtol_ode 0.5 \
    --d_model 160 \
    --d_inner 640 \
    --n_layers 4 \
    --n_head 4 \
    --d_k 40 \
    --output_dir $OUTPUT_DIR
# REDUCED-ARCHITECTURE ContiFormer: ~1.45M params (vs 7.8M shared budget).
# Full-size ContiFormer OOMs at L=360 on 80GB H100 regardless of batch/step
# knobs (4 independent tests, all failed at 4.95 GiB). The reduction matches
# the ContiFormer paper's own tested scale (hidden_dim 32-64, Chen et al.
# 2023 Section 4). This breaks param-budget fairness and will be documented
# as "ContiFormer was run at 1.45M parameters due to memory constraints".
# Per-layer ODE-attention tensor shrinks from ~5 GB to ~2.65 GB:
#   old: [32, 360, 360, 1, 4, 80] * 4 bytes = 5.3 GB
#   new: [32, 360, 360, 1, 4, 40] * 4 bytes = 2.65 GB
# Combined with n_layers 6->4, step_size 0.1->0.5, should fit in 80GB.
