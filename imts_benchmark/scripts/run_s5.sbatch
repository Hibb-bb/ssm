#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=s5_fc
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/s5/slurm_%A_%a.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/s5/slurm_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-20
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# S5 forecaster baseline (Smith et al. ICLR 2023) via Kwaijtaal's s5-pytorch
# port (v0.2.1). Per-variate SSM stacks consuming per-step Δt natively.
#   regimes : multisin_regular, multisin_low_irreg, multisin_med_irreg, multisin_high_irreg
#   seeds   : 1..5
# Total runs: 4 * 5 = 20.
#
# Param budget: d_model=384, n_layers=6, state_dim=96 -> 7.79M (-0.10% vs
# shared 7.8M target). Matches Mamba-MV/RoMAE/mTAN budget.

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
LOG_BASE=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2

REGIMES=("multisin_regular" "multisin_low_irreg" "multisin_med_irreg" "multisin_high_irreg")
SEEDS=(1 2 3 4 5)

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
REGIME_IDX=$(( K / 5 ))
SEED_IDX=$(( K % 5 ))

REGIME=${REGIMES[$REGIME_IDX]}
SEED=${SEEDS[$SEED_IDX]}

OUTPUT_DIR=${LOG_BASE}/s5/${REGIME}/default/seed${SEED}
mkdir -p $OUTPUT_DIR

echo "========================================="
echo "S5 forecaster  regime=$REGIME  seed=$SEED"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Time: $(date)"
echo ""

${MAMBA_ENV}/bin/python -c "import s5; print('s5-pytorch import OK')" \
    && echo "S5 deps: AVAILABLE" || echo "S5 deps: MISSING (run pip install s5-pytorch)"
echo ""

cd $CODE_DIR
START_TIME=$(date +%s)

srun ${MAMBA_ENV}/bin/python -m imts_benchmark.s5_forecaster.train_s5 \
    --regime $REGIME \
    --seed $SEED \
    --lr 1e-3 \
    --weight_decay 0.05 \
    --train_batch_size 32 \
    --accumulate_grad_batches 4 \
    --output_dir $OUTPUT_DIR
# Paper-native recipe (Smith et al. ICLR 2023, S5_pendulum/run_train.py):
#   lr=1e-3 (vs shared 5e-4), wd=0.05 applied only to non-SSM params
#   (Lambda/log_step/D/B/C excluded in configure_optimizers).
#   Architecture defaults: d_model=128, state_dim=256, n_layers=6  -> ~1.4M params.
#   Parallel-scan backward memory ~ B*V*L*state_dim: at B=128, V=3, state=256
#   this OOMs on H100 80GB. Keeping physical B=32 accum=4 (effective 128).

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "Exit code: $EXIT_CODE"
echo "Wall time: ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "Done at $(date)"
