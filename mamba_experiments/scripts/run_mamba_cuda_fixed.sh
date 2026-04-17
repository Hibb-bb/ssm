#!/bin/bash
#SBATCH --account=p32626
#SBATCH --job-name=mamba_fix
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/mamba_sinusoidal_cuda_fixed/%x_%A_%a.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/mamba_sinusoidal_cuda_fixed/%x_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-15

# Mamba sinusoidal forecasting with the CUDA selective-scan kernel and
# the prediction-region masking fix in mamba_forecaster.MambaForecaster.
#
# Trains 3 dt-variants × 5 seeds on the high_irreg synthetic dataset.
# Run from the repo root or use the absolute CODE_DIR below.

module purge
module load gcc/11.2.0

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
DATA_ROOT=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/tpatchgnn_data_nobs160
LOG_BASE=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/mamba_sinusoidal_cuda_fixed
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_model/ssm/mamba_experiments

VARIANTS=("vanilla_mamba" "mamba_true_dt" "mamba_hybrid_dt")
DT_MODES=("learned" "replace" "additive")
SEEDS=(1 2 3 4 5)

TASK_ID=$SLURM_ARRAY_TASK_ID

SEED_IDX=$(( (TASK_ID - 1) / 3 ))
VARIANT_IDX=$(( (TASK_ID - 1) % 3 ))

SEED=${SEEDS[$SEED_IDX]}
VARIANT=${VARIANTS[$VARIANT_IDX]}
DT_MODE=${DT_MODES[$VARIANT_IDX]}
IRREG="high_irreg"

OUTPUT_DIR=${LOG_BASE}/${VARIANT}/${IRREG}/seed${SEED}

echo "========================================="
echo "Mamba Sinusoidal (CUDA, leakage-fixed)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Variant: $VARIANT (dt_mode=$DT_MODE)"
echo "Irregularity: $IRREG"
echo "Seed: $SEED"
echo "Output: $OUTPUT_DIR"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Time: $(date)"
echo ""
echo "FIX: prediction-region values zeroed before forward pass"
echo ""

${MAMBA_ENV}/bin/python -c "
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
print('CUDA selective_scan_fn loaded OK')
" && echo "CUDA kernel: AVAILABLE" || echo "CUDA kernel: NOT AVAILABLE (falling back to PyTorch)"
echo ""

mkdir -p $OUTPUT_DIR

cd $CODE_DIR

START_TIME=$(date +%s)

srun ${MAMBA_ENV}/bin/python -m forecaster.train \
    --dt_mode $DT_MODE \
    --data_root $DATA_ROOT \
    --irregularity $IRREG \
    --seed $SEED \
    --output_dir $OUTPUT_DIR

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "Exit code: $EXIT_CODE"
echo "Wall time: ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "Done at $(date)"
