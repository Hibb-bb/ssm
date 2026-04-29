#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=mrope_p10_us
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/mamba_rope_p10/ushcn/slurm_%A_%a.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/mamba_rope_p10/ushcn/slurm_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-5
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Mamba-RoPE 5-seed confirm at HPO winner.
# USHCN HPO winner (val/mse-best from 9-cell sweep, seed=1): lr=5e-4, bs=64.
# dt_mode=learned, patience=10 (matches mamba_mv_p10 pattern).

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

if [[ -z "${WANDB_API_KEY:-}" ]] && [[ -f ~/.bashrc ]]; then
    source ~/.bashrc 2>/dev/null || true
fi
if [[ -z "${WANDB_API_KEY:-}" ]] && [[ -f ~/.netrc ]]; then
    export WANDB_API_KEY=$(awk '/api.wandb.ai/{f=1} f && /password/{print $2; exit}' ~/.netrc)
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set." >&2
    exit 1
fi
export WANDB_MODE=online

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
DATA_ROOT=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/tpatchgnn_data
LOG_BASE=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real

DS=ushcn
DT_MODE=learned
LR=5e-4
BS=64
PATIENCE=10
SEED=$SLURM_ARRAY_TASK_ID

OUTPUT_DIR=${LOG_BASE}/mamba_rope_p10/${DS}/${DT_MODE}_lr-${LR}_bs-${BS}/seed${SEED}
mkdir -p $OUTPUT_DIR
export WANDB_DIR=${OUTPUT_DIR}

echo "=========================================="
echo "Mamba-RoPE (p10 confirm)  ds=$DS  dt=$DT_MODE  lr=$LR  bs=$BS  seed=$SEED  patience=$PATIENCE"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Time: $(date)"

cd $CODE_DIR
START=$(date +%s)

srun ${MAMBA_ENV}/bin/python -m imts_benchmark.mamba_rope.train_mamba_rope \
    --data_root $DATA_ROOT \
    --regime $DS \
    --auto_meta \
    --dt_mode $DT_MODE \
    --lr $LR \
    --train_batch_size $BS \
    --patience $PATIENCE \
    --seed $SEED \
    --output_dir $OUTPUT_DIR \
    --use_wandb \
    --wandb_project TSKing \
    --wandb_run_name "p10_mrope_${DS}_${DT_MODE}_lr-${LR}_bs-${BS}_seed${SEED}"

EXIT_CODE=$?
END=$(date +%s)
echo "Exit code: $EXIT_CODE  Wall time: $((END-START))s  Done at $(date)"
