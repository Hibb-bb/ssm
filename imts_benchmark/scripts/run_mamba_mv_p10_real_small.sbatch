#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=mv_p10_real
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/mamba_mv_p10/slurm_%A_%a.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/mamba_mv_p10/slurm_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-10
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Mamba-MV at HPO winner config + patience=10 (matches T-PatchGNN protocol).
# HPO winners (val/mse-best from colleague's 27-cell sweep, dt_mode=replace):
#   activity: lr=2e-3, bs=128  (val/mse=0.002832)
#   ushcn:    lr=5e-4, bs=64   (val/mse=0.682333)
# Note: colleague's confirm used (lr=1e-4, bs=256) for ushcn — 2nd-best by val.
# We use the val/mse winner here per the documented selection protocol.
#
# Mapping: k = TASK_ID-1; ds_idx = k // 5; sd_idx = k % 5.

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

# Load WANDB_API_KEY: env first, then ~/.bashrc, finally ~/.netrc as fallback.
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

DATASETS=("activity" "ushcn")
SEEDS=(1 2 3 4 5)
DT_MODE=replace
PATIENCE=10

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
DS_IDX=$(( K / 5 ))
SD_IDX=$(( K % 5 ))

DS=${DATASETS[$DS_IDX]}
SEED=${SEEDS[$SD_IDX]}

# Per-dataset HPO winner (lr, bs) and grid_K
case $DS in
    activity)
        LR=2e-3
        BS=128
        GRID_K=128
        ;;
    ushcn)
        LR=5e-4
        BS=64
        GRID_K=128
        ;;
esac

OUTPUT_DIR=${LOG_BASE}/mamba_mv_p10/${DS}/${DT_MODE}_lr-${LR}_bs-${BS}/seed${SEED}
mkdir -p $OUTPUT_DIR
export WANDB_DIR=${OUTPUT_DIR}

echo "=========================================="
echo "Mamba-MV (p10)  ds=$DS  dt=$DT_MODE  lr=$LR  bs=$BS  grid_K=$GRID_K  seed=$SEED  patience=$PATIENCE"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Time: $(date)"

cd $CODE_DIR
START=$(date +%s)

srun ${MAMBA_ENV}/bin/python -m imts_benchmark.mamba_mv.train_mv \
    --data_root $DATA_ROOT \
    --regime $DS \
    --auto_meta \
    --dt_mode $DT_MODE \
    --grid_K $GRID_K \
    --lr $LR \
    --train_batch_size $BS \
    --patience $PATIENCE \
    --seed $SEED \
    --output_dir $OUTPUT_DIR \
    --use_wandb \
    --wandb_project TSKing \
    --wandb_run_name "p10_mv_${DS}_${DT_MODE}_lr-${LR}_bs-${BS}_seed${SEED}"

EXIT_CODE=$?
END=$(date +%s)
echo "Exit code: $EXIT_CODE  Wall time: $((END-START))s  Done at $(date)"
