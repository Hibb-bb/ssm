#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=s5_real
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/s5/slurm_%A_%a.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/s5/slurm_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-15
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# S5 baseline on real T-PatchGNN datasets (Phase 5).
#   datasets : physionet, activity, ushcn (auto_meta)
#   seeds    : 1..5
# Total: 3 * 5 = 15.
# Mapping: k = TASK_ID-1; ds_idx = k // 5; sd_idx = k % 5.
#
# Paper-native config (Smith et al. ICLR 2023): lr=1e-3, wd=0.05 (excl. SSM
# spectral params), d_model=128, state_dim=256, n_layers=6. Physical batch=32
# with accum=4 (effective 128) to avoid parallel-scan OOM.

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set. Run: export WANDB_API_KEY=... before sbatch." >&2
    exit 1
fi
_KEY_LEN=${#WANDB_API_KEY}
_KEY_CLEAN_LEN=$(printf '%s' "$WANDB_API_KEY" | tr -cd 'A-Za-z0-9_' | wc -c)
if [[ "$_KEY_LEN" -ne "$_KEY_CLEAN_LEN" ]] || [[ "$_KEY_LEN" -lt 30 ]]; then
    echo "ERROR: WANDB_API_KEY appears malformed (len=$_KEY_LEN, clean-len=$_KEY_CLEAN_LEN)." >&2
    exit 1
fi
export WANDB_MODE=online
unset WANDB_SILENT 2>/dev/null || true

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
DATA_ROOT=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/tpatchgnn_data
LOG_BASE=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real

DATASETS=("physionet" "activity" "ushcn")
SEEDS=(1 2 3 4 5)

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
DS_IDX=$(( K / 5 ))
SD_IDX=$(( K % 5 ))

DS=${DATASETS[$DS_IDX]}
SEED=${SEEDS[$SD_IDX]}

OUTPUT_DIR=${LOG_BASE}/s5/${DS}/default/seed${SEED}
mkdir -p $OUTPUT_DIR
export WANDB_DIR=${OUTPUT_DIR}

echo "=========================================="
echo "S5  ds=$DS  seed=$SEED"
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
    --data_root $DATA_ROOT \
    --regime $DS \
    --auto_meta \
    --seed $SEED \
    --lr 1e-3 \
    --weight_decay 0.05 \
    --train_batch_size 32 \
    --accumulate_grad_batches 4 \
    --output_dir $OUTPUT_DIR \
    --use_wandb \
    --wandb_project TSKing \
    --wandb_run_name "real_s5_${DS}_seed${SEED}"

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Exit code: $EXIT_CODE"
echo "Wall time: ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "Done at $(date)"
