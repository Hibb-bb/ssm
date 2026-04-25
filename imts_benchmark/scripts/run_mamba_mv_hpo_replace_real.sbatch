#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=mv_hpo_repl_real
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/hpo_mamba_mv/replace/slurm_%A_%a.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/hpo_mamba_mv/replace/slurm_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-27
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# HPO wave 1: Mamba-MV dt_mode=replace on real T-PatchGNN datasets, seed=1.
#   datasets : physionet, activity, ushcn (auto_meta)
#   lr       : 1e-4, 5e-4, 2e-3
#   bs       : 64, 128, 256
# Total: 3 * 3 * 3 = 27.
# Mapping: k = TASK_ID-1; ds_idx = k // 9; lr_idx = (k % 9) // 3; bs_idx = k % 3.
#
# Patience=10 (fail-fast). Bad configs die early on plateau, good configs
# survive. Winner-confirm uses patience=50 for full convergence.

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
LOG_BASE=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/hpo_mamba_mv/replace

DATASETS=("physionet" "activity" "ushcn")
LRS=("1e-4" "5e-4" "2e-3")
BATCH_SIZES=(64 128 256)
DT_MODE=replace
SEED=1

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
DS_IDX=$(( K / 9 ))
LR_IDX=$(( (K % 9) / 3 ))
BS_IDX=$(( K % 3 ))

DS=${DATASETS[$DS_IDX]}
LR=${LRS[$LR_IDX]}
BS=${BATCH_SIZES[$BS_IDX]}

case $DS in
    physionet) GRID_K=256 ;;
    activity)  GRID_K=128 ;;
    ushcn)     GRID_K=128 ;;
esac

TAG=lr-${LR}_bs-${BS}
OUTPUT_DIR=${LOG_BASE}/${DS}/${TAG}/seed${SEED}
mkdir -p $OUTPUT_DIR
export WANDB_DIR=${OUTPUT_DIR}

echo "=========================================="
echo "Mamba-MV HPO  ds=$DS  dt_mode=$DT_MODE  lr=$LR  bs=$BS  seed=$SEED  grid_K=$GRID_K"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Time: $(date)"
echo ""

cd $CODE_DIR
START_TIME=$(date +%s)

srun ${MAMBA_ENV}/bin/python -m imts_benchmark.mamba_mv.train_mv \
    --data_root $DATA_ROOT \
    --regime $DS \
    --auto_meta \
    --dt_mode $DT_MODE \
    --grid_K $GRID_K \
    --lr $LR \
    --train_batch_size $BS \
    --patience 10 \
    --seed $SEED \
    --output_dir $OUTPUT_DIR \
    --use_wandb \
    --wandb_project TSKing \
    --wandb_run_name "hpo_real_replace_${DS}_${TAG}_seed${SEED}"

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Exit code: $EXIT_CODE"
echo "Wall time: ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "Done at $(date)"
