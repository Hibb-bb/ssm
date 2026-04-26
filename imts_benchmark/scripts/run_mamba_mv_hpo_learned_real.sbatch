#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=mv_hpo_lrn_real
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/hpo_mamba_mv/learned/slurm_%A_%a.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/hpo_mamba_mv/learned/slurm_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-18
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# HPO wave 2: Mamba-MV dt_mode=learned on real T-PatchGNN datasets, seed=1.
#   datasets : activity, ushcn (auto_meta) -- PhysioNet skipped per user decision
#   lr       : 1e-4, 5e-4, 2e-3
#   bs       : 64, 128, 256
# Total: 2 * 3 * 3 = 18.
# Mapping: k = TASK_ID-1; ds_idx = k // 9; lr_idx = (k % 9) // 3; bs_idx = k % 3.
#
# Patience=10 (fail-fast, matches the colleague's 'replace' sweep).

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
LOG_BASE=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/hpo_mamba_mv/learned

DATASETS=("activity" "ushcn")
LRS=("1e-4" "5e-4" "2e-3")
BATCH_SIZES=(64 128 256)
DT_MODE=learned
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
    --patience 10 \
    --seed $SEED \
    --output_dir $OUTPUT_DIR \
    --use_wandb \
    --wandb_project TSKing \
    --wandb_run_name "hpo_real_learned_${DS}_${TAG}_seed${SEED}"

EXIT_CODE=$?
END=$(date +%s)
echo "Exit code: $EXIT_CODE  Wall time: $((END-START))s  Done at $(date)"
