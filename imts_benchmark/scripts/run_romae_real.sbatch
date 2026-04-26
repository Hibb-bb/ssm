#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=romae_real
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/romae/slurm_%A_%a.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/romae/slurm_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=1-15
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# RoMAE baseline on real T-PatchGNN datasets (Phase 5).
#   datasets : physionet, activity, ushcn (auto_meta)
#   seeds    : 1..5
# Total: 3 * 5 = 15.
# Mapping: k = TASK_ID-1; ds_idx = k // 5; sd_idx = k % 5.
#
# RoMAE flat-token attention is O(N²·d). For PhysioNet (V=41, N_max ~1500) at
# bs=128 the kernel OOMs on H100 80GB → physical bs=32 with accum=4 (effective
# 128, matches fair recipe). Activity (V=12) and USHCN (V=5) keep bs=128.

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
    echo "ERROR: WANDB_API_KEY is not set (env, ~/.bashrc, and ~/.netrc all empty)." >&2
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

# PhysioNet OOM mitigation: smaller physical batch, gradient accumulation to
# preserve effective batch=128. Activity/USHCN use the fair-default bs=128.
case $DS in
    physionet) BS_ARGS=(--train_batch_size 32 --val_batch_size 8 --accumulate_grad_batches 4) ;;
    *)         BS_ARGS=() ;;
esac

OUTPUT_DIR=${LOG_BASE}/romae/${DS}/default/seed${SEED}
mkdir -p $OUTPUT_DIR
export WANDB_DIR=${OUTPUT_DIR}

echo "=========================================="
echo "RoMAE  ds=$DS  seed=$SEED  bs_args='${BS_ARGS[@]}'"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Time: $(date)"
echo ""

${MAMBA_ENV}/bin/python -c "import romae; print('romae imported OK')" \
    && echo "RoMAE: AVAILABLE" || echo "RoMAE: MISSING (run pip install)"
echo ""

cd $CODE_DIR
START_TIME=$(date +%s)

srun ${MAMBA_ENV}/bin/python -m imts_benchmark.romae_forecaster.train_romae \
    --data_root $DATA_ROOT \
    --regime $DS \
    --auto_meta \
    --seed $SEED \
    --output_dir $OUTPUT_DIR \
    "${BS_ARGS[@]}" \
    --use_wandb \
    --wandb_project TSKing \
    --wandb_run_name "real_romae_${DS}_seed${SEED}"

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Exit code: $EXIT_CODE"
echo "Wall time: ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "Done at $(date)"
