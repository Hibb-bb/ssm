#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=mamba_hpo_p42
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/hpo_mamba_mv/phase4_2/slurm_%A_%a.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/hpo_mamba_mv/phase4_2/slurm_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-27
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Mamba-MV HPO sweep — Phase 4-2 (random_uniform gap), multisin_high_irreg only, seed=1.
# Grid: dt_mode ∈ {learned, replace, concat} × lr ∈ {1e-4, 5e-4, 2e-3} × train_batch_size ∈ {64, 128, 256}
# = 3 × 3 × 3 = 27 runs.
#
# Mapping:  k = TASK_ID-1;  dt_idx = k // 9;  lr_idx = (k % 9) // 3;  bs_idx = k % 3.

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

# W&B online (per colleague convention, run_mamba_mv_tsking_sandwich.sh).
# WANDB_API_KEY must be exported in the submitting shell before `sbatch ...`;
# SLURM inherits the env. Fail fast if it's missing or malformed so we don't
# waste GPU time (prior run died on a mangled key across all 27 cells).
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set. Run: export WANDB_API_KEY=... before sbatch." >&2
    exit 1
fi
_KEY_LEN=${#WANDB_API_KEY}
_KEY_CLEAN_LEN=$(printf '%s' "$WANDB_API_KEY" | tr -cd 'A-Za-z0-9_' | wc -c)
if [[ "$_KEY_LEN" -ne "$_KEY_CLEAN_LEN" ]] || [[ "$_KEY_LEN" -lt 30 ]]; then
    echo "ERROR: WANDB_API_KEY appears malformed (len=$_KEY_LEN, clean-len=$_KEY_CLEAN_LEN)." >&2
    echo "       Key must be alphanumeric+underscore only; no whitespace/quotes/newlines." >&2
    exit 1
fi
export WANDB_MODE=online
unset WANDB_SILENT 2>/dev/null || true

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
DATA_ROOT=${CODE_DIR}/data_correct_gap_random
LOG_BASE=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/hpo_mamba_mv/phase4_2

REGIME=multisin_high_irreg
SEED=1
DT_MODES=("learned" "replace" "concat")
LRS=("1e-4" "5e-4" "2e-3")
BATCH_SIZES=(64 128 256)

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
DT_IDX=$(( K / 9 ))
LR_IDX=$(( (K % 9) / 3 ))
BS_IDX=$(( K % 3 ))

DT_MODE=${DT_MODES[$DT_IDX]}
LR=${LRS[$LR_IDX]}
BS=${BATCH_SIZES[$BS_IDX]}

TAG=dt-${DT_MODE}_lr-${LR}_bs-${BS}
OUTPUT_DIR=${LOG_BASE}/${REGIME}/${TAG}/seed${SEED}
mkdir -p $OUTPUT_DIR
export WANDB_DIR=${OUTPUT_DIR}

echo "=========================================="
echo "Phase 4-2 (random_uniform gap) Mamba-MV HPO  regime=$REGIME  dt_mode=$DT_MODE  lr=$LR  bs=$BS  seed=$SEED"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Time: $(date)"
echo ""

cd $CODE_DIR
START_TIME=$(date +%s)

srun ${MAMBA_ENV}/bin/python -m imts_benchmark.mamba_mv.train_mv \
    --regime $REGIME \
    --dt_mode $DT_MODE \
    --lr $LR \
    --train_batch_size $BS \
    --seed $SEED \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --use_wandb \
    --wandb_project TSKing \
    --wandb_run_name "hpo_p42_${TAG}_seed${SEED}"

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Exit code: $EXIT_CODE"
echo "Wall time: ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "Done at $(date)"
