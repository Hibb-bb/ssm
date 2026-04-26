#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=mamba_winP3_d160
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/hpo_mamba_mv/phase3_winner_d160/slurm_%A_%a.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/hpo_mamba_mv/phase3_winner_d160/slurm_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-15
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# 5-seed x 3-dt_mode confirmation at d_model=160 (parameter-matched to S5).
# Reads per-dt_mode winner HPs from phase3_d160/winners_by_mode.json.
# Array mapping: T=TASK_ID-1; dt_idx=T/5; seed=(T%5)+1
#   tasks  1..5  -> learned, seeds 1..5
#   tasks  6..10 -> replace, seeds 1..5
#   tasks 11..15 -> concat,  seeds 1..5

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set." >&2
    exit 1
fi
export WANDB_MODE=online
unset WANDB_SILENT 2>/dev/null || true

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
DATA_ROOT=${CODE_DIR}/data_correct_async
LOG_BASE=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/hpo_mamba_mv/phase3_winner_d160
WINNERS_JSON=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/hpo_mamba_mv/phase3_d160/winners_by_mode.json

D_MODEL=160
D_HIDDEN=160

if [[ ! -f "${WINNERS_JSON}" ]]; then
    echo "ERROR: ${WINNERS_JSON} not found (aggregation must run first)." >&2
    exit 1
fi

DT_MODES=("learned" "replace" "concat")
T=$(( SLURM_ARRAY_TASK_ID - 1 ))
DT_IDX=$(( T / 5 ))
SEED=$(( (T % 5) + 1 ))
DT_MODE=${DT_MODES[$DT_IDX]}

LR=$(${MAMBA_ENV}/bin/python -c "import json; print(json.load(open('${WINNERS_JSON}'))['${DT_MODE}']['lr'])")
BS=$(${MAMBA_ENV}/bin/python -c "import json; print(json.load(open('${WINNERS_JSON}'))['${DT_MODE}']['train_batch_size'])")

REGIME=multisin_high_irreg

TAG=dt-${DT_MODE}_lr-${LR}_bs-${BS}
OUTPUT_DIR=${LOG_BASE}/${REGIME}/${TAG}/seed${SEED}
mkdir -p $OUTPUT_DIR
export WANDB_DIR=${OUTPUT_DIR}

echo "=========================================="
echo "Phase 3 WINNER (d_model=${D_MODEL})  regime=$REGIME  dt_mode=$DT_MODE  lr=$LR  bs=$BS  seed=$SEED"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
echo "Winners JSON: ${WINNERS_JSON}"
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
    --d_model $D_MODEL \
    --d_hidden $D_HIDDEN \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --use_wandb \
    --wandb_project TSKing \
    --wandb_run_name "hpo_p3_winner_d160_${TAG}_seed${SEED}"

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Exit code: $EXIT_CODE"
echo "Wall time: ${ELAPSED}s"
echo "Done at $(date)"
