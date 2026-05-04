#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --job-name=uea30d_full18
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --array=1-162
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/uea_cls_pretrain_30d_full18/final/slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/uea_cls_pretrain_30d_full18/final/slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Full ContiFormer Table 8 UEA classification at 30% Kidger sync drop, model_size=small.
#
# 18 datasets: the 5 we've already validated (BM, CT, EP, HB, LSST) plus 13
# more from ContiFormer Table 8 / 11. Excluded from the 20-dataset list:
#   - DuckDuckGeese (V=1345) — slot-anonymous architecture is O(V^2) in the
#     variable-axis attention; would OOM at any reasonable batch size.
#   - PEMS-SF (V=963) — same OOM concern. Both would need an architectural
#     change (variate packing per RoMAE App. C.2) to handle.
#
# 18 datasets × 3 dt_modes × 3 seeds = 162 cells.
#
# HPs: same protocol that won round 1 on the 5-dataset chain (job 18016813):
#   model_size=small (d=64, ~259K params), lr=1e-3 (AdamW-tuned, NOT RoMAE
#   Table 12 SGD magnitudes — those caused divergence in attempt 1),
#   patience=100, max_epochs=800, weight_decay=0, warmup_pct=0.05.
# Per-dataset bs / label_smoothing / grad_clip / head_dropout / grid_K from
# ROMAE_DATASET_DEFAULTS in train_cls.py (RoMAE Table 12 values for the 5
# original; pragmatic defaults for the 13 new — see commit message).
#
# class_weights ON (round-1 default; cw=0 ablation deferred to round 2).
# Seeds {0, 42, 123} match (a) the existing 30%-drop trees and (b) the
# ContiFormer paper §C.2.1 seed list.
#
# Index decoding: k = TASK_ID - 1
#   ds_idx = k // 9   (18 datasets × 9 cells/ds = 162)
#   rem    = k %  9
#   dt_idx = rem // 3 (3 dt: replace, learned, concat)
#   sd_idx = rem %  3 (3 seeds: 0, 42, 123)

set -uo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
DATA_ROOT=${CODE_DIR}/data_uea
LOG_BASE=${CODE_DIR}/output/log/uea_cls_pretrain_30d_full18/final

mkdir -p "${LOG_BASE}"

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

# Order matches our existing 5 first, then alphabetical for the 13 new.
DATASETS=(
    "BasicMotions"
    "CharacterTrajectories"
    "Epilepsy"
    "Heartbeat"
    "LSST"
    "ArticularyWordRecognition"
    "ERing"
    "FingerMovements"
    "HandMovementDirection"
    "Handwriting"
    "JapaneseVowels"
    "Libras"
    "NATOPS"
    "PenDigits"
    "RacketSports"
    "SelfRegulationSCP1"
    "SpokenArabicDigits"
    "UWaveGestureLibrary"
)
DT_MODES=("replace" "learned" "concat")
SEEDS=(0 42 123)

LR=0.001  # global; AdamW-tuned, see feedback_optimizer_lr_dont_transfer.md
SIZE=small

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
DS_IDX=$(( K / 9 ))
REM=$(( K % 9 ))
DT_IDX=$(( REM / 3 ))
SD_IDX=$(( REM % 3 ))

DATASET=${DATASETS[$DS_IDX]}
DT_MODE=${DT_MODES[$DT_IDX]}
SEED=${SEEDS[$SD_IDX]}

OUT_DIR=${LOG_BASE}/${SIZE}/${DATASET}/${DT_MODE}/seed${SEED}
mkdir -p "${OUT_DIR}"

export WANDB_DIR="${OUT_DIR}"

cd "${CODE_DIR}"

echo "=========================================="
echo "UEA 30%-drop full-18 (${SIZE}): ds=${DATASET}  dt=${DT_MODE}  seed=${SEED}"
echo "  HP: lr=${LR}, max_epochs=800, patience=100, drop_rate=0.3, cw=1"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Time: $(date)"
echo "=========================================="

srun python -m imts_benchmark.mamba_pretrain.train_cls \
    --dataset ${DATASET} \
    --data_root "${DATA_ROOT}" \
    --drop_rate 0.3 \
    --dt_mode ${DT_MODE} \
    --model_size ${SIZE} \
    --use_class_weights 1 \
    --lr ${LR} \
    --weight_decay 0.0 \
    --warmup_pct 0.05 \
    --max_epochs 800 \
    --patience 100 \
    --seed ${SEED} \
    --num_workers 4 \
    --output_dir "${OUT_DIR}" \
    --run_name final \
    --use_wandb \
    --wandb_project TSKing \
    --wandb_entity magicslabnorthwestern \
    --wandb_run_name "uea30d_full18_${SIZE}_${DATASET}_${DT_MODE}_seed${SEED}"

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}  Done at $(date)"
exit ${EXIT_CODE}
