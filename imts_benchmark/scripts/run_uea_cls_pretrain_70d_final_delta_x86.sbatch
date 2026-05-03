#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --job-name=uea70d_final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --array=1-45
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/uea_cls_pretrain_70d_repo/final/slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/uea_cls_pretrain_70d_repo/final/slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Pretrain-arch UEA classification @ 70% drop under ContiFormer REPO protocol
# (microsoft/physiopro/docs/configs/contiformer_mask_classification.yml verbatim;
# the paper text §C.2.1 describes a different setting that disagrees with the
# YAML — repo is the source-of-truth for the published Table 11 numbers):
#   optimizer = Adam (we use AdamW — small deviation; Lightning module is hard-coded)
#   lr = 1e-3 (NOT 1e-2 like the paper text)
#   batch_size = 16 (NOT 64 like the paper text)
#   max_epochs = 500 (NOT 1000 like the paper text)
#   patience = 50 on val/acc (NOT 100 on train/acc like the paper text)
#   weight_decay = 0
#   label_smoothing = 0, grad_clip = 1.0, head_dropout = 0
#   seeds = {27, 42, 1024} (from the README, NOT {42, 123, 0})
#   warmup_pct = 0 (minimize cosine warmup; closer to repo's no-scheduler)
# Residual deviations: AdamW vs Adam; cosine decay vs constant LR (both
# require Lightning-module edits to fully match — left as documented gaps).
#
# 5 ds × 3 dt_modes × 3 seeds = 45 cells.
#
# Index decoding: k = TASK_ID - 1
#   ds_idx = k // 9  (5 datasets × 9 cells/ds = 45)
#   rem    = k %  9
#   dt_idx = rem // 3  (3 dt: replace, learned, concat)
#   sd_idx = rem %  3  (0..2 -> seed in {27, 42, 1024})

set -uo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
DATA_ROOT=${CODE_DIR}/data_uea
LOG_BASE=${CODE_DIR}/output/log/uea_cls_pretrain_70d_repo/final

mkdir -p "${LOG_BASE}"

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

DATASETS=("BasicMotions" "CharacterTrajectories" "Epilepsy" "Heartbeat" "LSST")
DT_MODES=("replace" "learned" "concat")
SEEDS=(27 42 1024)

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
DS_IDX=$(( K / 9 ))
REM=$(( K % 9 ))
DT_IDX=$(( REM / 3 ))
SD_IDX=$(( REM % 3 ))

DATASET=${DATASETS[$DS_IDX]}
DT_MODE=${DT_MODES[$DT_IDX]}
SEED=${SEEDS[$SD_IDX]}

OUT_DIR=${LOG_BASE}/${DATASET}/${DT_MODE}/seed${SEED}
mkdir -p "${OUT_DIR}"

cd "${CODE_DIR}"

echo "=========================================="
echo "UEA final pretrain @ 70% drop (ContiFormer REPO protocol): ds=${DATASET}  dt=${DT_MODE}  seed=${SEED}"
echo "  HP: lr=1e-3 bs=16 max_epochs=500 patience=50 weight_decay=0 warmup_pct=0 label_smoothing=0 grad_clip=1.0 head_dropout=0 drop_rate=0.7"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Time: $(date)"
echo "=========================================="

srun python -m imts_benchmark.mamba_pretrain.train_cls \
    --dataset ${DATASET} \
    --data_root "${DATA_ROOT}" \
    --drop_rate 0.7 \
    --dt_mode ${DT_MODE} \
    --lr 1e-3 \
    --batch_size 16 \
    --weight_decay 0.0 \
    --warmup_pct 0.0 \
    --label_smoothing 0.0 \
    --grad_clip 1.0 \
    --head_dropout 0.0 \
    --max_epochs 500 \
    --patience 50 \
    --seed ${SEED} \
    --num_workers 4 \
    --output_dir "${OUT_DIR}" \
    --run_name final

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}  Done at $(date)"
exit ${EXIT_CODE}
