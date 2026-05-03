#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --job-name=uea30d_small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --array=1-45
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/uea_cls_pretrain_30d_size/final/slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/uea_cls_pretrain_30d_size/final/slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# 30%-drop UEA classification (Kidger / RoMAE Table 4 protocol). Round-1
# config: SMALL preset only (d=64, ~259K params), per the user's choice to
# keep round 1 to 45 cells.
#
# Layout:  5 datasets × 3 dt_modes × 3 seeds = 45 cells.
#
# Per-dataset LRs match RoMAE App. C.2 Table 12 (BM=1e-2, CT=8e-3, EP=2e-3,
# HB=2e-2, LSST=3e-2). Per-dataset batch_size / label_smoothing / grad_clip /
# head_dropout / grid_K still flow from ROMAE_DATASET_DEFAULTS in
# train_cls.py (also Table 12). class_weights ON (the historical default).
#
# Patience bumped to 100 (vs prior 10/50) and max_epochs to 800: the existing
# 30% chain at patience=10 was firing while still on the LR-warmup ramp
# (BM cells finishing in 4-11s). RoMAE caps epochs per-dataset (15-150);
# patience=100 lets us reach those caps naturally on the slow datasets.
#
# Re-running base (d=256) is intentionally deferred — comparison to the
# existing 30% tree (`output/log/uea_cls_pretrain/final/`, also d=256 but
# global lr=1e-3 + patience=10) will be partially confounded by the protocol
# difference. Acceptable for round 1; if `small` looks promising, run base
# under the new protocol in round 2 to confirm attribution.
#
# Index decoding: k = TASK_ID - 1
#   ds_idx = k // 9   (5 datasets × 9 cells/ds = 45)
#   rem    = k %  9
#   dt_idx = rem // 3 (3 dt: replace, learned, concat)
#   sd_idx = rem %  3 (3 seeds: 0, 42, 123)

set -uo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
DATA_ROOT=${CODE_DIR}/data_uea
LOG_BASE=${CODE_DIR}/output/log/uea_cls_pretrain_30d_size/final

mkdir -p "${LOG_BASE}"

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

DATASETS=("BasicMotions" "CharacterTrajectories" "Epilepsy" "Heartbeat" "LSST")
DT_MODES=("replace" "learned" "concat")
# Seeds {0, 42, 123} match (a) ContiFormer paper §C.2.1 explicitly and
# (b) the existing 30%-drop tree at output/log/uea_cls_pretrain/final/.
# RoMAE doesn't specify seeds, so this is the best-anchored 3-seed choice
# for the head-to-head.
SEEDS=(0 42 123)

# Per-dataset LRs from RoMAE Table 12 (App. C.2). Indices match DATASETS order.
LRS=(0.01 0.008 0.002 0.02 0.03)

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
LR=${LRS[$DS_IDX]}

OUT_DIR=${LOG_BASE}/${SIZE}/${DATASET}/${DT_MODE}/seed${SEED}
mkdir -p "${OUT_DIR}"

# wandb: online cloud logging into magicslabnorthwestern/TSKing.
# WANDB_DIR keeps the local sqlite buffer beside the run's other artifacts.
export WANDB_DIR="${OUT_DIR}"

cd "${CODE_DIR}"

echo "=========================================="
echo "UEA 30%-drop ROUND 1 (small only): size=${SIZE}  ds=${DATASET}  dt=${DT_MODE}  seed=${SEED}"
echo "  HP: lr=${LR} (RoMAE Table 12), max_epochs=800, patience=100, drop_rate=0.3, cw=1"
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
    --wandb_run_name "uea30d_${SIZE}_${DATASET}_${DT_MODE}_seed${SEED}"

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}  Done at $(date)"
exit ${EXIT_CODE}
