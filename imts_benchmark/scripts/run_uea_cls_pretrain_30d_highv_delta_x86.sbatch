#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --job-name=uea30d_highv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --array=1-18
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/uea_cls_pretrain_30d_full18/highv/slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/uea_cls_pretrain_30d_full18/highv/slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# High-V tail of the full ContiFormer Table 8 UEA chain: DuckDuckGeese (V=1345)
# and PEMS-SF (V=963). Smoke (job 18029621) confirmed the slot-anonymous
# architecture fits at bs=1 + accumulate_grad_batches=16 + grid_K=64 +
# bf16-mixed on H200 (143GB). DDG ~7s/epoch, PEMS ~25s/epoch.
#
# 2 datasets × 3 dt_modes × 3 seeds = 18 cells. Same protocol as the
# 18-dataset main chain (job 18029383) except for the high-V-specific
# knobs above (per-dataset bs/grid_K/precision come from
# ROMAE_DATASET_DEFAULTS; --accumulate_grad_batches forced to 16 here).
#
# Index decoding: k = TASK_ID - 1
#   ds_idx = k // 9   (2 datasets × 9 cells/ds = 18)
#   rem    = k %  9
#   dt_idx = rem // 3
#   sd_idx = rem %  3

set -uo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
DATA_ROOT=${CODE_DIR}/data_uea
LOG_BASE=${CODE_DIR}/output/log/uea_cls_pretrain_30d_full18/highv

mkdir -p "${LOG_BASE}"

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

DATASETS=("DuckDuckGeese" "PEMS-SF")
DT_MODES=("replace" "learned" "concat")
SEEDS=(0 42 123)
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
echo "UEA 30%-drop high-V (${SIZE}): ds=${DATASET}  dt=${DT_MODE}  seed=${SEED}"
echo "  HP: bs=1 accum=16 grid_K=64 bf16-mixed lr=1e-3 patience=100 max_epochs=800 cw=1"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Time: $(date)"
echo "=========================================="

srun python -m imts_benchmark.mamba_pretrain.train_cls \
    --dataset ${DATASET} \
    --data_root "${DATA_ROOT}" \
    --drop_rate 0.3 \
    --dt_mode ${DT_MODE} \
    --model_size ${SIZE} \
    --use_class_weights 1 \
    --batch_size 1 \
    --accumulate_grad_batches 16 \
    --precision bf16-mixed \
    --lr 1e-3 \
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
    --wandb_run_name "uea30d_highv_${SIZE}_${DATASET}_${DT_MODE}_seed${SEED}"

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}  Done at $(date)"
exit ${EXIT_CODE}
