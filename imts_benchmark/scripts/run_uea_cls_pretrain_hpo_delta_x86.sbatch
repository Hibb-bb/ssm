#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --job-name=uea_pre_hpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --array=1-90
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/uea_cls_pretrain/hpo/slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/uea_cls_pretrain/hpo/slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Pretrain-architecture UEA classification HPO sweep — all 5 datasets, uniform protocol.
# Grid: 5 ds × 3 dt × 3 LR × 2 BS × 1 HPO seed = 90 cells.
#   datasets : BasicMotions, CharacterTrajectories, Epilepsy, Heartbeat, LSST
#   dt_modes : replace, learned, concat
#   lrs      : 1e-4, 3e-4, 1e-3            (covers v2 winners across all 5 datasets:
#                                            HB→1e-4, BM/CT/EP→3e-4, LSST→3e-4)
#   batch    : 16, 32                       (BS=32 wins 4/5 in v2; BS=16 wins CT learned)
#   patience : 10  (uniform; tighter than v2's 50)
#   max_epochs: 200 (uniform; tighter than v2's 800)
# Other knobs (label_smoothing, grad_clip, grid_K, head_dropout) inherited from
# train_cls.py ROMAE_DATASET_DEFAULTS — kept at v2 defaults so the only new axes
# are the architecture + the patience/max_epochs schedule.
#
# Index decoding: k = TASK_ID - 1
#   ds_idx = k // 18           (5 ds × 18 cells/ds = 90)
#   rem    = k %  18
#   dt_idx = rem // 6          (3 dt × 6)
#   rem2   = rem %  6
#   lr_idx = rem2 // 2
#   bs_idx = rem2 %  2

set -uo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
DATA_ROOT=${CODE_DIR}/data_uea
LOG_BASE=${CODE_DIR}/output/log/uea_cls_pretrain/hpo

mkdir -p "${LOG_BASE}"

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

DATASETS=("BasicMotions" "CharacterTrajectories" "Epilepsy" "Heartbeat" "LSST")
DT_MODES=("replace" "learned" "concat")
LRS=(1e-4 3e-4 1e-3)
BSS=(16 32)

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
DS_IDX=$(( K / 18 ))
REM=$(( K % 18 ))
DT_IDX=$(( REM / 6 ))
REM2=$(( REM % 6 ))
LR_IDX=$(( REM2 / 2 ))
BS_IDX=$(( REM2 % 2 ))

DATASET=${DATASETS[$DS_IDX]}
DT_MODE=${DT_MODES[$DT_IDX]}
LR=${LRS[$LR_IDX]}
BS=${BSS[$BS_IDX]}

OUT_DIR=${LOG_BASE}/${DATASET}/${DT_MODE}/lr${LR}_bs${BS}
mkdir -p "${OUT_DIR}"

cd "${CODE_DIR}"

echo "=========================================="
echo "UEA HPO pretrain-arch: ds=${DATASET}  dt=${DT_MODE}  lr=${LR}  bs=${BS}  patience=10  max_epochs=200  seed=42"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Time: $(date)"
echo "=========================================="

srun python -m imts_benchmark.mamba_pretrain.train_cls \
    --dataset ${DATASET} \
    --data_root "${DATA_ROOT}" \
    --dt_mode ${DT_MODE} \
    --lr ${LR} \
    --batch_size ${BS} \
    --max_epochs 200 \
    --patience 10 \
    --seed 42 \
    --num_workers 4 \
    --output_dir "${OUT_DIR}" \
    --run_name hpo

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}  Done at $(date)"
exit ${EXIT_CODE}
