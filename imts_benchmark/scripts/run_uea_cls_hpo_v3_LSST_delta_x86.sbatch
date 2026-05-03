#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --job-name=uea_v3_lsst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --array=1-81
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/uea_cls_v3/hpo/LSST/slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/uea_cls_v3/hpo/LSST/slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# v3 HPO for LSST (Delta x86 H200).
# Shifts LR DOWN (back to v1's safe range — every v2 cell at LR>=3e-3 collapsed
# completely). Adds head_dropout sweep (v2 default was 0.2 inherited from RoMAE
# Table 12 without justification under our recipe).
# Adds concat dt_mode.
# Grid: 3 dt × 3 LR × 3 BS × 3 dropout = 81 cells.
#   dt_modes : replace, learned, concat
#   lrs      : 1e-4, 3e-4, 1e-3              (was {1e-3, 3e-3, 1e-2}; lr>=3e-3 was dead)
#   batch    : 16, 32, 64                    (V=6 + grid_K=64 fits; 64 is new)
#   dropout  : 0.0, 0.1, 0.2                 (0.2 was v2 default; 0.0/0.1 are new)
#
# Index decoding: k = TASK_ID - 1
#   dt_idx   = k // 27          (3 dt × 27 = 81; 27 = 3 LR × 3 BS × 3 dropout)
#   rem1     = k %  27
#   lr_idx   = rem1 // 9
#   rem2     = rem1 %  9
#   bs_idx   = rem2 // 3
#   drop_idx = rem2 %  3

set -uo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
DATA_ROOT=${CODE_DIR}/data_uea
LOG_BASE=${CODE_DIR}/output/log/uea_cls_v3/hpo

mkdir -p "${LOG_BASE}/LSST"

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

DT_MODES=(replace learned concat)
LRS=(1e-4 3e-4 1e-3)
BSS=(16 32 64)
DROPOUTS=(0.0 0.1 0.2)

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
DT_IDX=$(( K / 27 ))
REM1=$(( K % 27 ))
LR_IDX=$(( REM1 / 9 ))
REM2=$(( REM1 % 9 ))
BS_IDX=$(( REM2 / 3 ))
DROP_IDX=$(( REM2 % 3 ))

DT_MODE=${DT_MODES[$DT_IDX]}
LR=${LRS[$LR_IDX]}
BS=${BSS[$BS_IDX]}
DROP=${DROPOUTS[$DROP_IDX]}

OUT_DIR=${LOG_BASE}/LSST/${DT_MODE}/lr${LR}_bs${BS}_drop${DROP}
mkdir -p "${OUT_DIR}"

cd "${CODE_DIR}"

echo "=========================================="
echo "UEA HPO v3 LSST: dt=${DT_MODE}  lr=${LR}  bs=${BS}  head_dropout=${DROP}  seed=42"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Time: $(date)"
echo "=========================================="

srun python -m imts_benchmark.mamba_mv.train_cls \
    --dataset LSST \
    --data_root "${DATA_ROOT}" \
    --dt_mode ${DT_MODE} \
    --lr ${LR} \
    --batch_size ${BS} \
    --head_dropout ${DROP} \
    --seed 42 \
    --num_workers 4 \
    --output_dir "${OUT_DIR}" \
    --run_name hpo

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}  Done at $(date)"
exit ${EXIT_CODE}
