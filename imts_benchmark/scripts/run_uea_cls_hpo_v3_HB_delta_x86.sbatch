#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --job-name=uea_v3_hb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --array=1-48
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/uea_cls_v3/hpo/Heartbeat/slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/uea_cls_v3/hpo/Heartbeat/slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# v3 HPO for Heartbeat (Delta x86 H200).
# Extends LR DOWN from v2's lower edge (1e-4) and adds head_dropout sweep.
# Adds concat dt_mode (v2 had only learned/replace).
# Grid: 3 dt × 4 LR × 2 BS × 2 dropout = 48 cells.
#   dt_modes : replace, learned, concat
#   lrs      : 3e-5, 1e-4, 3e-4, 1e-3       (1e-4 = v2 winner, others are new)
#   batch    : 16, 32                        (32 = v2 OOM cap; no accum in train_cls)
#   dropout  : 0.0, 0.1                      (0.0 = v2 default; 0.1 is new)
#
# Index decoding: k = TASK_ID - 1
#   dt_idx   = k // 16          (3 dt × 16 = 48; 16 = 4 LR × 2 BS × 2 dropout)
#   rem1     = k %  16
#   lr_idx   = rem1 // 4
#   rem2     = rem1 %  4
#   bs_idx   = rem2 // 2
#   drop_idx = rem2 %  2

set -uo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
DATA_ROOT=${CODE_DIR}/data_uea
LOG_BASE=${CODE_DIR}/output/log/uea_cls_v3/hpo

mkdir -p "${LOG_BASE}/Heartbeat"

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

DT_MODES=(replace learned concat)
LRS=(3e-5 1e-4 3e-4 1e-3)
BSS=(16 32)
DROPOUTS=(0.0 0.1)

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
DT_IDX=$(( K / 16 ))
REM1=$(( K % 16 ))
LR_IDX=$(( REM1 / 4 ))
REM2=$(( REM1 % 4 ))
BS_IDX=$(( REM2 / 2 ))
DROP_IDX=$(( REM2 % 2 ))

DT_MODE=${DT_MODES[$DT_IDX]}
LR=${LRS[$LR_IDX]}
BS=${BSS[$BS_IDX]}
DROP=${DROPOUTS[$DROP_IDX]}

OUT_DIR=${LOG_BASE}/Heartbeat/${DT_MODE}/lr${LR}_bs${BS}_drop${DROP}
mkdir -p "${OUT_DIR}"

cd "${CODE_DIR}"

echo "=========================================="
echo "UEA HPO v3 Heartbeat: dt=${DT_MODE}  lr=${LR}  bs=${BS}  head_dropout=${DROP}  seed=42"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Time: $(date)"
echo "=========================================="

srun python -m imts_benchmark.mamba_mv.train_cls \
    --dataset Heartbeat \
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
