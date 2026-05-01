#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --job-name=mv_hpo_imm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuH200x8
#SBATCH --time=04:00:00
#SBATCH --array=1-9
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_imm/hpo_mamba_mv_%x/slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_imm/hpo_mamba_mv_%x/slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Mamba-MV HPO sweep on a TIME-IMM regime (lean design, 9 cells per dataset).
# 3 lr x 3 dt_mode at fixed eff_bs=128 (median of Phase-5 grid).
# Total compute: 8 datasets * 9 cells = 72 sweep runs.
#
# Selection: val/MSE-best per dt_mode (3 winners per dataset).
# Confirm: 3 seeds * 3 dt_modes per dataset = 72 confirm runs.
#
# Usage:
#   sbatch --export=ALL,DS_NAME=imm_fnspid \
#       --job-name=mv_hpo_${DS_NAME} \
#       imts_benchmark/scripts/run_mamba_mv_hpo_imm_delta_x86.sbatch

set -uo pipefail

DS_NAME="${DS_NAME:?Must set DS_NAME via --export, e.g. DS_NAME=imm_fnspid}"

LOG_BASE_ROOT=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_imm
mkdir -p "${LOG_BASE_ROOT}/hpo_mamba_mv_${DS_NAME}"

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
DATA_ROOT=${CODE_DIR}/time_imm_data
LOG_BASE=${LOG_BASE_ROOT}

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

if [[ -z "${WANDB_API_KEY:-}" ]] && [[ -f ~/.bashrc ]]; then
    source ~/.bashrc 2>/dev/null || true
fi
if [[ -z "${WANDB_API_KEY:-}" ]] && [[ -f ~/.netrc ]]; then
    export WANDB_API_KEY=$(awk '/api.wandb.ai/{f=1} f && /password/{print $2; exit}' ~/.netrc)
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "WARNING: WANDB_API_KEY not set; falling back to offline mode" >&2
    export WANDB_MODE=offline
else
    export WANDB_MODE=online
fi

# 9-cell grid: lr x dt_mode at fixed eff_bs=128
DT_MODES=(replace learned concat)
LRS=(1e-4 5e-4 1e-3)
EBS=128
PHYS_BS=32
ACCUM=4
GRID_K=256

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
DT_IDX=$(( K / 3 ))
LR_IDX=$(( K % 3 ))

DT_MODE=${DT_MODES[$DT_IDX]}
LR=${LRS[$LR_IDX]}

CELL_TAG="lr-${LR}_ebs-${EBS}"
OUTPUT_DIR=${LOG_BASE}/hpo_mamba_mv_${DS_NAME}/${DT_MODE}/${CELL_TAG}/seed1
mkdir -p "${OUTPUT_DIR}"
export WANDB_DIR="${OUTPUT_DIR}"

echo "=========================================="
echo "Mamba-MV IMM HPO  ds=$DS_NAME  dt=$DT_MODE  lr=$LR  ebs=$EBS (phys=${PHYS_BS} accum=${ACCUM})  grid_K=$GRID_K  seed=1"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID  cell ${TASK_ID}/9"
echo "host: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Time: $(date)"

cd "${CODE_DIR}"
START=$(date +%s)

srun python -m imts_benchmark.mamba_mv.train_mv \
    --data_root "${DATA_ROOT}" \
    --regime "${DS_NAME}" \
    --auto_meta \
    --dt_mode "${DT_MODE}" \
    --grid_K "${GRID_K}" \
    --lr "${LR}" \
    --train_batch_size "${PHYS_BS}" \
    --accumulate_grad_batches "${ACCUM}" \
    --patience 10 \
    --seed 1 \
    --output_dir "${OUTPUT_DIR}" \
    --use_wandb \
    --wandb_project TSKing \
    --wandb_run_name "hpo_mv_${DS_NAME}_${DT_MODE}_lr-${LR}_ebs-${EBS}_seed1_delta"

EXIT_CODE=$?
END=$(date +%s)
echo "Exit code: $EXIT_CODE  Wall time: $((END-START))s  Done at $(date)"
exit ${EXIT_CODE}
