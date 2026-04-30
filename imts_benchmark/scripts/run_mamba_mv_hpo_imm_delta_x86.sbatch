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
#SBATCH --array=1-27
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_imm/hpo_mamba_mv_%x/slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_imm/hpo_mamba_mv_%x/slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Mamba-MV HPO sweep on a TIME-IMM regime, parameterized by DS_NAME env var.
# Mirrors run_mamba_mv_hpo_physionet_delta_x86.sbatch (3 lr x 3 eff_bs x 3 dt_mode = 27 cells).
#
# Usage:
#   sbatch --export=ALL,DS_NAME=imm_fnspid,EFF_BS_OVERRIDE='' \
#       --job-name=mv_hpo_${DS_NAME} \
#       imts_benchmark/scripts/run_mamba_mv_hpo_imm_delta_x86.sbatch
#
# The ${LOG_BASE} dir uses ${SLURM_JOB_NAME} (set via --job-name) so different
# datasets land in separate log subdirs.
#
# For tiny datasets (CESNET ~23 train rows, ILINet ~85), pass smaller eff_bs:
#   EFF_BS_OVERRIDE="8 16 32"   # uses phys_bs=8/16/32 with accum=1

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

DT_MODES=(replace learned concat)
LRS=(1e-4 5e-4 2e-3)

# Default eff_bs grid (matches PhysioNet sweep). Override for tiny datasets.
if [[ -n "${EFF_BS_OVERRIDE:-}" ]]; then
    read -ra EFF_BS_ARR <<< "${EFF_BS_OVERRIDE}"
else
    EFF_BS_ARR=(64 128 256)
fi

if [[ ${#EFF_BS_ARR[@]} -ne 3 ]]; then
    echo "ERROR: EFF_BS_OVERRIDE must contain exactly 3 values, got: ${EFF_BS_ARR[*]}" >&2
    exit 1
fi

GRID_K=256

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
DT_IDX=$(( K / 9 ))
LR_IDX=$(( (K % 9) / 3 ))
EBS_IDX=$(( K % 3 ))

DT_MODE=${DT_MODES[$DT_IDX]}
LR=${LRS[$LR_IDX]}
EBS=${EFF_BS_ARR[$EBS_IDX]}

# eff_bs -> (phys_bs, accum). Two policies depending on EBS magnitude.
case $EBS in
    8)   PHYS_BS=8;  ACCUM=1 ;;
    16)  PHYS_BS=16; ACCUM=1 ;;
    32)  PHYS_BS=32; ACCUM=1 ;;
    64)  PHYS_BS=16; ACCUM=4 ;;
    128) PHYS_BS=32; ACCUM=4 ;;
    256) PHYS_BS=32; ACCUM=8 ;;
    *) echo "Unknown EBS=$EBS" >&2; exit 1 ;;
esac

CELL_TAG="lr-${LR}_ebs-${EBS}"
OUTPUT_DIR=${LOG_BASE}/hpo_mamba_mv_${DS_NAME}/${DT_MODE}/${CELL_TAG}/seed1
mkdir -p "${OUTPUT_DIR}"
export WANDB_DIR="${OUTPUT_DIR}"

echo "=========================================="
echo "Mamba-MV IMM HPO  ds=$DS_NAME  dt=$DT_MODE  lr=$LR  ebs=$EBS (phys=${PHYS_BS} accum=${ACCUM})  grid_K=$GRID_K  seed=1"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
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
