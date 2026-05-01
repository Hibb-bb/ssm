#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --job-name=mv_hpo_mim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuH200x8
#SBATCH --time=06:00:00
#SBATCH --array=1-27
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/hpo_mamba_mv_mimic/slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/hpo_mamba_mv_mimic/slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Mamba-MV HPO sweep on MIMIC (V=96), seed=1, patience=10.
# Mirrors the PhysioNet 27-cell sweep (3 dt_modes × 3 lr × 3 eff_bs).
# eff_bs → (phys_bs, accum) cut down ~4× from PhysioNet to fit V=96 memory:
#   eff=64  -> phys=4,  accum=16
#   eff=128 -> phys=8,  accum=16
#   eff=256 -> phys=8,  accum=32
# Total: 3 * 3 * 3 = 27 cells, single seed.
# Mapping: k = TASK_ID-1; dt_idx = k // 9; lr_idx = (k % 9) // 3; ebs_idx = k % 3.
#
# Winners are picked per-dt_mode by aggregate_hpo_winners_real.py and consumed by
# run_mamba_mv_confirm_mimic_delta_x86.sbatch.

set -uo pipefail

mkdir -p /projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/hpo_mamba_mv_mimic

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
DATA_ROOT=${CODE_DIR}/tpatchgnn_data
LOG_BASE=${CODE_DIR}/output/log/imts_benchmark_v2_real

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -z "${WANDB_API_KEY:-}" ]] && [[ -f ~/.bashrc ]]; then
    source ~/.bashrc 2>/dev/null || true
fi
if [[ -z "${WANDB_API_KEY:-}" ]] && [[ -f ~/.netrc ]]; then
    export WANDB_API_KEY=$(awk '/api.wandb.ai/{f=1} f && /password/{print $2; exit}' ~/.netrc)
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set." >&2
    exit 1
fi
export WANDB_MODE=online

DT_MODES=(replace learned concat)
LRS=(1e-4 5e-4 2e-3)
EFF_BS=(64 128 256)
DS=mimic
SEED=1
GRID_K=256

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
DT_IDX=$(( K / 9 ))
LR_IDX=$(( (K % 9) / 3 ))
EBS_IDX=$(( K % 3 ))

DT_MODE=${DT_MODES[$DT_IDX]}
LR=${LRS[$LR_IDX]}
EBS=${EFF_BS[$EBS_IDX]}

case $EBS in
    64)  PHYS_BS=4;  ACCUM=16 ;;
    128) PHYS_BS=8;  ACCUM=16 ;;
    256) PHYS_BS=8;  ACCUM=32 ;;
    *) echo "Unknown EBS=$EBS" >&2; exit 1 ;;
esac

CELL_TAG="lr-${LR}_ebs-${EBS}"
OUTPUT_DIR=${LOG_BASE}/hpo_mamba_mv_mimic/${DT_MODE}/${CELL_TAG}/seed${SEED}
mkdir -p "${OUTPUT_DIR}"
export WANDB_DIR="${OUTPUT_DIR}"

echo "=========================================="
echo "Mamba-MV HPO  ds=$DS  dt=$DT_MODE  lr=$LR  ebs=$EBS (phys=${PHYS_BS} accum=${ACCUM})  grid_K=$GRID_K  seed=$SEED"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
echo "host: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Time: $(date)"

cd "${CODE_DIR}"
START=$(date +%s)

srun python -m imts_benchmark.mamba_mv.train_mv \
    --data_root "${DATA_ROOT}" \
    --regime $DS \
    --auto_meta \
    --dt_mode $DT_MODE \
    --grid_K $GRID_K \
    --lr $LR \
    --train_batch_size $PHYS_BS \
    --accumulate_grad_batches $ACCUM \
    --patience 10 \
    --seed $SEED \
    --output_dir "${OUTPUT_DIR}" \
    --use_wandb \
    --wandb_project TSKing \
    --wandb_run_name "hpo_mv_${DS}_${DT_MODE}_lr-${LR}_ebs-${EBS}_seed${SEED}_delta"

EXIT_CODE=$?
END=$(date +%s)
echo "Exit code: $EXIT_CODE  Wall time: $((END-START))s  Done at $(date)"
exit ${EXIT_CODE}
