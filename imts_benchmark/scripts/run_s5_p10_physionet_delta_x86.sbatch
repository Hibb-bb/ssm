#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --job-name=s5_p10_phy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuH200x8
#SBATCH --time=08:00:00
#SBATCH --array=1-5
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/s5_p10/slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/s5_p10/slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# S5 + patience=10 on PhysioNet (V=41) — PAPER-NATIVE DEVIATION.
# Delta GH (GH200, aarch64, sm_90).
#
# S5's discretize_zoh (s5_model.py:113) materialises B_bars of shape
# (B, V, N, P, M). At V=41, N_max~1500, M=128, complex64:
#   state_dim=256 -> 16 GiB *per batch element* (bs=1 already tight)
#   state_dim=64  -> 4 GiB per batch element  (bs=8 fits in ~32 GiB)
#
# Quest never ran PhysioNet S5; the colleague's bs=32+accum=4 was untested
# and OOMs even at bs=16 on GH200's 95GB (job 2197568 confirmed).
#
# Pragmatic deviation:  state_dim=256 -> 64  (4x memory reduction)
# Compensation: bs=8 + accum=16 (eff bs=128 unchanged from Activity/USHCN).
# Allocator: expandable_segments=True to absorb fragmentation across epochs.
#
# Document this in RESULTS_phase5_real.md when reporting PhysioNet numbers —
# our PhysioNet S5 is at 4x smaller state than Activity/USHCN S5.

set -uo pipefail

mkdir -p /projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/s5_p10

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
DATA_ROOT=${CODE_DIR}/tpatchgnn_data
LOG_BASE=${CODE_DIR}/output/log/imts_benchmark_v2_real

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Load WANDB_API_KEY: env first, then ~/.bashrc, finally ~/.netrc fallback.
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

DS=physionet
SEED=$SLURM_ARRAY_TASK_ID

OUTPUT_DIR=${LOG_BASE}/s5_p10/${DS}/default/seed${SEED}
mkdir -p "${OUTPUT_DIR}"
export WANDB_DIR="${OUTPUT_DIR}"

echo "=========================================="
echo "S5 (p10)  ds=$DS  seed=$SEED  patience=10  bs=8+accum=16  state_dim=64 (paper deviation; OOM at 256)"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
echo "host: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Time: $(date)"

cd "${CODE_DIR}"
START=$(date +%s)

srun python -m imts_benchmark.s5_forecaster.train_s5 \
    --data_root "${DATA_ROOT}" \
    --regime $DS \
    --auto_meta \
    --seed $SEED \
    --lr 1e-3 \
    --weight_decay 0.05 \
    --train_batch_size 8 \
    --val_batch_size 8 \
    --accumulate_grad_batches 16 \
    --state_dim 64 \
    --patience 10 \
    --output_dir "${OUTPUT_DIR}" \
    --use_wandb \
    --wandb_project TSKing \
    --wandb_run_name "p10_s5_${DS}_seed${SEED}_delta"

EXIT_CODE=$?
END=$(date +%s)
echo "Exit code: $EXIT_CODE  Wall time: $((END-START))s  Done at $(date)"
exit ${EXIT_CODE}
