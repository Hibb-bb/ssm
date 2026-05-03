#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --job-name=mamba_smoke_x86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/bfrf/seojininus/ssm/logs/smoke_x86_%j.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/logs/smoke_x86_%j.err

# Smoke test on Delta (non-AI): same pipeline as smoke_test_delta.sbatch but
# pointed at the x86 env + Delta account/partition.
# A100 is roughly 1.5-2x slower than GH200, so 30 min wallclock instead of 20.

set -euo pipefail

ENV_PATH="${ENV_PATH:-/projects/bfrf/seojininus/envs/mamba_x86}"
REPO_DIR="/projects/bfrf/seojininus/ssm"
DATA_ROOT="${REPO_DIR}/data_correct_async"
OUTPUT_DIR="${REPO_DIR}/output/smoke_x86/$(date +%Y%m%d_%H%M%S)"
REGIME="multisin_high_irreg"

mkdir -p "${OUTPUT_DIR}" "${REPO_DIR}/logs"

module load miniforge3-python 2>/dev/null || true
CONDA_BIN="$(command -v conda)"
CONDA_REAL="$(readlink -f "${CONDA_BIN}")"
source "$(dirname "$(dirname "${CONDA_REAL}")")/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1
export WANDB_MODE=online
export WANDB_DIR="${OUTPUT_DIR}"

echo "host: $(hostname)"
nvidia-smi
which python
python --version

cd "${REPO_DIR}"

if [[ ! -d "${DATA_ROOT}/${REGIME}" || ! -f "${DATA_ROOT}/multisin_stats.json" ]]; then
    echo "[data] generating async multisin under ${DATA_ROOT}"
    python generate_multivariate_sinusodial_data_async.py --output_root "${DATA_ROOT}"
else
    echo "[data] reusing ${DATA_ROOT}"
fi

python -m imts_benchmark.mamba_mv.train_mv \
    --regime "${REGIME}" \
    --dt_mode replace \
    --seed 1 \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_epochs 2 \
    --patience 1 \
    --num_training_steps 50 \
    --num_warmup_steps 5 \
    --train_batch_size 32 \
    --val_batch_size 32 \
    --num_workers 2 \
    --use_wandb \
    --wandb_project TSKing-smoke \
    --wandb_run_name "smoke_x86_${SLURM_JOB_ID:-local}"

echo ""
echo "smoke test finished. output at ${OUTPUT_DIR}"
