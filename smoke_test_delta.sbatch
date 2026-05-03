#!/bin/bash
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --job-name=mamba_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=/projects/bfrf/seojininus/ssm/logs/smoke_%j.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/logs/smoke_%j.err

# Smoke test: generate one regime of data, train MultivariateMambaForecaster
# for 2 epochs on 1 seed. Confirms env + GPU + data + lightning + wandb path.

set -euo pipefail

ENV_PATH="${ENV_PATH:-/projects/bfrf/seojininus/envs/mamba}"
REPO_DIR="/projects/bfrf/seojininus/ssm"
DATA_ROOT="${REPO_DIR}/data_correct_async"
OUTPUT_DIR="${REPO_DIR}/output/smoke/$(date +%Y%m%d_%H%M%S)"
REGIME="multisin_high_irreg"

mkdir -p "${OUTPUT_DIR}" "${REPO_DIR}/logs"

source /sw/user/python/miniforge3-pytorch-2.5.0/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1
export WANDB_MODE=online
export WANDB_DIR="${OUTPUT_DIR}"

echo "host: $(hostname)"
nvidia-smi
which python
python --version

cd "${REPO_DIR}"

# Generate data if missing.
if [[ ! -d "${DATA_ROOT}/${REGIME}" || ! -f "${DATA_ROOT}/multisin_stats.json" ]]; then
    echo "[data] generating async multisin under ${DATA_ROOT}"
    python generate_multivariate_sinusodial_data_async.py --output_root "${DATA_ROOT}"
else
    echo "[data] reusing ${DATA_ROOT}"
fi

# Tiny training run: 2 epochs, patience 1, regime=high_irreg, seed 1.
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
    --wandb_run_name "smoke_${SLURM_JOB_ID:-local}"

echo ""
echo "smoke test finished. output at ${OUTPUT_DIR}"
