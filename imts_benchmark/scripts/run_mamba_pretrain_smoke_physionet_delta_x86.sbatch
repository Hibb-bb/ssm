#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --job-name=pre_smoke_phy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/_smoke_pretrain_phy/slurm_%j.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/_smoke_pretrain_phy/slurm_%j.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Pretrain-architecture Mamba-MV smoke on PhysioNet (Delta x86 H200).
# 2-epoch sanity check: validates env + GPU + datamodule + the new pretrain
# forecaster imports + forward/backward + test_metrics.csv write.
# Exit 0 = pass; chain HPO array on this jobid.

set -uo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
DATA_ROOT=${CODE_DIR}/tpatchgnn_data
OUT_DIR=${CODE_DIR}/output/log/imts_benchmark_v2_real/_smoke_pretrain_phy/${SLURM_JOB_ID:-local}
mkdir -p "${OUT_DIR}"

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

if [[ -z "${WANDB_API_KEY:-}" ]] && [[ -f ~/.bashrc ]]; then
    source ~/.bashrc 2>/dev/null || true
fi
if [[ -z "${WANDB_API_KEY:-}" ]] && [[ -f ~/.netrc ]]; then
    export WANDB_API_KEY=$(awk '/api.wandb.ai/{f=1} f && /password/{print $2; exit}' ~/.netrc)
fi
export WANDB_MODE=offline   # smoke: don't pollute the W&B project with 2-epoch runs

cd "${CODE_DIR}"

echo "=========================================="
echo "Pretrain-arch smoke on PhysioNet (replace, lr=2e-3, 2 epochs)"
echo "Job: ${SLURM_JOB_ID:-local}  host: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Time: $(date)"
echo "=========================================="

# --use_wandb is needed even in smoke because train_mv.py adds
# LearningRateMonitor unconditionally, which requires a logger. WANDB_MODE=offline
# above keeps the smoke from spamming the W&B project.
srun python -m imts_benchmark.mamba_pretrain.train_mv \
    --data_root "${DATA_ROOT}" \
    --regime physionet \
    --auto_meta \
    --dt_mode replace \
    --grid_K 256 \
    --lr 2e-3 \
    --train_batch_size 16 \
    --accumulate_grad_batches 4 \
    --max_epochs 2 \
    --patience 10 \
    --seed 1 \
    --output_dir "${OUT_DIR}" \
    --use_wandb \
    --wandb_project TSKing \
    --wandb_run_name "smoke_pre_phy_${SLURM_JOB_ID:-local}"

EXIT_CODE=$?

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "[smoke] FAILED with exit ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi

if [[ ! -f "${OUT_DIR}/test_metrics.csv" ]]; then
    echo "[smoke] missing test_metrics.csv at ${OUT_DIR}/"
    exit 1
fi

echo "[smoke] PASSED — ${OUT_DIR}/test_metrics.csv"
cat "${OUT_DIR}/test_metrics.csv"
exit 0
