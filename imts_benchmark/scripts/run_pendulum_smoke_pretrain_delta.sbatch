#!/bin/bash
#SBATCH --account=bfrf-dtai-gh
#SBATCH --job-name=pt_pend_smk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --partition=ghx4-interactive
#SBATCH --time=00:30:00
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/pendulum/smoke/slurm_%j.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/pendulum/smoke/slurm_%j.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Pendulum smoke: 1 epoch, dt_mode=replace, lr=2e-3 (PhysioNet winner), B=32, seed=1.
# head_type=gaussian (default; matches S5/CRU/RKN — our SSM family).
# Validates:
#   - PendulumDataModule loads {train,val,test}/ from data_root/pendulum/
#   - PendulumMambaForecaster forward + Gaussian NLL loss + test_metrics.csv
#   - Runtime warmup computation via trainer.estimated_stepping_batches
#   - W&B logging into magicslabnorthwestern/TSKing
# Pass = test_metrics.csv exists and contains a finite test_mse.
# Chain the HPO array with --dependency=afterok:$THIS_JOB.

set -uo pipefail

mkdir -p /projects/bfrf/seojininus/ssm/output/log/pendulum/smoke

ENV_PATH=/projects/bfrf/seojininus/envs/mamba
CODE_DIR=/u/seojininus/ssm
DATA_ROOT=${CODE_DIR}/pendulum_data
LOG_BASE=/projects/bfrf/seojininus/ssm/output/log/pendulum

source /sw/user/python/miniforge3-pytorch-2.5.0/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTPUT_DIR=${LOG_BASE}/smoke/${SLURM_JOB_ID}
mkdir -p "${OUTPUT_DIR}"
export WANDB_DIR="${OUTPUT_DIR}"

echo "=========================================="
echo "Pendulum SMOKE  dt=replace  lr=2e-3  B=32  seed=1  max_epochs=1  head=gaussian"
echo "Job: $SLURM_JOB_ID  host: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Time: $(date)"

cd "${CODE_DIR}"

srun python -m imts_benchmark.mamba_pretrain.train_pendulum \
    --data_root "${DATA_ROOT}" \
    --regime pendulum \
    --dt_mode replace \
    --lr 2e-3 \
    --train_batch_size 32 \
    --max_epochs 1 \
    --patience 1 \
    --seed 1 \
    --output_dir "${OUTPUT_DIR}" \
    --use_wandb \
    --wandb_project TSKing \
    --wandb_run_name "pendulum_pt_smoke_${SLURM_JOB_ID}"

EXIT_CODE=$?
TEST_CSV="${OUTPUT_DIR}/test_metrics.csv"
if [[ -f "${TEST_CSV}" ]]; then
    echo ""
    echo ">>> SMOKE PASS: ${TEST_CSV} written"
    cat "${TEST_CSV}"
    exit 0
fi
echo ">>> SMOKE FAIL: no test_metrics.csv at ${TEST_CSV} (exit=${EXIT_CODE})" >&2
exit ${EXIT_CODE}
