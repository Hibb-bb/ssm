#!/bin/bash
#SBATCH --account=bfrf-dtai-gh
#SBATCH --job-name=phy_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=ghx4-interactive
#SBATCH --time=01:30:00
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/_smoke/slurm_%j.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/_smoke/slurm_%j.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Single-job PhysioNet smoke: 2 epochs of S5, RoMAE, and Mamba-MV (replace),
# seed 1, exactly the same CLI shape as the production scripts but with
# --max_epochs 2 --patience 1 --num_training_steps 50. Verifies:
#   * env import path (mamba_ssm/causal_conv1d/romae)
#   * V=41 + bs=32 + accum=4 effective batch path
#   * Mamba-MV grid_K=256 + accumulate_grad_batches code path
#   * test_metrics.csv emits mse_tpg/mae_tpg
# Ends with a one-line PASS/FAIL gate. The full arrays should chain on this
# with --dependency=afterok:<this_jobid>.

set -uo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba
CODE_DIR=/projects/bfrf/seojininus/ssm
DATA_ROOT=${CODE_DIR}/tpatchgnn_data
LOG_BASE=${CODE_DIR}/output/log/imts_benchmark_v2_real
SMOKE_BASE=${LOG_BASE}/_smoke/${SLURM_JOB_ID:-local}

mkdir -p "${SMOKE_BASE}"

source /sw/user/python/miniforge3-pytorch-2.5.0/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1
# S5 needs expandable_segments to fit on PhysioNet (see run_s5_*_physionet_delta.sbatch).
# RoMAE / Mamba-MV are unaffected.
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
# Use online so LearningRateMonitor (in all three trainers) has a logger.
# Goes to TSKing-smoke project to avoid polluting the main TSKing dashboards.
export WANDB_MODE=online

cd "${CODE_DIR}"

echo "=========================================="
echo "PhysioNet smoke (3 models, 2 epochs each, seed=1)"
echo "Job: ${SLURM_JOB_ID:-local}  host: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Time: $(date)"
echo "=========================================="

ALL_OK=1

run_one () {
    local NAME=$1; local MODULE=$2; shift 2
    local OUT="${SMOKE_BASE}/${NAME}"
    mkdir -p "${OUT}"
    export WANDB_DIR="${OUT}"
    echo ""
    echo "--- ${NAME} ---  $(date)"
    python -m "${MODULE}" \
        --data_root "${DATA_ROOT}" \
        --regime physionet \
        --auto_meta \
        --seed 1 \
        --max_epochs 2 \
        --patience 1 \
        --num_training_steps 50 \
        --num_warmup_steps 5 \
        --num_workers 2 \
        --output_dir "${OUT}" \
        --use_wandb \
        --wandb_project TSKing-smoke \
        --wandb_run_name "physmoke_${NAME}_${SLURM_JOB_ID:-local}" \
        "$@"
    local CODE=$?
    if [[ $CODE -ne 0 ]]; then
        echo "[smoke] ${NAME} FAILED with exit ${CODE}"
        ALL_OK=0
        return
    fi
    if [[ ! -f "${OUT}/test_metrics.csv" ]]; then
        echo "[smoke] ${NAME} produced no test_metrics.csv"
        ALL_OK=0
        return
    fi
    # Check that the new tpg columns are present.
    if ! head -1 "${OUT}/test_metrics.csv" | grep -q "test_mse_tpg"; then
        echo "[smoke] ${NAME} test_metrics.csv missing test_mse_tpg column:"
        head -1 "${OUT}/test_metrics.csv"
        ALL_OK=0
        return
    fi
    echo "[smoke] ${NAME} OK -> ${OUT}/test_metrics.csv"
    head -1 "${OUT}/test_metrics.csv"
    tail -1 "${OUT}/test_metrics.csv"
}

# 1) S5 — state_dim=64 (paper deviation; OOM at 256 even at bs=1) ----------
#    bs=8 + accum=16 (eff 128). Job 2197538 OOMed at bs=32+state=256;
#    job 2197568 OOMed at bs=16+state=256. state_dim=64 is the only viable
#    config on PhysioNet's V=41+N=1500 within GH200's 95GB.
run_one "s5"        imts_benchmark.s5_forecaster.train_s5 \
    --lr 1e-3 \
    --weight_decay 0.05 \
    --train_batch_size 8 \
    --val_batch_size 8 \
    --accumulate_grad_batches 16 \
    --state_dim 64

# 2) RoMAE -------------------------------------------------------------
run_one "romae"     imts_benchmark.romae_forecaster.train_romae \
    --train_batch_size 32 \
    --val_batch_size 8 \
    --accumulate_grad_batches 4

# 3) Mamba-MV (replace, USHCN-borrowed lr=5e-4) -----------------------
run_one "mamba_mv"  imts_benchmark.mamba_mv.train_mv \
    --dt_mode replace \
    --grid_K 256 \
    --lr 5e-4 \
    --train_batch_size 32 \
    --accumulate_grad_batches 4

echo ""
echo "=========================================="
if [[ $ALL_OK -eq 1 ]]; then
    echo "[smoke] ALL THREE MODELS PASSED on PhysioNet."
    echo "[smoke] Outputs under: ${SMOKE_BASE}"
    exit 0
else
    echo "[smoke] FAILURE — see logs in ${SMOKE_BASE}"
    exit 1
fi
