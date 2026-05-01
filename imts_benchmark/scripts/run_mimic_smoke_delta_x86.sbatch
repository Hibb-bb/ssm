#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --job-name=mimic_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:45:00
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/_smoke_mimic/slurm_%j.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/_smoke_mimic/slurm_%j.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# 2-epoch smoke for all 3 models on MIMIC. Validates:
#   * Datamodule loads MIMIC HF Arrow correctly (V=96, history=24h)
#   * Models survive V=96 — much higher than PhysioNet's V=41
#   * test_metrics.csv with mse_tpg/mae_tpg columns is emitted
# Exit 0 = pass; chain real-data MIMIC arrays on this jobid.

set -uo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
DATA_ROOT=${CODE_DIR}/tpatchgnn_data
LOG_BASE=${CODE_DIR}/output/log/imts_benchmark_v2_real
SMOKE_BASE=${LOG_BASE}/_smoke_mimic/${SLURM_JOB_ID:-local}

mkdir -p "${SMOKE_BASE}"

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1
# S5 needs expandable_segments to fit on V=96 (PhysioNet V=41 already needed it).
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

cd "${CODE_DIR}"

echo "=========================================="
echo "MIMIC smoke (3 models, 2 epochs each, seed=1)"
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
        --regime mimic \
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
        --wandb_run_name "mimic_smoke_${NAME}_${SLURM_JOB_ID:-local}" \
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

# 1) S5 — even smaller phys_bs than PhysioNet (V=96 doubles state dim memory).
run_one "s5"        imts_benchmark.s5_forecaster.train_s5 \
    --lr 1e-3 \
    --weight_decay 0.05 \
    --train_batch_size 4 \
    --val_batch_size 4 \
    --accumulate_grad_batches 32 \
    --state_dim 64

# 2) RoMAE
run_one "romae"     imts_benchmark.romae_forecaster.train_romae \
    --train_batch_size 8 \
    --val_batch_size 4 \
    --accumulate_grad_batches 16

# 3) Mamba-MV (replace, borrowed PhysioNet HP scaled for V=96)
run_one "mamba_mv"  imts_benchmark.mamba_mv.train_mv \
    --dt_mode replace \
    --grid_K 256 \
    --lr 5e-4 \
    --train_batch_size 4 \
    --accumulate_grad_batches 16

echo ""
echo "=========================================="
if [[ $ALL_OK -eq 1 ]]; then
    echo "[smoke] ALL THREE MODELS PASSED on MIMIC."
    echo "[smoke] Outputs under: ${SMOKE_BASE}"
    exit 0
else
    echo "[smoke] FAILURE — see logs in ${SMOKE_BASE}"
    exit 1
fi
