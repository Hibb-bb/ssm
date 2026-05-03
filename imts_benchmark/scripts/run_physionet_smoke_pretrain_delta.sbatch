#!/bin/bash
#SBATCH --account=bfrf-dtai-gh
#SBATCH --job-name=pt_phy_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=ghx4-interactive
#SBATCH --time=00:45:00
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/_smoke/slurm_pt_%j.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/_smoke/slurm_pt_%j.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# 2-epoch PhysioNet smoke for mamba_pretrain (slot-anonymous variant).
# Validates end-to-end before launching the 27-cell HPO array.
# Verifies:
#   * env imports + new train_mv.py argparse / model construction
#   * MultivariateSinusoidalDataModule + --auto_meta path on physionet
#   * val/mse logged, ModelCheckpoint + EarlyStopping wire correctly
#   * trainer.test() succeeds + emits test_mse / test_mse_tpg in CSV
#   * /u/seojininus/ssm/ is visible from this compute partition
# Ends with PASS/FAIL gate. Chain the HPO pipeline on this with:
#   SMOKE_ID=<jobid> bash submit_hpo_physionet_pretrain_delta.sh

set -uo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba
CODE_DIR=/u/seojininus/ssm
DATA_ROOT=${CODE_DIR}/tpatchgnn_data
LOG_BASE=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real
SMOKE_BASE=${LOG_BASE}/_smoke/pretrain_${SLURM_JOB_ID:-local}

mkdir -p "${SMOKE_BASE}"

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

if [[ ! -d "${CODE_DIR}" ]]; then
    echo "ERROR: CODE_DIR not visible on $(hostname): ${CODE_DIR}" >&2
    exit 1
fi
if [[ ! -d "${DATA_ROOT}/physionet" ]]; then
    echo "ERROR: DATA_ROOT/physionet not found: ${DATA_ROOT}/physionet" >&2
    exit 1
fi

cd "${CODE_DIR}"

echo "=========================================="
echo "PhysioNet smoke (mamba_pretrain, 2 epochs, dt=replace, seed=1)"
echo "Job: ${SLURM_JOB_ID:-local}  host: $(hostname)"
echo "CODE_DIR: ${CODE_DIR}"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Time: $(date)"
echo "=========================================="

OUT="${SMOKE_BASE}/mamba_pretrain"
mkdir -p "${OUT}"
export WANDB_DIR="${OUT}"

START=$(date +%s)
python -m imts_benchmark.mamba_pretrain.train_mv \
    --data_root "${DATA_ROOT}" \
    --regime physionet \
    --auto_meta \
    --dt_mode replace \
    --grid_K 256 \
    --lr 5e-4 \
    --train_batch_size 32 \
    --accumulate_grad_batches 4 \
    --seed 1 \
    --max_epochs 2 \
    --patience 1 \
    --num_training_steps 50 \
    --num_warmup_steps 5 \
    --num_workers 2 \
    --output_dir "${OUT}" \
    --use_wandb \
    --wandb_project TSKing-smoke \
    --wandb_run_name "physmoke_pt_${SLURM_JOB_ID:-local}"

CODE=$?
END=$(date +%s)
echo ""
echo "wall: $((END-START))s  exit: ${CODE}"

if [[ ${CODE} -ne 0 ]]; then
    echo "[smoke] mamba_pretrain FAILED with exit ${CODE}"
    exit 1
fi
if [[ ! -f "${OUT}/test_metrics.csv" ]]; then
    echo "[smoke] no test_metrics.csv at ${OUT}/test_metrics.csv"
    exit 1
fi
if ! head -1 "${OUT}/test_metrics.csv" | grep -q "test_mse_tpg"; then
    echo "[smoke] test_metrics.csv missing test_mse_tpg column:"
    head -1 "${OUT}/test_metrics.csv"
    exit 1
fi
if [[ ! -f "${OUT}/checkpoints/best.ckpt" ]]; then
    echo "[smoke] no best.ckpt at ${OUT}/checkpoints/best.ckpt — ModelCheckpoint did not fire"
    exit 1
fi

echo ""
echo "=========================================="
echo "[smoke] mamba_pretrain PASSED on PhysioNet."
echo "[smoke] CSV header:"
head -1 "${OUT}/test_metrics.csv"
echo "[smoke] CSV row:"
tail -1 "${OUT}/test_metrics.csv"
echo "[smoke] best ckpt: ${OUT}/checkpoints/best.ckpt"
echo "[smoke] Output dir: ${OUT}"
echo ""
echo "Chain HPO with:  SMOKE_ID=${SLURM_JOB_ID:-?} bash imts_benchmark/scripts/submit_hpo_physionet_pretrain_delta.sh"
exit 0
