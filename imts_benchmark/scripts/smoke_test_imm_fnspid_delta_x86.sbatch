#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --job-name=mv_imm_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/projects/bfrf/seojininus/ssm/logs/smoke_imm_fnspid_x86_%j.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/logs/smoke_imm_fnspid_x86_%j.err

# 2-epoch Mamba-MV smoke test on TIME-IMM/FNSPID after running the
# data_pipeline/import_timeimm.py adapter. Validates:
#   1. norm_stats.json (n_vars=6, history=31, time_max=62) is read by --auto_meta.
#   2. MultivariateSparseFlat / collate_per_variate handle the imported Arrow.
#   3. Forward + backward succeed on Mamba-MV with dt_mode={replace,learned,concat}.
#   4. test_metrics.csv emits with mse_tpg / mae_tpg.
#
# This is NOT a full HPO/confirm run. Walltime 45 min on A100x4.

set -uo pipefail

ENV_PATH="${ENV_PATH:-/projects/bfrf/seojininus/envs/mamba_x86}"
REPO_DIR="/projects/bfrf/seojininus/ssm"
DATA_ROOT="${REPO_DIR}/time_imm_data"
REGIME="${REGIME:-imm_fnspid}"
DT_MODE="${DT_MODE:-replace}"
OUTPUT_DIR="${REPO_DIR}/output/smoke_imm/${REGIME}_${DT_MODE}_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUTPUT_DIR}" "${REPO_DIR}/logs"

module load miniforge3-python 2>/dev/null || true
CONDA_BIN="$(command -v conda)"
CONDA_REAL="$(readlink -f "${CONDA_BIN}")"
source "$(dirname "$(dirname "${CONDA_REAL}")")/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1
export WANDB_MODE=offline
export WANDB_DIR="${OUTPUT_DIR}"

echo "host: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
which python
python --version

cd "${REPO_DIR}"

if [[ ! -d "${DATA_ROOT}/${REGIME}/train" || ! -f "${DATA_ROOT}/${REGIME}/norm_stats.json" ]]; then
    echo "[data] adapter has not been run for ${REGIME}; running it now"
    python data_pipeline/import_timeimm.py --dataset "${REGIME#imm_}" \
        --time_imm_root "${REPO_DIR}/_time_imm_repo/data" \
        --imm_tsf_root "${REPO_DIR}/_imm_tsf_repo" \
        --out_root "${DATA_ROOT}"
else
    echo "[data] reusing ${DATA_ROOT}/${REGIME}"
fi

python -m imts_benchmark.mamba_mv.train_mv \
    --regime "${REGIME}" \
    --dt_mode "${DT_MODE}" \
    --seed 1 \
    --data_root "${DATA_ROOT}" \
    --auto_meta \
    --output_dir "${OUTPUT_DIR}" \
    --max_epochs 2 \
    --patience 1 \
    --num_training_steps 50 \
    --num_warmup_steps 5 \
    --train_batch_size 32 \
    --val_batch_size 32 \
    --num_workers 2 \
    --use_wandb \
    --wandb_project TSKing-imm-smoke \
    --wandb_run_name "smoke_imm_${REGIME}_${DT_MODE}_${SLURM_JOB_ID:-local}"

echo ""
echo "smoke test finished. output at ${OUTPUT_DIR}"
