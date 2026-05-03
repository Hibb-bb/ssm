#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --job-name=uea30d_size_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/uea_cls_pretrain_30d_size/_smoke/slurm_%j.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/uea_cls_pretrain_30d_size/_smoke/slurm_%j.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Smoke gate for the 30%-drop size ablation. One BasicMotions cell at
# size=small (the new untested config) with --smoke (caps to 30 epochs /
# patience 10). Validates that:
#   1. The d=64 model presets imports + forwards without dimension errors.
#   2. test_predictions.json is written (the new diagnostic pipeline).
#   3. CSVLogger metrics.csv is non-empty (loss curves).

set -uo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
DATA_ROOT=${CODE_DIR}/data_uea
OUT_DIR=${CODE_DIR}/output/log/uea_cls_pretrain_30d_size/_smoke/${SLURM_JOB_ID:-local}

mkdir -p "${OUT_DIR}"

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

export WANDB_DIR="${OUT_DIR}"

cd "${CODE_DIR}"

echo "=========================================="
echo "UEA 30%-drop size ablation SMOKE (BM, small=d64, --smoke)"
echo "Job: ${SLURM_JOB_ID:-local}  host: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Time: $(date)"
echo "=========================================="

srun python -m imts_benchmark.mamba_pretrain.train_cls \
    --dataset BasicMotions \
    --data_root "${DATA_ROOT}" \
    --drop_rate 0.3 \
    --dt_mode replace \
    --model_size small \
    --use_class_weights 1 \
    --lr 1e-2 \
    --weight_decay 0.0 \
    --warmup_pct 0.05 \
    --max_epochs 30 \
    --patience 10 \
    --seed 27 \
    --num_workers 4 \
    --output_dir "${OUT_DIR}" \
    --run_name smoke \
    --smoke \
    --use_wandb \
    --wandb_project TSKing \
    --wandb_entity magicslabnorthwestern \
    --wandb_run_name "uea30d_size_smoke_BM_small_seed27"

EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
    echo "[smoke] FAILED with exit ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi

# Validate the new diagnostic outputs exist.
if [[ ! -f "${OUT_DIR}/smoke/summary.json" ]]; then
    echo "[smoke] missing summary.json"
    exit 1
fi
if [[ ! -f "${OUT_DIR}/smoke/test_predictions.json" ]]; then
    echo "[smoke] missing test_predictions.json — train_cls.py prediction dump didn't fire"
    exit 1
fi
if [[ ! -f "${OUT_DIR}/smoke/csv/version_0/metrics.csv" ]]; then
    echo "[smoke] missing CSVLogger metrics.csv"
    exit 1
fi

echo "[smoke] PASSED — all 3 diagnostic outputs present"
echo "summary.json:"; cat "${OUT_DIR}/smoke/summary.json"
echo ""; echo "test_predictions.json head:"; head -c 600 "${OUT_DIR}/smoke/test_predictions.json"; echo
exit 0
