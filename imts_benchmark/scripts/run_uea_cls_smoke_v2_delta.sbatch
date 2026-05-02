#!/bin/bash
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=uea_cls_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/uea_cls_v2/_smoke/slurm_%j.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/uea_cls_v2/_smoke/slurm_%j.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# UEA classification smoke gate (DeltaAI / GH200, aarch64).
# Single canary on BasicMotions, --smoke flag (max 30 epochs, patience 10).
# Validates env activation, GPU, datamodule, head, summary.json write.
# Exit 0 = pass; chain v2 HPO array on this jobid.

set -uo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba
CODE_DIR=/projects/bfrf/seojininus/ssm
DATA_ROOT=${CODE_DIR}/data_uea
LOG_BASE=${CODE_DIR}/output/log/uea_cls_v2/_smoke
OUT_DIR=${LOG_BASE}/${SLURM_JOB_ID:-local}

mkdir -p "${OUT_DIR}"

source /sw/user/python/miniforge3-pytorch-2.5.0/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

cd "${CODE_DIR}"

echo "=========================================="
echo "UEA cls smoke (BasicMotions, dt=replace, lr=1e-3, --smoke)"
echo "Job: ${SLURM_JOB_ID:-local}  host: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Time: $(date)"
echo "=========================================="

srun python -m imts_benchmark.mamba_mv.train_cls \
    --dataset BasicMotions \
    --data_root "${DATA_ROOT}" \
    --dt_mode replace \
    --lr 1e-3 \
    --seed 42 \
    --num_workers 4 \
    --output_dir "${OUT_DIR}" \
    --run_name smoke \
    --smoke

EXIT_CODE=$?

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "[smoke] FAILED with exit ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi

if [[ ! -f "${OUT_DIR}/smoke/summary.json" ]]; then
    echo "[smoke] missing summary.json under ${OUT_DIR}/smoke/"
    exit 1
fi

echo "[smoke] PASSED — ${OUT_DIR}/smoke/summary.json"
cat "${OUT_DIR}/smoke/summary.json"
exit 0
