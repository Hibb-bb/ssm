#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --job-name=uea_pre_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/uea_cls_pretrain/_smoke/slurm_%j.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/uea_cls_pretrain/_smoke/slurm_%j.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Pretrain-architecture UEA classification smoke: BM canary, ~5 min on H200.
# Validates env + GPU + datamodule + the pretrain encoder's var_id wiring +
# summary.json write. Exit 0 = pass; chain HPO array on this jobid.

set -uo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
DATA_ROOT=${CODE_DIR}/data_uea
OUT_DIR=${CODE_DIR}/output/log/uea_cls_pretrain/_smoke/${SLURM_JOB_ID:-local}

mkdir -p "${OUT_DIR}"

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

cd "${CODE_DIR}"

echo "=========================================="
echo "Pretrain-arch UEA cls smoke (BasicMotions, dt=replace, lr=1e-3, patience=10, max_epochs=30)"
echo "Job: ${SLURM_JOB_ID:-local}  host: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Time: $(date)"
echo "=========================================="

srun python -m imts_benchmark.mamba_pretrain.train_cls \
    --dataset BasicMotions \
    --data_root "${DATA_ROOT}" \
    --dt_mode replace \
    --lr 1e-3 \
    --batch_size 16 \
    --max_epochs 30 \
    --patience 10 \
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
