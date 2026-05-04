#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --job-name=uea30d_hvsmoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --array=1-2
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/uea_cls_pretrain_30d_full18/_highv_smoke/slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/uea_cls_pretrain_30d_full18/_highv_smoke/slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# High-V smoke: just verify DuckDuckGeese (V=1345) and PEMS-SF (V=963) can
# train without OOM under physical bs=1 + grad accum=16 + bf16 + grid_K=64
# + model_size=small. --smoke caps to max_epochs=30, patience=10 — we just
# want to see ONE forward + backward pass land cleanly.
#
# If exit 0 → graduate to the full 18-cell mini-chain (2 ds × 3 dt × 3 seeds).
# If OOM → drop them.

set -uo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
DATA_ROOT=${CODE_DIR}/data_uea
LOG_BASE=${CODE_DIR}/output/log/uea_cls_pretrain_30d_full18/_highv_smoke

mkdir -p "${LOG_BASE}"

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

DATASETS=("DuckDuckGeese" "PEMS-SF")
DATASET=${DATASETS[$(( SLURM_ARRAY_TASK_ID - 1 ))]}

OUT_DIR=${LOG_BASE}/${DATASET}
mkdir -p "${OUT_DIR}"

cd "${CODE_DIR}"

echo "=========================================="
echo "High-V smoke: ds=${DATASET}  bs=1  accum=16  grid_K=64  precision=bf16-mixed"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Time: $(date)"
echo "=========================================="

srun python -m imts_benchmark.mamba_pretrain.train_cls \
    --dataset ${DATASET} \
    --data_root "${DATA_ROOT}" \
    --drop_rate 0.3 \
    --dt_mode replace \
    --model_size small \
    --use_class_weights 1 \
    --batch_size 1 \
    --accumulate_grad_batches 16 \
    --precision bf16-mixed \
    --lr 1e-3 \
    --weight_decay 0.0 \
    --warmup_pct 0.05 \
    --max_epochs 5 \
    --patience 5 \
    --seed 0 \
    --num_workers 2 \
    --output_dir "${OUT_DIR}" \
    --run_name smoke \
    --smoke

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}  Done at $(date)"
exit ${EXIT_CODE}
