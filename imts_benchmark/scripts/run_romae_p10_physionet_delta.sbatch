#!/bin/bash
#SBATCH --account=bfrf-dtai-gh
#SBATCH --job-name=romae_p10_phy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=ghx4
#SBATCH --time=12:00:00
#SBATCH --array=1-5
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/romae_p10/slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/romae_p10/slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# RoMAE paper-native + patience=10 on PhysioNet (V=41, N_max~1500).
# Delta GH (GH200, aarch64, sm_90) port of run_romae_p10_physionet.sbatch.
# RoMAE flat-token attention is O(N^2 * d). At PhysioNet's N_max with bs=128
# the kernel OOMs on H100 80GB (Quest); GH200 has 96GB but we keep the same
# bs=32 + accum=4 (eff bs=128) recipe for cross-cluster reproducibility.
# 1 dataset x 5 seeds = 5 jobs.

set -uo pipefail

mkdir -p /projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/romae_p10

ENV_PATH=/projects/bfrf/seojininus/envs/mamba
CODE_DIR=/projects/bfrf/seojininus/ssm
DATA_ROOT=${CODE_DIR}/tpatchgnn_data
LOG_BASE=${CODE_DIR}/output/log/imts_benchmark_v2_real

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

DS=physionet
SEED=$SLURM_ARRAY_TASK_ID

OUTPUT_DIR=${LOG_BASE}/romae_p10/${DS}/default/seed${SEED}
mkdir -p "${OUTPUT_DIR}"
export WANDB_DIR="${OUTPUT_DIR}"

echo "=========================================="
echo "RoMAE (p10)  ds=$DS  seed=$SEED  patience=10  bs=32+accum=4"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
echo "host: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Time: $(date)"

cd "${CODE_DIR}"
START=$(date +%s)

srun python -m imts_benchmark.romae_forecaster.train_romae \
    --data_root "${DATA_ROOT}" \
    --regime $DS \
    --auto_meta \
    --seed $SEED \
    --train_batch_size 32 \
    --val_batch_size 8 \
    --accumulate_grad_batches 4 \
    --patience 10 \
    --output_dir "${OUTPUT_DIR}" \
    --use_wandb \
    --wandb_project TSKing \
    --wandb_run_name "p10_romae_${DS}_seed${SEED}_delta"

EXIT_CODE=$?
END=$(date +%s)
echo "Exit code: $EXIT_CODE  Wall time: $((END-START))s  Done at $(date)"
exit ${EXIT_CODE}
