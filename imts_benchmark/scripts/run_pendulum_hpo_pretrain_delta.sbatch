#!/bin/bash
#SBATCH --account=bfrf-dtai-gh
#SBATCH --job-name=pt_pend_hpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --partition=ghx4
#SBATCH --time=02:00:00
#SBATCH --array=1-27
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/pendulum/hpo_mamba_pretrain/slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/pendulum/hpo_mamba_pretrain/slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Mamba-pretrain Pendulum HPO sweep, seed=1, patience=10.
#   dt_modes : replace, learned, concat        (3)  learned = negative control
#                                                   (no Delta_t -> tests whether
#                                                   true-Δt actually helps)
#   lrs      : 5e-4, 2e-3, 1e-2                (3)  wide bracket: anchors at
#                                                   RoMAE-near (5e-4), PhysioNet
#                                                   winner (2e-3), S5 pendulum
#                                                   LR (~1e-2)
#   batch_sz : 16, 32, 64                      (3)  spans RoMAE/S5/CRU choices
# Total: 27 cells, single seed.
# head_type=gaussian (matches S5/CRU/RKN; default in train_pendulum.py).
# wd=0.0, max_epochs=100 (S5/CRU pendulum protocol — defaults in trainer).
# warmup is computed at runtime as 10% of total steps (scales with B).
# Mapping: k = TASK_ID-1; dt_idx = k // 9; lr_idx = (k % 9) // 3; b_idx = k % 3.
#
# Winners are picked per-dt_mode by aggregate_hpo_winners_pendulum.py and
# consumed by run_pendulum_confirm_pretrain_delta.sbatch (20-seed confirm).

set -uo pipefail

mkdir -p /projects/bfrf/seojininus/ssm/output/log/pendulum/hpo_mamba_pretrain

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

DT_MODES=(replace learned concat)
LRS=(5e-4 2e-3 1e-2)
BATCH_SIZES=(16 32 64)
SEED=1
PATIENCE=10
MAX_EPOCHS=100

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
DT_IDX=$(( K / 9 ))
LR_IDX=$(( (K % 9) / 3 ))
B_IDX=$(( K % 3 ))

DT_MODE=${DT_MODES[$DT_IDX]}
LR=${LRS[$LR_IDX]}
B=${BATCH_SIZES[$B_IDX]}

CELL_TAG="lr-${LR}_b-${B}"
OUTPUT_DIR=${LOG_BASE}/hpo_mamba_pretrain/${DT_MODE}/${CELL_TAG}/seed${SEED}
mkdir -p "${OUTPUT_DIR}"
export WANDB_DIR="${OUTPUT_DIR}"

echo "=========================================="
echo "Pendulum HPO  dt=$DT_MODE  lr=$LR  B=$B  seed=$SEED  patience=$PATIENCE  max_epochs=$MAX_EPOCHS"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
echo "host: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Time: $(date)"

cd "${CODE_DIR}"

srun python -m imts_benchmark.mamba_pretrain.train_pendulum \
    --data_root "${DATA_ROOT}" \
    --regime pendulum \
    --dt_mode $DT_MODE \
    --lr $LR \
    --train_batch_size $B \
    --max_epochs $MAX_EPOCHS \
    --patience $PATIENCE \
    --seed $SEED \
    --output_dir "${OUTPUT_DIR}" \
    --use_wandb \
    --wandb_project TSKing \
    --wandb_run_name "pendulum_pt_hpo_${DT_MODE}_lr-${LR}_b-${B}_seed${SEED}"

EXIT_CODE=$?
echo "Final exit code: ${EXIT_CODE}  Done at $(date)"
exit ${EXIT_CODE}
