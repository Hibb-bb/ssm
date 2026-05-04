#!/bin/bash
#SBATCH --account=bfrf-dtai-gh
#SBATCH --job-name=pt_pend_v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --partition=ghx4
#SBATCH --time=02:00:00
#SBATCH --array=1-27
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/pendulum/hpo_mamba_pretrain_v3_cnn/slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/pendulum/hpo_mamba_pretrain_v3_cnn/slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# v3 Pendulum HPO sweep — SCHIRMER CNN ENCODER, addressing v1 noise-rejection failure.
# Diagnostic (RESULTS_pendulum.md §5.3): v1's Linear front-end has no spatial
# inductive bias → predicts FLAT zero on noisy frames (samples 1, 2, 3 in the
# per-sample plot). Schirmer 2022's CNN encoder (used by S5/CRU/RKN/etc.) does
# spatial smoothing via Conv+MaxPool — literally a noise filter.
#
# v3 changes EXACTLY ONE thing vs v1: image_embed = Linear → Schirmer CNN.
# Body, head, optimizer, schedule, training duration ALL identical to v1.
#   front_end : schirmer_cnn (vs v1's "linear")
#   d_model   : 64 (unchanged from v1; adapter Linear(30→64) inside encoder)
#   max_epochs: 100, patience: 10  (back to v1's budget; v2's 40 was the bug)
#   wd: 0.0  (v1 default)
#   dropout: 0.0  (v1 default)
#
# Same HPO axes as v1 (the floor at this new front-end is unknown):
#   dt_modes : replace, learned, concat        (3)
#   lrs      : 5e-4, 2e-3, 1e-2                (3)
#   batch_sz : 16, 32, 64                      (3)
# Total: 27 cells, single seed.
# Mapping: k = TASK_ID-1; dt_idx = k // 9; lr_idx = (k % 9) // 3; b_idx = k % 3.
#
# Reuses v1's aggregator (axes are identical) pointed at v3_cnn dir.
# Output: /projects/bfrf/seojininus/ssm/output/log/pendulum/hpo_mamba_pretrain_v3_cnn/
# (separate from v1 hpo_mamba_pretrain/ and v2 hpo_mamba_pretrain_v2_reg/).

set -uo pipefail

mkdir -p /projects/bfrf/seojininus/ssm/output/log/pendulum/hpo_mamba_pretrain_v3_cnn

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
OUTPUT_DIR=${LOG_BASE}/hpo_mamba_pretrain_v3_cnn/${DT_MODE}/${CELL_TAG}/seed${SEED}
mkdir -p "${OUTPUT_DIR}"
export WANDB_DIR="${OUTPUT_DIR}"

echo "=========================================="
echo "Pendulum v3_cnn HPO  dt=$DT_MODE  lr=$LR  B=$B  seed=$SEED  patience=$PATIENCE  max_epochs=$MAX_EPOCHS  front_end=schirmer_cnn"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
echo "host: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Time: $(date)"

cd "${CODE_DIR}"

# 3-attempt retry — protects against transient cluster launch failures.
ATTEMPTS=3
EXIT_CODE=99
TEST_CSV="${OUTPUT_DIR}/test_metrics.csv"
for ATTEMPT in $(seq 1 $ATTEMPTS); do
    echo ""
    echo ">>> Attempt ${ATTEMPT}/${ATTEMPTS} at $(date)"
    srun python -m imts_benchmark.mamba_pretrain.train_pendulum \
        --data_root "${DATA_ROOT}" \
        --regime pendulum \
        --dt_mode $DT_MODE \
        --lr $LR \
        --train_batch_size $B \
        --front_end schirmer_cnn \
        --max_epochs $MAX_EPOCHS \
        --patience $PATIENCE \
        --seed $SEED \
        --output_dir "${OUTPUT_DIR}" \
        --use_wandb \
        --wandb_project TSKing \
        --wandb_run_name "v3cnn_pendulum_pt_hpo_${DT_MODE}_lr-${LR}_b-${B}_seed${SEED}_a${ATTEMPT}"
    EXIT_CODE=$?
    if [[ ${EXIT_CODE} -eq 0 ]]; then break; fi
    if [[ -f "${TEST_CSV}" ]]; then EXIT_CODE=0; break; fi
    sleep 10
done

echo "Final exit code: ${EXIT_CODE}  Done at $(date)"
exit ${EXIT_CODE}
