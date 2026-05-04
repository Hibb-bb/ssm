#!/bin/bash
#SBATCH --account=bfrf-dtai-gh
#SBATCH --job-name=pt_pend_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --partition=ghx4
#SBATCH --time=01:00:00
#SBATCH --array=1-18
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/pendulum/hpo_mamba_pretrain_v2_reg/slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/pendulum/hpo_mamba_pretrain_v2_reg/slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# v2 Pendulum HPO sweep — REGULARIZATION sweep, addressing v1 overfitting.
# Diagnosis (RESULTS_pendulum.md §v1): val/mse floors at ~0.010 with train/val
# NLL gap of 0.3-0.6 (memorization). v1 used wd=0, dropout=0.
#
# v2 holds (lr, B) at the v1 confirm winner per dt_mode and sweeps:
#   dt_modes      : replace, learned, concat        (3)
#   weight_decay  : 0.0, 0.05, 0.2                  (3)
#   dropout       : 0.0, 0.1                        (2)
# Total: 18 cells, single seed.
# Mapping: k = TASK_ID-1; dt_idx = k // 6; wd_idx = (k % 6) // 2; dr_idx = k % 2.
#
# Also tightens training to address the late-epoch numerical collapse seen in
# v1 (gaussian head's log-var diverges around ep 40-80 → val/mse becomes None,
# wasting compute):
#   max_epochs = 40       (v1: 100 — but val/mse plateaus by ep 30-40)
#   patience    = 5       (v1: 10 — fire faster on the plateau)
#
# Output: /projects/bfrf/seojininus/ssm/output/log/pendulum/hpo_mamba_pretrain_v2_reg/
# (separate from v1's hpo_mamba_pretrain/ — v1 results preserved.)

set -uo pipefail

mkdir -p /projects/bfrf/seojininus/ssm/output/log/pendulum/hpo_mamba_pretrain_v2_reg

ENV_PATH=/projects/bfrf/seojininus/envs/mamba
CODE_DIR=/u/seojininus/ssm
DATA_ROOT=${CODE_DIR}/pendulum_data
LOG_BASE=/projects/bfrf/seojininus/ssm/output/log/pendulum
V1_WINNERS=${LOG_BASE}/hpo_mamba_pretrain/winners_by_dt_mode.json

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

if [[ ! -f "${V1_WINNERS}" ]]; then
    echo "ERROR: v1 winners JSON not found at ${V1_WINNERS}" >&2
    exit 1
fi

DT_MODES=(replace learned concat)
WDS=(0.0 0.05 0.2)
DROPOUTS=(0.0 0.1)
SEED=1
PATIENCE=5
MAX_EPOCHS=40

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
DT_IDX=$(( K / 6 ))
WD_IDX=$(( (K % 6) / 2 ))
DR_IDX=$(( K % 2 ))

DT_MODE=${DT_MODES[$DT_IDX]}
WD=${WDS[$WD_IDX]}
DROP=${DROPOUTS[$DR_IDX]}

# Read (lr, batch_size) from v1 winners JSON for this dt_mode — held fixed.
read LR B < <(python - "${V1_WINNERS}" "${DT_MODE}" <<'PY'
import json, sys
w = json.load(open(sys.argv[1]))
cfg = w[sys.argv[2]]
print(cfg["lr"], cfg["batch_size"])
PY
)
if [[ -z "${LR:-}" || -z "${B:-}" ]]; then
    echo "ERROR: could not parse v1 winner for dt_mode=${DT_MODE}" >&2
    exit 1
fi

CELL_TAG="wd-${WD}_drop-${DROP}"
OUTPUT_DIR=${LOG_BASE}/hpo_mamba_pretrain_v2_reg/${DT_MODE}/${CELL_TAG}/seed${SEED}
mkdir -p "${OUTPUT_DIR}"
export WANDB_DIR="${OUTPUT_DIR}"

echo "=========================================="
echo "Pendulum v2_reg HPO  dt=$DT_MODE  wd=$WD  drop=$DROP  (held: lr=$LR B=$B)  seed=$SEED  patience=$PATIENCE  max_epochs=$MAX_EPOCHS"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
echo "host: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Time: $(date)"

cd "${CODE_DIR}"

# Same retry shape as confirm — protects against transient launch failures
# (10/27 v1 cells failed without retries, mostly concat).
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
        --weight_decay $WD \
        --dropout $DROP \
        --max_epochs $MAX_EPOCHS \
        --patience $PATIENCE \
        --seed $SEED \
        --output_dir "${OUTPUT_DIR}" \
        --use_wandb \
        --wandb_project TSKing \
        --wandb_run_name "v2reg_pendulum_pt_hpo_${DT_MODE}_wd-${WD}_drop-${DROP}_seed${SEED}_a${ATTEMPT}"
    EXIT_CODE=$?
    if [[ ${EXIT_CODE} -eq 0 ]]; then break; fi
    if [[ -f "${TEST_CSV}" ]]; then EXIT_CODE=0; break; fi
    sleep 10
done

echo "Final exit code: ${EXIT_CODE}  Done at $(date)"
exit ${EXIT_CODE}
