#!/bin/bash
#SBATCH --account=bfrf-dtai-gh
#SBATCH --job-name=pt_pend_v2_cf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --partition=ghx4
#SBATCH --time=01:30:00
#SBATCH --array=1-60
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/pendulum/mamba_pretrain_p20_v2_reg/slurm_confirm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/pendulum/mamba_pretrain_p20_v2_reg/slurm_confirm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# v2 Pendulum 20-seed CONFIRM at v2_reg HPO winners (one per dt_mode).
# Depends on v2 aggregator. Output dir is *_v2_reg (separate from v1).
# 3 dt_modes × 20 seeds = 60 tasks. max_epochs=40, patience=5 (matches HPO).

set -uo pipefail

mkdir -p /projects/bfrf/seojininus/ssm/output/log/pendulum/mamba_pretrain_p20_v2_reg

ENV_PATH=/projects/bfrf/seojininus/envs/mamba
CODE_DIR=/u/seojininus/ssm
DATA_ROOT=${CODE_DIR}/pendulum_data
LOG_BASE=/projects/bfrf/seojininus/ssm/output/log/pendulum
WINNERS_JSON=${LOG_BASE}/hpo_mamba_pretrain_v2_reg/winners_by_dt_mode.json

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

if [[ ! -f "${WINNERS_JSON}" ]]; then
    echo "ERROR: v2 winners JSON not found at ${WINNERS_JSON}" >&2
    exit 1
fi

DT_MODES=(replace learned concat)
PATIENCE=${PATIENCE_OVERRIDE:-5}
MAX_EPOCHS=40

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
DT_IDX=$(( K / 20 ))
SD_IDX=$(( K % 20 ))

DT_MODE=${DT_MODES[$DT_IDX]}
SEED=$(( SD_IDX + 1 ))

read LR B WD DROP < <(python - "${WINNERS_JSON}" "${DT_MODE}" <<'PY'
import json, sys
w = json.load(open(sys.argv[1]))
cfg = w[sys.argv[2]]
print(cfg["lr"], cfg["batch_size"], cfg["weight_decay"], cfg["dropout"])
PY
)
if [[ -z "${LR:-}" || -z "${B:-}" || -z "${WD:-}" || -z "${DROP:-}" ]]; then
    echo "ERROR: failed to parse winner for dt_mode=${DT_MODE}" >&2
    exit 1
fi

VARIANT_DIR="${DT_MODE}_lr-${LR}_b-${B}_wd-${WD}_drop-${DROP}"
OUTPUT_DIR=${LOG_BASE}/mamba_pretrain_p20_v2_reg/${VARIANT_DIR}/seed${SEED}
mkdir -p "${OUTPUT_DIR}"
export WANDB_DIR="${OUTPUT_DIR}"

echo "=========================================="
echo "Pendulum v2_reg CONFIRM  dt=$DT_MODE  lr=$LR  B=$B  wd=$WD  drop=$DROP  seed=$SEED  patience=$PATIENCE"
echo "Winner source: ${WINNERS_JSON}"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
echo "host: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Time: $(date)"

cd "${CODE_DIR}"
START=$(date +%s)

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
        --wandb_run_name "v2reg_p20_pendulum_pt_${DT_MODE}_wd-${WD}_drop-${DROP}_seed${SEED}_a${ATTEMPT}"
    EXIT_CODE=$?
    if [[ ${EXIT_CODE} -eq 0 ]]; then break; fi
    if [[ -f "${TEST_CSV}" ]]; then EXIT_CODE=0; break; fi
    sleep 10
done

END=$(date +%s)
echo "Final exit code: ${EXIT_CODE}  Total wall: $((END-START))s  Done at $(date)"
exit ${EXIT_CODE}
