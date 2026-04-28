#!/bin/bash
#SBATCH --account=bfrf-dtai-gh
#SBATCH --job-name=mv_p10_phy_cf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=ghx4
#SBATCH --time=08:00:00
#SBATCH --array=1-15
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/mamba_mv_p10/slurm_confirm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/mamba_mv_p10/slurm_confirm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Mamba-MV PhysioNet 5-seed CONFIRM at HPO winners (one per dt_mode).
# Depends on the aggregator job that writes winners_by_dt_mode.json.
#
# 3 dt_modes (replace, learned, concat) × 5 seeds = 15 tasks.
# Mapping: k = TASK_ID-1; dt_idx = k // 5; sd_idx = k % 5.
# Each task reads the per-dt_mode (lr, phys_bs, accum) from the winners JSON
# at job start.

set -uo pipefail

mkdir -p /projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/mamba_mv_p10

ENV_PATH=/projects/bfrf/seojininus/envs/mamba
CODE_DIR=/projects/bfrf/seojininus/ssm
DATA_ROOT=${CODE_DIR}/tpatchgnn_data
LOG_BASE=${CODE_DIR}/output/log/imts_benchmark_v2_real
WINNERS_JSON=${LOG_BASE}/hpo_mamba_mv_phy/winners_by_dt_mode.json

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

if [[ ! -f "${WINNERS_JSON}" ]]; then
    echo "ERROR: winners JSON not found at ${WINNERS_JSON}" >&2
    exit 1
fi

DT_MODES=(replace learned concat)
SEEDS=(1 2 3 4 5)
DS=physionet
GRID_K=256
PATIENCE=10

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
DT_IDX=$(( K / 5 ))
SD_IDX=$(( K % 5 ))

DT_MODE=${DT_MODES[$DT_IDX]}
SEED=${SEEDS[$SD_IDX]}

# Read winner config for this dt_mode from JSON.
read LR PHYS_BS ACCUM EFF_BS < <(python - "${WINNERS_JSON}" "${DT_MODE}" <<'PY'
import json, sys
winners = json.load(open(sys.argv[1]))
cfg = winners[sys.argv[2]]
print(cfg["lr"], cfg["phys_bs"], cfg["accum"], cfg["eff_bs"])
PY
)

if [[ -z "${LR:-}" || -z "${PHYS_BS:-}" || -z "${ACCUM:-}" ]]; then
    echo "ERROR: failed to parse winner for dt_mode=${DT_MODE} from ${WINNERS_JSON}" >&2
    exit 1
fi

VARIANT_DIR="${DT_MODE}_lr-${LR}_bs-${PHYS_BS}_accum-${ACCUM}"
OUTPUT_DIR=${LOG_BASE}/mamba_mv_p10/${DS}/${VARIANT_DIR}/seed${SEED}
mkdir -p "${OUTPUT_DIR}"
export WANDB_DIR="${OUTPUT_DIR}"

echo "=========================================="
echo "Mamba-MV CONFIRM (p10)  ds=$DS  dt=$DT_MODE  lr=$LR  phys_bs=$PHYS_BS  accum=$ACCUM  eff_bs=$EFF_BS  grid_K=$GRID_K  seed=$SEED  patience=$PATIENCE"
echo "Winner source: ${WINNERS_JSON}"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
echo "host: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Time: $(date)"

cd "${CODE_DIR}"
START=$(date +%s)

srun python -m imts_benchmark.mamba_mv.train_mv \
    --data_root "${DATA_ROOT}" \
    --regime $DS \
    --auto_meta \
    --dt_mode $DT_MODE \
    --grid_K $GRID_K \
    --lr $LR \
    --train_batch_size $PHYS_BS \
    --accumulate_grad_batches $ACCUM \
    --patience $PATIENCE \
    --seed $SEED \
    --output_dir "${OUTPUT_DIR}" \
    --use_wandb \
    --wandb_project TSKing \
    --wandb_run_name "p10_mv_${DS}_${DT_MODE}_lr-${LR}_bs-${PHYS_BS}_accum-${ACCUM}_seed${SEED}_delta_confirm"

EXIT_CODE=$?
END=$(date +%s)
echo "Exit code: $EXIT_CODE  Wall time: $((END-START))s  Done at $(date)"
exit ${EXIT_CODE}
