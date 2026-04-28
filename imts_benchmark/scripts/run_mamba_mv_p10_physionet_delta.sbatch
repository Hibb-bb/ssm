#!/bin/bash
#SBATCH --account=bfrf-dtai-gh
#SBATCH --job-name=mv_p10_phy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=ghx4
#SBATCH --time=08:00:00
#SBATCH --array=1-15
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/mamba_mv_p10/slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/mamba_mv_p10/slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Mamba-MV on PhysioNet (V=41, N_max~1500), patience=10, 5 seeds * 3 dt_modes.
# Delta GH (GH200, aarch64, sm_90). No PhysioNet HPO winner exists yet —
# borrowing USHCN's per-dt_mode lr (USHCN is the closest analog: small,
# irregular, sparse). bs=32 + accum=4 (eff bs=128) because USHCN's bs=64/256
# would OOM at PhysioNet's V=41 + N_max~1500. GRID_K=256 per the p95 rule.
#
# Mapping: k = TASK_ID-1; dt_idx = k // 5; sd_idx = k % 5.
# 1-5  : replace
# 6-10 : learned
# 11-15: concat
#
# To switch to "single config, 5 seeds" (Option A): change --array=1-5,
# pin DT_MODE/LR/BS to one cell, and drop the dt_idx logic below.

set -uo pipefail

mkdir -p /projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/mamba_mv_p10

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

DT_MODES=(replace learned concat)
SEEDS=(1 2 3 4 5)
PATIENCE=10
BS=32
ACCUM=4
GRID_K=256
DS=physionet

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
DT_IDX=$(( K / 5 ))
SD_IDX=$(( K % 5 ))

DT_MODE=${DT_MODES[$DT_IDX]}
SEED=${SEEDS[$SD_IDX]}

# USHCN HPO winner lr per dt_mode (closest analog; see header).
case $DT_MODE in
    replace) LR=5e-4 ;;
    learned) LR=1e-4 ;;
    concat)  LR=1e-4 ;;
    *) echo "Unknown DT_MODE=$DT_MODE" >&2; exit 1 ;;
esac

VARIANT_DIR=${DT_MODE}_lr-${LR}_bs-${BS}_accum-${ACCUM}
OUTPUT_DIR=${LOG_BASE}/mamba_mv_p10/${DS}/${VARIANT_DIR}/seed${SEED}
mkdir -p "${OUTPUT_DIR}"
export WANDB_DIR="${OUTPUT_DIR}"

echo "=========================================="
echo "Mamba-MV (p10)  ds=$DS  dt=$DT_MODE  lr=$LR  bs=$BS  accum=$ACCUM  grid_K=$GRID_K  seed=$SEED  patience=$PATIENCE"
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
    --train_batch_size $BS \
    --accumulate_grad_batches $ACCUM \
    --patience $PATIENCE \
    --seed $SEED \
    --output_dir "${OUTPUT_DIR}" \
    --use_wandb \
    --wandb_project TSKing \
    --wandb_run_name "p10_mv_${DS}_${DT_MODE}_lr-${LR}_bs-${BS}_accum-${ACCUM}_seed${SEED}_delta"

EXIT_CODE=$?
END=$(date +%s)
echo "Exit code: $EXIT_CODE  Wall time: $((END-START))s  Done at $(date)"
exit ${EXIT_CODE}
