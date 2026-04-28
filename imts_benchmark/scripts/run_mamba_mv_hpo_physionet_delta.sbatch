#!/bin/bash
#SBATCH --account=bfrf-dtai-gh
#SBATCH --job-name=mv_hpo_phy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=ghx4
#SBATCH --time=04:00:00
#SBATCH --array=1-27
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/hpo_mamba_mv_phy/slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/hpo_mamba_mv_phy/slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Mamba-MV HPO sweep on PhysioNet, seed=1, patience=10.
# Mirrors the colleague's 9-cell HPO grid (3 lr × 3 effective bs) but per-dt_mode,
# adapted for PhysioNet OOM (V=41, N_max~1500).
#
# Grid:
#   dt_modes : replace, learned, concat        (3)
#   lrs      : 1e-4, 5e-4, 2e-3                (3)
#   eff_bs   : 64, 128, 256                    (3)  -> implemented as (phys_bs, accum):
#              eff=64  -> phys=16, accum=4
#              eff=128 -> phys=32, accum=4
#              eff=256 -> phys=32, accum=8
# Total: 3 * 3 * 3 = 27 cells, single seed.
# Mapping: k = TASK_ID-1; dt_idx = k // 9; lr_idx = (k % 9) // 3; ebs_idx = k % 3.
#
# Winners are picked per-dt_mode by aggregate_hpo_winners_physionet.py and
# consumed by run_mamba_mv_confirm_physionet_delta.sbatch (5-seed confirm).

set -uo pipefail

mkdir -p /projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/hpo_mamba_mv_phy

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
LRS=(1e-4 5e-4 2e-3)
EFF_BS=(64 128 256)
DS=physionet
SEED=1
GRID_K=256

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
DT_IDX=$(( K / 9 ))
LR_IDX=$(( (K % 9) / 3 ))
EBS_IDX=$(( K % 3 ))

DT_MODE=${DT_MODES[$DT_IDX]}
LR=${LRS[$LR_IDX]}
EBS=${EFF_BS[$EBS_IDX]}

# eff_bs -> (phys_bs, accum). Keeps phys_bs <= 32 (smoke-proven safe for V=41).
case $EBS in
    64)  PHYS_BS=16; ACCUM=4 ;;
    128) PHYS_BS=32; ACCUM=4 ;;
    256) PHYS_BS=32; ACCUM=8 ;;
    *) echo "Unknown EBS=$EBS" >&2; exit 1 ;;
esac

CELL_TAG="lr-${LR}_ebs-${EBS}"
OUTPUT_DIR=${LOG_BASE}/hpo_mamba_mv_phy/${DT_MODE}/${CELL_TAG}/seed${SEED}
mkdir -p "${OUTPUT_DIR}"
export WANDB_DIR="${OUTPUT_DIR}"

echo "=========================================="
echo "Mamba-MV HPO  ds=$DS  dt=$DT_MODE  lr=$LR  ebs=$EBS (phys=${PHYS_BS} accum=${ACCUM})  grid_K=$GRID_K  seed=$SEED"
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
    --patience 10 \
    --seed $SEED \
    --output_dir "${OUTPUT_DIR}" \
    --use_wandb \
    --wandb_project TSKing \
    --wandb_run_name "hpo_mv_${DS}_${DT_MODE}_lr-${LR}_ebs-${EBS}_seed${SEED}_delta"

EXIT_CODE=$?
END=$(date +%s)
echo "Exit code: $EXIT_CODE  Wall time: $((END-START))s  Done at $(date)"
exit ${EXIT_CODE}
