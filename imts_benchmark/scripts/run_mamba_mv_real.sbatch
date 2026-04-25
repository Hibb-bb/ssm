#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=mv_real
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/mamba_mv/slurm_%A_%a.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/mamba_mv/slurm_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-45
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Mamba-MV on real T-PatchGNN datasets (Phase 5).
#   datasets : physionet, activity, ushcn (auto_meta: n_vars/history/t_max from norm_stats.json)
#   dt_modes : learned, replace, concat
#   seeds    : 1..5
# Total: 3 * 3 * 5 = 45.
# Mapping: k = TASK_ID-1; ds_idx = k // 15; dt_idx = (k % 15) // 5; sd_idx = k % 5.
#
# Per-dataset grid_K (Nyquist-style: K = next_pow2(2 * p95(max_obs/var))):
#   physionet -> 256, activity -> 128, ushcn -> 128.
# Patience: 50 (matches fair_defaults; full convergence on default config).

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set. Run: export WANDB_API_KEY=... before sbatch." >&2
    exit 1
fi
_KEY_LEN=${#WANDB_API_KEY}
_KEY_CLEAN_LEN=$(printf '%s' "$WANDB_API_KEY" | tr -cd 'A-Za-z0-9_' | wc -c)
if [[ "$_KEY_LEN" -ne "$_KEY_CLEAN_LEN" ]] || [[ "$_KEY_LEN" -lt 30 ]]; then
    echo "ERROR: WANDB_API_KEY appears malformed (len=$_KEY_LEN, clean-len=$_KEY_CLEAN_LEN)." >&2
    exit 1
fi
export WANDB_MODE=online
unset WANDB_SILENT 2>/dev/null || true

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
DATA_ROOT=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/tpatchgnn_data
LOG_BASE=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real

DATASETS=("physionet" "activity" "ushcn")
DT_MODES=("learned" "replace" "concat")
SEEDS=(1 2 3 4 5)

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
DS_IDX=$(( K / 15 ))
DT_IDX=$(( (K % 15) / 5 ))
SD_IDX=$(( K % 5 ))

DS=${DATASETS[$DS_IDX]}
DT_MODE=${DT_MODES[$DT_IDX]}
SEED=${SEEDS[$SD_IDX]}

case $DS in
    physionet) GRID_K=256 ;;
    activity)  GRID_K=128 ;;
    ushcn)     GRID_K=128 ;;
esac

OUTPUT_DIR=${LOG_BASE}/mamba_mv/${DS}/${DT_MODE}/seed${SEED}
mkdir -p $OUTPUT_DIR
export WANDB_DIR=${OUTPUT_DIR}

echo "=========================================="
echo "Mamba-MV  ds=$DS  dt_mode=$DT_MODE  seed=$SEED  grid_K=$GRID_K"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Time: $(date)"
echo ""

${MAMBA_ENV}/bin/python -c "
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
print('CUDA selective_scan_fn loaded OK')
" && echo "CUDA kernel: AVAILABLE" || echo "CUDA kernel: NOT AVAILABLE (falling back to PyTorch)"
echo ""

cd $CODE_DIR
START_TIME=$(date +%s)

srun ${MAMBA_ENV}/bin/python -m imts_benchmark.mamba_mv.train_mv \
    --data_root $DATA_ROOT \
    --regime $DS \
    --auto_meta \
    --dt_mode $DT_MODE \
    --grid_K $GRID_K \
    --seed $SEED \
    --output_dir $OUTPUT_DIR \
    --use_wandb \
    --wandb_project TSKing \
    --wandb_run_name "real_mv_${DS}_${DT_MODE}_seed${SEED}"

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Exit code: $EXIT_CODE"
echo "Wall time: ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "Done at $(date)"
