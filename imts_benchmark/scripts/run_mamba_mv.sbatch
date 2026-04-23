#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=mamba_mv
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/mamba_mv/slurm_%A_%a.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/mamba_mv/slurm_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-60
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Multivariate Mamba forecaster (paper's full arch): VarAttn + TemporalMamba.
#   regimes : multisin_regular, multisin_low_irreg, multisin_med_irreg, multisin_high_irreg
#   dt_modes: learned, replace, additive
#   seeds   : 1..5
# Total runs: 4 * 3 * 5 = 60.
#
# v2 (Phase 2, rebuilt on correct convex-mix generator with shared timestamps).
#
# SLURM_ARRAY_TASK_ID in [1, 60]. Mapping (1-based):
#   k = TASK_ID - 1
#   regime_idx = k // 15        (0..3)
#   dt_idx     = (k % 15) // 5  (0..2)
#   seed_idx   = k % 5          (0..4)

module purge
module load gcc/11.2.0

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
LOG_BASE=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2

REGIMES=("multisin_regular" "multisin_low_irreg" "multisin_med_irreg" "multisin_high_irreg")
DT_MODES=("learned" "replace" "additive")
SEEDS=(1 2 3 4 5)

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
REGIME_IDX=$(( K / 15 ))
DT_IDX=$(( (K % 15) / 5 ))
SEED_IDX=$(( K % 5 ))

REGIME=${REGIMES[$REGIME_IDX]}
DT_MODE=${DT_MODES[$DT_IDX]}
SEED=${SEEDS[$SEED_IDX]}

OUTPUT_DIR=${LOG_BASE}/mamba_mv/${REGIME}/${DT_MODE}/seed${SEED}
mkdir -p $OUTPUT_DIR

echo "========================================="
echo "Mamba-MV  regime=$REGIME  dt_mode=$DT_MODE  seed=$SEED"
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
    --regime $REGIME \
    --dt_mode $DT_MODE \
    --seed $SEED \
    --output_dir $OUTPUT_DIR

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "Exit code: $EXIT_CODE"
echo "Wall time: ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "Done at $(date)"
