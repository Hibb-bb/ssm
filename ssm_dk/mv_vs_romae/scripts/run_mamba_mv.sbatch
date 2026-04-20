#!/bin/bash
#SBATCH --account=p32626
#SBATCH --job-name=mamba_mv
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/mvcompare_v1/mamba_mv/slurm_%A_%a.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/mvcompare_v1/mamba_mv/slurm_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-30

# Multivariate Mamba forecaster (paper's full arch): VarAttn + TemporalMamba.
#   regimes : sparse_independent, sparse_dependent
#   dt_modes: learned, replace, additive
#   seeds   : 1..5
# Total runs: 2 * 3 * 5 = 30.
#
# Environment and SLURM parameters intentionally match
# ssm_model/ssm/mamba_experiments/scripts/run_mamba_cuda_fixed.sh so that
# the CUDA selective-scan kernel is loadable and wall-clock timings are
# comparable across experiments.
#
# SLURM_ARRAY_TASK_ID in [1, 30]. Mapping (1-based):
#   k = TASK_ID - 1
#   regime_idx = k // 15        (0 or 1)
#   dt_idx     = (k % 15) // 5  (0, 1, 2)
#   seed_idx   = k % 5          (0..4)

module purge
module load gcc/11.2.0

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
LOG_BASE=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/mvcompare_v1

REGIMES=("sparse_independent" "sparse_dependent")
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

srun ${MAMBA_ENV}/bin/python -m mv_vs_romae.mamba_mv.train_mv \
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
