#!/bin/bash
#SBATCH --account=p32626
#SBATCH --job-name=romae_fc
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/mvcompare_v1/romae/slurm_%A_%a.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/mvcompare_v1/romae/slurm_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-10

# RoMAE forecasting baseline (transformer MAE with RoPEND on time+variate).
#   regimes : sparse_independent, sparse_dependent
#   seeds   : 1..5
# Total runs: 2 * 5 = 10.
#
# Uses the SAME conda env and SLURM parameters as run_mamba_mv.sbatch so
# parameter-matched wall-clock timings are directly comparable.
#
# SLURM_ARRAY_TASK_ID in [1, 10]. Mapping (1-based):
#   k = TASK_ID - 1
#   regime_idx = k // 5   (0 or 1)
#   seed_idx   = k % 5    (0..4)

module purge
module load gcc/11.2.0

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
LOG_BASE=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/mvcompare_v1

REGIMES=("sparse_independent" "sparse_dependent")
SEEDS=(1 2 3 4 5)

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
REGIME_IDX=$(( K / 5 ))
SEED_IDX=$(( K % 5 ))

REGIME=${REGIMES[$REGIME_IDX]}
SEED=${SEEDS[$SEED_IDX]}

OUTPUT_DIR=${LOG_BASE}/romae/${REGIME}/default/seed${SEED}
mkdir -p $OUTPUT_DIR

echo "========================================="
echo "RoMAE forecaster  regime=$REGIME  seed=$SEED"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Time: $(date)"
echo ""

${MAMBA_ENV}/bin/python -c "
import romae
print('romae imported OK')
" && echo "RoMAE: AVAILABLE" || echo "RoMAE: MISSING (run pip install)"
echo ""

cd $CODE_DIR
START_TIME=$(date +%s)

srun ${MAMBA_ENV}/bin/python -m mv_vs_romae.romae_forecaster.train_romae \
    --regime $REGIME \
    --seed $SEED \
    --output_dir $OUTPUT_DIR

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "Exit code: $EXIT_CODE"
echo "Wall time: ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "Done at $(date)"
