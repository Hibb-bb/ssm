#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=mtan_fc
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/mtan/slurm_%A_%a.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/mtan/slurm_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-20
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# mTAN forecaster baseline (Multi-Time Attention Networks, Shukla & Marlin
# ICLR 2021). Same training recipe as Mamba-MV / RoMAE via fair_defaults.
#   regimes : multisin_regular, multisin_low_irreg, multisin_med_irreg, multisin_high_irreg
#   seeds   : 1..5
# Total runs: 4 * 5 = 20.
#
# v2 (Phase 2, rebuilt on correct convex-mix generator).
#
# SLURM_ARRAY_TASK_ID in [1, 20]. Mapping (1-based):
#   k = TASK_ID - 1
#   regime_idx = k // 5   (0..3)
#   seed_idx   = k % 5    (0..4)

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
LOG_BASE=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2

REGIMES=("multisin_regular" "multisin_low_irreg" "multisin_med_irreg" "multisin_high_irreg")
SEEDS=(1 2 3 4 5)

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
REGIME_IDX=$(( K / 5 ))
SEED_IDX=$(( K % 5 ))

REGIME=${REGIMES[$REGIME_IDX]}
SEED=${SEEDS[$SEED_IDX]}

OUTPUT_DIR=${LOG_BASE}/mtan/${REGIME}/default/seed${SEED}
mkdir -p $OUTPUT_DIR

echo "========================================="
echo "mTAN forecaster  regime=$REGIME  seed=$SEED"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Time: $(date)"
echo ""

cd $CODE_DIR
START_TIME=$(date +%s)

srun ${MAMBA_ENV}/bin/python -m imts_benchmark.mtan_forecaster.train_mtan \
    --regime $REGIME \
    --seed $SEED \
    --lr 1e-3 \
    --weight_decay 0 \
    --output_dir $OUTPUT_DIR
# Paper-native recipe overrides (Shukla & Marlin 2021):
#   lr=1e-3 (vs shared default 5e-4) and wd=0 (vs shared 0.01).
#   Architecture defaults now: rec_hidden=32, gen_hidden=50, latent_dim=20
#   -> ~0.12M params. See docs/RESULTS_phase2.md.

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "Exit code: $EXIT_CODE"
echo "Wall time: ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "Done at $(date)"
