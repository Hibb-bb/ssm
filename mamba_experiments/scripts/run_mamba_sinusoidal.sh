#!/bin/bash
#SBATCH --account=p32626
#SBATCH --job-name=mamba_sin
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/mamba_sinusoidal_nobs160/%x_%A_%a.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/mamba_sinusoidal_nobs160/%x_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=1-60

module purge
module load gcc/11.2.0

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

CONDA_ENV=/projects/b1094/StarEmbed/pythonenvs/moirai_train
DATA_ROOT=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/tpatchgnn_data_nobs160
LOG_BASE=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/mamba_sinusoidal_nobs160
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_model/ssm/mamba_experiments

VARIANTS=("vanilla_mamba" "mamba_true_dt" "mamba_hybrid_dt")
DT_MODES=("learned" "replace" "additive")
IRREGS=("high_irreg" "med_irreg" "low_irreg" "regular")
SEEDS=(1 2 3 4 5)

TASK_ID=$SLURM_ARRAY_TASK_ID

SEED_IDX=$(( (TASK_ID - 1) / 12 ))
REMAIN=$(( (TASK_ID - 1) % 12 ))
VARIANT_IDX=$(( REMAIN / 4 ))
IRREG_IDX=$(( REMAIN % 4 ))

SEED=${SEEDS[$SEED_IDX]}
VARIANT=${VARIANTS[$VARIANT_IDX]}
DT_MODE=${DT_MODES[$VARIANT_IDX]}
IRREG=${IRREGS[$IRREG_IDX]}

OUTPUT_DIR=${LOG_BASE}/${VARIANT}/${IRREG}/seed${SEED}

echo "========================================="
echo "Mamba Sinusoidal Forecasting"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Variant: $VARIANT (dt_mode=$DT_MODE)"
echo "Irregularity: $IRREG"
echo "Seed: $SEED"
echo "Output: $OUTPUT_DIR"
echo "Time: $(date)"
echo ""

mkdir -p $OUTPUT_DIR

cd $CODE_DIR

srun ${CONDA_ENV}/bin/python -m forecaster.train \
    --dt_mode $DT_MODE \
    --data_root $DATA_ROOT \
    --irregularity $IRREG \
    --seed $SEED \
    --output_dir $OUTPUT_DIR

echo ""
echo "Exit code: $?"
echo "Done at $(date)"
