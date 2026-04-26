#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=plot_preds
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/aggregate/predictions_test/slurm_%j.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/aggregate/predictions_test/slurm_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Auto-fire plot_real_test_predictions.py --all once Mamba-MV learned + concat
# 5-seed jobs finish. Submit with:
#   sbatch --dependency=afterany:6441622:6441624 \
#          imts_benchmark/scripts/run_plot_predictions_after_mamba.sbatch
#
# Generates per-(model, variant, regime) sample prediction plots for all
# checkpoints found under output/log/imts_benchmark_v2_real/{s5,romae,mamba_mv}_p10/

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
OUT_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/aggregate/predictions_test

mkdir -p "${OUT_DIR}"

echo "=========================================="
echo "Plot test predictions for all available checkpoints"
echo "Job: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Time: $(date)"

cd "${CODE_DIR}"

${MAMBA_ENV}/bin/python -m imts_benchmark.eval.plot_real_test_predictions \
    --all \
    --out_dir "${OUT_DIR}" \
    --n_samples 4 \
    --n_plot_vars 6 \
    --seed 1 \
    --device cuda

EXIT_CODE=$?
echo "Exit code: $EXIT_CODE  Done at $(date)"
