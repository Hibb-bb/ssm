#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=export_preds_p34
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/export_preds_p34_%j.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/export_preds_p34_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Export test-set predictions for Phase 3 (async dense) and Phase 4-2
# (random-variate gap). Seed 1, best dt_mode per regime (read from the
# already-aggregated phase*_summary_wide.csv). Reuses
# export_test_predictions.py; it skips mTAN automatically (no ckpt).

set -euo pipefail

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk

cd "$CODE_DIR"

# ----- Phase 3 -----
srun ${MAMBA_ENV}/bin/python imts_benchmark/eval/export_test_predictions.py \
    --results_root /projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/phase3 \
    --data_root    "$CODE_DIR/data_correct_async" \
    --summary_csv  "$CODE_DIR/imts_benchmark/docs/phase3_results/phase3_summary_wide.csv" \
    --out_dir      "$CODE_DIR/imts_benchmark/docs/phase3_results/predictions" \
    --seed 1 \
    --device cuda

# ----- Phase 4-2 -----
srun ${MAMBA_ENV}/bin/python imts_benchmark/eval/export_test_predictions.py \
    --results_root /projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/phase4_2 \
    --data_root    "$CODE_DIR/data_correct_gap_random" \
    --summary_csv  "$CODE_DIR/imts_benchmark/docs/phase4_2_results/phase4_2_summary_wide.csv" \
    --out_dir      "$CODE_DIR/imts_benchmark/docs/phase4_2_results/predictions" \
    --seed 1 \
    --device cuda

echo "done"
