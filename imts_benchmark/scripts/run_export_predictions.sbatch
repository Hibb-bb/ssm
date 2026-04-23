#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=export_preds
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/export_preds_%j.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/export_preds_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# One-shot prediction export: load seed=1 ckpt for each (model, regime),
# run forward on test split, dump per-variate predictions to JSONL for
# downstream plotting.

set -euo pipefail

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk

cd "$CODE_DIR"

RESULTS_ROOT=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2
DATA_ROOT="$CODE_DIR/data_correct"
SUMMARY_CSV="$CODE_DIR/imts_benchmark/docs/phase2_assets/phase2_summary_wide.csv"
OUT_DIR="$CODE_DIR/imts_benchmark/docs/phase2_assets/predictions"

srun ${MAMBA_ENV}/bin/python imts_benchmark/eval/export_test_predictions.py \
    --results_root "$RESULTS_ROOT" \
    --data_root "$DATA_ROOT" \
    --summary_csv "$SUMMARY_CSV" \
    --out_dir "$OUT_DIR" \
    --seed 1 \
    --device cuda

echo "done"
