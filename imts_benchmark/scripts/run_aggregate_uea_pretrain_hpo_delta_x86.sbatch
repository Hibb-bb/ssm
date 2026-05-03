#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8-interactive
#SBATCH --job-name=uea_pre_agg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/uea_cls_pretrain/aggregate_%j.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/uea_cls_pretrain/aggregate_%j.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Pretrain-arch UEA HPO aggregator: walks output/log/uea_cls_pretrain/hpo/<ds>/<dt>/<cell>/hpo/summary.json,
# picks val/macro_f1-best per (dataset, dt_mode) using the same v2 picker (with
# collapse-gap exclusion), writes winner_configs.json with lr + batch_size.

set -euo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
HPO_ROOT=${CODE_DIR}/output/log/uea_cls_pretrain/hpo
OUT_PATH=${CODE_DIR}/output/log/uea_cls_pretrain/winner_configs.json

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

cd "${CODE_DIR}"
echo "Aggregating pretrain UEA HPO winners from ${HPO_ROOT}"
echo "Time: $(date)"

python -m imts_benchmark.eval.aggregate_uea_hpo \
    --hpo_root "${HPO_ROOT}" \
    --out      "${OUT_PATH}"

EXIT_CODE=$?
echo "aggregate exit code: ${EXIT_CODE}  $(date)"
exit ${EXIT_CODE}
