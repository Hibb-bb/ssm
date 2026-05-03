#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8-interactive
#SBATCH --job-name=uea_v3_addon_agg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/uea_cls_v3_addon/aggregate_%j.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/uea_cls_v3_addon/aggregate_%j.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# v3 add-on aggregator: picks val/macro_f1-best (LR, BS) per (ds, concat).

set -euo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
HPO_ROOT=${CODE_DIR}/output/log/uea_cls_v3_addon/hpo
OUT_PATH=${CODE_DIR}/output/log/uea_cls_v3_addon/winner_configs.json

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

cd "${CODE_DIR}"
echo "Aggregating UEA v3-addon HPO winners from ${HPO_ROOT}"
echo "Time: $(date)"

python -m imts_benchmark.eval.aggregate_uea_hpo \
    --hpo_root "${HPO_ROOT}" \
    --out      "${OUT_PATH}"

EXIT_CODE=$?
echo "aggregate exit code: ${EXIT_CODE}  $(date)"
exit ${EXIT_CODE}
