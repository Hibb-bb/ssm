#!/bin/bash
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=uea_cls_agg_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/uea_cls_v2/aggregate_%j.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/uea_cls_v2/aggregate_%j.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# CPU-bound aggregator (DeltaAI): walk hpo/<ds>/<dt>/<cell>/hpo/summary.json,
# pick winner per (dataset, dt_mode) using v2 picker (val/macro_f1 for HB/EP/LSST,
# val/acc for CT) with collapse-gap exclusion, write winner_configs.json.

set -euo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba
CODE_DIR=/projects/bfrf/seojininus/ssm
HPO_ROOT=${CODE_DIR}/output/log/uea_cls_v2/hpo
OUT_PATH=${CODE_DIR}/output/log/uea_cls_v2/winner_configs.json

source /sw/user/python/miniforge3-pytorch-2.5.0/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

cd "${CODE_DIR}"
echo "Aggregating UEA HPO winners from ${HPO_ROOT}"
echo "Time: $(date)"

python -m imts_benchmark.eval.aggregate_uea_hpo \
    --hpo_root "${HPO_ROOT}" \
    --out      "${OUT_PATH}"

EXIT_CODE=$?
echo "aggregate exit code: ${EXIT_CODE}  $(date)"
exit ${EXIT_CODE}
