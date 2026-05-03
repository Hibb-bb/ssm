#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8-interactive
#SBATCH --job-name=uea_v31_agg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/uea_cls_v3/aggregate_v31_%j.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/uea_cls_v3/aggregate_v31_%j.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# v3.1 aggregator: re-scans the merged v3 hpo tree (now contains learned/replace
# from v3 + concat from v3.1 cleanup) and writes a fresh winner_configs.json
# covering all 3 dt_modes per (HB, LSST). OVERWRITES the v3 winner_configs.json.

set -euo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
HPO_ROOT=${CODE_DIR}/output/log/uea_cls_v3/hpo
OUT_PATH=${CODE_DIR}/output/log/uea_cls_v3/winner_configs.json

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

cd "${CODE_DIR}"
echo "Re-aggregating v3 HPO winners (now incl concat) from ${HPO_ROOT}"
echo "Backing up old winner JSON..."
if [[ -f "${OUT_PATH}" ]]; then
    cp "${OUT_PATH}" "${OUT_PATH}.v3_pre_v31"
fi
echo "Time: $(date)"

python -m imts_benchmark.eval.aggregate_uea_hpo \
    --hpo_root "${HPO_ROOT}" \
    --out      "${OUT_PATH}"

EXIT_CODE=$?
echo "aggregate exit code: ${EXIT_CODE}  $(date)"
exit ${EXIT_CODE}
