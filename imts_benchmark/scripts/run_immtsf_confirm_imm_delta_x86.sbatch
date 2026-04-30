#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --job-name=immtsf_confirm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_imm/immtsf_confirm/%x/slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_imm/immtsf_confirm/%x/slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# 5-seed confirm with the architectural HPO winner from aggregate_immtsf_hpo_winners.py.
#
# Submit per (MODEL, DS_NAME). Array index 1-5 selects the seed.
#
# Usage:
#   sbatch --array=1-5 \
#       --export=ALL,MODEL=tPatchGNN,DATASET=FNSPID,DS_NAME=imm_fnspid \
#       --job-name=cf_tPatchGNN_imm_fnspid \
#       run_immtsf_confirm_imm_delta_x86.sbatch
#
# Reads <HPO_ROOT>/<MODEL>/<DS_NAME>/winner.json or all_winners.json,
# extracts the architectural HPs, runs at those values for seed = $SLURM_ARRAY_TASK_ID.

set -uo pipefail

MODEL="${MODEL:?Must set MODEL via --export}"
DATASET="${DATASET:?Must set DATASET via --export, e.g. DATASET=FNSPID}"
DS_NAME="${DS_NAME:?Must set DS_NAME via --export, e.g. DS_NAME=imm_fnspid}"

case $DS_NAME in
    imm_cesnet|imm_ilinet) BS_DEFAULT=8 ;;
    *) BS_DEFAULT=32 ;;
esac
BATCH_SIZE="${BATCH_SIZE:-${BS_DEFAULT}}"
LR="${LR:-1e-3}"
PATIENCE="${PATIENCE:-10}"

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
IMM_TSF_REPO=${CODE_DIR}/_imm_tsf_repo
TIME_IMM_DATA=${CODE_DIR}/_time_imm_repo/data
HPO_ROOT=${CODE_DIR}/output/log/imts_benchmark_v2_imm/immtsf_hpo
CONFIRM_ROOT=${CODE_DIR}/output/log/imts_benchmark_v2_imm/immtsf_confirm
WINNERS=${HPO_ROOT}/${MODEL}/winners_by_dataset.json

if [[ ! -f "${WINNERS}" ]]; then
    echo "ERROR: winners file ${WINNERS} not found. Run aggregate_immtsf_hpo_winners.py first." >&2
    exit 1
fi

ARCH_HPS_JSON=$(python -c "
import json
w = json.load(open('${WINNERS}'))
ds = w.get('${DS_NAME}')
if ds is None:
    raise SystemExit('no winner for ${DS_NAME} in ${WINNERS}')
print(json.dumps(ds.get('arch_hps', {})))
")

if [[ -z "${ARCH_HPS_JSON}" ]]; then
    echo "ERROR: failed to extract arch_hps for ${MODEL}/${DS_NAME}" >&2
    exit 2
fi

SEED=$SLURM_ARRAY_TASK_ID
OUT_DIR=${CONFIRM_ROOT}/${MODEL}/${DS_NAME}/seed${SEED}
mkdir -p "${OUT_DIR}"

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

echo "==========================================="
echo "IMM-TSF confirm  model=${MODEL}  dataset=${DATASET} (${DS_NAME})"
echo "  arch=${ARCH_HPS_JSON}"
echo "  lr=${LR}  patience=${PATIENCE}  bs=${BATCH_SIZE}  seed=${SEED}"
echo "  out=${OUT_DIR}"
echo "  job=${SLURM_JOB_ID}  array=${SLURM_ARRAY_TASK_ID}  host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
date
echo "==========================================="

cd "${CODE_DIR}"
START=$(date +%s)

python imts_benchmark/scripts/run_immtsf_baseline.py \
    --imm_tsf_repo "${IMM_TSF_REPO}" \
    --time_imm_data "${TIME_IMM_DATA}" \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --lr ${LR} \
    --patience ${PATIENCE} \
    --batch_size ${BATCH_SIZE} \
    --seed ${SEED} \
    --max_epochs 1000 \
    --arch_hps "${ARCH_HPS_JSON}" \
    --output_dir "${OUT_DIR}" \
    --output_json "${OUT_DIR}/result.json"

EXIT=$?
END=$(date +%s)
echo "Exit: ${EXIT}  Wall: $((END-START))s  Done: $(date)"
exit ${EXIT}
