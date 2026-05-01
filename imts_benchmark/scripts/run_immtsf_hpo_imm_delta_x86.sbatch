#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --job-name=immtsf_hpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_imm/immtsf_hpo/%x/slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_imm/immtsf_hpo/%x/slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# IMM-TSF baselines (CRU, LatentODE, NeuralFlow, tPatchGNN) HPO sweep on a
# TIME-IMM dataset.
#
# Lean unified design: 3-cell lr sweep, identical for all 4 models.
#   lr in {1e-4, 5e-4, 1e-3}
#   batch_size = 32 (T-PatchGNN paper default; bs=8 already covered by Table 1)
#   patience  = 10 (T-PatchGNN paper standard)
#   architectural HPs = IMM-TSF defaults (paper Tables 3-11 settings)
#
# Total compute: 4 models * 8 datasets * 3 cells = 96 sweep runs.
# Confirm: 4 * 8 * 3 seeds = 96 runs.
#
# Usage:
#   sbatch --array=1-3 --export=ALL,MODEL=tPatchGNN,DATASET=FNSPID,DS_NAME=imm_fnspid \
#       --job-name=hpo_tPatchGNN_imm_fnspid run_immtsf_hpo_imm_delta_x86.sbatch
#
# DS_NAME (lowercase imm_*) controls the output dir layout for our aggregator;
# DATASET (uppercase) is the IMM-TSF-internal name passed to its --dataset flag.

set -uo pipefail

MODEL="${MODEL:?Must set MODEL via --export, e.g. MODEL=tPatchGNN}"
DATASET="${DATASET:?Must set DATASET via --export, e.g. DATASET=FNSPID}"
DS_NAME="${DS_NAME:?Must set DS_NAME via --export, e.g. DS_NAME=imm_fnspid}"

# Allow per-launch override but defaults match the lean design.
BATCH_SIZE="${BATCH_SIZE:-32}"
PATIENCE="${PATIENCE:-10}"
SEED_HPO="${SEED_HPO:-1}"

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
IMM_TSF_REPO=${CODE_DIR}/_imm_tsf_repo
TIME_IMM_DATA=${CODE_DIR}/_time_imm_repo/data
LOG_BASE=${CODE_DIR}/output/log/imts_benchmark_v2_imm/immtsf_hpo

# 3-cell lr grid (unified across all 4 baselines)
LRS=(1e-4 5e-4 1e-3)
N_CELLS=${#LRS[@]}

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
if [[ $K -ge $N_CELLS ]]; then
    echo "TASK_ID=$TASK_ID exceeds N_CELLS=$N_CELLS; nothing to do" >&2
    exit 0
fi

LR=${LRS[$K]}
TAG="lr-${LR}"

# No --arch_hps passed: wrapper sends an empty dict, so IMM-TSF's
# update_args_for_dataset / update_args_for_model fill in the per-(model,
# dataset) architectural defaults that generated paper Tables 3-11.
ARCH_HPS_JSON='{}'

OUT_DIR=${LOG_BASE}/${MODEL}/${DS_NAME}/${TAG}/seed${SEED_HPO}
mkdir -p "${OUT_DIR}"

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

echo "==========================================="
echo "IMM-TSF HPO  model=${MODEL}  dataset=${DATASET} (${DS_NAME})"
echo "  cell ${TASK_ID}/${N_CELLS}  TAG=${TAG}  lr=${LR}"
echo "  patience=${PATIENCE}  bs=${BATCH_SIZE}  seed=${SEED_HPO}"
echo "  arch HPs: IMM-TSF defaults (no override)"
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
    --seed ${SEED_HPO} \
    --max_epochs 1000 \
    --arch_hps "${ARCH_HPS_JSON}" \
    --output_dir "${OUT_DIR}" \
    --output_json "${OUT_DIR}/result.json"

EXIT=$?
END=$(date +%s)
echo "Exit code: ${EXIT}  Wall: $((END-START))s  Done: $(date)"
exit ${EXIT}
