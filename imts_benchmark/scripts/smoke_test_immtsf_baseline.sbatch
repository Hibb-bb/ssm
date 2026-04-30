#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --job-name=immtsf_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_imm/immtsf_smoke/slurm_%j.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_imm/immtsf_smoke/slurm_%j.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Smoke test: run one cell of an IMM-TSF baseline on a TIME-IMM dataset using
# the matched-protocol HP (lr=1e-3, patience=10, seed=1) to validate that:
#   1. Dependencies resolve in our mamba_x86 env (with stribor/geotorch/torchdiffeq added)
#   2. The wrapper correctly invokes IMM-TSF's `trainable()`
#   3. The output JSON has the expected metrics structure
#
# Usage:
#   sbatch --export=ALL,MODEL=tPatchGNN,DATASET=FNSPID smoke_test_immtsf_baseline.sbatch
#   sbatch --export=ALL,MODEL=CRU,DATASET=FNSPID  smoke_test_immtsf_baseline.sbatch
#   sbatch --export=ALL,MODEL=LatentODE,DATASET=FNSPID smoke_test_immtsf_baseline.sbatch
#   sbatch --export=ALL,MODEL=NeuralFlow,DATASET=FNSPID smoke_test_immtsf_baseline.sbatch

set -uo pipefail

MODEL="${MODEL:-tPatchGNN}"
DATASET="${DATASET:-FNSPID}"
LR="${LR:-1e-3}"
PATIENCE="${PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-1}"

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
IMM_TSF_REPO=${CODE_DIR}/_imm_tsf_repo
TIME_IMM_DATA=${CODE_DIR}/_time_imm_repo/data
LOG_BASE=${CODE_DIR}/output/log/imts_benchmark_v2_imm/immtsf_smoke
OUT_DIR=${LOG_BASE}/${MODEL}_${DATASET}_lr${LR}_p${PATIENCE}_bs${BATCH_SIZE}_seed${SEED}_${SLURM_JOB_ID}
mkdir -p "${OUT_DIR}"

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

echo "==========================================="
echo "IMM-TSF baseline smoke test"
echo "  model=${MODEL}  dataset=${DATASET}"
echo "  lr=${LR}  patience=${PATIENCE}  bs=${BATCH_SIZE}  seed=${SEED}"
echo "  job=${SLURM_JOB_ID}  host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "  time=$(date)"
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
    --output_dir "${OUT_DIR}" \
    --output_json "${OUT_DIR}/result.json"

EXIT_CODE=$?
END=$(date +%s)
echo "Exit code: ${EXIT_CODE}  Wall time: $((END-START))s  Done at $(date)"
exit ${EXIT_CODE}
