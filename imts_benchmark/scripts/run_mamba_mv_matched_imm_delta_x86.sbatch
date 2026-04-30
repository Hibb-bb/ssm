#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --job-name=mv_matched_imm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuH200x8
#SBATCH --time=12:00:00
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_imm/matched_mv_%x/slurm_%j.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_imm/matched_mv_%x/slurm_%j.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Mamba-MV under the IMM-TSF "Without Textual Data" baseline protocol from
# _imm_tsf_repo/main_all.py + main.py, so our number is directly comparable to
# paper Tables 3-11 'Uni' rows. Single seed, no HPO, fixed config:
#   lr=1e-3, batch_size=8, weight_decay=0.01, patience=3, max_epochs=1000,
#   AMP off, no LR scheduler (warmup=0, training_steps≫run length)
# dt_mode is an architecture choice, not an HP; we fix it to `concat` (the
# PhysioNet winner). Other modes can be reported as ablations from the HPO tree.
#
# Usage:
#   sbatch --export=ALL,DS_NAME=imm_fnspid \
#       --job-name=mv_matched_${DS_NAME} \
#       imts_benchmark/scripts/run_mamba_mv_matched_imm_delta_x86.sbatch

set -uo pipefail

DS_NAME="${DS_NAME:?Must set DS_NAME via --export, e.g. DS_NAME=imm_fnspid}"
DT_MODE="${DT_MODE:-concat}"
SEED="${SEED:-1}"

LOG_BASE_ROOT=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_imm
ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
DATA_ROOT=${CODE_DIR}/time_imm_data
OUTPUT_DIR=${LOG_BASE_ROOT}/matched_mv_${DS_NAME}/${DT_MODE}/seed${SEED}
mkdir -p "${OUTPUT_DIR}"

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

if [[ -z "${WANDB_API_KEY:-}" ]] && [[ -f ~/.bashrc ]]; then
    source ~/.bashrc 2>/dev/null || true
fi
if [[ -z "${WANDB_API_KEY:-}" ]] && [[ -f ~/.netrc ]]; then
    export WANDB_API_KEY=$(awk '/api.wandb.ai/{f=1} f && /password/{print $2; exit}' ~/.netrc)
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "WARNING: WANDB_API_KEY not set; falling back to offline mode" >&2
    export WANDB_MODE=offline
else
    export WANDB_MODE=online
fi
export WANDB_DIR="${OUTPUT_DIR}"

echo "=========================================="
echo "Mamba-MV @ IMM-TSF protocol  ds=$DS_NAME  dt=$DT_MODE  seed=$SEED"
echo "Job: $SLURM_JOB_ID  host: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Time: $(date)"

cd "${CODE_DIR}"
START=$(date +%s)

# Matched to _imm_tsf_repo/main.py defaults:
#   lr=1e-3 (main.py:693)
#   batch_size=8 (main_all.py:67)
#   weight_decay=0.01 (main.py:699)
#   patience=3 (main.py:682)
#   max_epochs=1000 (main.py:678)
#   AMP off (main.py:709)
#   no scheduler -> set num_warmup_steps=0, num_training_steps=10**9 (constant lr)
srun python -m imts_benchmark.mamba_mv.train_mv \
    --data_root "${DATA_ROOT}" \
    --regime "${DS_NAME}" \
    --auto_meta \
    --dt_mode "${DT_MODE}" \
    --grid_K 256 \
    --lr 1e-3 \
    --weight_decay 0.01 \
    --train_batch_size 8 \
    --accumulate_grad_batches 1 \
    --max_epochs 1000 \
    --patience 3 \
    --num_warmup_steps 0 \
    --num_training_steps 1000000000 \
    --precision 32-true \
    --seed "${SEED}" \
    --output_dir "${OUTPUT_DIR}" \
    --use_wandb \
    --wandb_project TSKing \
    --wandb_run_name "matched_mv_${DS_NAME}_${DT_MODE}_seed${SEED}_delta"

EXIT_CODE=$?
END=$(date +%s)
echo "Exit code: $EXIT_CODE  Wall time: $((END-START))s  Done at $(date)"
exit ${EXIT_CODE}
