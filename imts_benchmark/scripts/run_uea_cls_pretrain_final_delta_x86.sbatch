#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --job-name=uea_pre_final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --array=1-45
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/uea_cls_pretrain/final/slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/uea_cls_pretrain/final/slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Pretrain-arch UEA classification final eval: 5 ds × 3 dt × 3 seeds = 45 cells.
# Replays HPO winner per (dataset, dt_mode) under the same patience=10 / max_epochs=200
# protocol as the HPO sweep.
#
# Index decoding: k = TASK_ID - 1
#   ds_idx = k // 9      (5 ds × 9 cells/ds)
#   rem    = k %  9
#   dt_idx = rem // 3    (3 dt: replace, learned, concat)
#   sd_idx = rem %  3    (0..2 -> seeds 42, 123, 0)

set -uo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
DATA_ROOT=${CODE_DIR}/data_uea
LOG_BASE=${CODE_DIR}/output/log/uea_cls_pretrain/final
WINNER_PATH=${CODE_DIR}/output/log/uea_cls_pretrain/winner_configs.json

mkdir -p "${LOG_BASE}"

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

if [[ ! -f "${WINNER_PATH}" ]]; then
    echo "ERROR: ${WINNER_PATH} not found. Run aggregator first." >&2
    exit 1
fi

DATASETS=("BasicMotions" "CharacterTrajectories" "Epilepsy" "Heartbeat" "LSST")
DT_MODES=("replace" "learned" "concat")
SEEDS=(42 123 0)

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
DS_IDX=$(( K / 9 ))
REM=$(( K % 9 ))
DT_IDX=$(( REM / 3 ))
SD_IDX=$(( REM % 3 ))

DATASET=${DATASETS[$DS_IDX]}
DT_MODE=${DT_MODES[$DT_IDX]}
SEED=${SEEDS[$SD_IDX]}

read -r LR BS < <(python - "${WINNER_PATH}" "${DATASET}" "${DT_MODE}" <<'PY'
import json, sys
w = json.load(open(sys.argv[1]))
cell = w[sys.argv[2]][sys.argv[3]]
print(cell["lr"], cell["batch_size"])
PY
)

if [[ -z "${LR:-}" || -z "${BS:-}" ]]; then
    echo "ERROR: failed to parse winner for ${DATASET}/${DT_MODE} from ${WINNER_PATH}" >&2
    exit 1
fi

OUT_DIR=${LOG_BASE}/${DATASET}/${DT_MODE}/seed${SEED}
mkdir -p "${OUT_DIR}"

cd "${CODE_DIR}"

echo "=========================================="
echo "UEA final pretrain-arch: ds=${DATASET}  dt=${DT_MODE}  lr=${LR}  bs=${BS}  seed=${SEED}  patience=10  max_epochs=200"
echo "Winner source: ${WINNER_PATH}"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Time: $(date)"
echo "=========================================="

srun python -m imts_benchmark.mamba_pretrain.train_cls \
    --dataset ${DATASET} \
    --data_root "${DATA_ROOT}" \
    --dt_mode ${DT_MODE} \
    --lr ${LR} \
    --batch_size ${BS} \
    --max_epochs 200 \
    --patience 10 \
    --seed ${SEED} \
    --num_workers 4 \
    --output_dir "${OUT_DIR}" \
    --run_name final

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}  Done at $(date)"
exit ${EXIT_CODE}
