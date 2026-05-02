#!/bin/bash
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=uea_cls_final_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --array=1-24
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/uea_cls_v2/final/slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/uea_cls_v2/final/slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# UEA classification final eval (DeltaAI / GH200).
# Replays HPO-picked (LR, batch_size) per (dataset, dt_mode), 3 seeds.
# 4 datasets x 2 dt_modes x 3 seeds = 24 cells.
# Index mapping: k = TASK_ID - 1
#   dataset_idx = k // 6
#   remainder   = k %  6
#   dt_mode_idx = remainder // 3   (0=replace, 1=learned)
#   seed_idx    = remainder %  3   (0..2 -> seed in {42, 123, 0})

set -uo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba
CODE_DIR=/projects/bfrf/seojininus/ssm
DATA_ROOT=${CODE_DIR}/data_uea
LOG_BASE=${CODE_DIR}/output/log/uea_cls_v2/final
WINNER_PATH=${CODE_DIR}/output/log/uea_cls_v2/winner_configs.json

mkdir -p "${LOG_BASE}"

source /sw/user/python/miniforge3-pytorch-2.5.0/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

if [[ ! -f "${WINNER_PATH}" ]]; then
    echo "ERROR: ${WINNER_PATH} not found. Run aggregator first." >&2
    exit 1
fi

DATASETS=("CharacterTrajectories" "Epilepsy" "Heartbeat" "LSST")
DT_MODES=("replace" "learned")
SEEDS=(42 123 0)

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
DATASET_IDX=$(( K / 6 ))
REM=$(( K % 6 ))
DT_IDX=$(( REM / 3 ))
SEED_IDX=$(( REM % 3 ))

DATASET=${DATASETS[$DATASET_IDX]}
DT_MODE=${DT_MODES[$DT_IDX]}
SEED=${SEEDS[$SEED_IDX]}

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
echo "UEA final v2 (DAI): dataset=${DATASET}  dt=${DT_MODE}  lr=${LR}  bs=${BS}  seed=${SEED}"
echo "Winner source: ${WINNER_PATH}"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Time: $(date)"
echo "=========================================="

srun python -m imts_benchmark.mamba_mv.train_cls \
    --dataset ${DATASET} \
    --data_root "${DATA_ROOT}" \
    --dt_mode ${DT_MODE} \
    --lr ${LR} \
    --batch_size ${BS} \
    --seed ${SEED} \
    --num_workers 4 \
    --output_dir "${OUT_DIR}" \
    --run_name final

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}  Done at $(date)"
exit ${EXIT_CODE}
