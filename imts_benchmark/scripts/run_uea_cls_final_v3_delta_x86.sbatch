#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --job-name=uea_v3_final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --array=1-18
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/uea_cls_v3/final/slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/uea_cls_v3/final/slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# v3 final eval: replays HPO winner per (Heartbeat,LSST) × (replace,learned,concat) × 3 seeds.
# 2 ds × 3 dt × 3 seeds = 18 cells.
# Reads winner JSON: lr, batch_size, head_dropout per (ds, dt).
# Index mapping: k = TASK_ID - 1
#   ds_idx  = k // 9      (0=Heartbeat, 1=LSST)
#   rem     = k %  9
#   dt_idx  = rem // 3    (0=replace, 1=learned, 2=concat)
#   sd_idx  = rem %  3    (0..2 -> seed in {42, 123, 0})

set -uo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
DATA_ROOT=${CODE_DIR}/data_uea
LOG_BASE=${CODE_DIR}/output/log/uea_cls_v3/final
WINNER_PATH=${CODE_DIR}/output/log/uea_cls_v3/winner_configs.json

mkdir -p "${LOG_BASE}"

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

if [[ ! -f "${WINNER_PATH}" ]]; then
    echo "ERROR: ${WINNER_PATH} not found. Run aggregator first." >&2
    exit 1
fi

DATASETS=("Heartbeat" "LSST")
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

# Read winner triple (lr, batch_size, head_dropout) for this (dataset, dt_mode).
read -r LR BS DROP < <(python - "${WINNER_PATH}" "${DATASET}" "${DT_MODE}" <<'PY'
import json, sys
w = json.load(open(sys.argv[1]))
cell = w[sys.argv[2]][sys.argv[3]]
# head_dropout may be missing for v2 winners; default to -1 (uses per-dataset default).
print(cell["lr"], cell["batch_size"], cell.get("head_dropout", -1.0))
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
echo "UEA final v3: ds=${DATASET}  dt=${DT_MODE}  lr=${LR}  bs=${BS}  head_dropout=${DROP}  seed=${SEED}"
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
    --head_dropout ${DROP} \
    --seed ${SEED} \
    --num_workers 4 \
    --output_dir "${OUT_DIR}" \
    --run_name final

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}  Done at $(date)"
exit ${EXIT_CODE}
