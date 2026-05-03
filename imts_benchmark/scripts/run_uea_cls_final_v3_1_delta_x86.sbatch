#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --job-name=uea_v31_final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --array=1-12
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/uea_cls_v3/final/_v31_slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/uea_cls_v3/final/_v31_slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# v3.1 cleanup final: re-runs the cells the v3 final couldn't deliver:
#   - HB × {learned, replace, concat} × 3 seeds = 9 cells
#     (5 of 6 learned/replace TIMEOUT'd at 2h; concat had no winner JSON entry)
#   - LSST × concat × 3 seeds = 3 cells
#     (concat had no winner JSON entry; learned/replace already ran in v3 final)
#   Total = 12 cells.
#
# Same tight budget as v3.1 HPO (patience=10, max_epochs=200) — within 1h walltime
# even at lr=3e-5. Reads the v3.1-refreshed winner_configs.json.
#
# Index decoding: k = TASK_ID - 1
#   ds_idx = k // 9      (0=Heartbeat 9 cells, 1=LSST 3 cells; LSST tasks have dt fixed=concat)
#   for HB (k in [0, 9)):
#     dt_idx = k // 3    (0=replace, 1=learned, 2=concat)
#     sd_idx = k % 3     (0..2 -> seeds 42, 123, 0)
#   for LSST (k in [9, 12)):
#     dt = concat
#     sd_idx = (k - 9) % 3

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
    echo "ERROR: ${WINNER_PATH} not found. Run v3.1 aggregator first." >&2
    exit 1
fi

DT_MODES_HB=("replace" "learned" "concat")
SEEDS=(42 123 0)

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))

if [[ $K -lt 9 ]]; then
    DATASET=Heartbeat
    DT_IDX=$(( K / 3 ))
    SD_IDX=$(( K % 3 ))
    DT_MODE=${DT_MODES_HB[$DT_IDX]}
else
    DATASET=LSST
    DT_MODE=concat
    SD_IDX=$(( (K - 9) % 3 ))
fi
SEED=${SEEDS[$SD_IDX]}

read -r LR BS DROP < <(python - "${WINNER_PATH}" "${DATASET}" "${DT_MODE}" <<'PY'
import json, sys
w = json.load(open(sys.argv[1]))
cell = w[sys.argv[2]][sys.argv[3]]
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
echo "UEA final v3.1: ds=${DATASET}  dt=${DT_MODE}  lr=${LR}  bs=${BS}  drop=${DROP}  seed=${SEED}  patience=10  max_epochs=200"
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
    --max_epochs 200 \
    --patience 10 \
    --seed ${SEED} \
    --num_workers 4 \
    --output_dir "${OUT_DIR}" \
    --run_name final

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}  Done at $(date)"
exit ${EXIT_CODE}
