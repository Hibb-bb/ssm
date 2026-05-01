#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=uea_cls_final_v2
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/uea_cls_v2/final/slurm_%A_%a.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/uea_cls_v2/final/slurm_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-24
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# UEA classification final eval at HPO-picked (LR, batch_size) per (dataset, dt_mode).
# Scope: 4 datasets (CT, EP, HB, LSST) x 2 dt_modes x 3 seeds = 24 cells.
# Index mapping: k = TASK_ID - 1
#   dataset_idx = k // 6      (0..3: CT, EP, HB, LSST)
#   remainder   = k %  6
#   dt_mode_idx = remainder // 3   (0=replace, 1=learned)
#   seed_idx    = remainder %  3   (0..2 -> seed in {42, 123, 0})

set -euo pipefail

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
DATA_ROOT=${CODE_DIR}/data_uea
LOG_BASE=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/uea_cls_v2/final
WINNER_PATH=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/uea_cls_v2/winner_configs.json
mkdir -p "${LOG_BASE}"

if [[ ! -f "${WINNER_PATH}" ]]; then
    echo "ERROR: ${WINNER_PATH} not found. Run run_aggregate_uea_hpo_v2.sbatch first." >&2
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

# Read picked LR + batch_size from winner_configs.json via Python one-liner.
# Falls back to per-dataset defaults if a key is missing (shouldn't happen
# under v2 aggregator output).
read -r LR BS <<< $(${MAMBA_ENV}/bin/python -c "
import json
w = json.load(open('${WINNER_PATH}'))
cell = w['${DATASET}']['${DT_MODE}']
print(cell['lr'], cell['batch_size'])
")

OUT_DIR=${LOG_BASE}/${DATASET}/${DT_MODE}/seed${SEED}
mkdir -p "${OUT_DIR}"

cd ${CODE_DIR}

echo "=== UEA final v2: dataset=${DATASET}, dt_mode=${DT_MODE}, lr=${LR}, bs=${BS}, seed=${SEED} ==="

srun ${MAMBA_ENV}/bin/python -m imts_benchmark.mamba_mv.train_cls \
    --dataset ${DATASET} \
    --data_root ${DATA_ROOT} \
    --dt_mode ${DT_MODE} \
    --lr ${LR} \
    --batch_size ${BS} \
    --seed ${SEED} \
    --num_workers 4 \
    --output_dir ${OUT_DIR} \
    --run_name final

echo "=== UEA final v2: dataset=${DATASET}, dt_mode=${DT_MODE}, seed=${SEED} done ==="
