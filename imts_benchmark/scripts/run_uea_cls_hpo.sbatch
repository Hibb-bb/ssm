#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=uea_cls_hpo
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/uea_cls/hpo/slurm_%A_%a.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/uea_cls/hpo/slurm_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-30
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# UEA classification HPO sweep.
# Grid: 5 datasets x 3 LRs x 2 dt_modes x 1 seed = 30 cells.
# Index mapping: k = TASK_ID - 1
#   dataset_idx = k // 6      (0..4: BM, CT, EP, HB, LSST)
#   remainder   = k %  6
#   dt_mode_idx = remainder // 3   (0 = replace, 1 = learned)
#   lr_idx      = remainder %  3   (0..2)

set -euo pipefail

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
DATA_ROOT=${CODE_DIR}/data_uea
LOG_BASE=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/uea_cls/hpo
mkdir -p "${LOG_BASE}"

DATASETS=("BasicMotions" "CharacterTrajectories" "Epilepsy" "Heartbeat" "LSST")
DT_MODES=("replace" "learned")
LRS=("3e-4" "1e-3" "3e-3")

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
DATASET_IDX=$(( K / 6 ))
REM=$(( K % 6 ))
DT_IDX=$(( REM / 3 ))
LR_IDX=$(( REM % 3 ))

DATASET=${DATASETS[$DATASET_IDX]}
DT_MODE=${DT_MODES[$DT_IDX]}
LR=${LRS[$LR_IDX]}

OUT_DIR=${LOG_BASE}/${DATASET}/${DT_MODE}/lr${LR}
mkdir -p "${OUT_DIR}"

cd ${CODE_DIR}

echo "=== UEA HPO: dataset=${DATASET}, dt_mode=${DT_MODE}, lr=${LR}, seed=42 (HPO seed) ==="

srun ${MAMBA_ENV}/bin/python -m imts_benchmark.mamba_mv.train_cls \
    --dataset ${DATASET} \
    --data_root ${DATA_ROOT} \
    --dt_mode ${DT_MODE} \
    --lr ${LR} \
    --seed 42 \
    --num_workers 4 \
    --output_dir ${OUT_DIR} \
    --run_name hpo

echo "=== UEA HPO: dataset=${DATASET}, dt_mode=${DT_MODE}, lr=${LR} done ==="
