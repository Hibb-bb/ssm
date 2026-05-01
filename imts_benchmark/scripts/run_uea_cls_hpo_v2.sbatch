#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=uea_cls_hpo_v2
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/uea_cls_v2/hpo/slurm_%A_%a.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/uea_cls_v2/hpo/slurm_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-72
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# UEA classification HPO sweep, v2.
# Scope: 4 datasets (CT, EP, HB, LSST) -- BM excluded (already at 1.0).
# Grid: 4 ds x 2 dt_modes x 3 LRs x 3 BSs x 1 HPO seed = 72 cells.
# Per-dataset LR ranges (data-aware, see HB-collapse and LSST-undertuned diagnosis):
#   CT:   {3e-4, 1e-3, 3e-3}    (unchanged; already working at 0.96)
#   EP:   {3e-4, 1e-3, 3e-3}    (unchanged; +head_dropout=0.2 from defaults)
#   HB:   {1e-4, 3e-4, 1e-3}    (shifted DOWN; 3e-3 caused majority-class collapse)
#   LSST: {1e-3, 3e-3, 1e-2}    (shifted UP;   RoMAE used 3e-2; we capped too low)
#
# Index decoding: k = TASK_ID - 1
#   dataset_idx = k // 18         (0..3: CT, EP, HB, LSST)
#   rem1        = k %  18
#   dt_idx      = rem1 // 9       (0=replace, 1=learned)
#   rem2        = rem1 %  9
#   lr_idx      = rem2 // 3       (0..2 within per-dataset list)
#   bs_idx      = rem2 %  3       (0..2 within {8, 16, 32})

set -euo pipefail

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
DATA_ROOT=${CODE_DIR}/data_uea
LOG_BASE=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/uea_cls_v2/hpo
mkdir -p "${LOG_BASE}"

DATASETS=("CharacterTrajectories" "Epilepsy" "Heartbeat" "LSST")
DT_MODES=("replace" "learned")
BSS=(8 16 32)

# Per-dataset LR triples (joined by space inside one string).
LRS_BY_DS=(
    "3e-4 1e-3 3e-3"      # CT
    "3e-4 1e-3 3e-3"      # EP
    "1e-4 3e-4 1e-3"      # HB
    "1e-3 3e-3 1e-2"      # LSST
)

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
DATASET_IDX=$(( K / 18 ))
REM1=$(( K % 18 ))
DT_IDX=$(( REM1 / 9 ))
REM2=$(( REM1 % 9 ))
LR_IDX=$(( REM2 / 3 ))
BS_IDX=$(( REM2 % 3 ))

DATASET=${DATASETS[$DATASET_IDX]}
DT_MODE=${DT_MODES[$DT_IDX]}
BS=${BSS[$BS_IDX]}

# Pull the LR triple for this dataset, split, index.
read -ra LRS <<< "${LRS_BY_DS[$DATASET_IDX]}"
LR=${LRS[$LR_IDX]}

OUT_DIR=${LOG_BASE}/${DATASET}/${DT_MODE}/lr${LR}_bs${BS}
mkdir -p "${OUT_DIR}"

cd ${CODE_DIR}

echo "=== UEA HPO v2: dataset=${DATASET}, dt_mode=${DT_MODE}, lr=${LR}, bs=${BS}, seed=42 ==="

srun ${MAMBA_ENV}/bin/python -m imts_benchmark.mamba_mv.train_cls \
    --dataset ${DATASET} \
    --data_root ${DATA_ROOT} \
    --dt_mode ${DT_MODE} \
    --lr ${LR} \
    --batch_size ${BS} \
    --seed 42 \
    --num_workers 4 \
    --output_dir ${OUT_DIR} \
    --run_name hpo

echo "=== UEA HPO v2: dataset=${DATASET}, dt_mode=${DT_MODE}, lr=${LR}, bs=${BS} done ==="
