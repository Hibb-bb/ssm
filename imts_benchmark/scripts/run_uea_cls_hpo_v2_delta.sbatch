#!/bin/bash
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=uea_cls_hpo_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --array=1-72
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/uea_cls_v2/hpo/slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/uea_cls_v2/hpo/slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# UEA classification HPO sweep, v2 (DeltaAI / GH200, aarch64).
# Scope: 4 datasets (CT, EP, HB, LSST) — BM excluded (already at 1.0 on Quest v1).
# Grid: 4 ds x 2 dt_modes x 3 LRs x 3 BSs x 1 HPO seed = 72 cells.
# Per-dataset LR ranges:
#   CT:   {3e-4, 1e-3, 3e-3}
#   EP:   {3e-4, 1e-3, 3e-3}
#   HB:   {1e-4, 3e-4, 1e-3}    (shifted DOWN; 3e-3 caused majority-class collapse)
#   LSST: {1e-3, 3e-3, 1e-2}    (shifted UP;   RoMAE used 3e-2)
#
# Index decoding: k = TASK_ID - 1
#   dataset_idx = k // 18         (0..3: CT, EP, HB, LSST)
#   rem1        = k %  18
#   dt_idx      = rem1 // 9       (0=replace, 1=learned)
#   rem2        = rem1 %  9
#   lr_idx      = rem2 // 3
#   bs_idx      = rem2 %  3       (0..2 within {8, 16, 32})

set -uo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba
CODE_DIR=/projects/bfrf/seojininus/ssm
DATA_ROOT=${CODE_DIR}/data_uea
LOG_BASE=${CODE_DIR}/output/log/uea_cls_v2/hpo

mkdir -p "${LOG_BASE}"

source /sw/user/python/miniforge3-pytorch-2.5.0/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

DATASETS=("CharacterTrajectories" "Epilepsy" "Heartbeat" "LSST")
DT_MODES=("replace" "learned")
BSS=(8 16 32)

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

read -ra LRS <<< "${LRS_BY_DS[$DATASET_IDX]}"
LR=${LRS[$LR_IDX]}

OUT_DIR=${LOG_BASE}/${DATASET}/${DT_MODE}/lr${LR}_bs${BS}
mkdir -p "${OUT_DIR}"

cd "${CODE_DIR}"

echo "=========================================="
echo "UEA HPO v2 (DAI): dataset=${DATASET}  dt=${DT_MODE}  lr=${LR}  bs=${BS}  seed=42"
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
    --seed 42 \
    --num_workers 4 \
    --output_dir "${OUT_DIR}" \
    --run_name hpo

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}  Done at $(date)"
exit ${EXIT_CODE}
