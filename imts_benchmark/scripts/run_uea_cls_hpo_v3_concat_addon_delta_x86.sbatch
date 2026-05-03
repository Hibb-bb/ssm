#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --job-name=uea_v3_addon_hpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --array=1-27
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/uea_cls_v3_addon/hpo/slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/uea_cls_v3_addon/hpo/slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# v3 add-on HPO: concat dt_mode for the picker-trustworthy datasets (BM, CT, EP)
# under EXACTLY the v2 HPO grid (no extra knobs). Lets us complete the
# 3-dt_mode column for all 5 datasets in the head-to-head table.
#
# Scope: 3 ds (BasicMotions, CharacterTrajectories, Epilepsy)
#        × 1 dt (concat)
#        × 3 LR ({3e-4, 1e-3, 3e-3})  — same LR set v2 used for CT/EP
#        × 3 BS ({8, 16, 32})           — same BS set v2 used
#        = 27 cells, single seed (HPO seed=42).
#
# head_dropout, label_smoothing, grad_clip, grid_K all kept at per-dataset
# defaults from RoMAE Table 12 (NOT swept) — matching the v2 protocol exactly.
# This keeps "concat add-on" a clean head-to-head with the existing v2
# learned/replace winners; the only changed axis is dt_mode.
#
# Index decoding: k = TASK_ID - 1
#   ds_idx = k // 9     (0=BasicMotions, 1=CharacterTrajectories, 2=Epilepsy)
#   rem    = k %  9
#   lr_idx = rem // 3
#   bs_idx = rem %  3

set -uo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
DATA_ROOT=${CODE_DIR}/data_uea
LOG_BASE=${CODE_DIR}/output/log/uea_cls_v3_addon/hpo

mkdir -p "${LOG_BASE}"

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

DATASETS=("BasicMotions" "CharacterTrajectories" "Epilepsy")
LRS=(3e-4 1e-3 3e-3)
BSS=(8 16 32)
DT_MODE=concat

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
DS_IDX=$(( K / 9 ))
REM=$(( K % 9 ))
LR_IDX=$(( REM / 3 ))
BS_IDX=$(( REM % 3 ))

DATASET=${DATASETS[$DS_IDX]}
LR=${LRS[$LR_IDX]}
BS=${BSS[$BS_IDX]}

OUT_DIR=${LOG_BASE}/${DATASET}/${DT_MODE}/lr${LR}_bs${BS}
mkdir -p "${OUT_DIR}"

cd "${CODE_DIR}"

echo "=========================================="
echo "UEA HPO v3-addon: ds=${DATASET}  dt=${DT_MODE}  lr=${LR}  bs=${BS}  seed=42"
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
