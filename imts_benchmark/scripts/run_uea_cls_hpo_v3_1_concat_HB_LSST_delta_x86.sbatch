#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --job-name=uea_v31_concat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --array=1-43
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/uea_cls_v3/hpo/_v31_slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/uea_cls_v3/hpo/_v31_slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# v3.1 cleanup HPO: rerun the 16 HB + 27 LSST concat cells that v3 dropped because
# train_cls.py CLI rejected --dt_mode concat. CLI is now patched.
# Tighter budget per cell (patience=10, max_epochs=200) so HB at low LRs fits in
# walltime AND so the HPO picker rewards configs that converge within the deployment
# budget. Applied uniformly across all 43 cells for protocol consistency with the
# upcoming v3.1 final array.
#
# Writes into the EXISTING v3 hpo tree under .../{Heartbeat,LSST}/concat/<cell>/hpo/
# so the v3.1 aggregator can pick winners across the union of v3 (learned/replace)
# + v3.1 (concat) by simply rescanning the v3 hpo root.
#
# Index decoding: k = TASK_ID - 1
#   ds=Heartbeat for k in [0, 16):
#     k_h     = k
#     lr_idx  = k_h // 4   (4 LR ∈ {3e-5, 1e-4, 3e-4, 1e-3})
#     rem     = k_h %  4
#     bs_idx  = rem // 2   (2 BS ∈ {16, 32})
#     drop_idx= rem %  2   (2 drop ∈ {0.0, 0.1})
#   ds=LSST for k in [16, 43):
#     k_l     = k - 16
#     lr_idx  = k_l // 9   (3 LR ∈ {1e-4, 3e-4, 1e-3})
#     rem     = k_l %  9
#     bs_idx  = rem // 3   (3 BS ∈ {16, 32, 64})
#     drop_idx= rem %  3   (3 drop ∈ {0.0, 0.1, 0.2})

set -uo pipefail

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
DATA_ROOT=${CODE_DIR}/data_uea
LOG_BASE=${CODE_DIR}/output/log/uea_cls_v3/hpo

mkdir -p "${LOG_BASE}/Heartbeat" "${LOG_BASE}/LSST"

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

DT_MODE=concat
TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))

if [[ $K -lt 16 ]]; then
    DATASET=Heartbeat
    LRS=(3e-5 1e-4 3e-4 1e-3)
    BSS=(16 32)
    DROPOUTS=(0.0 0.1)
    K_H=$K
    LR_IDX=$(( K_H / 4 ))
    REM=$(( K_H % 4 ))
    BS_IDX=$(( REM / 2 ))
    DROP_IDX=$(( REM % 2 ))
else
    DATASET=LSST
    LRS=(1e-4 3e-4 1e-3)
    BSS=(16 32 64)
    DROPOUTS=(0.0 0.1 0.2)
    K_L=$(( K - 16 ))
    LR_IDX=$(( K_L / 9 ))
    REM=$(( K_L % 9 ))
    BS_IDX=$(( REM / 3 ))
    DROP_IDX=$(( REM % 3 ))
fi

LR=${LRS[$LR_IDX]}
BS=${BSS[$BS_IDX]}
DROP=${DROPOUTS[$DROP_IDX]}

OUT_DIR=${LOG_BASE}/${DATASET}/${DT_MODE}/lr${LR}_bs${BS}_drop${DROP}
mkdir -p "${OUT_DIR}"

cd "${CODE_DIR}"

echo "=========================================="
echo "UEA HPO v3.1 (concat rerun): ds=${DATASET}  dt=${DT_MODE}  lr=${LR}  bs=${BS}  drop=${DROP}  patience=10  max_epochs=200"
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
    --seed 42 \
    --num_workers 4 \
    --output_dir "${OUT_DIR}" \
    --run_name hpo

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}  Done at $(date)"
exit ${EXIT_CODE}
