#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=uea_cls_smoke
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/uea_cls/smoke/slurm_%A_%a.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/uea_cls/smoke/slurm_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-2
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# UEA classification smoke runs — gates the rest of the pipeline.
# Task 1: BasicMotions (V=6, L=100, N_train=40) — pipeline & head sanity check.
# Task 2: Heartbeat (V=61, L=405, N_train=204) — memory check at high V.
# Both run dt_mode=replace, single seed, smoke=on (max 30 epochs, patience 10).

set -euo pipefail

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
DATA_ROOT=${CODE_DIR}/data_uea
LOG_BASE=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/uea_cls/smoke
mkdir -p "${LOG_BASE}"

DATASETS=("BasicMotions" "Heartbeat")
TASK_ID=$SLURM_ARRAY_TASK_ID
DATASET=${DATASETS[$(( TASK_ID - 1 ))]}

OUT_DIR=${LOG_BASE}/${DATASET}
mkdir -p "${OUT_DIR}"

cd ${CODE_DIR}

echo "=== UEA cls smoke: dataset=${DATASET}, task=${TASK_ID} ==="
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUT_DIR=${OUT_DIR}"
echo "PYTHON=${MAMBA_ENV}/bin/python"

# Smoke: dt_mode=replace, lr=1e-3, dropout=0.1, seed=42, max 30 epochs.
# Direct interpreter invocation (no conda activate) — matches the existing
# Mamba-MV phase3 SLURM pattern.
srun ${MAMBA_ENV}/bin/python -m imts_benchmark.mamba_mv.train_cls \
    --dataset ${DATASET} \
    --data_root ${DATA_ROOT} \
    --dt_mode replace \
    --lr 1e-3 \
    --seed 42 \
    --num_workers 4 \
    --output_dir ${OUT_DIR} \
    --run_name smoke \
    --smoke

echo "=== UEA cls smoke: dataset=${DATASET} done ==="
