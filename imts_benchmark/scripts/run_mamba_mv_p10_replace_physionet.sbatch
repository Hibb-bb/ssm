#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=mv_rpl_p10_phy
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/mamba_mv_p10/slurm_%A_%a.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/mamba_mv_p10/slurm_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=1-5
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Mamba-MV dt_mode=replace, PhysioNet only, 5 seeds, patience=10.
# HPO winner from server A (full 9-cell sweep finished):
#   test_mse winner:  lr=2e-3, ebs=64  (0.003112)
#   mse_tpg  winner:  lr=2e-3, ebs=128 (0.004554)  <- using this (TPG is paper headline)
#
# OOM mitigations on Quest H100 80GB:
#   * bs=8 + accumulate_grad_batches=16 (eff bs=128; matches HPO winner)
#   * PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   * grid_K=256 (per-step memory at bs=8 is 16x smaller than the bs=128 OOM)

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -z "${WANDB_API_KEY:-}" ]] && [[ -f ~/.bashrc ]]; then
    source ~/.bashrc 2>/dev/null || true
fi
if [[ -z "${WANDB_API_KEY:-}" ]] && [[ -f ~/.netrc ]]; then
    export WANDB_API_KEY=$(awk '/api.wandb.ai/{f=1} f && /password/{print $2; exit}' ~/.netrc)
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set." >&2
    exit 1
fi
export WANDB_MODE=online

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
DATA_ROOT=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/tpatchgnn_data
LOG_BASE=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real

DS=physionet
DT_MODE=replace
LR=2e-3
BS=8
ACCUM=16
GRID_K=256
PATIENCE=10
SEED=$SLURM_ARRAY_TASK_ID

OUTPUT_DIR=${LOG_BASE}/mamba_mv_p10/${DS}/${DT_MODE}_lr-${LR}_bs-${BS}x${ACCUM}/seed${SEED}
mkdir -p $OUTPUT_DIR
export WANDB_DIR=${OUTPUT_DIR}

echo "=========================================="
echo "Mamba-MV p10  ds=$DS  dt=$DT_MODE  lr=$LR  bs=${BS}x${ACCUM}=$((BS*ACCUM))  grid_K=$GRID_K  seed=$SEED  patience=$PATIENCE"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Time: $(date)"

cd $CODE_DIR
START=$(date +%s)

srun ${MAMBA_ENV}/bin/python -m imts_benchmark.mamba_mv.train_mv \
    --data_root $DATA_ROOT \
    --regime $DS \
    --auto_meta \
    --dt_mode $DT_MODE \
    --grid_K $GRID_K \
    --lr $LR \
    --train_batch_size $BS \
    --accumulate_grad_batches $ACCUM \
    --patience $PATIENCE \
    --seed $SEED \
    --output_dir $OUTPUT_DIR \
    --use_wandb \
    --wandb_project TSKing \
    --wandb_run_name "p10_mv_${DS}_${DT_MODE}_lr-${LR}_bs-${BS}x${ACCUM}_seed${SEED}"

EXIT_CODE=$?
END=$(date +%s)
echo "Exit code: $EXIT_CODE  Wall time: $((END-START))s  Done at $(date)"
