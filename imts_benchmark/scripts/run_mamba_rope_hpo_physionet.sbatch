#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=mrope_hpo_phy
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/hpo_mamba_rope/physionet/slurm_%A_%a.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/hpo_mamba_rope/physionet/slurm_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=1-9
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# HPO sweep: Mamba-RoPE on PhysioNet, dt_mode=learned, seed=1, patience=10.
# Grid: lr in {1e-4, 5e-4, 2e-3} x eff_bs in {64, 128, 256} = 9 cells.
# Achieved via bs=8 x accum in {8, 16, 32} for memory safety on V=41 / L<=128.
# Mamba-RoPE attention is O((V*L)^2) -> larger than Mamba-MV's O(K^2);
# bs=8 + expandable_segments avoids fragmentation OOM.

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
LOG_BASE=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/hpo_mamba_rope/physionet

LRS=("1e-4" "5e-4" "2e-3")
ACCUMS=(8 16 32)              # eff_bs = 8 * accum = {64, 128, 256}
EFF_BSS=(64 128 256)
DT_MODE=learned
DS=physionet
SEED=1
BS=8

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
LR_IDX=$(( K / 3 ))
ACC_IDX=$(( K % 3 ))

LR=${LRS[$LR_IDX]}
ACCUM=${ACCUMS[$ACC_IDX]}
EFF_BS=${EFF_BSS[$ACC_IDX]}

TAG=lr-${LR}_ebs-${EFF_BS}
OUTPUT_DIR=${LOG_BASE}/${TAG}/seed${SEED}
mkdir -p $OUTPUT_DIR
export WANDB_DIR=${OUTPUT_DIR}

echo "=========================================="
echo "Mamba-RoPE HPO  ds=$DS  dt=$DT_MODE  lr=$LR  bs=${BS}x${ACCUM}=${EFF_BS}  seed=$SEED"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Time: $(date)"

cd $CODE_DIR
START=$(date +%s)

srun ${MAMBA_ENV}/bin/python -m imts_benchmark.mamba_rope.train_mamba_rope \
    --data_root $DATA_ROOT \
    --regime $DS \
    --auto_meta \
    --dt_mode $DT_MODE \
    --lr $LR \
    --train_batch_size $BS \
    --accumulate_grad_batches $ACCUM \
    --patience 10 \
    --seed $SEED \
    --output_dir $OUTPUT_DIR \
    --use_wandb \
    --wandb_project TSKing \
    --wandb_run_name "hpo_mrope_${DS}_${DT_MODE}_${TAG}_seed${SEED}"

EXIT_CODE=$?
END=$(date +%s)
echo "Exit code: $EXIT_CODE  Wall time: $((END-START))s ($(( (END-START)/60 ))m)  Done at $(date)"
