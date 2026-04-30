#!/usr/bin/env bash
# Stage A under the any-variate-attention architecture (collaborator
# change + bug fixes; see README §7.J).
#
# This is a 1:1 A/B against the original axis4 Stage A
# (`stageA_chronos70_kernel30_d384_huber_50k_s42`):
#
#   * same data mix (70% chronos2_synth + 30% kernelsynth_irregular)
#   * same hparams (d_model=d_hidden=384, 3 perv layers, 3 fusion blocks,
#     bf16-mixed, lr=5e-4, batch_size=32, num_warmup_steps=200)
#   * same val cadence (every 2.5K steps; activity + ushcn + 8 IMM-TSF)
#   * same seed (42)
#
# Differences from the baseline:
#   * VariableAxisAttention: per-slot variate_embed -> Moirai-style
#     same/different binary attention bias (now wired through forward).
#   * SharedGridAligner: variate_embed removed.
#   * QueryReadout: per-variate (gamma, head) -> shared scalars.
#   * VariableAxisAttention: K/V projections back to full d_model
#     (we reverted the colleague's MQA conversion; see README §7.J).
#
# Headline metric: val/mse_z_imm_avg at the best step.
#   Baseline: 1.2722 @ step 15K.
# Tiebreaker: val/mse_activity + val/mse_ushcn at the same step.
#
# Output dir is *separate* from axis4 to avoid clobbering the baseline.
#
# Usage:
#
#   bash imts_benchmark/pretrain/scripts/run_anyvariate_stage_a.sh
#
# Override by env vars (same names as run_axis4_ablation.sh):
#
#   MAX_STEPS=50000
#   BATCH_SIZE=32
#   NUM_WORKERS=12
#   VAL_EVERY=2500
#   WANDB_MODE=online
#   WANDB_PROJECT=TSKing
#   SEED=42
#   PYTHON=/home/ubuntu/envs/mamba/bin/python
#
set -euo pipefail

REPO_ROOT="/home/ubuntu/hongyu/ssm"
PYTHON="${PYTHON:-/home/ubuntu/envs/mamba/bin/python}"
MAX_STEPS="${MAX_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-12}"
VAL_EVERY="${VAL_EVERY:-2500}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-TSKing}"
SEED="${SEED:-42}"

PRETRAIN_DIR="${REPO_ROOT}/imts_benchmark/pretrain"
OUT_ROOT="${PRETRAIN_DIR}/runs/anyvariate"
TPATCHGNN_ROOT="${REPO_ROOT}/tpatchgnn_data"
IMM_TSF_ROOT="${REPO_ROOT}/imts_benchmark/data/imm_tsf_sparse"

ARM_NAME="stage_a_synth_only"
RUN_NAME="stageA_chronos70_kernel30_anyvariate_d384_huber_50k_s${SEED}"
OUT_DIR="${OUT_ROOT}/${ARM_NAME}"
LOG_FILE="${OUT_DIR}/run.log"

mkdir -p "${OUT_DIR}"

echo "================================================================="
echo "  any-variate Stage A (synth-only)"
echo "    stage_cfg : ${PRETRAIN_DIR}/configs/stage_a.yaml"
echo "    out_dir   : ${OUT_DIR}"
echo "    max_steps : ${MAX_STEPS}"
echo "    wandb_run : ${RUN_NAME}"
echo "    log_file  : ${LOG_FILE}"
echo "    git_sha   : $(cd ${REPO_ROOT} && git rev-parse --short HEAD)"
echo "================================================================="

cd "${REPO_ROOT}"
"${PYTHON}" -m imts_benchmark.pretrain.train_pretrain \
    --stage a \
    --stage_cfg "${PRETRAIN_DIR}/configs/stage_a.yaml" \
    --output_dir "${OUT_DIR}" \
    --ablation_axis 5_arch_anyvariate \
    --ablation_level synth_only \
    --wandb_run_name "${RUN_NAME}" \
    --wandb_tags anyvariate stageA synth_only chronos2-70 kernelsynth-30 ab-vs-axis4 \
    --max_steps "${MAX_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --max_dim 20 \
    --seed "${SEED}" \
    --loss huber \
    --huber_delta 1.0 \
    --arch vanilla \
    --d_model 384 \
    --d_hidden 384 \
    --n_perv_layer 3 \
    --n_fusion_blocks 3 \
    --n_heads_varattn 4 \
    --grid_K 128 \
    --lr 5e-4 \
    --weight_decay 0.01 \
    --num_warmup_steps 200 \
    --precision bf16-mixed \
    --gradient_clip_val 1.0 \
    --log_every_n_steps 25 \
    --save_every_n_steps 5000 \
    --val_imts_data_root "${TPATCHGNN_ROOT}" \
    --val_imts_datasets activity ushcn \
    --val_imts_split val \
    --val_imts_subset 1024 \
    --val_imts_batch_size 64 \
    --val_imts_num_workers 2 \
    --val_imm_tsf_data_root "${IMM_TSF_ROOT}" \
    --val_imm_tsf_datasets EPA-Air ILINet GDELT FNSPID CESNET StudentLife RepoHealth ClusterTrace \
    --val_imm_tsf_split test \
    --val_imm_tsf_subset 1024 \
    --val_imm_tsf_batch_size 32 \
    --val_imm_tsf_num_workers 2 \
    --val_check_steps "${VAL_EVERY}" \
    --use_wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_mode "${WANDB_MODE}" \
    2>&1 | tee "${LOG_FILE}"

echo "[anyvariate] Stage A finished; output at ${OUT_DIR}"
