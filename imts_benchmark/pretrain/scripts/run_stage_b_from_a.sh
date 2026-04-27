#!/usr/bin/env bash
# Stage B starter — initialize weights from a Stage A checkpoint and start
# a fresh run with the Stage B data mix.
#
# This is the canonical "stage A -> stage B" handoff pattern (see
# pretrain/README.md §7.H):
#
#   - --init_from   loads ONLY the model state_dict from the source ckpt;
#                   optimizer / LR scheduler / step counter / RNG / data
#                   sampler cursor all start fresh.
#   - --ckpt        is reserved for crash recovery within the same stage.
#
# Why weights-only and not full Lightning resume:
#   - Stage B has a different source mix; Adam moments tuned to Stage A
#     synthetic-only data destabilize the first 100s of steps under the
#     Stage B (synthetic + degraded LOTSA + regular LOTSA) distribution.
#   - LR schedule should restart from warmup at step 0 with Stage B's
#     own step budget.
#   - W&B should track Stage B as its own run, not continue Stage A.
#
# Usage::
#
#   INIT_FROM=/path/to/stage_a/best-step00050000-mse0.7321.ckpt \
#       bash imts_benchmark/pretrain/scripts/run_stage_b_from_a.sh
#
# Override by env vars:
#
#   INIT_FROM=...                  required: source ckpt path
#   STAGE_CFG=path/to/stage_b.yaml (default: configs/stage_b.yaml)
#   MAX_STEPS=200000               (default: 200000)
#   BATCH_SIZE=32 NUM_WORKERS=12 VAL_EVERY=2500 SEED=42
#   WANDB_MODE=online WANDB_PROJECT=TSKing
#   PYTHON=/home/ubuntu/envs/mamba/bin/python

set -euo pipefail

if [[ -z "${INIT_FROM:-}" ]]; then
    echo "ERROR: INIT_FROM=... is required (path to a stage-A .ckpt)"
    echo ""
    echo "Pick the best stage-A ckpt:"
    echo "  ls /home/ubuntu/hongyu/ssm/imts_benchmark/pretrain/runs/axis4/axis4_synth_only/best-*.ckpt"
    echo ""
    echo "Then re-run as:"
    echo "  INIT_FROM=<...>.ckpt bash $0"
    exit 2
fi

REPO_ROOT="/home/ubuntu/hongyu/ssm"
PYTHON="${PYTHON:-/home/ubuntu/envs/mamba/bin/python}"
PRETRAIN_DIR="${REPO_ROOT}/imts_benchmark/pretrain"
STAGE_CFG="${STAGE_CFG:-${PRETRAIN_DIR}/configs/stage_b.yaml}"
OUT_ROOT="${PRETRAIN_DIR}/runs/stage_b"
TPATCHGNN_ROOT="${REPO_ROOT}/tpatchgnn_data"
IMM_TSF_ROOT="${REPO_ROOT}/imts_benchmark/data/imm_tsf_sparse"

MAX_STEPS="${MAX_STEPS:-200000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-12}"
VAL_EVERY="${VAL_EVERY:-2500}"
SEED="${SEED:-42}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-TSKing}"

# Encode the source-ckpt step in the run name so the W&B dashboard
# trivially reads "stage B started from step 50K of stage A".
SRC_STEP=$("${PYTHON}" - <<EOF
import torch
b = torch.load("${INIT_FROM}", map_location="cpu", weights_only=False)
print(b.get("global_step", "?"))
EOF
)
# Default run name encodes the Stage B mix percentages (35% chronos2 +
# 20% kernelsynth + 30% degraded LOTSA + 15% regular LOTSA) and the
# source step from Stage A.  Override with RUN_NAME=... if needed.
SRC_STEP_K=$(( ${SRC_STEP} / 1000 ))
MAX_K=$(( ${MAX_STEPS} / 1000 ))
RUN_NAME="${RUN_NAME:-stageB_chronos35_kernel20_lotsaDeg30_lotsaReg15_d384_huber_${MAX_K}k_fromA-${SRC_STEP_K}K_s${SEED}}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/${RUN_NAME}}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/run.log"

echo "================================================================="
echo "  Stage B (init from Stage A weights)"
echo "    init_from   : ${INIT_FROM}"
echo "    src step    : ${SRC_STEP}"
echo "    stage_cfg   : ${STAGE_CFG}"
echo "    out_dir     : ${OUT_DIR}"
echo "    max_steps   : ${MAX_STEPS}"
echo "    wandb_proj  : ${WANDB_PROJECT}"
echo "    wandb_run   : ${RUN_NAME}"
echo "================================================================="

cd "${REPO_ROOT}"
"${PYTHON}" -m imts_benchmark.pretrain.train_pretrain \
    --stage b \
    --stage_cfg "${STAGE_CFG}" \
    --output_dir "${OUT_DIR}" \
    --init_from "${INIT_FROM}" \
    --ablation_axis stage_transition \
    --ablation_level "stage_b_from_a-step${SRC_STEP}" \
    --wandb_run_name "${RUN_NAME}" \
    --wandb_tags stageB stage_transition "init_from-step${SRC_STEP}" \
        chronos2-35 kernelsynth-20 lotsa_degraded-30 lotsa_regular-15 \
    --max_steps "${MAX_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --max_dim 20 \
    --seed "${SEED}" \
    --loss huber \
    --huber_delta 1.0 \
    --arch vanilla \
    --d_model 384 --d_hidden 384 \
    --n_perv_layer 3 --n_fusion_blocks 3 --n_heads_varattn 4 \
    --grid_K 128 \
    --lr 5e-4 --weight_decay 0.01 --num_warmup_steps 200 \
    --precision bf16-mixed --gradient_clip_val 1.0 \
    --log_every_n_steps 25 --save_every_n_steps 5000 \
    --val_imts_data_root "${TPATCHGNN_ROOT}" \
    --val_imts_datasets activity ushcn \
    --val_imts_split val --val_imts_subset 1024 \
    --val_imts_batch_size 64 --val_imts_num_workers 2 \
    --val_imm_tsf_data_root "${IMM_TSF_ROOT}" \
    --val_imm_tsf_datasets EPA-Air ILINet GDELT FNSPID CESNET StudentLife RepoHealth ClusterTrace \
    --val_imm_tsf_split test --val_imm_tsf_subset 1024 \
    --val_imm_tsf_batch_size 32 --val_imm_tsf_num_workers 2 \
    --val_check_steps "${VAL_EVERY}" \
    --use_wandb --wandb_project "${WANDB_PROJECT}" --wandb_mode "${WANDB_MODE}" \
    2>&1 | tee "${LOG_FILE}"

echo "[stage_b] finished cleanly; outputs in ${OUT_DIR}"
