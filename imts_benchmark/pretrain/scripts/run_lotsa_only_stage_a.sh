#!/usr/bin/env bash
# Stage A — LOTSA-only (50% regular + 50% degraded), no synthetic.
#
# Question: do we actually need the synthetic warm-start at all?
#
# Identical to the anyvariate Stage A baseline EXCEPT:
#   * mix is 50% lotsa_regular + 50% lotsa_degraded (was 70% chronos + 30% kernel)
#   * early stopping enabled on val/mse_z_imm_avg, patience = 6 val cycles
#
# Architecture: any-variate (matches the anyvariate Stage A run).
# All other hparams (d_model=384, lr=5e-4, batch=32, val cadence,
# 50K steps, seed=42) are unchanged so the only varying axis is data.
#
# Usage:
#
#   bash imts_benchmark/pretrain/scripts/run_lotsa_only_stage_a.sh
#
# Override by env vars:
#   MAX_STEPS=50000  BATCH_SIZE=32  NUM_WORKERS=12
#   VAL_EVERY=2500   WANDB_MODE=online  WANDB_PROJECT=TSKing
#   SEED=42          PYTHON=/home/ubuntu/envs/mamba/bin/python
#   ES_PATIENCE=6    ES_METRIC=val/mse_z_imm_avg
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
ES_METRIC="${ES_METRIC:-val/mse_z_imm_avg}"
ES_PATIENCE="${ES_PATIENCE:-6}"

PRETRAIN_DIR="${REPO_ROOT}/imts_benchmark/pretrain"
OUT_ROOT="${PRETRAIN_DIR}/runs/lotsa_only"
TPATCHGNN_ROOT="${REPO_ROOT}/tpatchgnn_data"
IMM_TSF_ROOT="${REPO_ROOT}/imts_benchmark/data/imm_tsf_sparse"

ARM_NAME="stage_a_lotsa50reg_lotsa50deg"
RUN_NAME="stageA_lotsaReg50_lotsaDeg50_anyvariate_d384_huber_50k_es${ES_PATIENCE}_s${SEED}"
OUT_DIR="${OUT_ROOT}/${ARM_NAME}"
LOG_FILE="${OUT_DIR}/run.log"

mkdir -p "${OUT_DIR}"

echo "================================================================="
echo "  Stage A — LOTSA-only (50% reg + 50% degraded)"
echo "    stage_cfg : ${PRETRAIN_DIR}/configs/stage_a_lotsa_only.yaml"
echo "    out_dir   : ${OUT_DIR}"
echo "    max_steps : ${MAX_STEPS}"
echo "    early stop: ${ES_METRIC} (patience=${ES_PATIENCE} val cycles)"
echo "    wandb_run : ${RUN_NAME}"
echo "    log_file  : ${LOG_FILE}"
echo "    git_sha   : $(cd ${REPO_ROOT} && git rev-parse --short HEAD)"
echo "================================================================="

cd "${REPO_ROOT}"
"${PYTHON}" -m imts_benchmark.pretrain.train_pretrain \
    --stage a \
    --stage_cfg "${PRETRAIN_DIR}/configs/stage_a_lotsa_only.yaml" \
    --output_dir "${OUT_DIR}" \
    --ablation_axis 4_stage_a_mix \
    --ablation_level lotsa_only \
    --wandb_run_name "${RUN_NAME}" \
    --wandb_tags axis4 stageA lotsa_only lotsa_regular-50 lotsa_degraded-50 anyvariate \
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
    --early_stop_metric "${ES_METRIC}" \
    --early_stop_patience "${ES_PATIENCE}" \
    --early_stop_mode min \
    --use_wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_mode "${WANDB_MODE}" \
    2>&1 | tee "${LOG_FILE}"

echo "[lotsa_only] Stage A finished; output at ${OUT_DIR}"
