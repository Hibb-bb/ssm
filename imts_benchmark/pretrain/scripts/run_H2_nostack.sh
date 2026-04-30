#!/usr/bin/env bash
# H2: random-stacking off, mix back to 30/70.
#
# Three runs (30/70, 50/50, 10/90) all hit ~0.965 mse_z_imm_avg ceiling.
# H2 tests whether StackedLOTSASource's random-stacking is teaching the
# model a "treat variates as independent" wrong prior, blocking
# IMM-TSF transfer.  Same arch, same LR schedule as moirai_v2, only
# difference is sources.yaml::lotsa_stacking_datasets='none'.

set -euo pipefail

REPO_ROOT="/home/ubuntu/hongyu/ssm"
PYTHON="${PYTHON:-/home/ubuntu/envs/mamba/bin/python}"
MAX_STEPS="${MAX_STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-12}"
VAL_EVERY="${VAL_EVERY:-2000}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-TSKing}"
SEED="${SEED:-42}"
ES_METRIC="${ES_METRIC:-val/mse_z_imm_avg}"
ES_PATIENCE="${ES_PATIENCE:-10}"
SYNTH_VAL_N="${SYNTH_VAL_N:-512}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-500}"

PRETRAIN_DIR="${REPO_ROOT}/imts_benchmark/pretrain"
OUT_ROOT="${PRETRAIN_DIR}/runs/single_phase"
TPATCHGNN_ROOT="${REPO_ROOT}/tpatchgnn_data"
IMM_TSF_ROOT="${REPO_ROOT}/imts_benchmark/data/imm_tsf_sparse"

GIT_SHA=$(cd "${REPO_ROOT}" && git rev-parse --short HEAD)

RUN_NAME="H2_nostack_synth30_lotsa70_d384_cosine${MAX_STEPS}_warmup${NUM_WARMUP_STEPS}_minstd1e-3_s${SEED}"
OUT_DIR="${OUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

echo "================================================================="
echo "  H2 — stacking OFF, mix back to 30/70"
echo "    cfg          : stage_a_single_phase.yaml"
echo "    sources cfg  : sources_nostack.yaml (lotsa_stacking_datasets=none)"
echo "    mix          : 20% chronos2 + 10% kernel + 70% lotsa_degraded"
echo "    LR sched     : cosine — warmup ${NUM_WARMUP_STEPS}, peak 5e-4 → 0 over ${MAX_STEPS}"
echo "    max_steps    : ${MAX_STEPS}"
echo "    out_dir      : ${OUT_DIR}"
echo "    git_sha      : ${GIT_SHA}"
echo "================================================================="

cd "${REPO_ROOT}"
"${PYTHON}" -m imts_benchmark.pretrain.train_pretrain \
    --stage a \
    --stage_cfg "${PRETRAIN_DIR}/configs/stage_a_single_phase.yaml" \
    --sources_cfg "${PRETRAIN_DIR}/configs/sources_nostack.yaml" \
    --output_dir "${OUT_DIR}" \
    --ablation_axis 14_H2_nostack \
    --ablation_level main \
    --wandb_run_name "${RUN_NAME}" \
    --wandb_tags single_phase moirai_v2 H2 nostack synth30 lotsa70 anyvariate asinh cosine-lr minstd1e-3 metric-fix \
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
    --lr_schedule cosine \
    --weight_decay 0.01 \
    --num_warmup_steps "${NUM_WARMUP_STEPS}" \
    --precision bf16-mixed \
    --gradient_clip_val 1.0 \
    --log_every_n_steps 25 \
    --save_every_n_steps "${SAVE_EVERY}" \
    --val_imts_data_root "${TPATCHGNN_ROOT}" \
    --val_imts_datasets activity ushcn \
    --val_imts_split val \
    --val_imts_subset -1 \
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
    --synth_val_n_windows "${SYNTH_VAL_N}" \
    --use_wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_mode "${WANDB_MODE}" \
    2>&1 | tee "${OUT_DIR}/run.log"

echo "[H2] finished; outputs at ${OUT_DIR}"
