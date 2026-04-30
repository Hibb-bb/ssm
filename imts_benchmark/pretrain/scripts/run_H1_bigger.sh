#!/usr/bin/env bash
# H1: architectural capacity bump.
#
# Three single-phase runs at d=384 (synth30/lotsa70, synth50/lotsa50,
# synth10/lotsa90) all hit the same ~0.965 mse_z_imm_avg ceiling at
# step 16K.  The fact that data mix doesn't matter — and that loss
# stops improving while train/huber is still ~0.3 — points to model
# capacity as the binding constraint.
#
# H1 bumps:
#   d_model       384 → 512   (residual stream width)
#   d_hidden      384 → 512   (Mamba state width)
#   n_perv_layer    3 → 4     (per-variate SSM depth)
#   n_fusion_blocks 3 → 4     (cross-variate attention depth)
#   n_heads_varattn 4 → 8     (any-variate attention heads)
#   grid_K         128 → 128  (unchanged — readout grid)
#
# Param count: ~37M → ~75M (rough 2× scaling).
# Throughput: ~4.5 it/s → ~2.2 it/s on a single A100.
# Wall-clock @ 30K steps: ~3.8 h.
#
# Two important caveats before launching:
#   1. BATCH_SIZE may need to drop from 32 → 24 or 16 if VRAM is tight.
#      Watch nvidia-smi after step 10.
#   2. LR may want to scale.  Default keeps 5e-4 but a 0.5× LR (2.5e-4)
#      is safer for the bigger model.  Set LR via env var.
#
# This script intentionally inherits the SAME data config used by H2
# (no stacking, mix 30/70).  If H2 succeeds, H1 = H2 + bigger arch.
# If H2 fails, swap --sources_cfg back to sources.yaml here.

set -euo pipefail

REPO_ROOT="/home/ubuntu/hongyu/ssm"
PYTHON="${PYTHON:-/home/ubuntu/envs/mamba/bin/python}"
MAX_STEPS="${MAX_STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-24}"
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
LR="${LR:-5e-4}"

# H1 architecture.
D_MODEL="${D_MODEL:-512}"
D_HIDDEN="${D_HIDDEN:-512}"
N_PERV_LAYER="${N_PERV_LAYER:-4}"
N_FUSION_BLOCKS="${N_FUSION_BLOCKS:-4}"
N_HEADS_VARATTN="${N_HEADS_VARATTN:-8}"
GRID_K="${GRID_K:-128}"

# Data side: inherit H2's no-stacking config + 30/70 mix.
SOURCES_CFG="${SOURCES_CFG:-sources_nostack.yaml}"

PRETRAIN_DIR="${REPO_ROOT}/imts_benchmark/pretrain"
OUT_ROOT="${PRETRAIN_DIR}/runs/single_phase"
TPATCHGNN_ROOT="${REPO_ROOT}/tpatchgnn_data"
IMM_TSF_ROOT="${REPO_ROOT}/imts_benchmark/data/imm_tsf_sparse"

GIT_SHA=$(cd "${REPO_ROOT}" && git rev-parse --short HEAD)

RUN_NAME="H1_d${D_MODEL}_h${D_HIDDEN}_perv${N_PERV_LAYER}_fus${N_FUSION_BLOCKS}_head${N_HEADS_VARATTN}_bs${BATCH_SIZE}_lr${LR}_cosine${MAX_STEPS}_s${SEED}"
OUT_DIR="${OUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

echo "================================================================="
echo "  H1 — capacity bump (~75M params)"
echo "    arch         : d_model=${D_MODEL} d_hidden=${D_HIDDEN}"
echo "                   n_perv=${N_PERV_LAYER} n_fusion=${N_FUSION_BLOCKS}"
echo "                   n_heads_varattn=${N_HEADS_VARATTN} grid_K=${GRID_K}"
echo "    sources cfg  : ${SOURCES_CFG}  (no stacking, inherit from H2)"
echo "    mix          : 20% chronos2 + 10% kernel + 70% lotsa_degraded"
echo "    BATCH_SIZE   : ${BATCH_SIZE}  (override via env var if OOM)"
echo "    LR           : ${LR}          (consider 2.5e-4 for 2× model)"
echo "    LR sched     : cosine — warmup ${NUM_WARMUP_STEPS}, ${MAX_STEPS} steps"
echo "    out_dir      : ${OUT_DIR}"
echo "    git_sha      : ${GIT_SHA}"
echo "================================================================="

cd "${REPO_ROOT}"
"${PYTHON}" -m imts_benchmark.pretrain.train_pretrain \
    --stage a \
    --stage_cfg "${PRETRAIN_DIR}/configs/stage_a_single_phase.yaml" \
    --sources_cfg "${PRETRAIN_DIR}/configs/${SOURCES_CFG}" \
    --output_dir "${OUT_DIR}" \
    --ablation_axis 15_H1_capacity_bump \
    --ablation_level main \
    --wandb_run_name "${RUN_NAME}" \
    --wandb_tags single_phase H1 capacity-bump d${D_MODEL} synth30 lotsa70 anyvariate asinh cosine-lr nostack \
    --max_steps "${MAX_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --max_dim 20 \
    --seed "${SEED}" \
    --loss huber \
    --huber_delta 1.0 \
    --arch vanilla \
    --d_model "${D_MODEL}" \
    --d_hidden "${D_HIDDEN}" \
    --n_perv_layer "${N_PERV_LAYER}" \
    --n_fusion_blocks "${N_FUSION_BLOCKS}" \
    --n_heads_varattn "${N_HEADS_VARATTN}" \
    --grid_K "${GRID_K}" \
    --lr "${LR}" \
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

echo "[H1] finished; outputs at ${OUT_DIR}"
