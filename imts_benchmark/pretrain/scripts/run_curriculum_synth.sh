#!/usr/bin/env bash
# Curriculum Stage A — synth-only with regime curriculum.
#
# Two phases, both on the regenerated CLEAN synthetic arrows (no baked
# jitter/drop, no chronos2 sq-clip), and the new pipeline (asinh, no
# StartDelay, min_ctx_obs=8).
#
# Phase 1 (steps 0-10K):  regime [0.50, 0.15, 0.25, 0.10]  (easy)
# Phase 2 (steps 0-30K):  regime [0.30, 0.15, 0.35, 0.20]  (main)
#                         init_from = phase 1 best ckpt
#
# Architecture: any-variate (matches anyvariate_stage_a baseline).
# Early stopping on val/mse_z_imm_avg with patience=6 val cycles.
#
# Usage:
#   bash imts_benchmark/pretrain/scripts/run_curriculum_synth.sh
#
# Override by env vars:
#   PHASE1_STEPS=10000  PHASE2_STEPS=30000
#   BATCH_SIZE=32  NUM_WORKERS=12  VAL_EVERY=2500
#   WANDB_PROJECT=TSKing  SEED=42
#   ES_PATIENCE=6  ES_METRIC=val/mse_z_imm_avg
#
# Each phase writes to runs/curriculum/{phase1,phase2}/.
#
set -euo pipefail

REPO_ROOT="/home/ubuntu/hongyu/ssm"
PYTHON="${PYTHON:-/home/ubuntu/envs/mamba/bin/python}"
PHASE1_STEPS="${PHASE1_STEPS:-10000}"
PHASE2_STEPS="${PHASE2_STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-12}"
VAL_EVERY="${VAL_EVERY:-2500}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-TSKing}"
SEED="${SEED:-42}"
ES_METRIC="${ES_METRIC:-val/mse_z_imm_avg}"
ES_PATIENCE="${ES_PATIENCE:-6}"

PRETRAIN_DIR="${REPO_ROOT}/imts_benchmark/pretrain"
OUT_ROOT="${PRETRAIN_DIR}/runs/curriculum"
TPATCHGNN_ROOT="${REPO_ROOT}/tpatchgnn_data"
IMM_TSF_ROOT="${REPO_ROOT}/imts_benchmark/data/imm_tsf_sparse"

GIT_SHA=$(cd "${REPO_ROOT}" && git rev-parse --short HEAD)

# Common args shared across phases.
COMMON_ARGS=(
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --max_dim 20
    --seed "${SEED}"
    --loss huber
    --huber_delta 1.0
    --arch vanilla
    --d_model 384
    --d_hidden 384
    --n_perv_layer 3
    --n_fusion_blocks 3
    --n_heads_varattn 4
    --grid_K 128
    --lr 5e-4
    --weight_decay 0.01
    --num_warmup_steps 200
    --precision bf16-mixed
    --gradient_clip_val 1.0
    --log_every_n_steps 25
    --save_every_n_steps 5000
    --val_imts_data_root "${TPATCHGNN_ROOT}"
    --val_imts_datasets activity ushcn
    --val_imts_split val
    --val_imts_subset 1024
    --val_imts_batch_size 64
    --val_imts_num_workers 2
    --val_imm_tsf_data_root "${IMM_TSF_ROOT}"
    --val_imm_tsf_datasets EPA-Air ILINet GDELT FNSPID CESNET StudentLife RepoHealth ClusterTrace
    --val_imm_tsf_split test
    --val_imm_tsf_subset 1024
    --val_imm_tsf_batch_size 32
    --val_imm_tsf_num_workers 2
    --val_check_steps "${VAL_EVERY}"
    --early_stop_metric "${ES_METRIC}"
    --early_stop_patience "${ES_PATIENCE}"
    --early_stop_mode min
    --use_wandb
    --wandb_project "${WANDB_PROJECT}"
    --wandb_mode "${WANDB_MODE}"
)

# ----------------------- Phase 1: easy -----------------------
PHASE1_NAME="phase1_easy"
PHASE1_RUN_NAME="stageA_curriculum_p1easy_reg50_sync15_mix25_async10_d384_${PHASE1_STEPS}_s${SEED}"
PHASE1_DIR="${OUT_ROOT}/${PHASE1_NAME}"
mkdir -p "${PHASE1_DIR}"

echo "================================================================="
echo "  Curriculum PHASE 1 (easy regime, ${PHASE1_STEPS} steps)"
echo "    regime_dist : [0.50, 0.15, 0.25, 0.10]"
echo "    out_dir     : ${PHASE1_DIR}"
echo "    wandb_run   : ${PHASE1_RUN_NAME}"
echo "    git_sha     : ${GIT_SHA}"
echo "================================================================="

cd "${REPO_ROOT}"
"${PYTHON}" -m imts_benchmark.pretrain.train_pretrain \
    --stage a \
    --stage_cfg "${PRETRAIN_DIR}/configs/stage_a_curriculum_easy.yaml" \
    --output_dir "${PHASE1_DIR}" \
    --ablation_axis 6_curriculum \
    --ablation_level phase1_easy \
    --wandb_run_name "${PHASE1_RUN_NAME}" \
    --wandb_tags curriculum stageA phase1_easy synthonly chronos2-70 kernelsynth-30 anyvariate asinh \
    --max_steps "${PHASE1_STEPS}" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "${PHASE1_DIR}/run.log"

echo "[curriculum] phase 1 finished"

# ----------------------- Find phase 1 best ckpt -----------------------
# ModelCheckpoint(best by val/mse_z_imm_avg) writes filenames like
# best-step{N:08d}-mse{X.XXXX}.ckpt; pick the one with the LOWEST mse.
PHASE1_BEST=$(ls "${PHASE1_DIR}"/best-step*-mse*.ckpt 2>/dev/null \
    | awk -F'-mse' '{print $2"\t"$0}' \
    | sort -k1 \
    | head -1 \
    | cut -f2)

if [ -z "${PHASE1_BEST}" ]; then
    echo "[curriculum] ERROR: no best-*.ckpt found in ${PHASE1_DIR}" >&2
    echo "                   falling back to last.ckpt" >&2
    PHASE1_BEST="${PHASE1_DIR}/last.ckpt"
    if [ ! -f "${PHASE1_BEST}" ]; then
        echo "[curriculum] FATAL: no ckpt at all in ${PHASE1_DIR}" >&2
        exit 1
    fi
fi

echo "[curriculum] phase 2 will init_from: ${PHASE1_BEST}"

# ----------------------- Phase 2: main -----------------------
PHASE2_NAME="phase2_main"
PHASE2_RUN_NAME="stageA_curriculum_p2main_reg30_sync15_mix35_async20_d384_${PHASE2_STEPS}_fromP1_s${SEED}"
PHASE2_DIR="${OUT_ROOT}/${PHASE2_NAME}"
mkdir -p "${PHASE2_DIR}"

echo "================================================================="
echo "  Curriculum PHASE 2 (main regime, ${PHASE2_STEPS} steps)"
echo "    regime_dist : [0.30, 0.15, 0.35, 0.20]"
echo "    init_from   : ${PHASE1_BEST}"
echo "    out_dir     : ${PHASE2_DIR}"
echo "    wandb_run   : ${PHASE2_RUN_NAME}"
echo "================================================================="

cd "${REPO_ROOT}"
"${PYTHON}" -m imts_benchmark.pretrain.train_pretrain \
    --stage a \
    --stage_cfg "${PRETRAIN_DIR}/configs/stage_a_curriculum_main.yaml" \
    --output_dir "${PHASE2_DIR}" \
    --init_from "${PHASE1_BEST}" \
    --init_from_strict \
    --ablation_axis 6_curriculum \
    --ablation_level phase2_main \
    --wandb_run_name "${PHASE2_RUN_NAME}" \
    --wandb_tags curriculum stageA phase2_main synthonly chronos2-70 kernelsynth-30 anyvariate asinh \
    --max_steps "${PHASE2_STEPS}" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "${PHASE2_DIR}/run.log"

echo "[curriculum] phase 2 finished; outputs at ${PHASE2_DIR}"
