#!/usr/bin/env bash
# ALIGNED Strategy A — bigger model variant.
#
# Identical to run_aligned_strategy_a.sh EXCEPT d_model and d_hidden
# are bumped from 384 -> 768.  Tests "is the plateau caused by model
# capacity?" — same data, same regimes, same curriculum, same seed,
# only width changes.
#
# Param count: ~7.8M (d=384)  ->  ~31M (d=768).
# Throughput: roughly 4x compute per step => ~0.6 it/s vs 2.3 it/s.
# ETA: phase 1 (10K) ~4.5h, phase 2 (30K, early-stop likely) ~10-14h.

set -euo pipefail

REPO_ROOT="/home/ubuntu/hongyu/ssm"
PYTHON="${PYTHON:-/home/ubuntu/envs/mamba/bin/python}"
PHASE1_STEPS="${PHASE1_STEPS:-10000}"
PHASE2_STEPS="${PHASE2_STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-12}"
VAL_EVERY="${VAL_EVERY:-2000}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-TSKing}"
SEED="${SEED:-42}"
ES_METRIC="${ES_METRIC:-val/mse_z_imm_avg}"
ES_PATIENCE="${ES_PATIENCE:-6}"
SYNTH_VAL_N="${SYNTH_VAL_N:-512}"
D_MODEL="${D_MODEL:-768}"
D_HIDDEN="${D_HIDDEN:-768}"

PRETRAIN_DIR="${REPO_ROOT}/imts_benchmark/pretrain"
OUT_ROOT="${PRETRAIN_DIR}/runs/aligned_a_d${D_MODEL}"
TPATCHGNN_ROOT="${REPO_ROOT}/tpatchgnn_data"
IMM_TSF_ROOT="${REPO_ROOT}/imts_benchmark/data/imm_tsf_sparse"

GIT_SHA=$(cd "${REPO_ROOT}" && git rev-parse --short HEAD)

COMMON_ARGS=(
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --max_dim 20
    --seed "${SEED}"
    --loss huber
    --huber_delta 1.0
    --arch vanilla
    --d_model "${D_MODEL}"
    --d_hidden "${D_HIDDEN}"
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
    --save_every_n_steps "${SAVE_EVERY}"
    --val_imts_data_root "${TPATCHGNN_ROOT}"
    --val_imts_datasets activity ushcn
    --val_imts_split val
    --val_imts_subset -1
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
    --synth_val_n_windows "${SYNTH_VAL_N}"
    --use_wandb
    --wandb_project "${WANDB_PROJECT}"
    --wandb_mode "${WANDB_MODE}"
)

# ---------- Phase 1 ----------
P1_NAME="phase1_easy"
P1_RUN="alignedA_d${D_MODEL}_p1easy_lotsa70_synth30_regimeReg50_${PHASE1_STEPS}_s${SEED}"
P1_DIR="${OUT_ROOT}/${P1_NAME}"
mkdir -p "${P1_DIR}"

echo "================================================================="
echo "  ALIGNED Strategy A (d=${D_MODEL}) — PHASE 1 (easy, ${PHASE1_STEPS} steps)"
echo "    cfg     : stage_a_aligned_easy.yaml"
echo "    out_dir : ${P1_DIR}"
echo "    git_sha : ${GIT_SHA}"
echo "================================================================="

cd "${REPO_ROOT}"
"${PYTHON}" -m imts_benchmark.pretrain.train_pretrain \
    --stage a \
    --stage_cfg "${PRETRAIN_DIR}/configs/stage_a_aligned_easy.yaml" \
    --output_dir "${P1_DIR}" \
    --ablation_axis 8_capacity_d${D_MODEL} \
    --ablation_level phase1_easy \
    --wandb_run_name "${P1_RUN}" \
    --wandb_tags alignedA d${D_MODEL} stageA phase1_easy lotsa70 synth30 anyvariate asinh aligned-task aligned-V capacity-test \
    --max_steps "${PHASE1_STEPS}" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "${P1_DIR}/run.log"

echo "[alignedA d${D_MODEL}] phase 1 finished"

# ---------- Pick phase 1 best ckpt ----------
P1_BEST=$(ls "${P1_DIR}"/best-step*-mse*.ckpt 2>/dev/null \
    | awk -F'-mse' '{print $2"\t"$0}' | sort -k1 | head -1 | cut -f2)
if [ -z "${P1_BEST}" ]; then
    echo "[alignedA d${D_MODEL}] WARNING: no best-*.ckpt; falling back to last.ckpt" >&2
    P1_BEST="${P1_DIR}/last.ckpt"
    [ -f "${P1_BEST}" ] || { echo "[alignedA d${D_MODEL}] FATAL: no ckpt at all" >&2; exit 1; }
fi
echo "[alignedA d${D_MODEL}] phase 2 init_from: ${P1_BEST}"

# ---------- Phase 2 ----------
P2_NAME="phase2_main"
P2_RUN="alignedA_d${D_MODEL}_p2main_lotsa70_synth30_regimeMix35_${PHASE2_STEPS}_fromP1_s${SEED}"
P2_DIR="${OUT_ROOT}/${P2_NAME}"
mkdir -p "${P2_DIR}"

echo "================================================================="
echo "  ALIGNED Strategy A (d=${D_MODEL}) — PHASE 2 (main, ${PHASE2_STEPS} steps)"
echo "    cfg       : stage_a_aligned_main.yaml"
echo "    init_from : ${P1_BEST}"
echo "    out_dir   : ${P2_DIR}"
echo "================================================================="

cd "${REPO_ROOT}"
"${PYTHON}" -m imts_benchmark.pretrain.train_pretrain \
    --stage a \
    --stage_cfg "${PRETRAIN_DIR}/configs/stage_a_aligned_main.yaml" \
    --output_dir "${P2_DIR}" \
    --init_from "${P1_BEST}" \
    --init_from_strict \
    --ablation_axis 8_capacity_d${D_MODEL} \
    --ablation_level phase2_main \
    --wandb_run_name "${P2_RUN}" \
    --wandb_tags alignedA d${D_MODEL} stageA phase2_main lotsa70 synth30 anyvariate asinh aligned-task aligned-V capacity-test \
    --max_steps "${PHASE2_STEPS}" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "${P2_DIR}/run.log"

echo "[alignedA d${D_MODEL}] phase 2 finished; outputs at ${P2_DIR}"
