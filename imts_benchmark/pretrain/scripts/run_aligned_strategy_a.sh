#!/usr/bin/env bash
# ALIGNED Strategy A — curriculum on LOTSA-heavy mix with task-distribution
# alignment (95% all-target), V dist matched to IMM-TSF Table 1, asinh
# pipeline, lotsa_degraded clean_target_mode=False, Moirai per-subset
# weighting (auto-loaded from uni2ts/cli/conf/pretrain/data/lotsa_v1_weighted.yaml).
#
# Phase 1 (steps 0-10K, easy): regime [.50, .15, .25, .10],
#   mix 20% chronos2 + 10% kernel + 50% lotsa_reg + 20% lotsa_deg.
# Phase 2 (steps 0-30K, main): init_from phase 1 best,
#   regime [.30, .15, .35, .20],
#   mix 20% chronos2 + 10% kernel + 30% lotsa_reg + 40% lotsa_deg.
#
# Logging extras vs run_curriculum_synth.sh:
#   - synth_val_n_windows=512: in-distribution synth val every cycle
#     (logs val/synth_huber, val/synth_mse_z, val/synth_r2_overall,
#     val/synth_chronos2_synth_r2, val/synth_kernelsynth_r2).
#   - val_check_steps=2000 (was 2500) + save_every_n_steps=2000 +
#     ckpt save_top_k=5 — finer granularity past the saturation point.
#   - val_imts_subset=-1 — use all val windows on activity (1007) and
#     ushcn (5342) to halve sampling noise.
#
# IMM-TSF eval is on the full test split for all 8 datasets (every
# split has < 1024 windows, so subset=1024 means "use all").
#
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

PRETRAIN_DIR="${REPO_ROOT}/imts_benchmark/pretrain"
OUT_ROOT="${PRETRAIN_DIR}/runs/aligned_a"
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
P1_RUN="alignedA_p1easy_lotsa70_synth30_regimeReg50_d384_${PHASE1_STEPS}_s${SEED}"
P1_DIR="${OUT_ROOT}/${P1_NAME}"
mkdir -p "${P1_DIR}"

echo "================================================================="
echo "  ALIGNED Strategy A — PHASE 1 (easy, ${PHASE1_STEPS} steps)"
echo "    cfg     : stage_a_aligned_easy.yaml"
echo "    out_dir : ${P1_DIR}"
echo "    git_sha : ${GIT_SHA}"
echo "================================================================="

cd "${REPO_ROOT}"
"${PYTHON}" -m imts_benchmark.pretrain.train_pretrain \
    --stage a \
    --stage_cfg "${PRETRAIN_DIR}/configs/stage_a_aligned_easy.yaml" \
    --output_dir "${P1_DIR}" \
    --ablation_axis 7_aligned_strategy_a \
    --ablation_level phase1_easy \
    --wandb_run_name "${P1_RUN}" \
    --wandb_tags alignedA stageA phase1_easy lotsa70 synth30 anyvariate asinh aligned-task aligned-V \
    --max_steps "${PHASE1_STEPS}" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "${P1_DIR}/run.log"

echo "[alignedA] phase 1 finished"

# ---------- Pick phase 1 best ckpt ----------
P1_BEST=$(ls "${P1_DIR}"/best-step*-mse*.ckpt 2>/dev/null \
    | awk -F'-mse' '{print $2"\t"$0}' | sort -k1 | head -1 | cut -f2)
if [ -z "${P1_BEST}" ]; then
    echo "[alignedA] WARNING: no best-*.ckpt; falling back to last.ckpt" >&2
    P1_BEST="${P1_DIR}/last.ckpt"
    [ -f "${P1_BEST}" ] || { echo "[alignedA] FATAL: no ckpt at all" >&2; exit 1; }
fi
echo "[alignedA] phase 2 init_from: ${P1_BEST}"

# ---------- Phase 2 ----------
P2_NAME="phase2_main"
P2_RUN="alignedA_p2main_lotsa70_synth30_regimeMix35_d384_${PHASE2_STEPS}_fromP1_s${SEED}"
P2_DIR="${OUT_ROOT}/${P2_NAME}"
mkdir -p "${P2_DIR}"

echo "================================================================="
echo "  ALIGNED Strategy A — PHASE 2 (main, ${PHASE2_STEPS} steps)"
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
    --ablation_axis 7_aligned_strategy_a \
    --ablation_level phase2_main \
    --wandb_run_name "${P2_RUN}" \
    --wandb_tags alignedA stageA phase2_main lotsa70 synth30 anyvariate asinh aligned-task aligned-V \
    --max_steps "${PHASE2_STEPS}" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "${P2_DIR}/run.log"

echo "[alignedA] phase 2 finished; outputs at ${P2_DIR}"
