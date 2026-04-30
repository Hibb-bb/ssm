#!/usr/bin/env bash
# Single-phase pretraining (no curriculum), Moirai-aligned data sampling.
#
# Per user spec 2026-04-29:
#   - 30% synthetic + 70% degraded LOTSA, regime [0.30, 0.10, 0.40, 0.20]
#     applied uniformly to all sources.
#   - Cosine LR over 30K steps, 500-step warmup (replaces the 80K
#     multistep schedule that never fired before early-stopping).
#   - Patience=10 early stopping on val/mse_z_imm_avg.
#
# What's NEW vs single_synth30_lotsa70_regimeMix40_d384_warmup1000_multistep_80000_s42:
#   - Moirai-faithful weighting in mixed_dataset.make_logical_source:
#       p(D_k) ∝ num_ts_k × yaml_weight_k  (was: ∝ yaml_weight_k)
#       Drops solar_power+wind_power from 89.96% of LOTSA to ~2.2%.
#   - StackedLOTSASource: univariate-on-disk LOTSA datasets get K random
#     series stacked into a [K, T] sample (Moirai's
#     MultiSampleTimeSeriesDataset).  K drawn from variate_count_dist.
#   - variate_count_dist = [0.05, 0.35, 0.55, 0.05] (was 0.05, 0.40,
#     0.50, 0.05) — pulls more mass into bucket 3 (V=9..16) where 5/8
#     IMM-TSF eval datasets live.
#   - Eval-side guards from §7.L (preds_asinh clamp, EarlyStopping
#     check_finite=False, aggregate-callback non-finite filter) are in.
#
# All other settings carry forward:
#   d_model=384, asinh, min_ctx_obs=8, no StartDelay,
#   task_type_probs=[0.05,0.95,0.0], clean_target_mode=False on LOTSA.

set -euo pipefail

REPO_ROOT="/home/ubuntu/hongyu/ssm"
PYTHON="${PYTHON:-/home/ubuntu/envs/mamba/bin/python}"
MAX_STEPS="${MAX_STEPS:-60000}"
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

RUN_NAME="single_moirai_v2_synth10_lotsa90_d384_cosine${MAX_STEPS}_warmup${NUM_WARMUP_STEPS}_minstd1e-3_s${SEED}"
OUT_DIR="${OUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

echo "================================================================="
echo "  SINGLE-PHASE pretraining (Moirai-aligned, R1 lotsa-heavy, cosine)"
echo "    cfg          : stage_a_single_phase.yaml"
echo "    mix          : 5% chronos2 + 5% kernel + 90% lotsa_degraded (R1)"
echo "    sampling     : Moirai-faithful (num_ts × yaml_weight)"
echo "    stacking     : univariate LOTSA → MultiSampleTimeSeriesDataset"
echo "    V dist       : [0.05, 0.35, 0.55, 0.05]"
echo "    regime_dist  : [0.30, 0.10, 0.40, 0.20]"
echo "    min_std      : 1e-3 (F2 fix, from 1e-6)"
echo "    metric fix   : _SynthInDistValCallback clamps |a|<10 before sinh"
echo "    LR sched     : cosine — warmup ${NUM_WARMUP_STEPS}, peak 5e-4 → 0 over ${MAX_STEPS}"
echo "    max_steps    : ${MAX_STEPS}"
echo "    out_dir      : ${OUT_DIR}"
echo "    git_sha      : ${GIT_SHA}"
echo "================================================================="

cd "${REPO_ROOT}"
"${PYTHON}" -m imts_benchmark.pretrain.train_pretrain \
    --stage a \
    --stage_cfg "${PRETRAIN_DIR}/configs/stage_a_single_phase.yaml" \
    --output_dir "${OUT_DIR}" \
    --ablation_axis 13_single_phase_moirai_v2_R1_lotsa90 \
    --ablation_level main \
    --wandb_run_name "${RUN_NAME}" \
    --wandb_tags single_phase moirai_v2 R1 synth10 lotsa90 anyvariate asinh stacked-lotsa cosine-lr minstd1e-3 metric-fix \
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

echo "[single_phase] finished; outputs at ${OUT_DIR}"
