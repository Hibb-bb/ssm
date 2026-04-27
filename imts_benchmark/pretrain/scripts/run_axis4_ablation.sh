#!/usr/bin/env bash
# Axis-4 ablation runner: Stage-A synthetic-only  vs  +regular LOTSA tax.
#
# Both arms are 50K steps, identical model / optimizer / val cadence.
# Run sequentially on a single GPU; arm 2 only starts if arm 1 finishes
# cleanly.
#
# Outputs (under /home/ubuntu/hongyu/ssm/imts_benchmark/pretrain/runs/axis4/):
#   axis4_synth_only/{ckpt-*.ckpt, csv/, ablation.json, wandb/}
#   axis4_synth_lotsa/{...}
#
# W&B: project = TSKing (online by default once you've run `wandb login`).
# Run names encode stage + per-source mix percentages + key hparams, e.g.
#   stageA_chronos70_kernel30_d384_huber_50k_s42
#       (axis4_synth_only:  70% chronos2 + 30% kernelsynth)
#   stageA_chronos55_kernel25_lotsa20_d384_huber_50k_s42
#       (axis4_synth_lotsa: 55% chronos2 + 25% kernelsynth + 20% regular LOTSA)
#
# Usage::
#
#   bash imts_benchmark/pretrain/scripts/run_axis4_ablation.sh
#
# Override by env vars:
#
#   MAX_STEPS=50000  -- per-arm step budget (default 50000)
#   BATCH_SIZE=32    -- (default 32)
#   NUM_WORKERS=12   -- (default 12; matches §7.D Stage-B post-fix profile)
#   VAL_EVERY=2500   -- val cadence in steps (default 2500)
#   WANDB_MODE=offline  -- 'online' if you've run `wandb login` first
#   PYTHON=/home/ubuntu/envs/mamba/bin/python
#   ARMS="synth_only synth_lotsa"  -- subset to run
#                                       (default: synth_only only)
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
# Default to synth_only only; explicitly pass ARMS="synth_only synth_lotsa"
# to run both arms back-to-back.  See README §7.F.
ARMS="${ARMS:-synth_only}"

PRETRAIN_DIR="${REPO_ROOT}/imts_benchmark/pretrain"
OUT_ROOT="${PRETRAIN_DIR}/runs/axis4"
TPATCHGNN_ROOT="${REPO_ROOT}/tpatchgnn_data"
IMM_TSF_ROOT="${REPO_ROOT}/imts_benchmark/data/imm_tsf_sparse"

mkdir -p "${OUT_ROOT}"

# Common args shared between the two arms.
COMMON_ARGS=(
    --max_steps "${MAX_STEPS}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --max_dim 20
    --seed 42
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
    --use_wandb
    --wandb_project "${WANDB_PROJECT}"
    --wandb_mode "${WANDB_MODE}"
    --ablation_axis 4_stage_a_mix
)

run_arm() {
    local name="$1"        # synth_only | synth_lotsa
    local stage_cfg="$2"   # full path
    local run_name="$3"    # human-readable W&B run name
    local mix_tags=("${@:4}")  # extra W&B tags encoding the mix
    local out_dir="${OUT_ROOT}/axis4_${name}"
    local log_file="${out_dir}/run.log"

    mkdir -p "${out_dir}"
    echo "================================================================="
    echo "  axis4 arm : ${name}"
    echo "    stage_cfg : ${stage_cfg}"
    echo "    out_dir   : ${out_dir}"
    echo "    max_steps : ${MAX_STEPS}"
    echo "    wandb_run : ${run_name}"
    echo "    log_file  : ${log_file}"
    echo "================================================================="

    cd "${REPO_ROOT}"
    "${PYTHON}" -m imts_benchmark.pretrain.train_pretrain \
        --stage a \
        --stage_cfg "${stage_cfg}" \
        --output_dir "${out_dir}" \
        --ablation_level "${name}" \
        --wandb_run_name "${run_name}" \
        --wandb_tags axis4 stageA "${name}" "${mix_tags[@]}" \
        "${COMMON_ARGS[@]}" \
        2>&1 | tee "${log_file}"

    echo "[axis4] arm ${name} finished cleanly"
}

for arm in ${ARMS}; do
    case "${arm}" in
        synth_only)
            # 70% chronos2 + 30% kernelsynth = 100% synthetic
            run_arm \
                synth_only \
                "${PRETRAIN_DIR}/configs/stage_a.yaml" \
                "stageA_chronos70_kernel30_d384_huber_50k_s${SEED}" \
                chronos2-70 kernelsynth-30
            ;;
        synth_lotsa)
            # 55% chronos2 + 25% kernelsynth + 20% regular LOTSA
            run_arm \
                synth_lotsa \
                "${PRETRAIN_DIR}/configs/stage_a_with_lotsa.yaml" \
                "stageA_chronos55_kernel25_lotsa20_d384_huber_50k_s${SEED}" \
                chronos2-55 kernelsynth-25 lotsa_regular-20
            ;;
        *)
            echo "[axis4] unknown arm '${arm}'; expected one of: synth_only, synth_lotsa"
            exit 2
            ;;
    esac
done

echo "[axis4] all arms complete; outputs in ${OUT_ROOT}/"
