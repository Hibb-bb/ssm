#!/bin/bash
# 7.D profiling pass — see pretrain/README.md §7.D.
#
# Runs Stage A and Stage B for ~5 min wall-clock each with Lightning's
# SimpleProfiler enabled and an nvidia-smi sampler in parallel.  Val
# loaders are disabled so the throughput numbers reflect pure
# fwd+bwd+data steady state.
#
# Usage:
#     bash imts_benchmark/pretrain/scripts/profile_stages.sh [stage_a|stage_b|both]
#
# Outputs:
#     imts_benchmark/pretrain/runs/profile/<stage>/{
#         run.log,             # full stdout
#         profiler_simple-fit.txt,  # Lightning per-action timings
#         nvidia_smi.csv,      # GPU util / memory every 2s
#         ablation.json,       # all run parameters + git SHA
#         csv/*/metrics.csv    # Lightning step-level metrics
#     }
#
# After the run, summarize via:
#     python imts_benchmark/pretrain/scripts/summarize_profile.py \
#         --stage_dir imts_benchmark/pretrain/runs/profile/stage_a

set -u  # don't `-e`; we want the nvidia-smi sampler to be killed even
        # if Lightning errors

WHICH="${1:-both}"

# Resolve repo root (parent of imts_benchmark/...)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="/home/ubuntu/envs/mamba/bin/python"
RUN_ROOT="${REPO_ROOT}/imts_benchmark/pretrain/runs/profile"
mkdir -p "${RUN_ROOT}"

run_one_stage() {
    local stage="$1"        # "a" or "b"
    local batch_size="$2"
    local num_workers="$3"
    local minutes="$4"
    local out_dir="${RUN_ROOT}/stage_${stage}"

    rm -rf "${out_dir}"
    mkdir -p "${out_dir}"

    echo "============================================================"
    echo "Profiling Stage ${stage^^}  (bs=${batch_size}, nw=${num_workers}, ${minutes} min)"
    echo "  out: ${out_dir}"
    echo "============================================================"

    # nvidia-smi sampler in background. 2 s polling is fine; the GPU
    # util we want is a long-window average, not a transient spike.
    nvidia-smi \
        --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
        --format=csv -l 2 \
        > "${out_dir}/nvidia_smi.csv" 2>&1 &
    local nv_pid=$!
    # Always kill the sampler when this stage finishes (success or fail).
    trap "kill ${nv_pid} 2>/dev/null || true" RETURN

    "${PYTHON}" -m imts_benchmark.pretrain.train_pretrain \
        --stage "${stage}" \
        --batch_size "${batch_size}" \
        --num_workers "${num_workers}" \
        --max_steps 999999 \
        --max_minutes "${minutes}" \
        --val_check_steps 999999 \
        --save_every_n_steps 999999 \
        --num_warmup_steps 50 \
        --log_every_n_steps 10 \
        --profiler simple \
        --output_dir "${out_dir}" \
        --ablation_axis profiling \
        --ablation_level "stage_${stage}" \
        2>&1 | tee "${out_dir}/run.log"

    local exit_code=${PIPESTATUS[0]}
    kill ${nv_pid} 2>/dev/null || true
    trap - RETURN

    echo "Stage ${stage^^} done (exit ${exit_code}).  Profile:"
    ls -la "${out_dir}/" | sed 's/^/    /'
}

# Stage A: synthetic-only, lighter data path, push batch a bit.
# Stage B: includes LOTSA, more disk IO, more dataloader workers.
if [[ "${WHICH}" == "both" || "${WHICH}" == "stage_a" ]]; then
    run_one_stage a 64  8 5
fi
if [[ "${WHICH}" == "both" || "${WHICH}" == "stage_b" ]]; then
    run_one_stage b 32 12 5
fi

echo
echo "All requested stages profiled.  Summarize with:"
echo "    ${PYTHON} ${REPO_ROOT}/imts_benchmark/pretrain/scripts/summarize_profile.py \\"
echo "        --stage_dir ${RUN_ROOT}/stage_<a|b>"
