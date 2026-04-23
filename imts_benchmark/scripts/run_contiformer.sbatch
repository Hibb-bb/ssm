#!/bin/bash
# ==============================================================================
# DEFERRED FOR PHASE 2. Do not submit this sbatch in the current benchmark run.
#
# ContiFormer at our Phase-2 config (L=360, d_model=320, n_layers=6) exceeds
# H100 80GB regardless of batch size (tested 128/64/32/8) and regardless of
# ODE step size (tested 0.1 and 0.5, paper's full ablation range per Chen et
# al. 2023 Table 18). All four combinations OOM at the same 4.95 GiB
# allocation inside torchdiffeq's odeint_adjoint backward, confirming the
# failing tensor is L^2-dominated and batch/step-count-independent.
#
# Context for the deferral:
#   * Chen et al. (NeurIPS 2023, arXiv:2402.10635) ran ContiFormer on V100-16GB
#     but with input_length <= 96 for forecasting experiments (their Table 5);
#     their longest reported test (SCP1 at L=896) was classification at small
#     batch. The paper explicitly acknowledges "substantial ... GPU memory
#     overhead" (Section 5) and shows memory growing quadratically in L
#     (Figure 6 in Appendix E).
#   * The T-PATCHGNN IMTS forecasting benchmark (Zhang et al., ICML 2024)
#     did NOT include ContiFormer among its 17 baselines, despite it being
#     available -- consistent with the same memory limitation.
#
# Fallback: Phase 2 runs with Mamba-MV + RoMAE + mTAN. Document in RESULTS.
# If needed later, ContiFormer could be revisited with L-reduction via
# observation subsampling (breaks fairness) or on a >= 120GB GPU.
# ==============================================================================
#SBATCH --account=p33049
#SBATCH --job-name=contiformer_fc
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/contiformer/slurm_%A_%a.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/contiformer/slurm_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=1-20
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# ContiFormer forecaster baseline (Chen et al. NeurIPS 2023). ODE-based
# continuous-time attention; expected to be the slowest baseline due to
# odeint_adjoint cost in MultiHeadAttention. Walltime / mem doubled
# accordingly (4h, 64G) vs the other baselines.
#
#   regimes : multisin_regular, multisin_low_irreg, multisin_med_irreg, multisin_high_irreg
#   seeds   : 1..5
# Total runs: 4 * 5 = 20.
#
# v2 (Phase 2, rebuilt on correct convex-mix generator).
# actfn_ode default switched from "tanh" to "softplus" to match upstream physiopro default.
#
# SLURM_ARRAY_TASK_ID in [1, 20]. Mapping (1-based):
#   k = TASK_ID - 1
#   regime_idx = k // 5   (0..3)
#   seed_idx   = k % 5    (0..4)

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
LOG_BASE=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2

REGIMES=("multisin_regular" "multisin_low_irreg" "multisin_med_irreg" "multisin_high_irreg")
SEEDS=(1 2 3 4 5)

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
REGIME_IDX=$(( K / 5 ))
SEED_IDX=$(( K % 5 ))

REGIME=${REGIMES[$REGIME_IDX]}
SEED=${SEEDS[$SEED_IDX]}

OUTPUT_DIR=${LOG_BASE}/contiformer/${REGIME}/default/seed${SEED}
mkdir -p $OUTPUT_DIR

echo "========================================="
echo "ContiFormer forecaster  regime=$REGIME  seed=$SEED"
echo "Job: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Time: $(date)"
echo ""

${MAMBA_ENV}/bin/python -c "
import torchdiffeq, torchcde
print('torchdiffeq + torchcde imported OK')
" && echo "ContiFormer deps: AVAILABLE" || echo "ContiFormer deps: MISSING"
echo ""

cd $CODE_DIR
START_TIME=$(date +%s)

srun ${MAMBA_ENV}/bin/python -m imts_benchmark.contiformer_forecaster.train_contiformer \
    --regime $REGIME \
    --seed $SEED \
    --train_batch_size 64 \
    --accumulate_grad_batches 2 \
    --output_dir $OUTPUT_DIR
# ContiFormer OOMs at B=128 on H100 (confirmed in debug smoke).
# Effective batch = 64 * 2 = 128 preserves the shared optimization regime.

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "Exit code: $EXIT_CODE"
echo "Wall time: ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "Done at $(date)"
