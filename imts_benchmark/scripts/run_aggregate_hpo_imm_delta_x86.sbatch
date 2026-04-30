#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --job-name=mv_hpo_pick_imm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuH200x8-interactive
#SBATCH --time=00:15:00
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_imm/aggregate_%x_%j.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_imm/aggregate_%x_%j.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Read 27 HPO cells' best.ckpt val/mse, pick winner per dt_mode, write
# winners_by_dt_mode.json. CPU-only.
#
# Submitted with afterany dependency on the HPO sweep array (so partial-failure
# sweeps still produce winners as long as each dt_mode has at least one usable
# cell). The confirm array depends on this job with afterok.
#
# Usage:
#   sbatch --export=ALL,DS_NAME=imm_fnspid,EFF_BS_AGG='64 128 256' \
#       --job-name=mv_hpo_pick_${DS_NAME} \
#       imts_benchmark/scripts/run_aggregate_hpo_imm_delta_x86.sbatch

set -uo pipefail

DS_NAME="${DS_NAME:?Must set DS_NAME via --export, e.g. DS_NAME=imm_fnspid}"
EFF_BS_AGG="${EFF_BS_AGG:-64 128 256}"

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
HPO_DIR=${CODE_DIR}/output/log/imts_benchmark_v2_imm/hpo_mamba_mv_${DS_NAME}

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

cd "${CODE_DIR}"
echo "Aggregating HPO winners for DS=${DS_NAME} (eff_bs=${EFF_BS_AGG})"
echo "HPO_DIR=${HPO_DIR}"
echo "Time: $(date)"

python -m imts_benchmark.eval.aggregate_hpo_winners_imm \
    --hpo_log_dir "${HPO_DIR}" \
    --eff_bs ${EFF_BS_AGG}

EXIT_CODE=$?
echo "aggregate exit code: ${EXIT_CODE}  $(date)"
exit ${EXIT_CODE}
