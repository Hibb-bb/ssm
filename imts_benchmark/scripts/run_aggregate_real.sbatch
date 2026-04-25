#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=agg_real
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/aggregate/slurm_%j.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/aggregate/slurm_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=short
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# Two-stage aggregation for the real-data IMTS benchmark.
#   Stage A: HPO winner-pick from .../hpo_mamba_mv/replace -> winners_by_dataset.json
#   Stage B: Cross-model summary from imts_benchmark_v2_real -> summary_real.csv
#
# Stage A is a no-op until the HPO sweep has produced cells; Stage B always runs
# over whatever is currently on disk. Both are cheap, so we always run both.

module purge
module load gcc/11.2.0
export PYTHONNOUSERSITE=1

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
LOG_BASE=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real

mkdir -p ${LOG_BASE}/aggregate

cd $CODE_DIR

# --- Stage A: pick HPO winners (dt_mode=replace, per-dataset) ----------------
HPO_DIR=${LOG_BASE}/hpo_mamba_mv/replace
if [[ -d "${HPO_DIR}" ]]; then
    echo "=== Stage A: HPO winner pick (replace, per-dataset) ==="
    ${MAMBA_ENV}/bin/python -m imts_benchmark.eval.aggregate_hpo_real \
        --hpo_log_dir ${HPO_DIR} \
        --datasets physionet activity ushcn \
        --dt_mode replace
    echo ""
else
    echo "[skip] Stage A: ${HPO_DIR} does not exist yet."
fi

# --- Stage B: cross-model summary --------------------------------------------
echo "=== Stage B: cross-model summary ==="
${MAMBA_ENV}/bin/python -m imts_benchmark.eval.aggregate_results \
    --results_root ${LOG_BASE} \
    --output_csv ${LOG_BASE}/aggregate/summary_real.csv \
    --long_csv ${LOG_BASE}/aggregate/long_real.csv

echo ""
echo "Done at $(date)"
