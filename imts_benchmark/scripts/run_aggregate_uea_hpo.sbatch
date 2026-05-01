#!/bin/bash
#SBATCH --account=p33049
#SBATCH --job-name=uea_cls_agg
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/uea_cls/slurm_agg_%j.out
#SBATCH --error=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/uea_cls/slurm_agg_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
HPO_ROOT=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/uea_cls/hpo
OUT_PATH=/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/uea_cls/winner_configs.json

cd ${CODE_DIR}

${MAMBA_ENV}/bin/python -m imts_benchmark.eval.aggregate_uea_hpo \
    --hpo_root ${HPO_ROOT} \
    --out ${OUT_PATH}
