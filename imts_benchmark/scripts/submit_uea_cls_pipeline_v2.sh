#!/bin/bash
# Submit the v2 UEA classification pipeline:
#   1. HPO array (72 tasks: 4 datasets x 2 dt_modes x 3 LRs x 3 BSs)
#   2. Aggregator with per-dataset metric + collapse detector (afterok HPO)
#   3. Final eval array (24 tasks: 4 ds x 2 dt x 3 seeds, picks LR+BS from winner_configs.json)
#
# Scope vs v1: BM excluded (already at 1.0); per-dataset LR ranges; batch_size as a real axis.
# Output root: .../output/log/imts_benchmark_v2/uea_cls_v2/
#
# Usage: bash submit_uea_cls_pipeline_v2.sh

set -euo pipefail

CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
SCRIPT_DIR=${CODE_DIR}/imts_benchmark/scripts

cd ${CODE_DIR}

# Stage 1: HPO array (72 cells).
HPO_JID=$(sbatch --parsable ${SCRIPT_DIR}/run_uea_cls_hpo_v2.sbatch)
echo "[submit-v2] HPO array: ${HPO_JID}"

# Stage 2: Aggregator (afterok on entire HPO array).
AGG_JID=$(sbatch --parsable --dependency=afterok:${HPO_JID} ${SCRIPT_DIR}/run_aggregate_uea_hpo_v2.sbatch)
echo "[submit-v2] Aggregator: ${AGG_JID} (afterok ${HPO_JID})"

# Stage 3: Final eval array (afterok on aggregator).
FINAL_JID=$(sbatch --parsable --dependency=afterok:${AGG_JID} ${SCRIPT_DIR}/run_uea_cls_final_v2.sbatch)
echo "[submit-v2] Final array: ${FINAL_JID} (afterok ${AGG_JID})"

echo ""
echo "Pipeline v2 submitted:"
echo "  HPO        ${HPO_JID}  (72 tasks: 4 ds x 2 dt x 3 LR x 3 BS)"
echo "  Aggregator ${AGG_JID}  (1 task; per-dataset metric + collapse filter)"
echo "  Final      ${FINAL_JID}  (24 tasks: 4 ds x 2 dt x 3 seeds)"
echo ""
echo "Watch with: squeue -u \$USER"
echo "Outputs at: /projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/uea_cls_v2/"
