#!/bin/bash
# Submit the full UEA classification pipeline:
#   1. HPO array (30 tasks)
#   2. Aggregator (depends on HPO afterok)
#   3. Final eval array (depends on aggregator afterok)
# Usage: bash submit_uea_cls_pipeline.sh

set -euo pipefail

CODE_DIR=/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
SCRIPT_DIR=${CODE_DIR}/imts_benchmark/scripts

cd ${CODE_DIR}

# Stage 1: HPO array.
HPO_JID=$(sbatch --parsable ${SCRIPT_DIR}/run_uea_cls_hpo.sbatch)
echo "[submit] HPO array: ${HPO_JID}"

# Stage 2: Aggregator (afterok on entire HPO array).
AGG_JID=$(sbatch --parsable --dependency=afterok:${HPO_JID} ${SCRIPT_DIR}/run_aggregate_uea_hpo.sbatch)
echo "[submit] Aggregator: ${AGG_JID} (afterok ${HPO_JID})"

# Stage 3: Final eval array (afterok on aggregator).
FINAL_JID=$(sbatch --parsable --dependency=afterok:${AGG_JID} ${SCRIPT_DIR}/run_uea_cls_final.sbatch)
echo "[submit] Final array: ${FINAL_JID} (afterok ${AGG_JID})"

echo ""
echo "Pipeline submitted:"
echo "  HPO        ${HPO_JID}  (30 tasks)"
echo "  Aggregator ${AGG_JID}  (1 task)"
echo "  Final      ${FINAL_JID}  (30 tasks)"
echo ""
echo "Watch with: squeue -u \$USER"
