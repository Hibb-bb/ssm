#!/bin/bash
# Download the 5 UEA datasets used in the RoMAE Table 4 protocol.
# Run from a node with internet access (Quest login node should work).
#
# Usage:  bash download_uea_datasets.sh [DATA_ROOT]
# Default DATA_ROOT: $CODE_DIR/data_uea where
#   CODE_DIR = /projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk

set -euo pipefail

DATA_ROOT="${1:-/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk/data_uea}"
DATASETS=(BasicMotions CharacterTrajectories Epilepsy Heartbeat LSST)
BASE_URL="https://www.timeseriesclassification.com/aeon-toolkit"

mkdir -p "${DATA_ROOT}"
cd "${DATA_ROOT}"

for ds in "${DATASETS[@]}"; do
    if [[ -f "${ds}/${ds}_TRAIN.ts" && -f "${ds}/${ds}_TEST.ts" ]]; then
        echo "[skip] ${ds}: already present"
        continue
    fi
    echo "[download] ${ds}"
    if [[ ! -f "${ds}.zip" ]]; then
        wget -q --show-progress "${BASE_URL}/${ds}.zip" -O "${ds}.zip"
    fi
    rm -rf "${ds}"
    unzip -q "${ds}.zip" -d "${ds}"
    # Some archives unzip into a nested directory of the same name; flatten.
    if [[ -d "${ds}/${ds}" ]]; then
        mv "${ds}/${ds}"/* "${ds}/"
        rmdir "${ds}/${ds}"
    fi
    if [[ ! -f "${ds}/${ds}_TRAIN.ts" || ! -f "${ds}/${ds}_TEST.ts" ]]; then
        echo "ERROR: ${ds}_TRAIN.ts / _TEST.ts not found after extracting ${ds}.zip" >&2
        ls -la "${ds}/" >&2
        exit 1
    fi
    rm -f "${ds}.zip"
    echo "  ${ds}: OK"
done

echo "Done. Datasets at: ${DATA_ROOT}"
