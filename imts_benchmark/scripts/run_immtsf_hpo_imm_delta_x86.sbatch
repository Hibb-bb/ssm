#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --job-name=immtsf_hpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_imm/immtsf_hpo/%x/slurm_%A_%a.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_imm/immtsf_hpo/%x/slurm_%A_%a.err
#SBATCH --mail-user=dhjrzzang@gmail.com
#SBATCH --mail-type=END,FAIL

# IMM-TSF IMTS baseline architectural HPO sweep on a TIME-IMM dataset.
# Mirrors T-PatchGNN paper's HPO design: lr=1e-3 and patience=10 fixed; only
# architectural HPs are tuned per (model, dataset). Per-model grids:
#
#   tPatchGNN   12 cells: patch_size {ds-specific × 3} × K {1,2} × hid_dim {32,64}
#   LatentODE    8 cells: rec_layers {1,3} × gen_layers {1,3} × hid_dim {32,64}
#   CRU          8 cells: scaling_factor {0.2,0.3} × bandwidth {3,10} × bias {15,20}
#   NeuralFlow   8 cells: flow_model {GRU,coupling} × time_net {TimeLinear,TimeTanh} × rec_dims {20,40}
#
# Submit per (MODEL, DATASET); array index selects the cell. Then 5-seed confirm.
#
# Usage:
#   sbatch --array=1-12 --export=ALL,MODEL=tPatchGNN,DATASET=FNSPID,DS_NAME=imm_fnspid \
#       --job-name=hpo_tPatchGNN_imm_fnspid run_immtsf_hpo_imm_delta_x86.sbatch
#
# DS_NAME (lowercase imm_*) controls the output dir layout for our aggregator;
# DATASET (uppercase) is the IMM-TSF-internal name passed to its `--dataset` flag.

set -uo pipefail

MODEL="${MODEL:?Must set MODEL via --export, e.g. MODEL=tPatchGNN}"
DATASET="${DATASET:?Must set DATASET via --export, e.g. DATASET=FNSPID}"
DS_NAME="${DS_NAME:?Must set DS_NAME via --export, e.g. DS_NAME=imm_fnspid}"

# Per-(model, dataset) batch_size lookup: tiny train sets get bs=8.
case $DS_NAME in
    imm_cesnet|imm_ilinet) BS_DEFAULT=8 ;;
    *) BS_DEFAULT=32 ;;
esac
BATCH_SIZE="${BATCH_SIZE:-${BS_DEFAULT}}"
LR="${LR:-1e-3}"
PATIENCE="${PATIENCE:-10}"
SEED_HPO="${SEED_HPO:-1}"

ENV_PATH=/projects/bfrf/seojininus/envs/mamba_x86
CODE_DIR=/projects/bfrf/seojininus/ssm
IMM_TSF_REPO=${CODE_DIR}/_imm_tsf_repo
TIME_IMM_DATA=${CODE_DIR}/_time_imm_repo/data
LOG_BASE=${CODE_DIR}/output/log/imts_benchmark_v2_imm/immtsf_hpo

# ---- Per-model architectural grid -------------------------------------------
# Each line: ARCH_HPS_JSON tag-suffix
case $MODEL in
    tPatchGNN)
        # patch_size depends on dataset's history (see TIME-IMM paper Table 2)
        case $DS_NAME in
            imm_gdelt)        PS=(2 4 7) ;;
            imm_repohealth)   PS=(4 8 16) ;;
            imm_fnspid)       PS=(4 8 16) ;;
            imm_clustertrace) PS=(2 4 6) ;;
            imm_studentlife)  PS=(4 8 16) ;;
            imm_ilinet)       PS=(4 8 18) ;;
            imm_cesnet)       PS=(1 2 4) ;;
            imm_epa_air)      PS=(1 2 4) ;;
            *) PS=(4 8 16) ;;
        esac
        K_VALS=(1 2)
        HID_VALS=(32 64)
        N_CELLS=$(( ${#PS[@]} * ${#K_VALS[@]} * ${#HID_VALS[@]} ))
        ;;
    LatentODE)
        REC_LAYERS_VALS=(1 3)
        GEN_LAYERS_VALS=(1 3)
        HID_VALS=(32 64)
        N_CELLS=$(( ${#REC_LAYERS_VALS[@]} * ${#GEN_LAYERS_VALS[@]} * ${#HID_VALS[@]} ))
        ;;
    CRU)
        SCALE_VALS=(0.2 0.3)
        BANDWIDTH_VALS=(3 10)
        BIAS_VALS=(15 20)
        N_CELLS=$(( ${#SCALE_VALS[@]} * ${#BANDWIDTH_VALS[@]} * ${#BIAS_VALS[@]} ))
        ;;
    NeuralFlow)
        FLOW_MODEL_VALS=(gru coupling)
        TIME_NET_VALS=(TimeLinear TimeTanh)
        REC_DIMS_VALS=(20 40)
        N_CELLS=$(( ${#FLOW_MODEL_VALS[@]} * ${#TIME_NET_VALS[@]} * ${#REC_DIMS_VALS[@]} ))
        ;;
    *)
        echo "Unknown model $MODEL" >&2; exit 2
        ;;
esac

TASK_ID=$SLURM_ARRAY_TASK_ID
K=$(( TASK_ID - 1 ))
if [[ $K -ge $N_CELLS ]]; then
    echo "TASK_ID=$TASK_ID exceeds N_CELLS=$N_CELLS; nothing to do" >&2
    exit 0
fi

# Decode K into the per-model arch HPs JSON. Tag is for output dir naming.
case $MODEL in
    tPatchGNN)
        n_hid=${#HID_VALS[@]}; n_k=${#K_VALS[@]}
        i_ps=$(( K / (n_k * n_hid) ))
        i_k=$(( (K % (n_k * n_hid)) / n_hid ))
        i_hid=$(( K % n_hid ))
        PS_VAL=${PS[$i_ps]}; K_VAL=${K_VALS[$i_k]}; HID_VAL=${HID_VALS[$i_hid]}
        ARCH_HPS_JSON="{\"patch_size\": ${PS_VAL}, \"K\": ${K_VAL}, \"hid_dim\": ${HID_VAL}}"
        TAG="ps${PS_VAL}_K${K_VAL}_h${HID_VAL}"
        ;;
    LatentODE)
        n_hid=${#HID_VALS[@]}; n_gen=${#GEN_LAYERS_VALS[@]}
        i_rec=$(( K / (n_gen * n_hid) ))
        i_gen=$(( (K % (n_gen * n_hid)) / n_hid ))
        i_hid=$(( K % n_hid ))
        REC=${REC_LAYERS_VALS[$i_rec]}; GEN=${GEN_LAYERS_VALS[$i_gen]}; HID=${HID_VALS[$i_hid]}
        ARCH_HPS_JSON="{\"rec_layers\": ${REC}, \"gen_layers\": ${GEN}, \"latents\": ${HID}, \"rec_dims\": ${HID}, \"gen_dims\": ${HID}}"
        TAG="rec${REC}_gen${GEN}_h${HID}"
        ;;
    CRU)
        n_bw=${#BANDWIDTH_VALS[@]}; n_b=${#BIAS_VALS[@]}
        i_sf=$(( K / (n_bw * n_b) ))
        i_bw=$(( (K % (n_bw * n_b)) / n_b ))
        i_b=$(( K % n_b ))
        SF=${SCALE_VALS[$i_sf]}; BW=${BANDWIDTH_VALS[$i_bw]}; B=${BIAS_VALS[$i_b]}
        ARCH_HPS_JSON="{\"ts\": ${SF}, \"bandwidth\": ${BW}, \"num_basis\": ${B}}"
        TAG="sf${SF}_bw${BW}_b${B}"
        ;;
    NeuralFlow)
        n_tn=${#TIME_NET_VALS[@]}; n_rd=${#REC_DIMS_VALS[@]}
        i_fm=$(( K / (n_tn * n_rd) ))
        i_tn=$(( (K % (n_tn * n_rd)) / n_rd ))
        i_rd=$(( K % n_rd ))
        FM=${FLOW_MODEL_VALS[$i_fm]}; TN=${TIME_NET_VALS[$i_tn]}; RD=${REC_DIMS_VALS[$i_rd]}
        ARCH_HPS_JSON="{\"flow_model\": \"${FM}\", \"time_net\": \"${TN}\", \"rec_dims\": ${RD}}"
        TAG="fm${FM}_tn${TN}_rd${RD}"
        ;;
esac

OUT_DIR=${LOG_BASE}/${MODEL}/${DS_NAME}/${TAG}/seed${SEED_HPO}
mkdir -p "${OUT_DIR}"

module load miniforge3-python 2>/dev/null || true; eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
export PYTHONNOUSERSITE=1

echo "==========================================="
echo "IMM-TSF HPO  model=${MODEL}  dataset=${DATASET} (${DS_NAME})"
echo "  cell ${TASK_ID}/${N_CELLS}  TAG=${TAG}  ARCH=${ARCH_HPS_JSON}"
echo "  lr=${LR}  patience=${PATIENCE}  bs=${BATCH_SIZE}  seed=${SEED_HPO}"
echo "  out=${OUT_DIR}"
echo "  job=${SLURM_JOB_ID}  array=${SLURM_ARRAY_TASK_ID}  host=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
date
echo "==========================================="

cd "${CODE_DIR}"
START=$(date +%s)

python imts_benchmark/scripts/run_immtsf_baseline.py \
    --imm_tsf_repo "${IMM_TSF_REPO}" \
    --time_imm_data "${TIME_IMM_DATA}" \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --lr ${LR} \
    --patience ${PATIENCE} \
    --batch_size ${BATCH_SIZE} \
    --seed ${SEED_HPO} \
    --max_epochs 1000 \
    --arch_hps "${ARCH_HPS_JSON}" \
    --output_dir "${OUT_DIR}" \
    --output_json "${OUT_DIR}/result.json"

EXIT=$?
END=$(date +%s)
echo "Exit code: ${EXIT}  Wall: $((END-START))s  Done: $(date)"
exit ${EXIT}
