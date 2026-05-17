#!/bin/bash

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <MODEL_TYPE> <GPU_ID> [Config] [Datasets] [Shots]"
    echo "Supported models: PromptFL, FedOPT, FedTPG, FedMGP, FedPGP, IVLP, VPT, PromptFolio"
    echo "Config: default base2novel_vit_b16"
    echo "Datasets: default 'oxford_pets dtd oxford_flowers food101 caltech101'"
    echo "Shots: default '16'"
    echo "Example: $0 FedTPG 8 base2novel_vit_b16 oxford_flowers 4"
    exit 1
fi

MODEL_TYPE=$1
GPU=$2
CONFIG_FILE=${3:-"base2novel_vit_b16"}

DEFAULT_DATASETS="oxford_pets dtd oxford_flowers food101 caltech101"
DEFAULT_SHOTS="16"

CUSTOM_DATASETS=${4:-"$DEFAULT_DATASETS"}
CUSTOM_SHOTS=${5:-"$DEFAULT_SHOTS"}

IFS=' ' read -r -a DATASETS <<< "$CUSTOM_DATASETS"
IFS=' ' read -r -a SHOTS_LIST <<< "$CUSTOM_SHOTS"

# Validate model type
VALID_MODELS=("PromptFL" "FedOPT" "FedTPG" "FedMGP" "FedPGP" "IVLP" "VPT" "PromptFolio")
VALID_MODEL=false

for model in "${VALID_MODELS[@]}"; do
    if [ "$MODEL_TYPE" == "$model" ]; then
        VALID_MODEL=true
        break
    fi
done

if [ "$VALID_MODEL" == "false" ]; then
    echo "Error: Invalid model type '$MODEL_TYPE'"
    echo "Supported models: ${VALID_MODELS[*]}"
    exit 1
fi

check_output_dir() {
    local dataset=$1
    local model=$2
    local config=$3
    local shots=$4
    local phase=$5

    local output_dir="output/${dataset}/${model}/${config}/${shots}shots"

    if [ -d "$output_dir" ]; then
        if [ "$phase" == "train" ]; then
            echo "Warning: Output directory '$output_dir' exists, will overwrite"
        else
            echo "Warning: Output directory '$output_dir' exists, will overwrite"
        fi
        return 0
    else
        mkdir -p "$(dirname "$output_dir")"
        return 1
    fi
}

echo "====================================================="
echo "Starting training and testing on specified datasets"
echo "Model: ${MODEL_TYPE}"
echo "GPU: ${GPU}"
echo "Config: ${CONFIG_FILE}"
echo "Shots: ${SHOTS_LIST[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "====================================================="

SCRIPT_DIR="scripts/${MODEL_TYPE}"
if [ ! -d "$SCRIPT_DIR" ]; then
    echo "Error: Script directory '$SCRIPT_DIR' not found"
    exit 1
fi

TRAIN_SCRIPT="${SCRIPT_DIR}/base2novel_train.sh"
TEST_SCRIPT="${SCRIPT_DIR}/base2novel_test.sh"

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: Train script '$TRAIN_SCRIPT' not found"
    exit 1
fi

if [ ! -f "$TEST_SCRIPT" ]; then
    echo "Error: Test script '$TEST_SCRIPT' not found"
    exit 1
fi

chmod +x "$TRAIN_SCRIPT" "$TEST_SCRIPT"

for SHOTS in "${SHOTS_LIST[@]}"; do
    echo ""
    echo "====================================================="
    echo "Processing SHOTS: ${SHOTS}"
    echo "====================================================="

    for DATASET in "${DATASETS[@]}"; do
        echo ""
        echo "====================================================="
        echo "Processing dataset: ${DATASET} with ${SHOTS} shots"
        echo "====================================================="

        echo ""
        echo "1. Training on base classes for ${DATASET} (${SHOTS} shots)..."
        check_output_dir "$DATASET" "$MODEL_TYPE" "$CONFIG_FILE" "$SHOTS" "train"
        bash "$TRAIN_SCRIPT" ${GPU} ${DATASET} ${SHOTS} ${CONFIG_FILE}

        echo ""
        echo "2. Testing on base classes for ${DATASET} (${SHOTS} shots)..."
        check_output_dir "$DATASET" "$MODEL_TYPE" "$CONFIG_FILE" "$SHOTS" "test_base"
        bash "$TEST_SCRIPT" ${GPU} ${DATASET} base ${SHOTS} ${CONFIG_FILE}

        echo ""
        echo "3. Testing on new classes for ${DATASET} (${SHOTS} shots)..."
        check_output_dir "$DATASET" "$MODEL_TYPE" "$CONFIG_FILE" "$SHOTS" "test_new"
        bash "$TEST_SCRIPT" ${GPU} ${DATASET} new ${SHOTS} ${CONFIG_FILE}

        echo ""
        echo "${DATASET} with ${SHOTS} shots completed"
    done

    echo ""
    echo "All datasets with ${SHOTS} shots completed"
done

echo ""
echo "====================================================="
echo "All datasets completed"
echo "====================================================="

# if [ -f "/ifs/root/ipa01/108/user_108006/get_gpu.py" ]; then
#     python /ifs/root/ipa01/108/user_108006/get_gpu.py ${GPU}
# fi
