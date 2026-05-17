#!/bin/bash

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <GPU_ID> <MODEL_TYPE> [Seeds] [Datasets] [Config]"
    echo "Supported models: PromptFL, FedOPT, FedTPG, FedMGP, FedPGP, IVLP, VPT, PromptFolio, FedCoCoOp, MaPLe"
    echo "Seeds: single (e.g., 0) or multiple (e.g., '0,1,2'), default: 0"
    echo "Datasets: single or multiple (e.g., 'caltech101,dtd'), default: caltech101,dtd,oxford_pets,oxford_flowers,food101"
    echo "Config: default base2novel_vit_b16"
    exit 1
fi
# bash scripts/run_multi_seeds.sh 1 PromptFL "1,2" "dtd"
# bash scripts/run_multi_seeds.sh 1 FedOPT "1,2" "dtd"
# bash scripts/run_multi_seeds.sh 2 FedTPG "1,2" "dtd"
# bash scripts/run_multi_seeds.sh 2 FedPGP "1,2" "dtd"
# bash scripts/run_multi_seeds.sh 3 FedMGP "1,2" "food101"
# bash scripts/run_multi_seeds.sh 4 FedMGP "1,2" "oxford_flowers"

GPU=$1
MODEL_TYPE=$2
SEEDS=${3:-"0"}
DATASETS_INPUT=${4:-"caltech101,dtd,oxford_pets,oxford_flowers,food101"}
CONFIG_FILE=${5:-"base2novel_vit_b16"}

# Convert space-separated datasets to comma-separated
if [[ "$DATASETS_INPUT" == *" "* ]]; then
    DATASETS_INPUT=$(echo "$DATASETS_INPUT" | tr ' ' ',')
fi

# Validate model type
VALID_MODELS=("PromptFL" "FedOPT" "FedTPG" "FedMGP" "FedPGP" "IVLP" "VPT" "PromptFolio" "FedCoCoOp" "MaPLe")
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

# Parse seeds and datasets
IFS=',' read -ra SEEDS_ARRAY <<< "$SEEDS"
IFS=',' read -ra DATASETS_ARRAY <<< "$DATASETS_INPUT"

SHOTS_LIST=(16)

# Check if output directory exists
check_output_dir() {
    local dataset=$1
    local model=$2
    local config=$3
    local shots=$4
    local seed=$5
    local phase=$6

    local output_dir="output/${dataset}/${model}/${config}/${shots}shots/seed${seed}"

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

# Modify SEED value in script
modify_seed_in_script() {
    local script_path=$1
    local new_seed=$2
    local temp_script="${script_path}.tmp"

    sed "s/SEED=0/SEED=${new_seed}/g" "$script_path" > "$temp_script"
    mv "$temp_script" "$script_path"
    chmod +x "$script_path"
}

# Restore SEED value in script
restore_seed_in_script() {
    local script_path=$1
    local temp_script="${script_path}.tmp"

    sed "s/SEED=[0-9]\+/SEED=0/g" "$script_path" > "$temp_script"
    mv "$temp_script" "$script_path"
    chmod +x "$script_path"
}

echo "====================================================="
echo "Starting training and testing on multiple datasets, seeds and shots"
echo "Model: ${MODEL_TYPE}"
echo "GPU: ${GPU}"
echo "Config: ${CONFIG_FILE}"
echo "Shots: ${SHOTS_LIST[*]}"
echo "Seeds: ${SEEDS_ARRAY[*]}"
echo "Datasets: ${DATASETS_ARRAY[*]}"
echo "====================================================="

# Check script directory
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

# Process each seed
for SEED in "${SEEDS_ARRAY[@]}"; do
    echo ""
    echo "====================================================="
    echo "Processing SEED: ${SEED}"
    echo "====================================================="

    modify_seed_in_script "$TRAIN_SCRIPT" "$SEED"
    modify_seed_in_script "$TEST_SCRIPT" "$SEED"

    for SHOTS in "${SHOTS_LIST[@]}"; do
        echo ""
        echo "====================================================="
        echo "Processing SHOTS: ${SHOTS} (SEED: ${SEED})"
        echo "====================================================="

        for DATASET in "${DATASETS_ARRAY[@]}"; do
            echo ""
            echo "====================================================="
            echo "Processing dataset: ${DATASET} with ${SHOTS} shots (SEED: ${SEED})"
            echo "====================================================="

            # 1. Train on base classes
            echo ""
            echo "1. Training on base classes for ${DATASET} (${SHOTS} shots, SEED: ${SEED})..."
            check_output_dir "$DATASET" "$MODEL_TYPE" "$CONFIG_FILE" "$SHOTS" "$SEED" "train"
            bash "$TRAIN_SCRIPT" ${GPU} ${DATASET} ${SHOTS} ${CONFIG_FILE}

            # 2. Test on base classes
            echo ""
            echo "2. Testing on base classes for ${DATASET} (${SHOTS} shots, SEED: ${SEED})..."
            check_output_dir "$DATASET" "$MODEL_TYPE" "$CONFIG_FILE" "$SHOTS" "$SEED" "test_base"
            bash "$TEST_SCRIPT" ${GPU} ${DATASET} base ${SHOTS} ${CONFIG_FILE}

            # 3. Test on new classes
            echo ""
            echo "3. Testing on new classes for ${DATASET} (${SHOTS} shots, SEED: ${SEED})..."
            check_output_dir "$DATASET" "$MODEL_TYPE" "$CONFIG_FILE" "$SHOTS" "$SEED" "test_new"
            bash "$TEST_SCRIPT" ${GPU} ${DATASET} new ${SHOTS} ${CONFIG_FILE}

            echo ""
            echo "${DATASET} with ${SHOTS} shots (SEED: ${SEED}) completed"
        done

        echo ""
        echo "All datasets with ${SHOTS} shots (SEED: ${SEED}) completed"
    done

    restore_seed_in_script "$TRAIN_SCRIPT"
    restore_seed_in_script "$TEST_SCRIPT"

    echo ""
    echo "SEED ${SEED} completed"
done

echo ""
echo "====================================================="
echo "All seeds, shots and datasets completed"
echo "====================================================="

# Release GPU
if [ -f "/ifs/root/ipa01/108/user_108006/get_gpu.py" ]; then
    python /ifs/root/ipa01/108/user_108006/get_gpu.py ${GPU}
fi
