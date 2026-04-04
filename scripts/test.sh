#!/usr/bin/env bash

MODEL_NAME=$1
VERSION=$2

DEFAULT_MODEL_NAME=hvf_system

# 1. Set model name from args or use default
if [ -z "${MODEL_NAME}" ]; then
    export MODEL_NAME=${DEFAULT_MODEL_NAME}
else
    export MODEL_NAME=${MODEL_NAME}
fi

LOG_DIR="logs/${MODEL_NAME}"

# 2. Resolve Version (if not provided, find the latest directory)
if [ -z "$VERSION" ]; then
    echo "No version specified. Searching for latest in $LOG_DIR..."
    # Sort numerically/alphabetically and take the last one
    VERSION=$(ls -1 "$LOG_DIR" | sort | tail -n 1)
    
    if [ -z "$VERSION" ]; then
        echo "Error: No versions found in $LOG_DIR"
        exit 1
    fi
fi


echo "🚀 Testing Model: $MODEL_NAME"
echo "📂 Version: $VERSION"

RESULTS_DIR=logs/${MODEL_NAME}/${VERSION}/test_results
mkdir -p $RESULTS_DIR
python cli/main.py test \
    --config=configs/${MODEL_NAME}/test.yaml \
    --ckpt_path=logs/${MODEL_NAME}/${VERSION}/checkpoints/best.ckpt \
    --trainer.logger.init_args.name=${MODEL_NAME} \
    --trainer.logger.init_args.version=${VERSION} > ${RESULTS_DIR}/results.txt
