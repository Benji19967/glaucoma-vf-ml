MODEL_NAME=$1

DEFAULT_MODEL_NAME=hvf_classifier

# 1. Set model name from args or use default
if [ -z "${MODEL_NAME}" ]; then
    export MODEL_NAME=${DEFAULT_MODEL_NAME}
else
    export MODEL_NAME=${MODEL_NAME}
fi

echo "🚀 Training Model: $MODEL_NAME"

python cli/main.py fit \
    --config=configs/train/${MODEL_NAME}.yaml \
    --trainer.logger.init_args.name=${MODEL_NAME}
