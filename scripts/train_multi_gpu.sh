#!/bin/bash
# MedAgent Agentic RL Multi-GPU Training Launch Script
#
# 使用 accelerate 进行多卡并行训练
#
# 使用方法:
#   bash scripts/train_multi_gpu.sh [OPTIONS]
#
# 示例:
#   bash scripts/train_multi_gpu.sh --model /dev/shm/model/Qwen3-4B --max-steps 1000
#   bash scripts/train_multi_gpu.sh --config deepspeed --model /dev/shm/model/Qwen3-4B

set -e

# 默认配置
CONFIG="multi_gpu"  # multi_gpu | deepspeed_zero2 | deepspeed_zero3
NUM_GPUS=2
MODEL="/dev/shm/model/Qwen3-4B"
MAX_STEPS=500
BATCH_SIZE=4
DATA="data/train.jsonl"
DISABLE_KB=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --data)
            DATA="$2"
            shift 2
            ;;
        --disable-kb)
            DISABLE_KB=true
            shift
            ;;
        --help)
            echo "Usage: bash scripts/train_multi_gpu.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config        Accelerate config: multi_gpu | deepspeed_zero2 | deepspeed_zero3 (default: multi_gpu)"
            echo "  --num-gpus      Number of GPUs (default: 2)"
            echo "  --model         Base model name or path (default: /dev/shm/model/Qwen3-4B)"
            echo "  --max-steps     Maximum training steps (default: 500)"
            echo "  --batch-size    Per-device batch size (default: 4)"
            echo "  --data          Training data path (default: data/train.jsonl)"
            echo "  --disable-kb    Disable knowledge base (use simulated responses)"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 选择配置文件
CONFIG_FILE="scripts/accelerate_configs/${CONFIG}.yaml"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# 动态调整 num_processes
sed -i "s/num_processes: [0-9]/num_processes: $NUM_GPUS/" "$CONFIG_FILE"

echo "============================================================"
echo "MedAgent Agentic RL Multi-GPU Training"
echo "============================================================"
echo "Config:     $CONFIG"
echo "Config File: $CONFIG_FILE"
echo "Num GPUs:   $NUM_GPUS"
echo "Model:      $MODEL"
echo "Max Steps:  $MAX_STEPS"
echo "Batch Size: $BATCH_SIZE (per device)"
echo "Data:       $DATA"
echo "Disable KB: $DISABLE_KB"
echo "============================================================"

# 构建额外参数
EXTRA_ARGS=""
if [[ "$DISABLE_KB" == "true" ]]; then
    EXTRA_ARGS="--disable-kb"
fi

# 启动训练
accelerate launch \
    --config_file "$CONFIG_FILE" \
    scripts/agentic_rl.py \
    --model "$MODEL" \
    --data "$DATA" \
    --max-steps "$MAX_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --use-lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --gradient-accumulation-steps 4 \
    --output-dir "output/agentic_rl_multi_gpu" \
    $EXTRA_ARGS

echo "============================================================"
echo "Training completed!"
echo "============================================================"
