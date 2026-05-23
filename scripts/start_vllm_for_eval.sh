#!/bin/bash
# 启动 vLLM 服务评测 1000步训练后的 4个模型
#
# 使用方法:
#   bash scripts/start_vllm_for_eval.sh [GPU_ID]
#
# GPU_ID: 指定使用哪张GPU (默认0)
#
# 注意: 4个模型需要分4次启动服务并分别评测
#      因为 vLLM 单服务只能加载有限数量的 LoRA

set -e

GPU_ID="${GPU_ID:-0}"
BASE_MODEL="/dev/shm/model/Qwen3-4B"
PORT="${PORT:-8000}"
LOG_DIR="output/agentic_rl/rl_eval_logs"

# 1000步训练后的LoRA模型
LORA_MODELS=(
    "medical_only"
    "medical_cost"
    "medical_efficiency"
    "medical_efficiency_cost"
)

echo "============================================================"
echo "vLLM 服务启动脚本 (评测 1000步模型)"
echo "============================================================"
echo "基础模型: $BASE_MODEL"
echo "GPU: $GPU_ID"
echo "端口: $PORT"
echo "日志目录: $LOG_DIR"
echo "============================================================"
echo ""
echo "可用 LoRA 模型:"
for i in "${!LORA_MODELS[@]}"; do
    echo "  [$i] ${LORA_MODELS[$i]} -> output/agentic_rl/${LORA_MODELS[$i]}"
done
echo ""
echo "============================================================"
echo "使用说明:"
echo "============================================================"
echo ""
echo "方式1: 启动单个LoRA服务"
echo "  bash scripts/start_vllm_for_eval.sh 0"
echo "  # 然后选择要评测的模型编号"
echo ""
echo "方式2: 同时加载所有LoRA (如果显存足够)"
echo "  bash scripts/start_vllm_for_eval.sh 0 --all-loras"
echo ""
echo "============================================================"

# 检查参数
if [[ "$1" == "--all-loras" ]]; then
    # 加载所有LoRA
    LORA_ARGS=""
    for model in "${LORA_MODELS[@]}"; do
        LORA_ARGS="${LORA_ARGS} --lora-modules name=qwen3-4b-${model},path=output/agentic_rl/${model}"
    done

    echo "启动 vLLM 服务 (加载所有4个LoRA)..."
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${LOG_DIR}/vllm_gpu${GPU_ID}_all_loras_1000steps_${TIMESTAMP}.log"

    mkdir -p "$LOG_DIR"

    CUDA_VISIBLE_DEVICES=$GPU_ID vllm serve "$BASE_MODEL" \
        --port $PORT \
        --enable-lora \
        --max-lora-rank 16 \
        --max-loras 4 \
        $LORA_ARGS \
        --gpu-memory-utilization 0.9 \
        --max-model-len 4096 \
        --trust-remote-code \
        > "$LOG_FILE" 2>&1 &

    VLLM_PID=$!
    echo "vLLM 服务已启动 (PID: $VLLM_PID)"
    echo "日志: $LOG_FILE"
    echo ""
    echo "等待服务就绪..."
    sleep 10

    # 检查服务状态
    if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
        echo "✅ 服务已就绪!"
        echo ""
        echo "可用模型:"
        curl -s http://localhost:$PORT/v1/models | python -c "import json,sys; d=json.load(sys.stdin); [print(f'  - {m[\"id\"]}') for m in d['data']]"
        echo ""
        echo "评测命令示例:"
        echo "  python exp/benchmark.py --models qwen3-4b-medical-only --base-url http://localhost:$PORT/v1 --data exp/data/benchmark_1000.jsonl --n 1000"
    else
        echo "⚠️ 服务可能未完全就绪，请检查日志"
    fi

else
    # 选择单个LoRA
    echo "请选择要评测的模型编号 (0-3):"
    read -r MODEL_IDX

    if [[ ! "$MODEL_IDX" =~ ^[0-3]$ ]]; then
        echo "错误: 请输入 0-3 之间的数字"
        exit 1
    fi

    MODEL_NAME="${LORA_MODELS[$MODEL_IDX]}"
    echo "选择模型: $MODEL_NAME"

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${LOG_DIR}/vllm_gpu${GPU_ID}_${MODEL_NAME}_1000steps_${TIMESTAMP}.log"

    mkdir -p "$LOG_DIR"

    CUDA_VISIBLE_DEVICES=$GPU_ID vllm serve "$BASE_MODEL" \
        --port $PORT \
        --enable-lora \
        --max-lora-rank 16 \
        --max-loras 1 \
        --lora-modules name=qwen3-4b-${MODEL_NAME},path=output/agentic_rl/${MODEL_NAME} \
        --gpu-memory-utilization 0.9 \
        --max-model-len 4096 \
        --trust-remote-code \
        > "$LOG_FILE" 2>&1 &

    VLLM_PID=$!
    echo "vLLM 服务已启动 (PID: $VLLM_PID)"
    echo "日志: $LOG_FILE"
    echo ""
    echo "等待服务就绪..."
    sleep 10

    # 检查服务状态
    for i in {1..30}; do
        if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
            echo "✅ 服务已就绪!"
            break
        fi
        echo "等待中... ($i/30)"
        sleep 2
    done

    echo ""
    echo "评测命令:"
    echo "  python exp/benchmark.py --models qwen3-4b-${MODEL_NAME} --base-url http://localhost:$PORT/v1 --data exp/data/benchmark_1000.jsonl --n 1000 --output exp/output/${MODEL_NAME}_1000steps"
    echo ""
    echo "停止服务:"
    echo "  kill $VLLM_PID"
fi