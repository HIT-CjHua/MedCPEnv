#!/bin/bash
# 模型部署与 Benchmark 测试脚本
#
# 用法:
#   # 测试 base 模型（无 LoRA）
#   bash scripts/model_test.sh --model-path /dev/shm/model/Qwen3-4B --n 100
#
#   # 测试 LoRA 微调模型
#   bash scripts/model_test.sh --model-path /dev/shm/model/Qwen3-4B --lora-path output/agentic_rl/medical_only --n 100
#
#   # 完整评测 (1000 条)
#   bash scripts/model_test.sh --model-path /dev/shm/model/Qwen3-4B --lora-path output/agentic_rl/medical_only --n 1000

set -e

# ============================ 默认配置 ============================
MODEL_PATH=""
LORA_PATH=""
N=100
DATA="exp/data/benchmark_1000.jsonl"
DEPLOY_PORT=30002
MODEL_NAME=""

# ============================ 解析参数 ============================
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --lora-path)
            LORA_PATH="$2"
            shift 2
            ;;
        --n)
            N="$2"
            shift 2
            ;;
        --data)
            DATA="$2"
            shift 2
            ;;
        --port)
            DEPLOY_PORT="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: bash scripts/model_test.sh --model-path <path> [--lora-path <path>] [--n <num>] [--model-name <name>]"
            exit 1
            ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "错误: 必须指定 --model-path"
    echo "用法: bash scripts/model_test.sh --model-path <path> [--lora-path <path>] [--n <num>] [--model-name <name>]"
    exit 1
fi

# 设置模型名称
if [ -z "$MODEL_NAME" ]; then
    # 从 model-path 提取模型名称
    MODEL_NAME=$(basename "$MODEL_PATH")
    echo "自动设置模型名称: $MODEL_NAME"
fi

# # ============================ 部署模型 ============================
# echo "============================================================"
# echo "模型部署"
# echo "============================================================"
# echo "Base 模型: ${MODEL_PATH}"
# if [ -n "$LORA_PATH" ]; then
#     echo "LoRA 适配器: ${LORA_PATH}"
#     DEPLOY_CMD="CUDA_VISIBLE_DEVICES=1 vllm serve ${MODEL_PATH} --lora-modules lora=${LORA_PATH} --port ${DEPLOY_PORT} --enable-lora --max-lora-rank 16"
# else
#     echo "无 LoRA 适配器"
#     DEPLOY_CMD="CUDA_VISIBLE_DEVICES=1 vllm serve ${MODEL_PATH} --port ${DEPLOY_PORT}"
# fi
# echo "部署命令: ${DEPLOY_CMD}"
# echo ""

# # 启动 vLLM 服务（后台）
# eval ${DEPLOY_CMD} &
# VLLM_PID=$!
# echo "vLLM 服务已启动 (PID: ${VLLM_PID})"

# # 等待服务就绪
# echo "等待服务就绪..."
# for i in $(seq 1 60); do
#     if curl -s http://localhost:${DEPLOY_PORT}/v1/models > /dev/null 2>&1; then
#         echo "服务就绪!"
#         break
#     fi
#     if [ $i -eq 60 ]; then
#         echo "服务启动超时 (60s)"
#         kill $VLLM_PID 2>/dev/null
#         exit 1
#     fi
#     sleep 1
# done

# ============================ 运行 Benchmark ============================
echo ""
echo "============================================================"
echo "运行 Benchmark 评测"
echo "============================================================"

BENCHMARK_CMD="python exp/benchmark.py --models ${MODEL_NAME} --base-url http://localhost:${DEPLOY_PORT}/v1 --n ${N} --data ${DATA}"
echo "执行: ${BENCHMARK_CMD}"
eval ${BENCHMARK_CMD}

# ============================ 运行 Rejudge ============================
echo ""
echo "============================================================"
echo "运行 Judger 重测"
echo "============================================================"

REJUDGE_CMD="python exp/rejudge.py --models ${MODEL_NAME} --n ${N} --output exp/output"
echo "执行: ${REJUDGE_CMD}"
eval ${REJUDGE_CMD}

# # ============================ 清理 ============================
# echo ""
# echo "============================================================"
# echo "清理"
# echo "============================================================"
# echo "停止 vLLM 服务 (PID: ${VLLM_PID})..."
# kill $VLLM_PID 2>/dev/null
# wait $VLLM_PID 2>/dev/null
# echo "服务已停止"
# echo "============================================================"
# echo "测试完成!"
# echo "============================================================"
