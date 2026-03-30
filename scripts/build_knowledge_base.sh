#!/bin/bash
# MedAgent/scripts/build_knowledge_base.sh
# 启动 embedding 服务，然后构建知识库（不做精排）

set -e

PROJECT_ROOT="/home/jiazixiao.jzx/MedAgent"
LOGS_DIR="$PROJECT_ROOT/logs"
EMBEDDING_PORT=8300
GPU_ID=1
GPU_MEMORY=0.8

mkdir -p "$LOGS_DIR"

echo "=========================================="
echo "MedAgent 知识库构建脚本 (仅 Embedding)"
echo "=========================================="

# 清理旧进程
echo "[1/3] 清理旧服务进程..."
pkill -f "vllm serve.*:$EMBEDDING_PORT" 2>/dev/null || true
sleep 2

# 启动 Embedding 服务
echo "[2/3] 启动 Embedding 服务 (端口 $EMBEDDING_PORT)..."
CUDA_VISIBLE_DEVICES=$GPU_ID nohup vllm serve /dev/shm/models/Qwen/Qwen3-Embedding-4B \
    --port $EMBEDDING_PORT \
    --served-model-name Qwen3-embedding \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization $GPU_MEMORY \
    > "$LOGS_DIR/embedding_server.log" 2>&1 &

EMBEDDING_PID=$!
echo "Embedding PID: $EMBEDDING_PID"

# 等待服务就绪
echo "[3/3] 等待服务启动..."
wait_for_server() {
    local port=$1
    local name=$2
    local max_wait=180
    local waited=0

    while [ $waited -lt $max_wait ]; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "✓ $name 服务就绪 (端口 $port)"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
        echo "  等待 $name... ($waited/$max_wait 秒)"
    done

    echo "✗ $name 服务启动超时"
    return 1
}

wait_for_server $EMBEDDING_PORT "Embedding"

# 构建知识库
echo ""
echo "开始构建知识库..."
cd "$PROJECT_ROOT"
python scripts/rag.py --build \
    --data data/knowledge_dataset/ResponseMed.json \
    --db data/knowledge_db

echo ""
echo "=========================================="
echo "知识库构建完成!"
echo "=========================================="
echo "服务进程:"
echo "  Embedding PID: $EMBEDDING_PID"
echo ""
echo "日志文件:"
echo "  $LOGS_DIR/embedding_server.log"
echo ""
echo "知识库位置: $PROJECT_ROOT/data/knowledge_db"
echo "=========================================="