#!/bin/bash
# MedAgent 批量评测脚本
# 配置：数据量 100, 最大轮次 10, topk 5

set -e

MODELS=("qwen3.5-plus" "glm-5" "kimi-k2.5" "MiniMax-M2.5")
DATA="output/generate_2000/sampled_300.jsonl"
N=300
MAX_STEPS=10
TOPK=5

echo "=============================================="
echo "MedAgent 批量评测"
echo "=============================================="
echo "待测模型: ${MODELS[*]}"
echo "数据量: $N"
echo "最大轮次: $MAX_STEPS"
echo "Top-K: $TOPK"
echo "=============================================="

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "=============================================="
    echo "开始测试模型: $MODEL"
    echo "=============================================="

    # 删除旧的 checkpoint 文件
    CHECKPOINT_FILE="results/eval/checkpoint_${MODEL}.jsonl"
    if [ -f "$CHECKPOINT_FILE" ]; then
        echo "删除旧 checkpoint: $CHECKPOINT_FILE"
        rm "$CHECKPOINT_FILE"
    fi

    python scripts/eval.py \
        --data "$DATA" \
        --model "$MODEL" \
        --n $N \
        --max-steps $MAX_STEPS \
        --topk $TOPK \
        --output "results/eval/${MODEL}"

    echo "模型 $MODEL 测试完成"
done

echo ""
echo "=============================================="
echo "所有模型测试完成!"
echo "=============================================="