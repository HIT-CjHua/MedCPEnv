#!/bin/bash
# MedAgent 分批评测脚本
# 每测完一个模型间隔2小时再测下一个

set -e

DATA="output/generate_2000/sampled_300.jsonl"
N=300
MAX_STEPS=10
TOPK=5
DELAY_SECONDS=7200  # 2小时 = 7200秒

# 剩余待测模型
MODELS=("glm-5" "kimi-k2.5" "MiniMax-M2.5")

echo "=============================================="
echo "MedAgent 分批评测"
echo "=============================================="
echo "待测模型: ${MODELS[*]}"
echo "数据量: $N"
echo "最大轮次: $MAX_STEPS"
echo "Top-K: $TOPK"
echo "模型间隔: 2小时"
echo "=============================================="

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "=============================================="
    echo "开始测试模型: $MODEL"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================="

    python scripts/eval.py \
        --data "$DATA" \
        --model "$MODEL" \
        --n $N \
        --max-steps $MAX_STEPS \
        --topk $TOPK \
        --output "results/eval/${MODEL}"

    echo ""
    echo "模型 $MODEL 测试完成"
    echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"

    # 如果不是最后一个模型,等待2小时
    if [ "$MODEL" != "${MODELS[-1]}" ]; then
        echo ""
        echo "=============================================="
        echo "等待 2 小时后继续下一个模型..."
        echo "下次测试时间: $(date -d '+2 hours' '+%Y-%m-%d %H:%M:%S')"
        echo "=============================================="
        sleep $DELAY_SECONDS
    fi
done

echo ""
echo "=============================================="
echo "所有模型测试完成!"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="