#!/bin/bash
# MedAgent/scripts/batch_eval.sh
# 批量评测多个模型

set -e

PROJECT_ROOT="/home/jiazixiao.jzx/MedAgent"
cd $PROJECT_ROOT

MODELS=("qwen3.5-plus" "glm-5" "kimi-k2.5" "MiniMax-M2.5")
JUDGE_MODEL="qwen3.5-plus"  # Judge 模型固定，独立于被评测模型
N_CASES=100
OUTPUT_DIR="results/eval"
KB_PATH="data/knowledge_db"
DATA_PATH="output/generate_2000/merged_selected.jsonl"

mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "MedAgent 批量评测"
echo "=========================================="
echo "模型列表: ${MODELS[*]}"
echo "Judge 模型: ${JUDGE_MODEL}"
echo "每个模型评测: ${N_CASES} 条"
echo "并发线程数: 16"
echo "知识库: ${KB_PATH}"
echo "数据: ${DATA_PATH}"
echo "=========================================="

for model in "${MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "开始评测模型: ${model}"
    echo "=========================================="

    # 清理旧的 checkpoint（从头开始）
    rm -f ${OUTPUT_DIR}/${model}/checkpoint_${model}.jsonl

    start_time=$(date +%s)

    python scripts/eval.py \
        --model ${model} \
        --judge-model ${JUDGE_MODEL} \
        --n ${N_CASES} \
        --kb ${KB_PATH} \
        --data ${DATA_PATH} \
        --output ${OUTPUT_DIR}/${model} \
        --max-steps 10

    end_time=$(date +%s)
    elapsed=$((end_time - start_time))

    echo ""
    echo "模型 ${model} 评测完成"
    echo "耗时: ${elapsed} 秒 ($(($elapsed / 60)) 分钟)"
done

echo ""
echo "=========================================="
echo "所有模型评测完成!"
echo "=========================================="

# 汇总结果
echo ""
echo "=== 结果汇总 ==="
for model in "${MODELS[@]}"; do
    echo ""
    echo "--- ${model} ---"
    # 查找最新的报告文件
    latest_report=$(ls -t ${OUTPUT_DIR}/${model}/eval_summary_*.txt 2>/dev/null | head -1)
    if [ -f "$latest_report" ]; then
        cat "$latest_report"
    else
        echo "报告未找到"
    fi
done