#!/bin/bash
# Agentic RL 训练脚本 - 4 组实验编排
#
# 实验1: 只使用医疗奖励（诊断+治疗）
# 实验2: 医疗奖励 + 效率奖励
# 实验3: 医疗奖励 + cost奖励
# 实验4: 医疗奖励 + 效率奖励 + cost奖励

set -e

# ============================ 配置 ============================
MODEL="${MODEL:-/dev/shm/model/Qwen3-4B}"
DATA="${DATA:-data/datasets/train.jsonl}"
OUTPUT_BASE="${OUTPUT_BASE:-output/agentic_rl}"
MAX_STEPS="${MAX_STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-5}"
USE_VLLM="${USE_VLLM:-false}"
PYDEPS_DIR="${PYDEPS_DIR:-/tmp/medagent_pydeps}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF

if [ -d "$PYDEPS_DIR" ]; then
    export PYTHONPATH="${PYDEPS_DIR}:${PYTHONPATH:-}"
fi

# ============================ 工具函数 ============================
run_experiment() {
    local exp_name="$1"
    local enable_efficiency="$2"
    local enable_cost="$3"
    local output_dir="${OUTPUT_BASE}/${exp_name}"

    echo "============================================================"
    echo "实验: ${exp_name}"
    echo "输出目录: ${output_dir}"
    echo "============================================================"

    CMD="python scripts/agentic_rl.py \
        --model ${MODEL} \
        --data ${DATA} \
        --output-dir ${output_dir} \
        --max-steps ${MAX_STEPS} \
        --batch-size ${BATCH_SIZE} \
        --learning-rate ${LR} \
        --use-lora"

    if [ "$USE_VLLM" = "true" ]; then
        CMD="${CMD} --use-vllm"
    fi

    if [ "$enable_efficiency" = "true" ]; then
        CMD="${CMD}"
        # 默认启用 tool_efficiency
    else
        CMD="${CMD} --disable-tool-efficiency"
    fi

    if [ "$enable_cost" = "true" ]; then
        CMD="${CMD} --enable-cost-reward"
    fi

    echo "执行: ${CMD}"
    eval ${CMD}

    echo "============================================================"
    echo "实验 ${exp_name} 完成"
    echo "============================================================"
}

# ============================ 四组实验 ============================

# # 实验1: 只使用医疗奖励
# run_experiment "medical_only" "false" "false"

# # 实验2: 医疗奖励 + 效率奖励
# run_experiment "medical_efficiency" "true" "false"

# 实验3: 医疗奖励 + cost奖励
run_experiment "medical_cost" "false" "true"

# 实验4: 医疗奖励 + 效率奖励 + cost奖励
run_experiment "medical_efficiency_cost" "true" "true"

echo "============================================================"
echo "全部实验完成！"
echo "============================================================"
