#!/bin/bash
# Train Agentic RL with the mixed quality/cost/efficiency reward.
# This script intentionally calls scripts/agentic_rl_mixed_reward.py, not
# scripts/agentic_rl.py, so the original training entrypoint stays untouched.

set -euo pipefail

MODEL="${MODEL:-/dev/shm/model/Qwen3-4B}"
DATA="${DATA:-data/datasets/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/agentic_rl/medical_mixed_reward}"
MAX_STEPS="${MAX_STEPS:-700}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-5}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_GPUS="${NUM_GPUS:-4}"
CONFIG_FILE="${CONFIG_FILE:-scripts/accelerate_configs/multi_gpu.yaml}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29610}"
PYDEPS_DIR="${PYDEPS_DIR:-/tmp/medagent_pydeps}"
USE_VLLM="${USE_VLLM:-0}"
VLLM_ENV="${VLLM_ENV:-/home/huachangjie.hcj/.conda/envs/vllm_env}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

QUALITY_GATE_DIAGNOSIS="${QUALITY_GATE_DIAGNOSIS:-3.0}"
QUALITY_GATE_TREATMENT="${QUALITY_GATE_TREATMENT:-2.0}"
EXAM_PENALTY_THRESHOLD="${EXAM_PENALTY_THRESHOLD:-3}"
TOOL_CALL_PENALTY_THRESHOLD="${TOOL_CALL_PENALTY_THRESHOLD:-6}"
TOKEN_PENALTY_THRESHOLD="${TOKEN_PENALTY_THRESHOLD:-900}"
EXAM_PENALTY_SCALE="${EXAM_PENALTY_SCALE:-0.1}"
TOOL_CALL_PENALTY_SCALE="${TOOL_CALL_PENALTY_SCALE:-0.1}"
TOKEN_PENALTY_SCALE="${TOKEN_PENALTY_SCALE:-0.05}"
MIXED_COST_BONUS_SCALE="${MIXED_COST_BONUS_SCALE:-0.1}"
MIXED_EFFICIENCY_BONUS_SCALE="${MIXED_EFFICIENCY_BONUS_SCALE:-0.1}"

export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if [ "$USE_VLLM" = "1" ]; then
    PYTHON_BIN="${VLLM_ENV}/bin/python"
    VLLM_SITE="${VLLM_ENV}/lib/python3.10/site-packages"
    export PYTHONPATH="${VLLM_SITE}:${PYTHONPATH:-}"
    export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
    export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
    export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"
elif [ -d "$PYDEPS_DIR" ]; then
    export PYTHONPATH="${PYDEPS_DIR}:${PYTHONPATH:-}"
fi

mkdir -p "$OUTPUT_DIR/logs"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/logs/train_mixed_reward_$(date +%Y%m%d_%H%M%S).log}"

extra_args=()
if [ "$USE_VLLM" = "1" ]; then
    extra_args+=(--use-vllm)
fi
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    extra_args+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

echo "============================================================"
echo "MedAgent Agentic RL Mixed Reward 4-GPU Training"
echo "Script:     scripts/agentic_rl_mixed_reward.py"
echo "Model:      ${MODEL}"
echo "Data:       ${DATA}"
echo "Output:     ${OUTPUT_DIR}"
echo "Max steps:  ${MAX_STEPS}"
echo "Batch size: ${BATCH_SIZE} per device"
echo "Grad accum: ${GRAD_ACCUM}"
echo "GPUs:       ${CUDA_VISIBLE_DEVICES}"
echo "Config:     ${CONFIG_FILE}"
echo "Use vLLM:   ${USE_VLLM}"
echo "Python:     ${PYTHON_BIN}"
echo "Log:        ${LOG_FILE}"
echo "============================================================"

"$PYTHON_BIN" -m accelerate.commands.launch \
    --config_file "$CONFIG_FILE" \
    --num_processes "$NUM_GPUS" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    scripts/agentic_rl_mixed_reward.py \
    --model "$MODEL" \
    --data "$DATA" \
    --output-dir "$OUTPUT_DIR" \
    --max-steps "$MAX_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LR" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --use-lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --quality-gate-diagnosis "$QUALITY_GATE_DIAGNOSIS" \
    --quality-gate-treatment "$QUALITY_GATE_TREATMENT" \
    --exam-penalty-threshold "$EXAM_PENALTY_THRESHOLD" \
    --tool-call-penalty-threshold "$TOOL_CALL_PENALTY_THRESHOLD" \
    --token-penalty-threshold "$TOKEN_PENALTY_THRESHOLD" \
    --exam-penalty-scale "$EXAM_PENALTY_SCALE" \
    --tool-call-penalty-scale "$TOOL_CALL_PENALTY_SCALE" \
    --token-penalty-scale "$TOKEN_PENALTY_SCALE" \
    --mixed-cost-bonus-scale "$MIXED_COST_BONUS_SCALE" \
    --mixed-efficiency-bonus-scale "$MIXED_EFFICIENCY_BONUS_SCALE" \
    "${extra_args[@]}" \
    > "$LOG_FILE" 2>&1

echo "[DONE] mixed reward training completed"
