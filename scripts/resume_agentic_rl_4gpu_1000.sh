#!/bin/bash
# Resume the four Agentic RL reward ablations from checkpoint-500 to step 1000
# using all four local GPUs through accelerate.

set -euo pipefail

MODEL="${MODEL:-/dev/shm/model/Qwen3-4B}"
DATA="${DATA:-data/datasets/train.jsonl}"
OUTPUT_BASE="${OUTPUT_BASE:-output/agentic_rl}"
MAX_STEPS="${MAX_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-5}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_GPUS="${NUM_GPUS:-4}"
CONFIG_FILE="${CONFIG_FILE:-scripts/accelerate_configs/multi_gpu.yaml}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29600}"
PYDEPS_DIR="${PYDEPS_DIR:-/tmp/medagent_pydeps}"
EXPERIMENTS="${EXPERIMENTS:-medical_only,medical_cost,medical_efficiency,medical_efficiency_cost}"
USE_VLLM="${USE_VLLM:-0}"
VLLM_ENV="${VLLM_ENV:-/home/huachangjie.hcj/.conda/envs/vllm_env}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESUME_FROM_LATEST="${RESUME_FROM_LATEST:-1}"

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

LOG_DIR="${OUTPUT_BASE}/logs"
mkdir -p "$LOG_DIR"

latest_checkpoint() {
    local output_dir="$1"
    find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' \
        | sort -V \
        | tail -n 1
}

run_experiment() {
    local exp_name="$1"
    local enable_efficiency="$2"
    local enable_cost="$3"
    local output_dir="${OUTPUT_BASE}/${exp_name}"
    local checkpoint_dir="${output_dir}/checkpoint-500"
    local log_suffix="resume_500_to_${MAX_STEPS}"

    if [ "$RESUME_FROM_LATEST" = "1" ]; then
        checkpoint_dir="$(latest_checkpoint "$output_dir")"
        log_suffix="resume_latest_to_${MAX_STEPS}"
    fi

    local log_file="${LOG_DIR}/${exp_name}_${log_suffix}.log"

    if [ ! -d "$checkpoint_dir" ]; then
        echo "[ERROR] Missing checkpoint: ${checkpoint_dir}" >&2
        exit 1
    fi

    echo "============================================================"
    echo "Experiment: ${exp_name}"
    echo "Output:     ${output_dir}"
    echo "Resume:     ${checkpoint_dir}"
    echo "Log:        ${log_file}"
    echo "============================================================"

    local extra_args=()
    if [ "$enable_efficiency" != "true" ]; then
        extra_args+=(--disable-tool-efficiency)
    fi
    if [ "$enable_cost" = "true" ]; then
        extra_args+=(--enable-cost-reward)
    fi
    if [ "$USE_VLLM" = "1" ]; then
        extra_args+=(--use-vllm)
    fi

    "$PYTHON_BIN" -m accelerate.commands.launch \
        --config_file "$CONFIG_FILE" \
        --num_processes "$NUM_GPUS" \
        --main_process_port "$MAIN_PROCESS_PORT" \
        scripts/agentic_rl.py \
        --model "$MODEL" \
        --data "$DATA" \
        --output-dir "$output_dir" \
        --max-steps "$MAX_STEPS" \
        --batch-size "$BATCH_SIZE" \
        --learning-rate "$LR" \
        --gradient-accumulation-steps "$GRAD_ACCUM" \
        --use-lora \
        --lora-r 16 \
        --lora-alpha 32 \
        --resume-from-checkpoint "$checkpoint_dir" \
        "${extra_args[@]}" \
        > "$log_file" 2>&1

    echo "[DONE] ${exp_name}"
}

echo "============================================================"
echo "MedAgent Agentic RL 4-GPU Resume Training"
echo "Model:      ${MODEL}"
echo "Data:       ${DATA}"
echo "Output:     ${OUTPUT_BASE}"
echo "Max steps:  ${MAX_STEPS}"
echo "Batch size: ${BATCH_SIZE} per device"
echo "Grad accum: ${GRAD_ACCUM}"
echo "GPUs:       ${CUDA_VISIBLE_DEVICES}"
echo "Config:     ${CONFIG_FILE}"
echo "Use vLLM:   ${USE_VLLM}"
echo "Python:     ${PYTHON_BIN}"
echo "Resume:     latest=${RESUME_FROM_LATEST}"
echo "============================================================"

IFS="," read -r -a EXPERIMENT_LIST <<< "$EXPERIMENTS"
for exp_name in "${EXPERIMENT_LIST[@]}"; do
    case "$exp_name" in
        medical_only)
            run_experiment "medical_only" "false" "false"
            ;;
        medical_cost)
            run_experiment "medical_cost" "false" "true"
            ;;
        medical_efficiency)
            run_experiment "medical_efficiency" "true" "false"
            ;;
        medical_efficiency_cost)
            run_experiment "medical_efficiency_cost" "true" "true"
            ;;
        *)
            echo "[ERROR] Unknown experiment: ${exp_name}" >&2
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "All experiments completed."
echo "============================================================"
