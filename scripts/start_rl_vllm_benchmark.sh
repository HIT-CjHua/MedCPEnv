#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VLLM_BIN="/home/huachangjie.hcj/.conda/envs/vllm_env/bin/vllm"
BASE_MODEL="/dev/shm/model/Qwen3-4B"
LOG_DIR="${LOG_DIR:-output/agentic_rl/benchmark_logs}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"

mkdir -p "$LOG_DIR"

run_service() {
  local gpu="$1"
  local port="$2"
  local name="$3"
  local lora_path="$4"
  local log_file="$5"
  local pid_file="$6"
  local lora_json

  lora_json="{\"name\":\"${name}\",\"path\":\"${lora_path}\"}"

  echo "$$" > "$pid_file"
  exec env \
    PYTHONNOUSERSITE=1 \
    CUDA_VISIBLE_DEVICES="$gpu" \
    VLLM_USE_DEEP_GEMM=0 \
    VLLM_MOE_USE_DEEP_GEMM=0 \
    VLLM_DEEP_GEMM_WARMUP=skip \
    "$VLLM_BIN" serve "$BASE_MODEL" \
      --host 0.0.0.0 \
      --port "$port" \
      --enable-lora \
      --max-lora-rank 16 \
      --max-loras 1 \
      --lora-modules "$lora_json" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --trust-remote-code \
      > "$log_file" 2>&1
}

if [[ "${1:-}" == "run" ]]; then
  shift
  run_service "$@"
  exit 0
fi

start_service() {
  local gpu="$1"
  local port="$2"
  local name="$3"
  local lora_path="$4"
  local log_file="$5"
  local pid_file="$6"

  setsid bash "$0" run "$gpu" "$port" "$name" "$lora_path" "$log_file" "$pid_file" &
  sleep 0.2
  echo "started ${name} on GPU ${gpu}, port ${port}, pid $(cat "$pid_file")"
}

start_service 0 8100 qwen3-4b-medical-only \
  output/agentic_rl/checkpoint/medical_only/checkpoint-1000 \
  "$LOG_DIR/vllm_medical_only_8100.log" \
  "$LOG_DIR/vllm_medical_only_8100.pid"

start_service 1 8101 qwen3-4b-medical-cost \
  output/agentic_rl/checkpoint/medical_cost/checkpoint-1000 \
  "$LOG_DIR/vllm_medical_cost_8101.log" \
  "$LOG_DIR/vllm_medical_cost_8101.pid"

start_service 2 8102 qwen3-4b-medical-efficiency \
  output/agentic_rl/checkpoint/medical_efficiency/checkpoint-1000 \
  "$LOG_DIR/vllm_medical_efficiency_8102.log" \
  "$LOG_DIR/vllm_medical_efficiency_8102.pid"

start_service 3 8103 qwen3-4b-medical-efficiency-cost \
  output/agentic_rl/checkpoint/medical_efficiency_cost/checkpoint-1000 \
  "$LOG_DIR/vllm_medical_efficiency_cost_8103.log" \
  "$LOG_DIR/vllm_medical_efficiency_cost_8103.pid"
