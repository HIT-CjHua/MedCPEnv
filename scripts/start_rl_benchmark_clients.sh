#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_PATH="${DATA_PATH:-exp/data/benchmark_1000.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/agentic_rl}"
LOG_DIR="${LOG_DIR:-output/agentic_rl/benchmark_logs}"
N_CASES="${N_CASES:-1000}"
MAX_WORKERS="${MAX_WORKERS:-8}"
MAX_TOKENS="${MEDAGENT_MAX_TOKENS:-4096}"
PYTHON_BIN="${PYTHON_BIN:-/home/huachangjie.hcj/.conda/envs/vllm_env/bin/python}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

run_client() {
  local model_name="$1"
  local base_url="$2"
  local log_file="$3"
  local pid_file="$4"

  echo "$$" > "$pid_file"
  exec env MEDAGENT_MAX_TOKENS="$MAX_TOKENS" \
    "$PYTHON_BIN" exp/benchmark.py \
      --data "$DATA_PATH" \
      --output "$OUTPUT_DIR" \
      --models "$model_name" \
      --base-url "$base_url" \
      --api-key dummy \
      --n "$N_CASES" \
      --max-workers "$MAX_WORKERS" \
      --checkpoint-every 10 \
      > "$log_file" 2>&1
}

if [[ "${1:-}" == "run" ]]; then
  shift
  run_client "$@"
  exit 0
fi

start_client() {
  local model_name="$1"
  local base_url="$2"
  local log_file="$3"
  local pid_file="$4"

  setsid bash "$0" run "$model_name" "$base_url" "$log_file" "$pid_file" &
  sleep 0.2
  echo "started ${model_name}, pid $(cat "$pid_file"), log ${log_file}"
}

start_client qwen3-4b-medical-only \
  http://localhost:8100/v1 \
  "$LOG_DIR/bench_full_medical_only.log" \
  "$LOG_DIR/bench_full_medical_only.pid"

start_client qwen3-4b-medical-cost \
  http://localhost:8101/v1 \
  "$LOG_DIR/bench_full_medical_cost.log" \
  "$LOG_DIR/bench_full_medical_cost.pid"

start_client qwen3-4b-medical-efficiency \
  http://localhost:8102/v1 \
  "$LOG_DIR/bench_full_medical_efficiency.log" \
  "$LOG_DIR/bench_full_medical_efficiency.pid"

start_client qwen3-4b-medical-efficiency-cost \
  http://localhost:8103/v1 \
  "$LOG_DIR/bench_full_medical_efficiency_cost.log" \
  "$LOG_DIR/bench_full_medical_efficiency_cost.pid"
