#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:?Please set MODEL_PATH (e.g. /path/to/Qwen3-ASR-1.7B)}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-ASR-1.7B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.10}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

echo "Starting Qwen vLLM server..."
echo "MODEL_PATH: ${MODEL_PATH}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "HOST:  ${HOST}"
echo "PORT:  ${PORT}"
echo "DTYPE: ${DTYPE}"
echo "GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION}"
echo "TENSOR_PARALLEL_SIZE: ${TENSOR_PARALLEL_SIZE}"
echo "MAX_MODEL_LEN: ${MAX_MODEL_LEN}"
echo "MAX_NUM_SEQS: ${MAX_NUM_SEQS}"

COMMON_ARGS=(
  --served-model-name "${MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --dtype "${DTYPE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
)

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  COMMON_ARGS+=(--trust-remote-code)
fi

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  COMMON_ARGS+=(--enforce-eager)
fi

if command -v qwen-asr-serve >/dev/null 2>&1; then
  echo "Launching with qwen-asr-serve (recommended for Qwen3-ASR)..."
  exec qwen-asr-serve "${MODEL_PATH}" "${COMMON_ARGS[@]}"
fi

echo "qwen-asr-serve not found, falling back to plain vllm serve."
echo "If startup fails with qwen3_asr architecture error, install qwen-asr[vllm]."
exec vllm serve "${MODEL_PATH}" "${COMMON_ARGS[@]}"
