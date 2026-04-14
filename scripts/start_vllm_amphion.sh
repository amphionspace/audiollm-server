#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:?Please set MODEL_PATH (e.g. /path/to/Amphion-3B)}"
MODEL_NAME="${MODEL_NAME:-Amphion/Amphion-3B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.25}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

echo "Starting Amphion vLLM server..."
echo "MODEL_PATH: ${MODEL_PATH}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "HOST:  ${HOST}"
echo "PORT:  ${PORT}"
echo "DTYPE: ${DTYPE}"
echo "GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION}"
echo "TENSOR_PARALLEL_SIZE: ${TENSOR_PARALLEL_SIZE}"
echo "MAX_MODEL_LEN: ${MAX_MODEL_LEN}"
echo "MAX_NUM_SEQS: ${MAX_NUM_SEQS}"

VLLM_ARGS=(
  serve "${MODEL_PATH}"
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
  VLLM_ARGS+=(--trust-remote-code)
fi

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  VLLM_ARGS+=(--enforce-eager)
fi

exec vllm "${VLLM_ARGS[@]}"
