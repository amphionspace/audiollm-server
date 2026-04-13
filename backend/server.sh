#!/usr/bin/env bash
set -euo pipefail

# Fill values directly here.
MODEL_PATH="/home/ubuntu/models/hf/checkpoint-5500_completed/"
MODEL_NAME="Amphion/Amphion-3B"
HOST="0.0.0.0"
PORT="8000"
DTYPE="bfloat16"
# Must be <= current free_ratio from nvidia-smi.
# Your log shows free memory is about 38.75 / 95.08 ~= 0.41,
# so keep this below 0.41 unless other GPU processes are released.
GPU_MEMORY_UTILIZATION="0.25"
TENSOR_PARALLEL_SIZE="1"
MAX_MODEL_LEN="4096"
MAX_NUM_SEQS="8"
TRUST_REMOTE_CODE="1"
ENFORCE_EAGER="0"

echo "Starting vLLM server..."
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
