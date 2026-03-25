import os

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen2.5-Omni-7B")
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
SILENCE_DURATION_MS = int(os.getenv("SILENCE_DURATION_MS", "600"))
HOP_SIZE = 160  # 10ms at 16kHz, TEN VAD recommended
SAMPLE_RATE = 16000
