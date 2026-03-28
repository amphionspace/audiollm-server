import os

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Amphion/Amphion-3B")
SECONDARY_VLLM_BASE_URL = os.getenv("SECONDARY_VLLM_BASE_URL", "http://localhost:8001")
SECONDARY_VLLM_MODEL_NAME = os.getenv(
    "SECONDARY_VLLM_MODEL_NAME", "Qwen/Qwen3-ASR-1.7B"
)
ENABLE_SECONDARY_ASR = os.getenv("ENABLE_SECONDARY_ASR", "1") == "1"
ENABLE_PRIMARY_ASR = os.getenv("ENABLE_PRIMARY_ASR", "1") == "1"
PRIMARY_ASR_TIMEOUT = float(os.getenv("PRIMARY_ASR_TIMEOUT", "4.0"))
DEBUG_SHOW_DUAL_ASR = os.getenv("DEBUG_SHOW_DUAL_ASR", "1") == "1"
FUSION_SIMILARITY_THRESHOLD = float(os.getenv("FUSION_SIMILARITY_THRESHOLD", "0.85"))
FUSION_MIN_PRIMARY_SCORE = float(os.getenv("FUSION_MIN_PRIMARY_SCORE", "0.55"))
FUSION_MAX_REPETITION_RATIO = float(os.getenv("FUSION_MAX_REPETITION_RATIO", "0.35"))
FUSION_DISAGREEMENT_THRESHOLD = float(
    os.getenv("FUSION_DISAGREEMENT_THRESHOLD", "0.55")
)
FUSION_HOTWORD_BOOST = float(os.getenv("FUSION_HOTWORD_BOOST", "0.12"))
FUSION_PRIMARY_SCORE_MARGIN = float(os.getenv("FUSION_PRIMARY_SCORE_MARGIN", "0.08"))
ASR_REQUEST_TIMEOUT = float(os.getenv("ASR_REQUEST_TIMEOUT", "120"))

VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.6"))
SILENCE_DURATION_MS = int(os.getenv("SILENCE_DURATION_MS", "200"))
# Exponential moving average smoothing for frame-level VAD probability.
# 0 means no smoothing; closer to 1 means stronger smoothing.
VAD_SMOOTHING_ALPHA = float(os.getenv("VAD_SMOOTHING_ALPHA", "0.35"))
# Require N consecutive speech frames before entering speaking state.
VAD_START_FRAMES = int(os.getenv("VAD_START_FRAMES", "3"))
# Include this much pre-roll audio before speech start for each segment.
VAD_PRE_SPEECH_MS = int(os.getenv("VAD_PRE_SPEECH_MS", "500"))
# Require N consecutive non-speech frames before ending speaking state.
# Keep backward compatibility by defaulting to SILENCE_DURATION_MS / 10.
VAD_END_FRAMES = int(os.getenv("VAD_END_FRAMES", str(max(1, SILENCE_DURATION_MS // 10))))
VAD_KEEP_TAIL_MS = int(os.getenv("VAD_KEEP_TAIL_MS", "40"))
HOP_SIZE = 160  # 10ms at 16kHz, TEN VAD recommended
SAMPLE_RATE = 16000
# Drop very short VAD segments (helps suppress tiny noise bursts).
MIN_SEGMENT_DURATION_MS = int(os.getenv("MIN_SEGMENT_DURATION_MS", "350"))
