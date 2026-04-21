from __future__ import annotations

import json
import logging
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

HOP_SIZE = 160  # 10ms at 16kHz, TEN VAD recommended
SAMPLE_RATE = 16000


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        logger.warning("Config file not found: %s, using built-in defaults", path)
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class Config:
    # ---- ASR: primary / secondary vLLM endpoints --------------------------
    vllm_base_url: str = "http://localhost:8000"
    vllm_model_name: str = "Amphion/Amphion-3B"
    secondary_vllm_base_url: str = "http://localhost:8001"
    secondary_vllm_model_name: str = "Qwen/Qwen3-ASR-1.7B"
    enable_secondary_asr: bool = True
    enable_primary_asr: bool = True
    primary_asr_timeout: float = 4.0
    debug_show_dual_asr: bool = True

    # ---- ASR: dual-model fusion knobs ------------------------------------
    fusion_similarity_threshold: float = 0.85
    fusion_min_primary_score: float = 0.55
    fusion_max_repetition_ratio: float = 0.35
    fusion_disagreement_threshold: float = 0.55
    fusion_hotword_boost: float = 0.12
    fusion_primary_score_margin: float = 0.08

    # ---- Common: HTTP / pseudo-streaming ---------------------------------
    asr_request_timeout: float = 120
    enable_pseudo_stream: bool = True
    pseudo_stream_interval_ms: int = 500

    # ---- ASR: VAD segmentation -------------------------------------------
    vad_threshold: float = 0.5
    silence_duration_ms: int = 200
    vad_smoothing_alpha: float = 0.35
    vad_start_frames: int = 3
    vad_pre_speech_ms: int = 500
    vad_end_frames: int = 20
    vad_keep_tail_ms: int = 40
    min_segment_duration_ms: int = 350

    # ---- Target-Speaker ASR (TS-ASR) -------------------------------------
    # TS-ASR runs on the same Amphion vLLM endpoint as the primary ASR by
    # default. When ``tsasr_base_url`` / ``tsasr_model_name`` are non-empty
    # they override that default, which lets us swap in a dedicated TS-ASR
    # checkpoint later without touching the standard ASR configuration.
    tsasr_base_url: str = ""
    tsasr_model_name: str = ""
    tsasr_request_timeout: float = 30.0
    tsasr_enrollment_min_sec: float = 1.0
    # Long uploads are trimmed via VAD to the leading ``max_sec`` of voiced
    # audio rather than being rejected — see ``decode_enrollment``. Five
    # seconds is enough for TS-ASR voice conditioning and keeps the
    # dual-audio prompt short.
    tsasr_enrollment_max_sec: float = 5.0
    tsasr_max_audio_seconds: float = 30.0
    tsasr_enable_partial: bool = False
    tsasr_enable_hotwords: bool = False
    # Second-stage speech-presence gate. The Amphion TS-ASR engine has no
    # silence/noise filter beyond the upstream VAD (the secondary ASR is
    # disabled because Qwen3-ASR-1.7B can't take dual-audio input), so
    # transient noise like keyboard taps that VAD misclassifies as speech
    # would otherwise be sent straight to the model. We re-analyze each
    # segmented clip with a stricter per-frame probability threshold and
    # require a minimum cumulative voiced duration before invoking vLLM.
    tsasr_speech_gate_enabled: bool = True
    tsasr_speech_gate_prob_threshold: float = 0.6
    tsasr_speech_gate_min_voiced_ms: int = 200

    # ---- Emotion recognition: vLLM endpoint ------------------------------
    # The Amphion multi-task model (Amphion/Amphion-3B) is trained to handle
    # SER/SEC alongside ASR via different text prompts, so by default we point
    # the emotion endpoint at the same backend as the primary ASR. Override
    # ``emotion_vllm_base_url`` if you serve a dedicated emotion model.
    emotion_vllm_base_url: str = "http://localhost:8000"
    emotion_vllm_model_name: str = "Amphion/Amphion-3B"
    emotion_request_timeout: float = 30.0
    # Amphion SER/SEC training uses 1-20s utterances, so we cap longer audio
    # to the trailing 20 seconds (where the most recent speech lives).
    emotion_max_audio_seconds: float = 20.0
    # Default task variant when the client doesn't specify one in start.mode.
    # "ser" -> single label classification; "sec" -> free-form caption.
    emotion_task_mode: str = "ser"

    def override(self, **kwargs: Any) -> Config:
        """Return a new Config with the given fields replaced (unknown keys ignored)."""
        valid_names = {f.name for f in fields(self)}
        accepted: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k not in valid_names:
                continue
            expected = type(getattr(self, k))
            try:
                accepted[k] = expected(v)
            except (TypeError, ValueError):
                logger.warning("Ignoring invalid config override %s=%r", k, v)
        return replace(self, **accepted) if accepted else self


def load_config(path: Path | None = None) -> Config:
    raw = _load_json(path or _CONFIG_PATH)
    valid_names = {f.name for f in fields(Config)}
    filtered = {k: v for k, v in raw.items() if k in valid_names}
    return Config(**filtered) if filtered else Config()


_default = load_config()

# ---------------------------------------------------------------------------
# Module-level constants for backward compatibility.
# Modules that don't need per-session override can keep importing these.
# ---------------------------------------------------------------------------
VLLM_BASE_URL = _default.vllm_base_url
VLLM_MODEL_NAME = _default.vllm_model_name
SECONDARY_VLLM_BASE_URL = _default.secondary_vllm_base_url
SECONDARY_VLLM_MODEL_NAME = _default.secondary_vllm_model_name
ENABLE_SECONDARY_ASR = _default.enable_secondary_asr
ENABLE_PRIMARY_ASR = _default.enable_primary_asr
PRIMARY_ASR_TIMEOUT = _default.primary_asr_timeout
DEBUG_SHOW_DUAL_ASR = _default.debug_show_dual_asr

FUSION_SIMILARITY_THRESHOLD = _default.fusion_similarity_threshold
FUSION_MIN_PRIMARY_SCORE = _default.fusion_min_primary_score
FUSION_MAX_REPETITION_RATIO = _default.fusion_max_repetition_ratio
FUSION_DISAGREEMENT_THRESHOLD = _default.fusion_disagreement_threshold
FUSION_HOTWORD_BOOST = _default.fusion_hotword_boost
FUSION_PRIMARY_SCORE_MARGIN = _default.fusion_primary_score_margin

ASR_REQUEST_TIMEOUT = _default.asr_request_timeout
ENABLE_PSEUDO_STREAM = _default.enable_pseudo_stream
PSEUDO_STREAM_INTERVAL_MS = _default.pseudo_stream_interval_ms

VAD_THRESHOLD = _default.vad_threshold
SILENCE_DURATION_MS = _default.silence_duration_ms
VAD_SMOOTHING_ALPHA = _default.vad_smoothing_alpha
VAD_START_FRAMES = _default.vad_start_frames
VAD_PRE_SPEECH_MS = _default.vad_pre_speech_ms
VAD_END_FRAMES = _default.vad_end_frames
VAD_KEEP_TAIL_MS = _default.vad_keep_tail_ms
MIN_SEGMENT_DURATION_MS = _default.min_segment_duration_ms

EMOTION_VLLM_BASE_URL = _default.emotion_vllm_base_url
EMOTION_VLLM_MODEL_NAME = _default.emotion_vllm_model_name
EMOTION_REQUEST_TIMEOUT = _default.emotion_request_timeout
EMOTION_MAX_AUDIO_SECONDS = _default.emotion_max_audio_seconds
EMOTION_TASK_MODE = _default.emotion_task_mode

TSASR_BASE_URL = _default.tsasr_base_url
TSASR_MODEL_NAME = _default.tsasr_model_name
TSASR_REQUEST_TIMEOUT = _default.tsasr_request_timeout
TSASR_ENROLLMENT_MIN_SEC = _default.tsasr_enrollment_min_sec
TSASR_ENROLLMENT_MAX_SEC = _default.tsasr_enrollment_max_sec
TSASR_MAX_AUDIO_SECONDS = _default.tsasr_max_audio_seconds
TSASR_ENABLE_PARTIAL = _default.tsasr_enable_partial
TSASR_ENABLE_HOTWORDS = _default.tsasr_enable_hotwords
TSASR_SPEECH_GATE_ENABLED = _default.tsasr_speech_gate_enabled
TSASR_SPEECH_GATE_PROB_THRESHOLD = _default.tsasr_speech_gate_prob_threshold
TSASR_SPEECH_GATE_MIN_VOICED_MS = _default.tsasr_speech_gate_min_voiced_ms
