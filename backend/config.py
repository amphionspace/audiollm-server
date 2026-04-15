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
    vllm_base_url: str = "http://localhost:8000"
    vllm_model_name: str = "Amphion/Amphion-3B"
    secondary_vllm_base_url: str = "http://localhost:8001"
    secondary_vllm_model_name: str = "Qwen/Qwen3-ASR-1.7B"
    enable_secondary_asr: bool = True
    enable_primary_asr: bool = True
    primary_asr_timeout: float = 4.0
    debug_show_dual_asr: bool = True

    fusion_similarity_threshold: float = 0.85
    fusion_min_primary_score: float = 0.55
    fusion_max_repetition_ratio: float = 0.35
    fusion_disagreement_threshold: float = 0.55
    fusion_hotword_boost: float = 0.12
    fusion_primary_score_margin: float = 0.08

    asr_request_timeout: float = 120
    enable_pseudo_stream: bool = True
    pseudo_stream_interval_ms: int = 500

    vad_threshold: float = 0.5
    silence_duration_ms: int = 200
    vad_smoothing_alpha: float = 0.35
    vad_start_frames: int = 3
    vad_pre_speech_ms: int = 500
    vad_end_frames: int = 20
    vad_keep_tail_ms: int = 40
    min_segment_duration_ms: int = 350

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
