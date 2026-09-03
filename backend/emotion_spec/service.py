"""Shared whole-utterance inference for the AmphionSPEC HTTP job API.

Sibling of :mod:`backend.emotion.service` — same shape, different vLLM
endpoint and prompt set. ``decode_wav_capped`` / ``EmotionDecodeError``
are re-used from the baseline package (they handle generic 16 kHz WAV
re-encoding, no SPEC-specific behaviour required).
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from ..config import Config
from ..emotion.service import (  # noqa: F401 — re-export so callers don't need a second import
    EmotionDecodeError,
    decode_wav_capped,
)
from .client import query_emotion_spec_model
from .prompt import EmotionSpecMode, normalize_mode

logger = logging.getLogger(__name__)


def build_final_emotion_spec_payload(
    result: dict[str, Any],
    *,
    mode: str,
    duration_sec: float,
    language: str = "",
) -> dict[str, Any]:
    """Build the canonical ``final_emotion`` response object for SPEC.

    Reuses the same ``type: "final_emotion"`` envelope as the baseline
    emotion service. Model identity is not exposed in the protocol.
    """
    payload: dict[str, Any] = {
        "type": "final_emotion",
        "mode": mode,
        "label": result.get("label", ""),
        "text": result.get("text", ""),
        "duration_sec": round(duration_sec, 3),
    }
    raw_text = result.get("raw_text", "")
    if raw_text and raw_text != payload["text"]:
        payload["raw_text"] = raw_text
    if language:
        payload["language"] = language
    return payload


def empty_final_emotion_spec(*, mode: str, language: str = "") -> dict[str, Any]:
    """Empty result when no usable audio was submitted."""
    return build_final_emotion_spec_payload(
        {"label": "", "text": ""},
        mode=mode,
        duration_sec=0.0,
        language=language,
    )


async def infer_emotion_spec_from_wav(
    raw_wav: bytes,
    *,
    mode: str,
    language: str,
    cfg: Config,
) -> dict[str, Any]:
    """Run whole-utterance SPEC inference and return ``final_emotion`` dict."""
    fallback = getattr(cfg, "emotion_spec_task_mode", "sec")
    chosen_mode: EmotionSpecMode = normalize_mode(mode or fallback)
    if not raw_wav:
        return empty_final_emotion_spec(mode=chosen_mode, language=language)
    cap = float(getattr(cfg, "emotion_spec_max_audio_seconds", 0.0))

    wav_bytes, duration_sec = decode_wav_capped(raw_wav, cap)

    if duration_sec <= 0:
        return empty_final_emotion_spec(mode=chosen_mode, language=language)

    wav_b64 = base64.b64encode(wav_bytes).decode("ascii")
    result = await query_emotion_spec_model(
        wav_b64,
        mode=chosen_mode,
        base_url=cfg.emotion_spec_vllm_base_url,
        model_name=cfg.emotion_spec_vllm_model_name,
        timeout=cfg.emotion_spec_request_timeout,
    )
    return build_final_emotion_spec_payload(
        result,
        mode=chosen_mode,
        duration_sec=duration_sec,
        language=language,
    )
