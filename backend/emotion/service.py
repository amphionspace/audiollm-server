"""Shared whole-utterance emotion inference for HTTP job API."""

from __future__ import annotations

import base64
import io
import logging
import wave
from typing import Any

import numpy as np

from ..audio.utils import pcm_to_wav_base64, wav_base64_to_pcm_16k_mono
from ..config import SAMPLE_RATE, Config
from .client import query_emotion_model
from .prompt import normalize_mode

logger = logging.getLogger(__name__)


class EmotionDecodeError(ValueError):
    """WAV could not be decoded for emotion inference."""


def decode_wav_capped(raw: bytes, max_seconds: float) -> tuple[bytes, float]:
    """Decode WAV to 16 kHz mono; tail-trim to ``max_seconds``.

    Returns (re-encoded_wav_bytes, duration_sec). Raises :class:`EmotionDecodeError`
    when the blob is invalid; returns zero duration when PCM is empty.
    """
    if not raw:
        return b"", 0.0
    wav_b64 = base64.b64encode(raw).decode("ascii")
    try:
        pcm = wav_base64_to_pcm_16k_mono(wav_b64)
    except ValueError as exc:
        raise EmotionDecodeError(str(exc)) from exc
    if pcm.size == 0:
        return raw, 0.0
    duration = pcm.size / SAMPLE_RATE
    if max_seconds > 0 and duration > max_seconds:
        keep = int(SAMPLE_RATE * max_seconds)
        pcm = pcm[-keep:]
        duration = pcm.size / SAMPLE_RATE
    new_b64 = pcm_to_wav_base64(pcm.astype(np.float32, copy=False))
    new_bytes = base64.b64decode(new_b64)
    with wave.open(io.BytesIO(new_bytes), "rb") as wf:
        if wf.getframerate() != SAMPLE_RATE:
            raise EmotionDecodeError("unexpected sample rate after re-encode")
    return new_bytes, duration


def build_final_emotion_payload(
    result: dict[str, Any],
    *,
    mode: str,
    duration_sec: float,
    language: str = "",
) -> dict[str, Any]:
    """Build the canonical ``final_emotion`` response object."""
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
    top_emotions = result.get("top_emotions")
    if mode == "ser" and isinstance(top_emotions, list) and top_emotions:
        payload["top_emotions"] = top_emotions
        payload["best_label"] = result.get("best_label", payload["label"])
        payload["best_score"] = result.get("best_score", 0.0)
    return payload


def empty_final_emotion(*, mode: str, language: str = "") -> dict[str, Any]:
    """Empty result when no usable audio was submitted (matches WS on_stop)."""
    return build_final_emotion_payload(
        {"label": "", "text": ""},
        mode=mode,
        duration_sec=0.0,
        language=language,
    )


async def infer_emotion_from_wav(
    raw_wav: bytes,
    *,
    mode: str,
    language: str,
    cfg: Config,
) -> dict[str, Any]:
    """Run whole-utterance emotion inference and return ``final_emotion`` dict."""
    chosen_mode = normalize_mode(mode or getattr(cfg, "emotion_task_mode", "ser"))
    if not raw_wav:
        return empty_final_emotion(mode=chosen_mode, language=language)
    cap = float(getattr(cfg, "emotion_max_audio_seconds", 0.0))

    try:
        wav_bytes, duration_sec = decode_wav_capped(raw_wav, cap)
    except EmotionDecodeError:
        raise

    if duration_sec <= 0:
        return empty_final_emotion(mode=chosen_mode, language=language)

    wav_b64 = base64.b64encode(wav_bytes).decode("ascii")
    result = await query_emotion_model(
        wav_b64,
        mode=chosen_mode,
        base_url=cfg.emotion_vllm_base_url,
        model_name=cfg.emotion_vllm_model_name,
        timeout=cfg.emotion_request_timeout,
    )
    return build_final_emotion_payload(
        result,
        mode=chosen_mode,
        duration_sec=duration_sec,
        language=language,
    )
