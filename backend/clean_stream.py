"""Public, auth-free pseudo-streaming ASR protocol for the Playground."""

from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import re
import secrets
import wave
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .config import Upstream, get_service_upstream, load_config
from .emotion_spec.service import infer_emotion_spec_from_wav
from .http_client import get_client

logger = logging.getLogger(__name__)
router = APIRouter()

SAMPLE_RATE = 16_000
BYTES_PER_SECOND = SAMPLE_RATE * 2
PARTIAL_INTERVAL_BYTES = BYTES_PER_SECOND * 2
MAX_AUDIO_BYTES = BYTES_PER_SECOND * 60
SPEECH_RMS_THRESHOLD = 120.0
_ASR_PREFIX = re.compile(r"^language\s+[^<]+<asr_text>", re.IGNORECASE)

BUILTIN_HOTWORDS: dict[str, list[str]] = {
    "finance": ["AUM", "ETF", "IPO", "净值", "市盈率"],
    "education": ["慕课", "教研", "学情", "生成式人工智能"],
    "internet": ["Amphion", "Qwen3-ASR", "OpenTelemetry", "WebSocket"],
}


@dataclass
class SessionOptions:
    language: str = "auto"
    cleanup_level: str = "light"
    text_emotion: bool = False
    translate_mode: bool = False
    target_language: str | None = None
    builtin: list[str] = field(default_factory=list)
    custom: list[str] = field(default_factory=list)

    @property
    def glossary(self) -> list[str]:
        values: list[str] = []
        for name in self.builtin:
            values.extend(BUILTIN_HOTWORDS.get(name, []))
        values.extend(self.custom)
        return list(dict.fromkeys(word.strip() for word in values if word.strip()))[:100]


def _pcm_to_wav(pcm: bytes) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    return output.getvalue()


def _has_speech(pcm: bytes) -> bool:
    if len(pcm) < 2:
        return False
    samples = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32)
    return bool(samples.size and np.sqrt(np.mean(samples * samples)) >= SPEECH_RMS_THRESHOLD)


def _parse_options(message: dict[str, Any]) -> SessionOptions:
    language = str(message.get("language") or "auto").strip().lower()
    cleanup = message.get("cleanup") or {}
    hotwords = message.get("hotwords") or {}
    if not isinstance(cleanup, dict) or not isinstance(hotwords, dict):
        raise ValueError("cleanup and hotwords must be objects.")
    level = str(cleanup.get("level") or "light").lower()
    if level not in {"off", "light", "standard"}:
        raise ValueError("cleanup.level must be off, light, or standard.")
    translate = message.get("translate_mode", False)
    if not isinstance(translate, bool):
        raise ValueError("translate_mode must be boolean.")
    target = str(message.get("target_language") or "").strip().lower() or None
    if translate and (not target or target == "auto"):
        raise ValueError("target_language is required when translate_mode=true.")
    builtin = hotwords.get("builtin") or []
    custom = hotwords.get("custom") or []
    if not isinstance(builtin, list) or not isinstance(custom, list):
        raise ValueError("hotword lists must be arrays.")
    builtin = [str(item).strip() for item in builtin if str(item).strip()]
    if len(builtin) > 1 or any(item not in BUILTIN_HOTWORDS for item in builtin):
        raise ValueError("hotwords.builtin accepts at most one known list.")
    text_emotion = cleanup.get("text_emotion", False)
    if not isinstance(text_emotion, bool):
        raise ValueError("cleanup.text_emotion must be boolean.")
    return SessionOptions(
        language=language,
        cleanup_level=level,
        text_emotion=text_emotion,
        translate_mode=translate,
        target_language=target,
        builtin=builtin,
        custom=[str(item).strip() for item in custom if str(item).strip()][:100],
    )


async def transcribe_qwen(pcm: bytes, options: SessionOptions) -> str:
    upstream = get_service_upstream("clean_stream_asr")
    if upstream is None:
        raise RuntimeError("clean_stream_asr upstream is not configured")
    data = {"model": upstream.model_name}
    if options.language != "auto":
        data["language"] = options.language
    response = await get_client().post(
        f"{upstream.base_url.rstrip('/')}/v1/audio/transcriptions",
        data=data,
        files={"file": ("audio.wav", _pcm_to_wav(pcm), "audio/wav")},
        timeout=upstream.timeout,
    )
    response.raise_for_status()
    text = str(response.json().get("text") or "").strip()
    return _ASR_PREFIX.sub("", text).strip()


def _refine_prompt(text: str, options: SessionOptions, emotion: dict[str, Any] | None) -> list[dict[str, str]]:
    mode = "translation" if options.translate_mode else "cleanup"
    if mode == "translation":
        instruction = (
            f"Translate the ASR transcript faithfully into {options.target_language}. "
            "Return only the translation. Do not explain or add information."
        )
    else:
        strictness = "只修正标点、空格、数字格式和术语" if options.cleanup_level == "light" else "可同时删除明显口头语和重复词，但不得改写或增加事实"
        instruction = f"精修 ASR 文本：{strictness}。只返回最终文本，不要解释。"
    payload = {
        "asr_text": text,
        "language": options.language,
        "glossary": options.glossary,
        "emotion": emotion or {},
    }
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


async def refine_text(text: str, options: SessionOptions, emotion: dict[str, Any] | None) -> str:
    upstream = get_service_upstream("clean_stream_refine")
    if upstream is None or not upstream.api_key:
        raise RuntimeError("text_cleanup upstream is not configured")
    payload: dict[str, Any] = {
        "model": upstream.model_name,
        "messages": _refine_prompt(text, options, emotion),
        "temperature": 0.1,
        "max_tokens": upstream.max_tokens or 1024,
    }
    payload["thinking"] = {"type": "disabled"}
    response = await get_client().post(
        f"{upstream.base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {upstream.api_key}"},
        json=payload,
        timeout=upstream.timeout,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return str(content or "").strip()


async def infer_emotion(pcm: bytes, options: SessionOptions) -> dict[str, Any]:
    result = await infer_emotion_spec_from_wav(
        _pcm_to_wav(pcm), mode="sec", language=options.language, cfg=load_config()
    )
    return {
        "mode": "sec",
        "label": result.get("label") or result.get("text") or "",
        "text": result.get("text") or "",
    }


async def _send_error(websocket: WebSocket, session_id: str, code: str, message: str) -> None:
    await websocket.send_json(
        {"type": "error", "session_id": session_id, "code": code, "message": message}
    )


def compute_delta(previous_text: str, current_text: str) -> str:
    """Match the referenced clean-stream protocol's cumulative-text delta."""
    if current_text.startswith(previous_text):
        return current_text[len(previous_text):]
    if previous_text.startswith(current_text):
        return ""
    prefix_len = 0
    for previous_char, current_char in zip(previous_text, current_text):
        if previous_char != current_char:
            break
        prefix_len += 1
    return current_text[prefix_len:]


@router.websocket("/asr/v1/clean-stream")
async def clean_stream_ws(websocket: WebSocket) -> None:
    """Gateway-compatible wire protocol, owned and served by this application."""
    await websocket.accept()
    session_id = f"asr-clean-{secrets.token_hex(6)}"
    await websocket.send_json({"type": "session.created", "session": {"id": session_id}})
    options: SessionOptions | None = None
    audio = bytearray()
    last_partial_bytes = 0
    last_partial_text = ""
    try:
        while True:
            message = await websocket.receive_json()
            kind = message.get("type")
            if kind == "session.update":
                if options is not None:
                    await _send_error(websocket, session_id, "invalid_state", "session.update may only be sent once.")
                    continue
                try:
                    options = _parse_options(message)
                except ValueError as exc:
                    await _send_error(websocket, session_id, "invalid_request", str(exc))
                    continue
                await websocket.send_json({
                    "type": "session.updated",
                    "session": {"id": session_id},
                    "language": options.language,
                    "cleanup": {"level": options.cleanup_level, "text_emotion": options.text_emotion},
                    "translate_mode": options.translate_mode,
                    "postprocess_mode": "translation" if options.translate_mode else "cleanup",
                    "target_language": options.target_language,
                    "hotwords": {"builtin": options.builtin, "custom_count": len(options.custom)},
                    "fallback_active": False,
                })
                continue
            if options is None:
                await _send_error(websocket, session_id, "invalid_state", "Send session.update before streaming audio.")
                continue
            if kind == "input_audio_buffer.append":
                try:
                    chunk = base64.b64decode(message.get("audio") or "", validate=True)
                except (binascii.Error, ValueError, TypeError):
                    await _send_error(websocket, session_id, "invalid_audio", "audio must be valid base64 PCM.")
                    continue
                if not chunk:
                    await _send_error(websocket, session_id, "invalid_audio", "audio field is required.")
                    continue
                if len(audio) + len(chunk) > MAX_AUDIO_BYTES:
                    await _send_error(websocket, session_id, "audio_too_long", "Audio exceeds 60 seconds.")
                    await websocket.close(code=1009)
                    return
                audio.extend(chunk)
                if len(audio) - last_partial_bytes >= PARTIAL_INTERVAL_BYTES and _has_speech(audio):
                    partial = await transcribe_qwen(bytes(audio), options)
                    if partial and partial != last_partial_text:
                        delta = compute_delta(last_partial_text, partial)
                        await websocket.send_json({"type": "transcription.delta", "delta": delta, "text": partial})
                        last_partial_text = partial
                    last_partial_bytes = len(audio)
                continue
            if kind == "input_audio_buffer.commit":
                if message.get("final") is not True:
                    await _send_error(websocket, session_id, "invalid_request", "Only final=true is supported.")
                    continue
                if not audio:
                    await _send_error(websocket, session_id, "no_audio", "No audio data received.")
                    return
                if not _has_speech(audio):
                    await _send_error(websocket, session_id, "no_speech_detected", "No speech detected.")
                    return
                final_text = await transcribe_qwen(bytes(audio), options)
                if final_text != last_partial_text:
                    delta = compute_delta(last_partial_text, final_text)
                    await websocket.send_json({"type": "transcription.delta", "delta": delta, "text": final_text})
                emotion: dict[str, Any] | None = None
                if options.text_emotion:
                    emotion = await infer_emotion(bytes(audio), options)
                    await websocket.send_json({"type": "emotion.bucket", "emotion": emotion})
                result: dict[str, Any] = {
                    "type": "transcription.done", "session_id": session_id,
                    "text": final_text,
                    "usage": {"type": "duration", "seconds": round(len(audio) / BYTES_PER_SECOND, 3)},
                    "language": options.language,
                    "builtin_hotword_lists": options.builtin,
                    "custom_hotword_count": len(options.custom),
                    "translate_mode": options.translate_mode,
                    "postprocess_mode": "translation" if options.translate_mode else "cleanup",
                    "cleanup_level": options.cleanup_level,
                    "asr_fallback_used": False,
                }
                if options.translate_mode or options.cleanup_level != "off":
                    status_key = "translation_status" if options.translate_mode else "cleanup_status"
                    text_key = "translated_text" if options.translate_mode else "cleaned_text"
                    try:
                        processed = await refine_text(final_text, options, emotion)
                        result[text_key] = processed
                        result[status_key] = "completed"
                        await websocket.send_json({
                            "type": "postprocess.delta",
                            "postprocess_mode": result["postprocess_mode"],
                            "delta": processed, "text": processed,
                        })
                    except Exception as exc:  # raw ASR remains usable if refine is down
                        logger.exception("clean-stream refine failed: %s", exc)
                        result[status_key] = "degraded_raw_only"
                await websocket.send_json(result)
                return
            await _send_error(websocket, session_id, "invalid_request", "Unknown message type.")
    except WebSocketDisconnect:
        return
    except Exception as exc:
        logger.exception("clean-stream session failed: %s", exc)
        try:
            await _send_error(websocket, session_id, "server_error", "Temporary server error.")
        except RuntimeError:
            pass
