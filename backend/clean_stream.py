"""Public, auth-free pseudo-streaming ASR protocol for the Playground."""

from __future__ import annotations

import asyncio
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

from .config import get_service_upstream, load_config
from .emotion_spec.service import infer_emotion_spec_from_wav
from .http_client import get_client
from .streaming.audio_stream import VadSegmentedStream
from .streaming.events import PartialSnapshot, SegmentReady

logger = logging.getLogger(__name__)
router = APIRouter()

SAMPLE_RATE = 16_000
BYTES_PER_SECOND = SAMPLE_RATE * 2
MAX_AUDIO_BYTES = BYTES_PER_SECOND * 60
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


def _float_pcm_to_bytes(pcm: np.ndarray) -> bytes:
    return (np.clip(pcm, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def _join_segments(parts: list[str]) -> str:
    result = ""
    for part in parts:
        value = part.strip()
        if not value:
            continue
        separator = " " if result and result[-1].isascii() and value[0].isascii() else ""
        result += separator + value
    return result


@router.websocket("/asr/v1/clean-stream")
async def clean_stream_ws(websocket: WebSocket) -> None:
    """Gateway-compatible wire protocol, owned and served by this application."""
    await websocket.accept()
    session_id = f"asr-clean-{secrets.token_hex(6)}"
    await websocket.send_json({"type": "session.created", "session": {"id": session_id}})
    options: SessionOptions | None = None
    stream: VadSegmentedStream | None = None
    event_queue: asyncio.Queue[PartialSnapshot | SegmentReady | None] = asyncio.Queue()
    worker: asyncio.Task[None] | None = None
    postprocess_tasks: list[asyncio.Task[None]] = []
    send_lock = asyncio.Lock()
    raw_segments: list[str] = []
    processed_segments: dict[int, str] = {}
    postprocess_failed = False
    emitted_text = ""
    total_audio_bytes = 0
    worker_error: Exception | None = None

    async def send(payload: dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def send_error(code: str, message: str) -> None:
        await send({"type": "error", "session_id": session_id, "code": code, "message": message})

    async def postprocess_segment(index: int, text: str, pcm: bytes) -> None:
        nonlocal postprocess_failed
        assert options is not None
        emotion: dict[str, Any] | None = None
        try:
            if options.text_emotion:
                emotion = await infer_emotion(pcm, options)
                await send({"type": "emotion.bucket", "segment_index": index, "emotion": emotion})
            if options.translate_mode or options.cleanup_level != "off":
                processed = await refine_text(text, options, emotion)
                processed_segments[index] = processed
                await send({
                    "type": "postprocess.delta",
                    "postprocess_mode": "translation" if options.translate_mode else "cleanup",
                    "delta": processed,
                    "text": processed,
                    "segment_index": index,
                })
        except Exception as exc:
            postprocess_failed = True
            processed_segments[index] = text
            logger.exception("clean-stream segment %d postprocess failed: %s", index, exc)

    async def process_events() -> None:
        nonlocal emitted_text, worker_error
        assert options is not None
        while True:
            event = await event_queue.get()
            try:
                if event is None:
                    return
                if worker_error is not None:
                    continue
                pcm = _float_pcm_to_bytes(event.pcm)
                if isinstance(event, PartialSnapshot):
                    try:
                        partial_tail = await transcribe_qwen(pcm, options)
                    except Exception as exc:
                        logger.warning("clean-stream partial ASR failed: %s", exc)
                        continue
                    current = _join_segments([*raw_segments, partial_tail])
                    if current and current != emitted_text:
                        await send({
                            "type": "transcription.delta",
                            "delta": compute_delta(emitted_text, current),
                            "text": current,
                        })
                        emitted_text = current
                    continue
                segment_text = await transcribe_qwen(pcm, options)
                if not segment_text:
                    continue
                raw_segments.append(segment_text)
                current = _join_segments(raw_segments)
                if current != emitted_text:
                    await send({
                        "type": "transcription.delta",
                        "delta": compute_delta(emitted_text, current),
                        "text": current,
                    })
                    emitted_text = current
                index = len(raw_segments) - 1
                postprocess_tasks.append(
                    asyncio.create_task(postprocess_segment(index, segment_text, pcm))
                )
            except Exception as exc:
                worker_error = exc
                logger.exception("clean-stream ASR worker failed: %s", exc)
            finally:
                event_queue.task_done()

    async def enqueue_events(events: list[Any]) -> None:
        for event in events:
            if isinstance(event, SegmentReady):
                await event_queue.put(event)
            elif isinstance(event, PartialSnapshot) and event_queue.empty():
                await event_queue.put(event)

    try:
        while True:
            message = await websocket.receive_json()
            kind = message.get("type")
            if kind == "session.update":
                if options is not None:
                    await send_error("invalid_state", "session.update may only be sent once.")
                    continue
                try:
                    options = _parse_options(message)
                except ValueError as exc:
                    await send_error("invalid_request", str(exc))
                    continue
                stream = VadSegmentedStream(enable_partial=True)
                stream.configure(load_config())
                worker = asyncio.create_task(process_events())
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
                await send_error("invalid_state", "Send session.update before streaming audio.")
                continue
            if kind == "input_audio_buffer.append":
                try:
                    chunk = base64.b64decode(message.get("audio") or "", validate=True)
                except (binascii.Error, ValueError, TypeError):
                    await send_error("invalid_audio", "audio must be valid base64 PCM.")
                    continue
                if not chunk:
                    await send_error("invalid_audio", "audio field is required.")
                    continue
                if total_audio_bytes + len(chunk) > MAX_AUDIO_BYTES:
                    await send_error("audio_too_long", "Audio exceeds 60 seconds.")
                    await websocket.close(code=1009)
                    return
                total_audio_bytes += len(chunk)
                assert stream is not None
                await enqueue_events(list(stream.feed(chunk)))
                continue
            if kind == "input_audio_buffer.commit":
                if message.get("final") is not True:
                    await send_error("invalid_request", "Only final=true is supported.")
                    continue
                if not total_audio_bytes:
                    await send_error("no_audio", "No audio data received.")
                    return
                assert stream is not None and worker is not None
                await enqueue_events(list(stream.flush(force=True)))
                await event_queue.join()
                await event_queue.put(None)
                await worker
                if worker_error is not None:
                    await send_error("server_error", "ASR service unavailable.")
                    return
                if not raw_segments:
                    await send_error("no_speech_detected", "No speech detected.")
                    return
                if postprocess_tasks:
                    await asyncio.gather(*postprocess_tasks)
                final_text = _join_segments(raw_segments)
                result: dict[str, Any] = {
                    "type": "transcription.done", "session_id": session_id,
                    "text": final_text,
                    "usage": {"type": "duration", "seconds": round(total_audio_bytes / BYTES_PER_SECOND, 3)},
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
                    result[text_key] = _join_segments(
                        [processed_segments.get(index, text) for index, text in enumerate(raw_segments)]
                    )
                    result[status_key] = "degraded_raw_only" if postprocess_failed else "completed"
                await websocket.send_json(result)
                return
            await send_error("invalid_request", "Unknown message type.")
    except WebSocketDisconnect:
        return
    except Exception as exc:
        logger.exception("clean-stream session failed: %s", exc)
        try:
            await send_error("server_error", "Temporary server error.")
        except RuntimeError:
            pass
    finally:
        if worker is not None and not worker.done():
            worker.cancel()
        for task in postprocess_tasks:
            if not task.done():
                task.cancel()
