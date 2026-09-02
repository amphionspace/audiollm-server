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
import unicodedata
import wave
from dataclasses import dataclass, field
from difflib import SequenceMatcher
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
_ASR_PREFIX = re.compile(r"^language\s+[^<]+<asr_text>", re.IGNORECASE)
_ASCII_TOKEN = re.compile(r"\b[A-Z][A-Z0-9-]+\b")

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


def _refine_prompt(
    text: str, options: SessionOptions, emotion: dict[str, Any] | None
) -> list[dict[str, str]]:
    mode = "translation" if options.translate_mode else "cleanup"
    if mode == "translation":
        instruction = (
            f"Translate the ASR transcript faithfully into {options.target_language}. "
            "Return only the translation. Do not explain or add information."
        )
    else:
        instructions = [
            "你是实时语音识别结果的保守清洗器。",
            "如果原文已经正确，必须原样返回；宁可少改，不要猜测。",
            "不得增加原文没有的信息，不得改写句意，不得润色或总结。",
            "不得改变数字的数值、英文缩写以及术语表之外的人名、机构名、品牌名和产品名。",
            "只返回清洗后的正文，不要解释、标题、引号或 Markdown。",
        ]
        if options.cleanup_level == "light":
            instructions.append(
                "light 模式只允许调整标点、空格、大小写、数字格式，"
                "以及有充分依据的术语表纠错；不得替换普通词语。"
            )
        else:
            instructions.append(
                "standard 模式可额外删除非常明显的语气词和紧邻重复词，但仍不得重写句子。"
            )
        if options.glossary:
            instructions.extend(
                [
                    "术语表中的词是用户指定的规范写法。必须逐项检查原文是否存在"
                    "明显同音、近音、中文音译或英文拼写误识别；确认对应时替换为规范写法。",
                    "英文术语可能被 ASR 写成发音相近的汉字，这属于应当纠正的术语误识别。",
                    "无法确认原文指向该术语时保持原文，禁止为了使用术语表而凭空插入词语。",
                    f"术语表：{', '.join(options.glossary)}",
                ]
            )
        if emotion:
            instructions.append(
                "情感描述只可用于选择句号、问号、感叹号或停顿；"
                "绝对不能据此增删或替换任何字词，也不要添加 emoji。"
            )
        instruction = "\n".join(instructions)
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


def _normalize_for_compare(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(char for char in normalized if char.isalnum())


def _is_scoped_glossary_replacement(
    original_cmp: str, candidate_cmp: str, canonical_terms: list[str]
) -> bool:
    """Allow a canonical term to replace a small local ASR error, not be inserted."""
    remainder = candidate_cmp
    canonical_size = 0
    for term in canonical_terms:
        normalized_term = _normalize_for_compare(term)
        canonical_size += len(normalized_term)
        remainder = remainder.replace(normalized_term, "", 1)
    if not remainder:
        return False
    blocks = SequenceMatcher(None, original_cmp, remainder).get_matching_blocks()
    matched = sum(block.size for block in blocks)
    unmatched_original = len(original_cmp) - matched
    candidate_coverage = matched / len(remainder)
    return candidate_coverage >= 0.90 and 1 <= unmatched_original <= max(4, canonical_size)


def evaluate_cleanup_result(
    original: str,
    candidate: str,
    cleanup_level: str,
    glossary: list[str],
) -> tuple[bool, str]:
    """Reject cleanup output that is more likely to be a rewrite than a correction."""
    original_cmp = _normalize_for_compare(original)
    candidate_cmp = _normalize_for_compare(candidate)
    if not candidate_cmp:
        return False, "empty"
    if not original_cmp:
        return False, "empty_original"

    original_digits = "".join(
        char for char in unicodedata.normalize("NFKC", original) if char.isdigit()
    )
    candidate_digits = "".join(
        char for char in unicodedata.normalize("NFKC", candidate) if char.isdigit()
    )
    if original_digits != candidate_digits:
        return False, "digits_changed"

    for token in _ASCII_TOKEN.findall(original):
        if token not in candidate:
            return False, f"ascii_token_dropped:{token}"
    for term in glossary:
        if term.casefold() in original.casefold() and term.casefold() not in candidate.casefold():
            return False, f"glossary_term_dropped:{term}"

    similarity = SequenceMatcher(None, original_cmp, candidate_cmp).ratio()
    length_ratio = len(candidate_cmp) / len(original_cmp)
    if cleanup_level == "light":
        min_similarity, min_length, max_length = (0.94, 0.88, 1.12)
    else:
        min_similarity, min_length, max_length = (0.82, 0.75, 1.20)
    newly_canonical_terms = [
        term
        for term in glossary
        if term.casefold() in candidate.casefold() and term.casefold() not in original.casefold()
    ]
    if newly_canonical_terms:
        if _is_scoped_glossary_replacement(original_cmp, candidate_cmp, newly_canonical_terms):
            if 0.65 <= length_ratio <= 1.60:
                return True, "ok"
            return False, f"length_ratio:{length_ratio:.3f}"
        min_similarity = min(min_similarity, 0.78)
    if similarity < min_similarity:
        return False, f"similarity:{similarity:.3f}"
    if not min_length <= length_ratio <= max_length:
        return False, f"length_ratio:{length_ratio:.3f}"
    return True, "ok"


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
        return current_text[len(previous_text) :]
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
                guardrail_status = "not_applicable"
                if not options.translate_mode:
                    accepted, reason = evaluate_cleanup_result(
                        text, processed, options.cleanup_level, options.glossary
                    )
                    if not accepted:
                        logger.warning(
                            "clean-stream segment %d cleanup rejected: %s", index, reason
                        )
                        postprocess_failed = True
                        processed = text
                        guardrail_status = f"rejected:{reason}"
                    else:
                        guardrail_status = "accepted"
                processed_segments[index] = processed
                await send(
                    {
                        "type": "postprocess.delta",
                        "postprocess_mode": "translation" if options.translate_mode else "cleanup",
                        "delta": processed,
                        "text": processed,
                        "segment_index": index,
                        "guardrail_status": guardrail_status,
                    }
                )
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
                        await send(
                            {
                                "type": "transcription.delta",
                                "delta": compute_delta(emitted_text, current),
                                "text": current,
                            }
                        )
                        emitted_text = current
                    continue
                segment_text = await transcribe_qwen(pcm, options)
                if not segment_text:
                    continue
                raw_segments.append(segment_text)
                current = _join_segments(raw_segments)
                if current != emitted_text:
                    await send(
                        {
                            "type": "transcription.delta",
                            "delta": compute_delta(emitted_text, current),
                            "text": current,
                        }
                    )
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
                await websocket.send_json(
                    {
                        "type": "session.updated",
                        "session": {"id": session_id},
                        "language": options.language,
                        "cleanup": {
                            "level": options.cleanup_level,
                            "text_emotion": options.text_emotion,
                        },
                        "translate_mode": options.translate_mode,
                        "postprocess_mode": "translation" if options.translate_mode else "cleanup",
                        "target_language": options.target_language,
                        "hotwords": {
                            "builtin": options.builtin,
                            "custom_count": len(options.custom),
                        },
                        "fallback_active": False,
                    }
                )
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
                    "type": "transcription.done",
                    "session_id": session_id,
                    "text": final_text,
                    "usage": {
                        "type": "duration",
                        "seconds": round(total_audio_bytes / BYTES_PER_SECOND, 3),
                    },
                    "language": options.language,
                    "builtin_hotword_lists": options.builtin,
                    "custom_hotword_count": len(options.custom),
                    "translate_mode": options.translate_mode,
                    "postprocess_mode": "translation" if options.translate_mode else "cleanup",
                    "cleanup_level": options.cleanup_level,
                    "asr_fallback_used": False,
                }
                if options.translate_mode or options.cleanup_level != "off":
                    status_key = (
                        "translation_status" if options.translate_mode else "cleanup_status"
                    )
                    text_key = "translated_text" if options.translate_mode else "cleaned_text"
                    result[text_key] = _join_segments(
                        [
                            processed_segments.get(index, text)
                            for index, text in enumerate(raw_segments)
                        ]
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
