import asyncio
import base64
import logging
import re
import threading
import time
from typing import Any, TypedDict

import httpx
import numpy as np
from pypinyin import Style, lazy_pinyin

from ..config import SAMPLE_RATE, default_config
from ..http_client import get_client
from .enrollment import current_embedding_fingerprint, get_enrollment_store, wav_digest
from .prompt_templates import audio_item
from .prompt_templates import build_primary_messages as _build_primary_messages
from .prompt_templates import sanitize_hotwords
from .recall import recall_audio_sync, recall_projector_sync

logger = logging.getLogger(__name__)
_sync_local = threading.local()
_CHINESE_WORD_RE = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+$")


class ASRResult(TypedDict):
    transcription: str
    reported_hotwords: list[str]
    raw_text: str
    detected_language: str | None


def build_primary_messages(
    target_wav_base64: str,
    *,
    hotwords: list[str] | None = None,
    enrollment_wav_base64: str | None = None,
    template: str | None = None,
) -> list[dict]:
    """Build primary ASR messages for the selected model prompt template."""
    return _build_primary_messages(
        target_wav_base64,
        hotwords=hotwords,
        enrollment_wav_base64=enrollment_wav_base64,
        template=template or default_config.vllm_prompt_template,
    )


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks).strip()
    return str(content or "")


def build_audio_only_messages(audio_wav_base64: str) -> list[dict]:
    """Single-audio prompt without any text — used by the Qwen3 secondary
    path, which is trained as a pure ASR model and ignores text guidance."""
    return [
        {
            "role": "user",
            "content": [audio_item(audio_wav_base64)],
        }
    ]


def _build_payload(
    messages: list[dict],
    *,
    model_name: str,
    repetition_penalty: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 512,
    }
    if repetition_penalty > 1.0:
        payload["repetition_penalty"] = repetition_penalty
    return payload


def _log_http_timing(
    *,
    request_kind: str,
    trace_id: str,
    status_code: int,
    post_elapsed: float,
    parse_elapsed: float,
    total_elapsed: float,
    raw_text: str,
    model_name: str,
    mode: str,
) -> None:
    if not getattr(default_config, "asr_post_chat_timing_enabled", False):
        return
    logger.info(
        "ASR_HTTP_TIMING kind=%s traceId=%s status=%s post_ms=%.1f "
        "parse_ms=%.1f total_ms=%.1f raw_chars=%d model=%s mode=%s",
        request_kind,
        trace_id or "-",
        status_code,
        post_elapsed * 1000.0,
        parse_elapsed * 1000.0,
        total_elapsed * 1000.0,
        len(raw_text),
        model_name,
        mode,
    )


def _log_split_timings(
    *,
    request_kind: str,
    trace_id: str,
    timings: dict[str, Any],
) -> None:
    if not getattr(default_config, "asr_post_chat_timing_enabled", False):
        return
    keys = (
        "audio_decode",
        "feature_extract",
        "feature_to_device",
        "encoder_wait",
        "encoder",
        "encoder_batch_size",
        "encoder_queue_ms",
        "microbatch_encode_dispatch",
        "serialize",
        "prepare",
        "decode_queue_wait",
        "generate",
        "generate_first_output",
        "generate_outputs",
        "generate_active_at_start",
        "total",
        "microbatch_wait",
        "microbatch_batch_size",
        "microbatch_dedupe_size",
        "partial_first_cache_hit",
        "partial_first_cache_wait",
    )
    parts = []
    for key in keys:
        value = timings.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"{key}={float(value):.1f}")
    if parts:
        logger.info(
            "ASR_SPLIT_TIMING kind=%s traceId=%s %s",
            request_kind,
            trace_id or "-",
            " ".join(parts),
        )


def _log_split_gap(
    *,
    request_kind: str,
    trace_id: str,
    mode: str,
    post_elapsed: float,
    total_elapsed: float,
    timings: dict[str, Any] | None,
) -> None:
    if not getattr(default_config, "asr_post_chat_timing_enabled", False):
        return
    if not isinstance(timings, dict):
        return
    sidecar_total = timings.get("total")
    if not isinstance(sidecar_total, (int, float)):
        return
    post_ms = post_elapsed * 1000.0
    total_ms = total_elapsed * 1000.0
    sidecar_total_ms = float(sidecar_total)
    logger.info(
        "ASR_SPLIT_GAP kind=%s traceId=%s mode=%s post_ms=%.1f "
        "api_total_ms=%.1f sidecar_total_ms=%.1f downstream_gap_ms=%.1f",
        request_kind,
        trace_id or "-",
        mode,
        post_ms,
        total_ms,
        sidecar_total_ms,
        max(0.0, post_ms - sidecar_total_ms),
    )


def _system_prompt_for_split(
    hotwords: list[str] | None,
) -> tuple[str, list[str]]:
    hws = sanitize_hotwords(hotwords)
    return "", hws


def _hotword_pronunciation_key(word: str) -> tuple[str, ...] | None:
    """Return a whole-word pinyin key for pure Chinese hotwords."""
    if not _CHINESE_WORD_RE.fullmatch(word):
        return None
    syllables = lazy_pinyin(
        word,
        style=Style.NORMAL,
        errors="ignore",
        strict=False,
    )
    if len(syllables) != len(word) or not all(syllables):
        return None
    return tuple(syllables)


def _merge_recalled_and_request_hotwords(
    recalled: list[str] | None,
    request_hotwords: list[str] | None,
    *,
    request_limit: int | None = None,
) -> list[str]:
    """Prefer request hotwords and suppress same-pronunciation recalls."""
    merged: list[str] = []
    seen: set[str] = set()
    request_pronunciations: set[tuple[str, ...]] = set()

    def add_word(raw: object) -> bool:
        word = str(raw or "").strip()
        if not word or word in seen:
            return False
        seen.add(word)
        merged.append(word)
        return True

    limit = (
        int(request_limit)
        if request_limit is not None
        else int(getattr(default_config, "recall_custom_hotword_limit", 32))
    )
    remaining = max(limit, 0)
    if remaining > 0:
        for word in request_hotwords or []:
            before = len(merged)
            add_word(word)
            if len(merged) > before:
                key = _hotword_pronunciation_key(merged[-1])
                if key is not None:
                    request_pronunciations.add(key)
                remaining -= 1
                if remaining <= 0:
                    break

    for word in recalled or []:
        normalized = str(word or "").strip()
        key = _hotword_pronunciation_key(normalized)
        if key is not None and key in request_pronunciations:
            continue
        add_word(normalized)

    return sanitize_hotwords(merged)


def _split_max_tokens_for_request(request_kind: str) -> int:
    if request_kind == "partial_first":
        return int(getattr(default_config, "asr_partial_first_max_tokens", 64))
    if request_kind == "partial_refresh":
        return int(getattr(default_config, "asr_partial_refresh_max_tokens", 128))
    return int(getattr(default_config, "asr_final_max_tokens", 512))


def _should_use_split_asr(
    *,
    prompt_template: str | None,
    enrollment_wav_base64: str | None,
    request_kind: str,
) -> bool:
    # NOTE: ``enrollment_wav_base64`` intentionally does NOT disqualify the split
    # path. Target-speaker (voiceprint) is implemented ON the split path: the
    # enrollment clip is encoded once by the split vLLM encoder into projector
    # frames and injected as the FIRST audio segment of ``decode_embeddings`` (see
    # ``_enrollment_embeds_sync`` and ``enrollment_audio_embeds_base64`` threading
    # below). The split decoder service exposes no /v1/chat/completions, so routing
    # enrollment finals to the non-split _post_chat path would 404 -> empty final.
    # For a genuine non-split vLLM chat deployment ``split_asr_enabled`` is False and
    # this returns False anyway, preserving the enrollment-in-prompt path there.
    _ = enrollment_wav_base64  # split enrollment handled via encoder embeds, not here
    template = prompt_template or default_config.vllm_prompt_template
    if (
        not bool(getattr(default_config, "split_asr_enabled", False))
        or template != "amphion_asr_1.7b"
    ):
        return False
    if not bool(getattr(default_config, "split_asr_final_only", False)):
        return True
    return request_kind in {"final", "final_primary", "stop_flush", "stop_flush_primary"}


def _split_asr_base_url(fallback_base_url: str | None) -> str:
    return (
        str(getattr(default_config, "split_asr_base_url", "") or "").strip()
        or fallback_base_url
        or default_config.vllm_base_url
    )


async def _post_split_asr(
    audio_wav_base64: str,
    hotwords: list[str] | None,
    *,
    base_url: str,
    timeout: float,
    trace_id: str,
    request_kind: str,
    audio_pcm: np.ndarray | None = None,
) -> ASRResult:
    return await asyncio.to_thread(
        _post_split_encode_decode_sync,
        audio_wav_base64,
        hotwords,
        base_url=base_url,
        timeout=timeout,
        trace_id=trace_id,
        request_kind=request_kind,
        audio_pcm=audio_pcm,
    )


def _post_split_asr_sync(
    audio_wav_base64: str,
    hotwords: list[str] | None,
    *,
    base_url: str,
    timeout: float,
    trace_id: str,
    request_kind: str,
    audio_pcm: np.ndarray | None = None,
    enrollment_audio_embeds_base64: str | None = None,
) -> ASRResult:
    return _post_split_encode_decode_sync(
        audio_wav_base64,
        hotwords,
        base_url=base_url,
        timeout=timeout,
        trace_id=trace_id,
        request_kind=request_kind,
        audio_pcm=audio_pcm,
        enrollment_audio_embeds_base64=enrollment_audio_embeds_base64,
    )


def _post_split_decode_embeddings_sync(
    audio_embeds_base64: str,
    hotwords: list[str] | None,
    *,
    base_url: str,
    timeout: float,
    trace_id: str,
    request_kind: str,
    enrollment_audio_embeds_base64: str | None = None,
) -> ASRResult:
    client = _get_sync_client()
    base = base_url.rstrip("/")
    system_prompt, _ = _system_prompt_for_split(hotwords)
    hws = sanitize_hotwords(hotwords)
    payload: dict[str, Any] = {
        "audio_embeds_base64": audio_embeds_base64,
        "max_tokens": _split_max_tokens_for_request(request_kind),
        "system_prompt": system_prompt,
        "hotwords": hws,
        "trace_id": trace_id,
        "request_kind": request_kind,
    }
    if enrollment_audio_embeds_base64:
        # Target-speaker: the decoder prepends these enrollment projector frames as
        # the first audio item and switches to the enrollment system prompt.
        payload["enrollment_audio_embeds_base64"] = enrollment_audio_embeds_base64
    total_start = time.monotonic()
    post_start = time.monotonic()
    resp = client.post(
        f"{base}/v1/asr/decode_embeddings",
        json=payload,
        timeout=timeout,
    )
    post_elapsed = time.monotonic() - post_start
    resp.raise_for_status()
    parse_start = time.monotonic()
    body = resp.json()
    raw_text = str(body.get("text") or "")
    parsed = parse_model_output(raw_text)
    parse_elapsed = time.monotonic() - parse_start
    split_timings = body.get("timings_ms")
    if isinstance(split_timings, dict):
        _log_split_timings(
            request_kind=f"{request_kind}_decode_embeddings",
            trace_id=trace_id,
            timings=split_timings,
        )
    total_elapsed = time.monotonic() - total_start
    _log_split_gap(
        request_kind=f"{request_kind}_decode_embeddings",
        trace_id=trace_id,
        mode="split_decode_embeddings_sync_thread",
        post_elapsed=post_elapsed,
        total_elapsed=total_elapsed,
        timings=split_timings if isinstance(split_timings, dict) else None,
    )
    _log_http_timing(
        request_kind=request_kind,
        trace_id=trace_id,
        status_code=resp.status_code,
        post_elapsed=post_elapsed,
        parse_elapsed=parse_elapsed,
        total_elapsed=total_elapsed,
        raw_text=raw_text,
        model_name=default_config.vllm_model_name,
        mode="split_decode_embeddings_sync_thread",
    )
    return parsed


def _post_split_encode_sync(
    audio_wav_base64: str,
    *,
    base_url: str,
    timeout: float,
    trace_id: str,
    request_kind: str,
    audio_pcm: np.ndarray | None = None,
) -> tuple[str, dict[str, Any]]:
    client = _get_sync_client()
    base = base_url.rstrip("/")
    audio_format = "wav"
    audio_base64 = audio_wav_base64
    if audio_pcm is not None:
        pcm = np.asarray(audio_pcm, dtype=np.float32)
        pcm_i16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2", copy=False)
        audio_base64 = base64.b64encode(pcm_i16.tobytes()).decode("ascii")
        audio_format = "pcm_s16le"
    payload: dict[str, Any] = {
        "audio_base64": audio_base64,
        "audio_format": audio_format,
        "output_dtype": "float16",
        "trace_id": trace_id,
    }
    total_start = time.monotonic()
    resp = client.post(
        f"{base}/v1/asr/encode",
        json=payload,
        timeout=timeout,
    )
    post_elapsed = time.monotonic() - total_start
    resp.raise_for_status()
    body = resp.json()
    split_timings = body.get("timings_ms")
    if isinstance(split_timings, dict):
        _log_split_timings(
            request_kind=f"{request_kind}_encode",
            trace_id=trace_id,
            timings=split_timings,
        )
    total_elapsed = time.monotonic() - total_start
    _log_split_gap(
        request_kind=f"{request_kind}_encode",
        trace_id=trace_id,
        mode="split_encode_sync_thread",
        post_elapsed=post_elapsed,
        total_elapsed=total_elapsed,
        timings=split_timings if isinstance(split_timings, dict) else None,
    )
    logger.info(
        "ASR_HTTP_TIMING kind=%s traceId=%s status=%s post_ms=%.1f "
        "parse_ms=0.0 total_ms=%.1f raw_chars=0 model=%s mode=split_encode_sync_thread",
        request_kind,
        trace_id or "-",
        resp.status_code,
        post_elapsed * 1000.0,
        total_elapsed * 1000.0,
        default_config.vllm_model_name,
    )
    audio_embeds_base64 = str(body.get("audio_embeds_base64") or "")
    if not audio_embeds_base64:
        raise RuntimeError("split encoder did not return audio_embeds_base64")
    return audio_embeds_base64, body


_ENROLLMENT_EMBEDS_CACHE: dict[str, str] = {}
_ENROLLMENT_EMBEDS_LOCK = threading.Lock()
_ENROLLMENT_EMBEDS_CACHE_MAX = 256


def _enrollment_cache_key(enrollment_id: str | None, enrollment_wav_base64: str) -> str:
    digest = wav_digest(enrollment_wav_base64)
    ident = enrollment_id or digest
    return f"{current_embedding_fingerprint()}|{ident}|{digest}"


def clear_enrollment_embedding_cache_for_tests() -> None:
    with _ENROLLMENT_EMBEDS_LOCK:
        _ENROLLMENT_EMBEDS_CACHE.clear()


def _enrollment_embeds_sync(
    enrollment_wav_base64: str | None,
    *,
    enrollment_id: str | None = None,
    base_url: str,
    timeout: float,
    trace_id: str,
    request_kind: str,
) -> str | None:
    """Encode the target-speaker enrollment clip into projector frames (once).

    Reuses the split vLLM encoder — the same encoder that produces utterance
    embeds — so no separate encoder is started. Lookup order is process cache,
    compatible persisted embedding, then lazy encode from the canonical WAV.
    """
    if not enrollment_wav_base64:
        return None
    key = _enrollment_cache_key(enrollment_id, enrollment_wav_base64)
    with _ENROLLMENT_EMBEDS_LOCK:
        cached = _ENROLLMENT_EMBEDS_CACHE.get(key)
    if cached is not None:
        logger.info(
            "ENROLLMENT_EMBEDDING source=memory_cache id=%s traceId=%s",
            enrollment_id or "-",
            trace_id or "-",
        )
        return cached

    store = get_enrollment_store()
    if enrollment_id:
        persisted = store.load_embedding(enrollment_id, enrollment_wav_base64)
        if persisted:
            with _ENROLLMENT_EMBEDS_LOCK:
                if len(_ENROLLMENT_EMBEDS_CACHE) >= _ENROLLMENT_EMBEDS_CACHE_MAX:
                    _ENROLLMENT_EMBEDS_CACHE.clear()
                _ENROLLMENT_EMBEDS_CACHE[key] = persisted
            logger.info(
                "ENROLLMENT_EMBEDDING source=persisted_cache id=%s traceId=%s",
                enrollment_id,
                trace_id or "-",
            )
            return persisted

    try:
        embeds_b64, encode_response = _post_split_encode_sync(
            enrollment_wav_base64,
            base_url=base_url,
            timeout=timeout,
            trace_id=f"{trace_id}-enroll" if trace_id else "enroll",
            request_kind=f"{request_kind}_enrollment_encode",
        )
    except Exception as exc:  # noqa: BLE001 - enrollment must never break the final
        logger.warning(
            "ENROLLMENT_EMBEDDING source=unavailable id=%s traceId=%s error=%s",
            enrollment_id or "-",
            trace_id or "-",
            exc,
        )
        return None
    with _ENROLLMENT_EMBEDS_LOCK:
        if len(_ENROLLMENT_EMBEDS_CACHE) >= _ENROLLMENT_EMBEDS_CACHE_MAX:
            _ENROLLMENT_EMBEDS_CACHE.clear()
        _ENROLLMENT_EMBEDS_CACHE[key] = embeds_b64
    if enrollment_id:
        store.persist_embedding(
            enrollment_id,
            enrollment_wav_base64,
            embeds_b64,
            encode_response=encode_response,
        )
    logger.info(
        "ENROLLMENT_EMBEDDING source=lazy_encode id=%s traceId=%s",
        enrollment_id or "-",
        trace_id or "-",
    )
    return embeds_b64


def precompute_enrollment_embedding_sync(
    enrollment_id: str,
    enrollment_wav_base64: str,
    *,
    base_url: str,
    timeout: float,
    trace_id: str = "enrollment-upload",
) -> bool:
    """Best-effort upload-time/background enrollment embedding precompute."""
    embeds = _enrollment_embeds_sync(
        enrollment_wav_base64,
        enrollment_id=enrollment_id,
        base_url=base_url,
        timeout=timeout,
        trace_id=trace_id,
        request_kind="enrollment_precompute",
    )
    return bool(embeds)


def _post_split_asr_with_projector_recall_sync(
    audio_wav_base64: str,
    hotwords: list[str] | None,
    *,
    base_url: str,
    timeout: float,
    trace_id: str,
    request_kind: str,
    audio_pcm: np.ndarray | None = None,
    hotword_pool_id: str = "",
    enrollment_audio_embeds_base64: str | None = None,
) -> ASRResult:
    total_start = time.monotonic()
    encode_start = time.monotonic()
    audio_embeds_base64, _ = _post_split_encode_sync(
        audio_wav_base64,
        base_url=base_url,
        timeout=timeout,
        trace_id=trace_id,
        request_kind=request_kind,
        audio_pcm=audio_pcm,
    )
    encode_ms = (time.monotonic() - encode_start) * 1000.0

    recall_start = time.monotonic()
    recalled = recall_projector_sync(
        audio_embeds_base64, default_config, hotword_pool_id=hotword_pool_id
    )
    recall_ms = (time.monotonic() - recall_start) * 1000.0
    logger.info(
        "HOTWORD_RECALL_PROJECTOR traceId=%s count=%d projector_len=%s",
        trace_id or "-",
        len(recalled.words),
        recalled.projector_len,
    )
    decode_start = time.monotonic()
    effective_hotwords = _merge_recalled_and_request_hotwords(
        recalled.words,
        hotwords,
    )
    result = _post_split_decode_embeddings_sync(
        audio_embeds_base64,
        effective_hotwords,
        base_url=base_url,
        timeout=timeout,
        trace_id=trace_id,
        request_kind=request_kind,
        enrollment_audio_embeds_base64=enrollment_audio_embeds_base64,
    )
    decode_ms = (time.monotonic() - decode_start) * 1000.0
    logger.info(
        "ASR_TIMING type=split_projector_recall traceId=%s "
        "encode_ms=%.1f recall_ms=%.1f "
        "decode_embeddings_ms=%.1f total_ms=%.1f recalled=%s "
        "request_hotwords=%s effective_hotwords=%s projector_len=%s",
        trace_id or "-",
        encode_ms,
        recall_ms,
        decode_ms,
        (time.monotonic() - total_start) * 1000.0,
        len(recalled.words),
        len(hotwords or []),
        len(effective_hotwords),
        recalled.projector_len,
    )
    return result


def _post_split_encode_decode_sync(
    audio_wav_base64: str,
    hotwords: list[str] | None,
    *,
    base_url: str,
    timeout: float,
    trace_id: str,
    request_kind: str,
    audio_pcm: np.ndarray | None = None,
    enrollment_audio_embeds_base64: str | None = None,
) -> ASRResult:
    total_start = time.monotonic()
    encode_start = time.monotonic()
    audio_embeds_base64, _ = _post_split_encode_sync(
        audio_wav_base64,
        base_url=base_url,
        timeout=timeout,
        trace_id=trace_id,
        request_kind=request_kind,
        audio_pcm=audio_pcm,
    )
    encode_ms = (time.monotonic() - encode_start) * 1000.0
    decode_start = time.monotonic()
    result = _post_split_decode_embeddings_sync(
        audio_embeds_base64,
        hotwords,
        base_url=base_url,
        timeout=timeout,
        trace_id=trace_id,
        request_kind=request_kind,
        enrollment_audio_embeds_base64=enrollment_audio_embeds_base64,
    )
    decode_ms = (time.monotonic() - decode_start) * 1000.0
    logger.info(
        "ASR_TIMING type=split_encode_decode traceId=%s "
        "encode_ms=%.1f decode_embeddings_ms=%.1f total_ms=%.1f hotwords=%s",
        trace_id or "-",
        encode_ms,
        decode_ms,
        (time.monotonic() - total_start) * 1000.0,
        len(hotwords or []),
    )
    return result


def _get_sync_client() -> httpx.Client:
    client = getattr(_sync_local, "client", None)
    if client is not None and not client.is_closed:
        return client
    cfg = default_config
    max_conn = max(1, int(getattr(cfg, "http_max_connections", 32)))
    max_keepalive = max(
        0, int(getattr(cfg, "http_max_keepalive_connections", 16))
    )
    limits = httpx.Limits(
        max_connections=max_conn,
        max_keepalive_connections=min(max_keepalive, max_conn),
    )
    client = httpx.Client(timeout=120.0, limits=limits)
    _sync_local.client = client
    return client


def _recall_hotwords_for_final(
    hotwords: list[str] | None,
    *,
    audio_pcm: np.ndarray | None,
    audio_sample_rate: int,
    request_kind: str,
    trace_id: str = "",
    hotword_pool_id: str = "",
) -> list[str] | None:
    if request_kind not in {"final", "final_primary", "stop_flush", "stop_flush_primary"}:
        return hotwords
    if audio_pcm is None or not bool(getattr(default_config, "enable_hotword_recall", False)):
        return hotwords
    start = time.monotonic()
    try:
        recalled = recall_audio_sync(
            audio_pcm,
            default_config,
            sample_rate=audio_sample_rate,
            hotword_pool_id=hotword_pool_id,
        )
        logger.info(
            "HOTWORD_RECALL_FINAL traceId=%s kind=%s elapsed_ms=%.1f "
            "input_hotwords=%s recalled=%s projector_len=%s audio_ms=%.1f",
            trace_id or "-",
            request_kind,
            (time.monotonic() - start) * 1000.0,
            len(hotwords or []),
            len(recalled.words),
            recalled.projector_len,
            len(audio_pcm) * 1000.0 / audio_sample_rate if audio_sample_rate > 0 else -1.0,
        )
        return _merge_recalled_and_request_hotwords(recalled.words, hotwords)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Triton hotword recall failed; using request hotwords: %s "
            "traceId=%s kind=%s elapsed_ms=%.1f",
            exc,
            trace_id or "-",
            request_kind,
            (time.monotonic() - start) * 1000.0,
        )
        return hotwords


def _parse_hotwords_field(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []
    lowered = text.lower()
    if lowered in {"n/a", "na", "none", "null", "-"}:
        return []
    return [item.strip() for item in re.split(r"[,，;；]", text) if item.strip()]


def _parse_language_field(value: str) -> str | None:
    v = str(value or "").strip()
    if not v:
        return None
    if v.lower() in {"n/a", "na", "none", "null", "-"}:
        return None
    return v


def _postprocess_asr_text(text: str) -> str:
    """Normalize provider-specific wrappers to plain transcription text."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(
        r"^\s*language\s+[A-Za-z\u4e00-\u9fff_-]+\s*<asr_text>\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^\s*language\s+[A-Za-z\u4e00-\u9fff_-]+\s*[:：-]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def detect_and_fix_repetitions(text: str, threshold: int = 20) -> str:
    """Collapse pathological decode loops while leaving normal text untouched."""
    source = str(text or "")
    if len(source) <= threshold:
        return source

    fixed = source
    max_unit = min(16, max(1, len(fixed) // (threshold + 1)))
    for unit_len in range(1, max_unit + 1):
        out: list[str] = []
        i = 0
        changed = False
        while i < len(fixed):
            unit = fixed[i : i + unit_len]
            if len(unit) < unit_len:
                out.append(fixed[i:])
                break
            count = 1
            j = i + unit_len
            while fixed[j : j + unit_len] == unit:
                count += 1
                j += unit_len
            if count > threshold:
                out.append(unit)
                i = j
                changed = True
            else:
                out.append(fixed[i])
                i += 1
        if changed:
            fixed = "".join(out)
    return fixed


def parse_model_output(
    raw_text: str,
    *,
    enable_repetition_fix: bool | None = None,
) -> ASRResult:
    """Parse model output wrappers and normalize to plain transcription text."""
    raw = str(raw_text or "").strip()
    if not raw:
        return ASRResult(
            transcription="",
            reported_hotwords=[],
            raw_text="",
            detected_language=None,
        )

    normalized = raw.replace("\\r\\n", "\n").replace("\\n", "\n")

    lang_m = re.search(
        r"(?:^|\n)\s*language\s*:\s*([^\n]*)",
        normalized,
        flags=re.IGNORECASE,
    )
    detected_language = (
        _parse_language_field(lang_m.group(1)) if lang_m else None
    )
    if detected_language is None:
        qwen_lang_m = re.search(
            r"(?:^|\n)\s*language\s+([A-Za-z\u4e00-\u9fff_-]+)\s*<asr_text>",
            normalized,
            flags=re.IGNORECASE,
        )
        detected_language = (
            _parse_language_field(qwen_lang_m.group(1)) if qwen_lang_m else None
        )

    hw_m = re.search(
        r"(?:^|\n)\s*hotwords\s*:\s*(.+?)(?=\n\s*(?:language|transcription)\s*:|\Z)",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not hw_m:
        hw_m = re.search(
            r"(?:^|\n)\s*hotwords\s*:\s*(.+?)(?=\n\s*[A-Za-z_]+\s*:|\Z)",
            normalized,
            flags=re.IGNORECASE | re.DOTALL,
        )
    reported_hotwords = (
        _parse_hotwords_field(hw_m.group(1)) if hw_m else []
    )

    hm = re.search(r"(?i)hotwords\s*:", normalized)
    tm = re.search(r"(?i)transcription\s*:", normalized)
    h_start = hm.start() if hm else -1
    t_start = tm.start() if tm else -1

    transcription = ""
    if tm:
        if h_start >= 0 and h_start < t_start:
            m_tr = re.search(
                r"(?:^|\n)\s*transcription\s*:\s*(.*)\Z",
                normalized,
                flags=re.IGNORECASE | re.DOTALL,
            )
            transcription = m_tr.group(1).strip() if m_tr else ""
        else:
            m_tr = re.search(
                r"(?:^|\n)\s*transcription\s*:\s*(.+?)(?=\n\s*hotwords\s*:|\Z)",
                normalized,
                flags=re.IGNORECASE | re.DOTALL,
            )
            transcription = (
                m_tr.group(1).strip() if m_tr else normalized.strip()
            )
    else:
        transcription = normalized.strip()

    transcription = _postprocess_asr_text(transcription)
    if enable_repetition_fix is None:
        enable_repetition_fix = default_config.enable_asr_repetition_fix
    if enable_repetition_fix:
        transcription = detect_and_fix_repetitions(transcription)

    return ASRResult(
        transcription=transcription,
        reported_hotwords=reported_hotwords,
        raw_text=raw,
        detected_language=detected_language,
    )


async def _post_chat(
    messages: list[dict],
    *,
    base_url: str,
    model_name: str,
    timeout: float,
    repetition_penalty: float = 1.0,
    trace_id: str = "",
    request_kind: str = "asr",
) -> ASRResult:
    client = get_client()
    base = base_url.rstrip("/")
    payload = _build_payload(
        messages,
        model_name=model_name,
        repetition_penalty=repetition_penalty,
    )
    transient_errors = (
        httpx.ConnectError,
        httpx.ReadError,
        httpx.RemoteProtocolError,
    )
    total_start = time.monotonic()
    post_elapsed = 0.0
    parse_elapsed = 0.0
    status_code = 0
    raw_text = ""
    for attempt in range(2):
        try:
            post_start = time.monotonic()
            resp = await client.post(
                f"{base}/v1/chat/completions",
                json=payload,
                timeout=timeout,
            )
            post_elapsed = time.monotonic() - post_start
            status_code = resp.status_code
            break
        except transient_errors:
            if attempt == 1:
                raise
            await asyncio.sleep(0.05)
    resp.raise_for_status()
    parse_start = time.monotonic()
    raw_text = _content_to_text(resp.json()["choices"][0]["message"]["content"])
    parsed = parse_model_output(raw_text)
    parse_elapsed = time.monotonic() - parse_start
    _log_http_timing(
        request_kind=request_kind,
        trace_id=trace_id,
        status_code=status_code,
        post_elapsed=post_elapsed,
        parse_elapsed=parse_elapsed,
        total_elapsed=time.monotonic() - total_start,
        raw_text=raw_text,
        model_name=model_name,
        mode="async",
    )
    return parsed


def _post_chat_sync(
    messages: list[dict],
    *,
    base_url: str,
    model_name: str,
    timeout: float,
    repetition_penalty: float = 1.0,
    trace_id: str = "",
    request_kind: str = "asr",
) -> ASRResult:
    client = _get_sync_client()
    base = base_url.rstrip("/")
    payload = _build_payload(
        messages,
        model_name=model_name,
        repetition_penalty=repetition_penalty,
    )
    transient_errors = (
        httpx.ConnectError,
        httpx.ReadError,
        httpx.RemoteProtocolError,
    )
    total_start = time.monotonic()
    post_elapsed = 0.0
    parse_elapsed = 0.0
    status_code = 0
    raw_text = ""
    for attempt in range(2):
        try:
            post_start = time.monotonic()
            resp = client.post(
                f"{base}/v1/chat/completions",
                json=payload,
                timeout=timeout,
            )
            post_elapsed = time.monotonic() - post_start
            status_code = resp.status_code
            break
        except transient_errors:
            if attempt == 1:
                raise
            time.sleep(0.05)
    resp.raise_for_status()
    parse_start = time.monotonic()
    raw_text = _content_to_text(resp.json()["choices"][0]["message"]["content"])
    parsed = parse_model_output(raw_text)
    parse_elapsed = time.monotonic() - parse_start
    _log_http_timing(
        request_kind=request_kind,
        trace_id=trace_id,
        status_code=status_code,
        post_elapsed=post_elapsed,
        parse_elapsed=parse_elapsed,
        total_elapsed=time.monotonic() - total_start,
        raw_text=raw_text,
        model_name=model_name,
        mode="sync_thread",
    )
    return parsed


async def query_audio_model(
    audio_wav_base64: str,
    hotwords: list[str] | None = None,
    *,
    src_lang: str = "N/A",  # accepted for callsite compatibility, ignored
    audio_pcm: np.ndarray | None = None,
    audio_sample_rate: int = SAMPLE_RATE,
    enrollment_wav_base64: str | None = None,
    enrollment_id: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    prompt_template: str | None = None,
    timeout: float | None = None,
    repetition_penalty: float | None = None,
    trace_id: str = "",
    request_kind: str = "primary",
    hotword_pool_id: str = "",
) -> ASRResult:
    """Primary ASR call.

    ``src_lang`` is intentionally not forwarded into the prompt. The
    primary model's prompt format is selected by ``prompt_template`` (or
    the configured default) so Amphion 4B and 1.7B can coexist without
    duplicating call sites.
    """
    _ = src_lang  # noqa: F841 — preserved for compatibility, see docstring
    effective_hotwords = _recall_hotwords_for_final(
        hotwords,
        audio_pcm=audio_pcm,
        audio_sample_rate=audio_sample_rate,
        request_kind=request_kind,
        trace_id=trace_id,
        hotword_pool_id=hotword_pool_id,
    )
    if _should_use_split_asr(
        prompt_template=prompt_template,
        enrollment_wav_base64=enrollment_wav_base64,
        request_kind=request_kind,
    ):
        split_base_url = _split_asr_base_url(base_url)
        split_timeout = timeout if timeout is not None else default_config.asr_request_timeout
        enrollment_audio_embeds_base64 = None
        if enrollment_wav_base64:
            enrollment_audio_embeds_base64 = await asyncio.to_thread(
                _enrollment_embeds_sync,
                enrollment_wav_base64,
                enrollment_id=enrollment_id,
                base_url=split_base_url,
                timeout=split_timeout,
                trace_id=trace_id,
                request_kind=request_kind,
            )
        return await asyncio.to_thread(
            _post_split_asr_sync,
            audio_wav_base64,
            effective_hotwords,
            base_url=split_base_url,
            timeout=split_timeout,
            trace_id=trace_id,
            request_kind=request_kind,
            audio_pcm=audio_pcm,
            enrollment_audio_embeds_base64=enrollment_audio_embeds_base64,
        )
    messages = build_primary_messages(
        audio_wav_base64,
        hotwords=effective_hotwords,
        enrollment_wav_base64=enrollment_wav_base64,
        template=prompt_template,
    )
    return await _post_chat(
        messages,
        base_url=base_url or default_config.vllm_base_url,
        model_name=model_name or default_config.vllm_model_name,
        timeout=timeout if timeout is not None else default_config.asr_request_timeout,
        repetition_penalty=(
            repetition_penalty
            if repetition_penalty is not None
            else default_config.asr_repetition_penalty
        ),
        trace_id=trace_id,
        request_kind=request_kind,
    )


def query_audio_model_sync(
    audio_wav_base64: str,
    hotwords: list[str] | None = None,
    *,
    src_lang: str = "N/A",
    audio_pcm: np.ndarray | None = None,
    audio_sample_rate: int = SAMPLE_RATE,
    enrollment_wav_base64: str | None = None,
    enrollment_id: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    prompt_template: str | None = None,
    timeout: float | None = None,
    repetition_penalty: float | None = None,
    trace_id: str = "",
    request_kind: str = "primary",
    hotword_pool_id: str = "",
) -> ASRResult:
    _ = src_lang
    if _should_use_split_asr(
        prompt_template=prompt_template,
        enrollment_wav_base64=enrollment_wav_base64,
        request_kind=request_kind,
    ):
        split_base_url = _split_asr_base_url(base_url)
        split_timeout = timeout if timeout is not None else default_config.asr_request_timeout
        is_final_kind = request_kind in {
            "final",
            "final_primary",
            "stop_flush",
            "stop_flush_primary",
        }
        # Target-speaker: encode the enrollment clip once (cached) into projector
        # frames and inject them into the VAD-final decode. Partials stay speaker-
        # agnostic (K2/CTC streaming); only finals go through the split decoder.
        enrollment_audio_embeds_base64 = None
        if is_final_kind and enrollment_wav_base64:
            enrollment_audio_embeds_base64 = _enrollment_embeds_sync(
                enrollment_wav_base64,
                enrollment_id=enrollment_id,
                base_url=split_base_url,
                timeout=split_timeout,
                trace_id=trace_id,
                request_kind=request_kind,
            )
        if is_final_kind and bool(getattr(default_config, "enable_hotword_recall", False)):
            try:
                return _post_split_asr_with_projector_recall_sync(
                    audio_wav_base64,
                    hotwords,
                    base_url=split_base_url,
                    timeout=split_timeout,
                    trace_id=trace_id,
                    request_kind=request_kind,
                    audio_pcm=audio_pcm,
                    hotword_pool_id=hotword_pool_id,
                    enrollment_audio_embeds_base64=enrollment_audio_embeds_base64,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "split projector hotword recall failed; falling back to split ASR: %s",
                    exc,
                )
        return _post_split_asr_sync(
            audio_wav_base64,
            hotwords,
            base_url=split_base_url,
            timeout=split_timeout,
            trace_id=trace_id,
            request_kind=request_kind,
            audio_pcm=audio_pcm,
            enrollment_audio_embeds_base64=enrollment_audio_embeds_base64,
        )
    effective_hotwords = _recall_hotwords_for_final(
        hotwords,
        audio_pcm=audio_pcm,
        audio_sample_rate=audio_sample_rate,
        request_kind=request_kind,
        trace_id=trace_id,
        hotword_pool_id=hotword_pool_id,
    )
    messages = build_primary_messages(
        audio_wav_base64,
        hotwords=effective_hotwords,
        enrollment_wav_base64=enrollment_wav_base64,
        template=prompt_template,
    )
    return _post_chat_sync(
        messages,
        base_url=base_url or default_config.vllm_base_url,
        model_name=model_name or default_config.vllm_model_name,
        timeout=timeout if timeout is not None else default_config.asr_request_timeout,
        repetition_penalty=(
            repetition_penalty
            if repetition_penalty is not None
            else default_config.asr_repetition_penalty
        ),
        trace_id=trace_id,
        request_kind=request_kind,
    )


async def query_audio_model_secondary(
    audio_wav_base64: str,
    hotwords: list[str] | None = None,
    *,
    base_url: str | None = None,
    model_name: str | None = None,
    timeout: float | None = None,
    trace_id: str = "",
    request_kind: str = "secondary",
) -> ASRResult:
    _ = hotwords
    messages = build_audio_only_messages(audio_wav_base64)
    return await _post_chat(
        messages,
        base_url=base_url or default_config.secondary_vllm_base_url,
        model_name=model_name or default_config.secondary_vllm_model_name,
        timeout=timeout if timeout is not None else default_config.asr_request_timeout,
        trace_id=trace_id,
        request_kind=request_kind,
    )


def query_audio_model_secondary_sync(
    audio_wav_base64: str,
    hotwords: list[str] | None = None,
    *,
    base_url: str | None = None,
    model_name: str | None = None,
    timeout: float | None = None,
    trace_id: str = "",
    request_kind: str = "secondary",
) -> ASRResult:
    _ = hotwords
    messages = build_audio_only_messages(audio_wav_base64)
    return _post_chat_sync(
        messages,
        base_url=base_url or default_config.secondary_vllm_base_url,
        model_name=model_name or default_config.secondary_vllm_model_name,
        timeout=timeout if timeout is not None else default_config.asr_request_timeout,
        trace_id=trace_id,
        request_kind=request_kind,
    )
