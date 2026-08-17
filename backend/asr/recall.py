"""Triton hotword recall client for RAG-ASR."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import threading
import time
import wave
from dataclasses import dataclass
from urllib.parse import urlparse
import urllib.request

import numpy as np

from ..config import SAMPLE_RATE, Config, Upstream, get_service_upstream

DEFAULT_RECALL_MODEL = "rag_asr_retrieve"
logger = logging.getLogger(__name__)
_SEMAPHORE_LOCK = threading.Lock()
_SEMAPHORES: dict[int, threading.BoundedSemaphore] = {}

# hotword_pool_id isolation key (contract V0.4-review). Empty string means the
# default pool (unchanged legacy behavior). A non-empty id is validated to a
# conservative charset so it can safely key per-pool storage/paths downstream.
_POOL_ID_MAX_LEN = 128
_POOL_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")


def normalize_hotword_pool_id(raw: object) -> str:
    """Normalize/validate a ``hotword_pool_id``.

    Returns ``""`` for an absent/blank id (the default pool). A non-empty id
    must match ``[A-Za-z0-9._:-]`` and be at most ``_POOL_ID_MAX_LEN`` chars;
    anything else raises ``ValueError`` so callers can surface a 400 rather
    than silently routing to the wrong (or default) pool.
    """
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    if len(text) > _POOL_ID_MAX_LEN:
        raise ValueError(
            f"hotword_pool_id too long (>{_POOL_ID_MAX_LEN} chars)"
        )
    if not _POOL_ID_RE.match(text):
        raise ValueError(
            "hotword_pool_id may only contain letters, digits, and . _ : -"
        )
    return text


@dataclass(frozen=True)
class RecallResult:
    words: list[str]
    projector_len: int | None = None


@dataclass(frozen=True)
class HotwordPoolResult:
    action: str
    status: str
    message: dict[str, object]
    hotwords: list[str]
    total_count: int | None = None
    hotword_pool_id: str = ""
    # Per-operation statistics passed through from the RAG-ASR backend
    # (e.g. added / deleted / missing / invalid / duplicates / matched_count).
    stats: dict[str, object] | None = None


# Backend CRUD statistics keys forwarded verbatim in HotwordPoolResult.stats.
_POOL_STAT_KEYS: tuple[str, ...] = (
    "added",
    "deleted",
    "cleared",
    "reloaded",
    "skipped_duplicates",
    "invalid",
    "duplicates",
    "missing",
    "matched_count",
)


def _extract_pool_stats(response: dict[str, object]) -> dict[str, object]:
    return {k: response[k] for k in _POOL_STAT_KEYS if k in response}


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _triton_url(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    parsed = urlparse(value if "://" in value else f"http://{value}")
    return parsed.netloc or parsed.path


def _recall_upstream() -> Upstream:
    upstream = get_service_upstream("recall")
    if upstream is None or not upstream.base_url:
        raise RuntimeError("Triton recall service is not configured")
    return upstream


def _client_for(upstream: Upstream):
    import tritonclient.http as httpclient

    return httpclient, httpclient.InferenceServerClient(url=_triton_url(upstream.base_url))


def _recall_backend() -> str:
    value = os.getenv("RAG_ASR_SERVICE_BACKEND", "auto").strip().lower()
    if value == "auto":
        value = "http"
    if value not in {"triton", "http"}:
        logger.warning("unknown RAG_ASR_SERVICE_BACKEND=%s; using triton", value)
        return "triton"
    return value


def _http_json_post(
    upstream: Upstream,
    path: str,
    payload: dict[str, object],
    *,
    timeout: float | None = None,
) -> dict[str, object]:
    base = upstream.base_url.strip().rstrip("/")
    url = f"{base}{path}"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(
        req,
        timeout=float(timeout if timeout is not None else (upstream.timeout or 30.0)),
    ) as resp:
        raw = resp.read().decode("utf-8")
    parsed = json.loads(raw) if raw else {}
    if not isinstance(parsed, dict):
        raise RuntimeError(f"unexpected RAG-ASR HTTP response from {url}: {type(parsed).__name__}")
    return parsed


def _http_retrieve_audio(
    upstream: Upstream,
    pcm: np.ndarray,
    *,
    sample_rate: int,
    top_k: int,
    hotword_pool_id: str = "",
) -> dict[str, object]:
    base = upstream.base_url.strip().rstrip("/")
    url = f"{base}/retrieve"
    wav = np.asarray(pcm, dtype=np.float32).reshape(-1)
    wav_i16 = (np.clip(wav, -1.0, 1.0) * 32767.0).astype(np.int16)

    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(wav_i16.tobytes())

    boundary = f"----rag-asr-{int(time.time() * 1000)}"
    parts: list[bytes] = [
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="top_k"\r\n\r\n'
            f"{top_k}\r\n"
        ).encode("utf-8"),
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="sample_rate"\r\n\r\n'
            f"{sample_rate}\r\n"
        ).encode("utf-8"),
    ]
    # Only send the id for non-default pools so the default pool keeps the exact
    # legacy request shape.
    if hotword_pool_id:
        parts.append(
            (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="hotword_pool_id"\r\n\r\n'
                f"{hotword_pool_id}\r\n"
            ).encode("utf-8")
        )
    body = b"".join(
        parts
        + [
            (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
                "Content-Type: audio/wav\r\n\r\n"
            ).encode("utf-8"),
            wav_buf.getvalue(),
            f"\r\n--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=float(upstream.timeout or 30.0)) as resp:
        raw = resp.read().decode("utf-8")
    parsed = json.loads(raw) if raw else {}
    if not isinstance(parsed, dict):
        raise RuntimeError(f"unexpected RAG-ASR HTTP response from {url}: {type(parsed).__name__}")
    return parsed


def _string_input(httpclient, name: str, value: str):
    tensor = httpclient.InferInput(name, [1], "BYTES")
    tensor.set_data_from_numpy(np.array([value], dtype=object))
    return tensor


def _int_input(httpclient, name: str, value: int):
    tensor = httpclient.InferInput(name, [1], "INT32")
    tensor.set_data_from_numpy(np.array([int(value)], dtype=np.int32))
    return tensor


def _optional_int_input(httpclient, name: str, value: int | None):
    if value is None:
        return None
    return _int_input(httpclient, name, value)


def _model_name(upstream: Upstream) -> str:
    return upstream.model_name or DEFAULT_RECALL_MODEL


def _semaphore(limit: int) -> threading.BoundedSemaphore:
    limit = max(1, int(limit))
    sem = _SEMAPHORES.get(limit)
    if sem is not None:
        return sem
    with _SEMAPHORE_LOCK:
        sem = _SEMAPHORES.get(limit)
        if sem is None:
            sem = threading.BoundedSemaphore(limit)
            _SEMAPHORES[limit] = sem
        return sem


def _decode_projector_base64(value: str) -> np.ndarray:
    import torch

    raw = base64.b64decode(value, validate=True)
    tensor = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if tensor.ndim == 3:
        if tensor.shape[0] != 1:
            raise ValueError(f"expected batch size 1 projector tensor, got {tuple(tensor.shape)}")
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"expected [T,H] projector tensor, got {tuple(tensor.shape)}")
    return tensor.detach().to(torch.float32).cpu().numpy().astype(np.float32, copy=False)


def recall_audio_sync(
    pcm: np.ndarray,
    cfg: Config,
    *,
    sample_rate: int = SAMPLE_RATE,
    hotword_pool_id: str = "",
) -> RecallResult:
    top_k = max(int(cfg.recall_top_k), 0)
    if top_k == 0:
        return RecallResult(words=[])
    pool_id = normalize_hotword_pool_id(hotword_pool_id)
    limit = max(1, int(getattr(cfg, "recall_max_concurrent", 4)))
    timeout_ms = max(0, int(getattr(cfg, "recall_queue_timeout_ms", 200)))
    sem = _semaphore(limit)
    queue_start = time.monotonic()
    acquired = sem.acquire(timeout=timeout_ms / 1000.0)
    queue_ms = (time.monotonic() - queue_start) * 1000.0
    if not acquired:
        raise TimeoutError(
            f"hotword recall queue timeout after {timeout_ms}ms (limit={limit})"
        )
    start = time.monotonic()
    upstream = _recall_upstream()
    prepare_ms = 0.0
    infer_ms = 0.0
    parse_ms = 0.0
    words_count = 0
    projector_len: int | None = None
    try:
        if _recall_backend() == "http":
            prepare_start = time.monotonic()
            payload = _http_retrieve_audio(
                upstream,
                pcm,
                sample_rate=sample_rate,
                top_k=top_k,
                hotword_pool_id=pool_id,
            )
            prepare_ms = (time.monotonic() - prepare_start) * 1000.0
            parse_start = time.monotonic()
            parsed_words = [str(word) for word in payload.get("word_list", [])]
            projector_len_raw = payload.get("projector_len")
            projector_len = None if projector_len_raw is None else int(projector_len_raw)
            words_count = len(parsed_words)
            parse_ms = (time.monotonic() - parse_start) * 1000.0
            infer_ms = prepare_ms
            prepare_ms = 0.0
            return RecallResult(words=parsed_words, projector_len=projector_len)
        if pool_id:
            raise RuntimeError(
                "hotword_pool_id recall requires the HTTP RAG-ASR backend; "
                "the Triton backend does not support per-pool retrieval"
            )
        httpclient, client = _client_for(upstream)
        prepare_start = time.monotonic()
        wav = np.asarray(pcm, dtype=np.float32).reshape(-1)
        inputs = [
            _string_input(httpclient, "ACTION", "infer"),
            httpclient.InferInput("WAV", wav.shape, "FP32"),
            _int_input(httpclient, "SAMPLE_RATE", sample_rate),
            _int_input(httpclient, "TOP_K", top_k),
        ]
        inputs[1].set_data_from_numpy(wav)
        outputs = [
            httpclient.InferRequestedOutput("WORD_LIST"),
            httpclient.InferRequestedOutput("PROJECTOR_LEN"),
        ]
        prepare_ms = (time.monotonic() - prepare_start) * 1000.0
        infer_start = time.monotonic()
        result = client.infer(_model_name(upstream), inputs, outputs=outputs)
        infer_ms = (time.monotonic() - infer_start) * 1000.0
        parse_start = time.monotonic()
        words = json.loads(_decode(result.as_numpy("WORD_LIST")[0]))
        projector_len = int(result.as_numpy("PROJECTOR_LEN")[0])
        parsed_words = [str(word) for word in words]
        words_count = len(parsed_words)
        parse_ms = (time.monotonic() - parse_start) * 1000.0
        return RecallResult(words=parsed_words, projector_len=projector_len)
    finally:
        sem.release()
        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "HOTWORD_RECALL_TIMING mode=audio queue_ms=%.1f prepare_ms=%.1f "
            "infer_ms=%.1f parse_ms=%.1f elapsed_ms=%.1f limit=%s "
            "timeout_ms=%s top_k=%s words=%s projector_len=%s audio_ms=%.1f",
            queue_ms,
            prepare_ms,
            infer_ms,
            parse_ms,
            elapsed_ms,
            limit,
            timeout_ms,
            top_k,
            words_count,
            projector_len,
            len(pcm) * 1000.0 / sample_rate if sample_rate > 0 else -1.0,
        )


def recall_projector_sync(
    audio_embeds_base64: str,
    cfg: Config,
    *,
    hotword_pool_id: str = "",
) -> RecallResult:
    top_k = max(int(cfg.recall_top_k), 0)
    if top_k == 0:
        return RecallResult(words=[])
    pool_id = normalize_hotword_pool_id(hotword_pool_id)
    limit = max(1, int(getattr(cfg, "recall_max_concurrent", 4)))
    timeout_ms = max(0, int(getattr(cfg, "recall_queue_timeout_ms", 200)))
    sem = _semaphore(limit)
    queue_start = time.monotonic()
    acquired = sem.acquire(timeout=timeout_ms / 1000.0)
    queue_ms = (time.monotonic() - queue_start) * 1000.0
    if not acquired:
        raise TimeoutError(
            f"hotword recall queue timeout after {timeout_ms}ms (limit={limit})"
        )
    start = time.monotonic()
    prepare_ms = 0.0
    infer_ms = 0.0
    parse_ms = 0.0
    words_count = 0
    projector_len: int | None = None
    try:
        upstream = _recall_upstream()
        if _recall_backend() == "http":
            infer_start = time.monotonic()
            projector_payload: dict[str, object] = {
                "audio_embeds_base64": audio_embeds_base64,
                "top_k": top_k,
            }
            # Only send the id for non-default pools so the default pool keeps
            # the exact legacy request shape.
            if pool_id:
                projector_payload["hotword_pool_id"] = pool_id
            response = _http_json_post(
                upstream,
                "/retrieve/projector",
                projector_payload,
            )
            infer_ms = (time.monotonic() - infer_start) * 1000.0
            parse_start = time.monotonic()
            words = response.get("word_list", [])
            projector_len = response.get("projector_len")
            parsed_words = [str(word) for word in words]
            words_count = len(parsed_words)
            parse_ms = (time.monotonic() - parse_start) * 1000.0
            return RecallResult(
                words=parsed_words,
                projector_len=None if projector_len is None else int(projector_len),
            )
        # Triton path: the packaged rag_asr_retrieve model has no pool-id input.
        # 910b delivery runs the HTTP backend; fail loudly rather than silently
        # recalling from the wrong (default) pool.
        if pool_id:
            raise RuntimeError(
                "hotword_pool_id recall requires the HTTP RAG-ASR backend; "
                "the Triton backend does not support per-pool retrieval"
            )
        httpclient, client = _client_for(upstream)
        prepare_start = time.monotonic()
        projector = _decode_projector_base64(audio_embeds_base64)
        inputs = [
            _string_input(httpclient, "ACTION", "projector"),
            httpclient.InferInput("PROJECTOR_IN", projector.shape, "FP32"),
            _int_input(httpclient, "PROJECTOR_IN_LEN", int(projector.shape[0])),
            _int_input(httpclient, "TOP_K", top_k),
        ]
        inputs[1].set_data_from_numpy(projector)
        outputs = [
            httpclient.InferRequestedOutput("WORD_LIST"),
            httpclient.InferRequestedOutput("PROJECTOR_LEN"),
        ]
        prepare_ms = (time.monotonic() - prepare_start) * 1000.0
        infer_start = time.monotonic()
        result = client.infer(_model_name(upstream), inputs, outputs=outputs)
        infer_ms = (time.monotonic() - infer_start) * 1000.0
        parse_start = time.monotonic()
        words = json.loads(_decode(result.as_numpy("WORD_LIST")[0]))
        projector_len = int(result.as_numpy("PROJECTOR_LEN")[0])
        parsed_words = [str(word) for word in words]
        words_count = len(parsed_words)
        parse_ms = (time.monotonic() - parse_start) * 1000.0
        return RecallResult(words=parsed_words, projector_len=projector_len)
    finally:
        sem.release()
        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "HOTWORD_RECALL_TIMING mode=projector queue_ms=%.1f "
            "prepare_ms=%.1f infer_ms=%.1f parse_ms=%.1f elapsed_ms=%.1f "
            "limit=%s timeout_ms=%s top_k=%s words=%s projector_len=%s",
            queue_ms,
            prepare_ms,
            infer_ms,
            parse_ms,
            elapsed_ms,
            limit,
            timeout_ms,
            top_k,
            words_count,
            projector_len,
        )


def manage_hotword_pool_sync(
    action: str,
    *,
    hotwords: list[str] | None = None,
    query: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    hotword_pool_id: str = "",
) -> HotwordPoolResult:
    """Manage the RAG-ASR hotword pool for ``hotword_pool_id`` (default pool
    when empty)."""
    normalized_action = action.strip().lower()
    if normalized_action not in {"list", "add", "delete", "remove", "reload", "clear"}:
        raise ValueError(f"unsupported hotword pool action: {action}")
    pool_id = normalize_hotword_pool_id(hotword_pool_id)

    upstream = _recall_upstream()
    if _recall_backend() == "http":
        request_timeout = None
        if normalized_action in {"reload", "clear"}:
            request_timeout = max(
                float(upstream.timeout or 30.0),
                float(os.getenv("RAG_ASR_MANAGE_TIMEOUT", "300")),
            )
        payload: dict[str, object] = {
            "action": normalized_action,
            "offset": int(offset),
        }
        # Only send the id for non-default pools so the default pool keeps the
        # exact legacy request shape (backward-compatible with older RAG-ASR).
        if pool_id:
            payload["hotword_pool_id"] = pool_id
        if normalized_action in {"add", "delete", "remove"}:
            payload["hotwords"] = list(hotwords or [])
        if normalized_action == "list":
            if query:
                payload["query"] = query
            if limit is not None:
                payload["limit"] = int(limit)
        response = _http_json_post(
            upstream,
            "/hotword-pool/action",
            payload,
            timeout=request_timeout,
        )
        status = str(response.get("status", "ok"))
        message_obj = response.get("message", {})
        if isinstance(message_obj, dict):
            message = message_obj
        else:
            message = {"message": str(message_obj)}
        pool_words = response.get("hotwords", [])
        total_count = response.get("total_count")
        # Echo back the caller's normalized pool id; older backends that don't
        # yet return it still report the id the request targeted.
        resp_pool_id = normalize_hotword_pool_id(
            response.get("hotword_pool_id") or pool_id
        )
        return HotwordPoolResult(
            action=normalized_action,
            status=status,
            message=message,
            hotwords=[str(word) for word in pool_words],
            total_count=None if total_count is None else int(total_count),
            hotword_pool_id=resp_pool_id,
            stats=_extract_pool_stats(response),
        )
    # Triton path: the packaged rag_asr_retrieve model has no pool-id input, so
    # a non-default pool cannot be isolated here. Per delivery rules the running
    # backend is HTTP; fail loudly rather than silently mixing pools.
    if pool_id:
        raise RuntimeError(
            "hotword_pool_id isolation requires the HTTP RAG-ASR backend; "
            "the Triton backend does not support per-pool management"
        )
    httpclient, client = _client_for(upstream)
    inputs = [_string_input(httpclient, "ACTION", normalized_action)]
    if normalized_action in {"add", "delete", "remove"}:
        inputs.append(
            _string_input(
                httpclient,
                "HOTWORDS",
                json.dumps(list(hotwords or []), ensure_ascii=False),
            )
        )
    if normalized_action == "list":
        if query:
            inputs.append(_string_input(httpclient, "QUERY", query))
        limit_input = _optional_int_input(httpclient, "LIMIT", limit)
        if limit_input is not None:
            inputs.append(limit_input)
        inputs.append(_int_input(httpclient, "OFFSET", offset))

    outputs = [
        httpclient.InferRequestedOutput("STATUS"),
        httpclient.InferRequestedOutput("MESSAGE"),
        httpclient.InferRequestedOutput("HOTWORD_COUNT"),
        httpclient.InferRequestedOutput("HOTWORD_LIST"),
    ]
    result = client.infer(_model_name(upstream), inputs, outputs=outputs)
    status = _decode(result.as_numpy("STATUS")[0])
    message_raw = _decode(result.as_numpy("MESSAGE")[0])
    try:
        message = json.loads(message_raw)
    except json.JSONDecodeError:
        message = {"message": message_raw}
    hotword_list_raw = _decode(result.as_numpy("HOTWORD_LIST")[0])
    try:
        pool_words = json.loads(hotword_list_raw)
    except json.JSONDecodeError:
        pool_words = []
    total_count_arr = result.as_numpy("HOTWORD_COUNT")
    total_count = int(total_count_arr[0]) if total_count_arr is not None else None
    return HotwordPoolResult(
        action=normalized_action,
        status=status,
        message=message,
        hotwords=[str(word) for word in pool_words],
        total_count=total_count,
    )


async def recall_audio(
    pcm: np.ndarray,
    cfg: Config,
    *,
    sample_rate: int = SAMPLE_RATE,
    hotword_pool_id: str = "",
) -> RecallResult:
    return await asyncio.to_thread(
        recall_audio_sync,
        pcm,
        cfg,
        hotword_pool_id=hotword_pool_id,
        sample_rate=sample_rate,
    )


async def recall_projector(
    audio_embeds_base64: str,
    cfg: Config,
    *,
    hotword_pool_id: str = "",
) -> RecallResult:
    return await asyncio.to_thread(
        recall_projector_sync,
        audio_embeds_base64,
        cfg,
        hotword_pool_id=hotword_pool_id,
    )


async def manage_hotword_pool(
    action: str,
    *,
    hotwords: list[str] | None = None,
    query: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    hotword_pool_id: str = "",
) -> HotwordPoolResult:
    return await asyncio.to_thread(
        manage_hotword_pool_sync,
        action,
        hotwords=hotwords,
        query=query,
        limit=limit,
        offset=offset,
        hotword_pool_id=hotword_pool_id,
    )
