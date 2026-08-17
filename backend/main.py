import asyncio
import base64
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import websockets
from fastapi import (
    Body,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.types import Receive, Scope, Send

from .asr.enrollment import (
    EnrollmentError,
    decode_and_validate,
    get_enrollment_store,
)
from .asr.client import precompute_enrollment_embedding_sync
from .asr.jobs import get_transcription_job_store
from .asr.oneshot import OneshotAsrError, run_oneshot_asr
from .asr.recall import manage_hotword_pool, normalize_hotword_pool_id
from .asr.ctc_streaming import maybe_warmup_from_app as maybe_warmup_ctc_from_app
from .asr.sherpa_streaming import maybe_warmup_from_app
from .asr.transcribe import float_pcm_to_i16_bytes
from .audio.utils import wav_base64_to_pcm_16k_mono, wav_bytes_to_pcm_16k_mono
from .config import SAMPLE_RATE, load_config, load_transcribe_config
from .emotion.client import query_emotion_model
from .emotion.jobs import JobQueueFullError, get_emotion_job_store
from .emotion.service import EmotionDecodeError, decode_wav_capped
from .emotion_spec.jobs import get_emotion_spec_job_store
from .http_client import close_client, get_client
from .session import AudioSession
from .asr.ctc_streaming import config_from_app as ctc_config_from_app
from .streaming import AstV3Protocol, StreamingSession, VadSegmentedStream
from .streaming.ascend_k2_stream import AscendK2Stream
from .tasks import AsrTaskEngine, EmotionTaskEngine
from .text_cleanup import clean_asr_text
from .text_cleanup.client import TextCleanupConfigError

logging.basicConfig(level=logging.INFO)


# High-frequency per-segment / per-partial timing logs (ASR_TIMING, VAD_TIMING,
# STREAM_QUEUE_TIMING, ASR_SPLIT/HTTP_TIMING, ...) dominate log output under
# BS52 — a 10-minute run emits ~270k lines (~450/s). Formatting each record
# holds the GIL and the docker json-file handler serializes the writes, which
# measurably slows the single event loop / feed pipeline and feeds the ~1.2%/s
# server real-time deficit that accumulates vad_lag (design F12). Drop these
# records in a logging Filter *before* the (expensive) %-formatting and I/O.
# Set ASR_PERF_LOG=1 to restore full timing traces for debugging.
if os.environ.get("ASR_PERF_LOG", "").strip().lower() not in ("1", "true", "yes"):
    _PERF_LOG_TOKENS = (
        "ASR_TIMING",
        "ASR_SPLIT_TIMING",
        "ASR_SPLIT_GAP",
        "ASR_HTTP_TIMING",
        "HOTWORD_RECALL_TIMING",
        "VAD_TIMING",
        "STREAM_QUEUE_TIMING",
        "STREAM_INPUT_TIMING",
        "CTC_ONLINE_STATUS",
        "CTC_ONLINE_BATCH_TIMING",
    )

    class _PerfLogFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.msg
            if isinstance(msg, str) and msg[:24].lstrip().startswith(
                _PERF_LOG_TOKENS
            ):
                return False
            return True

    _perf_filter = _PerfLogFilter()
    for _h in logging.getLogger().handlers:
        _h.addFilter(_perf_filter)

logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def _make_asr_stream(cfg=None) -> VadSegmentedStream | AscendK2Stream:
    """Return the appropriate AudioStream for ASR based on config."""
    if cfg is None:
        cfg = load_config()
    if not bool(getattr(cfg, "ascend_k2_enabled", False)):
        return VadSegmentedStream()
    ctc_cfg = ctc_config_from_app(cfg)
    stream = AscendK2Stream(ctc_cfg)
    logger.info("AscendK2Stream selected (ascend_k2_enabled=True)")
    return stream


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    backend = str(getattr(cfg, "streaming_partial_backend", "vllm")).lower()
    if backend in {"sherpa", "k2_om"}:
        logger.info("Warming streaming partial recognizer")
        await asyncio.to_thread(maybe_warmup_from_app, cfg)
    elif backend == "ctc_om":
        logger.info("Warming CTC OM streaming partial runtime")
        await asyncio.to_thread(maybe_warmup_ctc_from_app, cfg)
    yield
    await close_client()


app = FastAPI(title="AudioLLM Server", lifespan=lifespan)


class HotwordPoolUpdateRequest(BaseModel):
    hotwords: list[str] = Field(default_factory=list)
    hotword_pool_id: str = ""


class HotwordPoolScopeRequest(BaseModel):
    """Body for pool-scoped actions (clear/reload) that carry only the pool id."""

    hotword_pool_id: str | None = None


def _normalize_pool_id_or_400(raw: object) -> str:
    """Normalize a hotword_pool_id, mapping validation errors to HTTP 400."""
    try:
        return normalize_hotword_pool_id(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _resolve_pool_id_or_400(body_raw: object, query_raw: object) -> str:
    """Resolve ``hotword_pool_id`` from an optional body value and query param.

    Contract (最终版 §5/§6): both absent -> default pool (""); exactly one
    present -> use it; both present and inconsistent -> 400 parameter error.
    """
    body_present = body_raw is not None and str(body_raw).strip() != ""
    query_present = query_raw is not None and str(query_raw).strip() != ""
    if body_present and query_present:
        body_id = _normalize_pool_id_or_400(body_raw)
        query_id = _normalize_pool_id_or_400(query_raw)
        if body_id != query_id:
            raise HTTPException(
                status_code=400,
                detail="conflicting hotword_pool_id in body and query",
            )
        return body_id
    if body_present:
        return _normalize_pool_id_or_400(body_raw)
    if query_present:
        return _normalize_pool_id_or_400(query_raw)
    return ""


def _hotword_pool_payload(result) -> dict[str, object]:
    # Some backends double-wrap the human message as {"message": "..."}; flatten
    # it so the customer response carries a plain string per the interface doc.
    message = result.message
    if isinstance(message, dict):
        message = message.get("message", message)
    payload: dict[str, object] = {
        "action": result.action,
        "status": result.status,
        "message": message,
        "hotwords": result.hotwords,
        "hotword_count": result.total_count,
        "total_count": result.total_count,
        "hotword_pool_id": result.hotword_pool_id,
    }
    # Pass through per-operation statistics (added/deleted/invalid/...) so
    # clients get the same CRUD detail the RAG-ASR backend reports.
    stats = dict(result.stats) if result.stats else {}
    if stats:
        payload.update(stats)
        # Interface-doc (TMGenius 语音识别接口 §3/§4) aliases. The RAG-ASR backend
        # reports terse keys (added/skipped_duplicates/invalid/duplicates/deleted/
        # missing); the customer contract names them *_count / ignored_hotwords /
        # missing_hotwords. Emit both so CAgent can parse the documented names
        # while existing consumers of the terse keys keep working.
        invalid = list(stats.get("invalid") or [])
        duplicates = list(stats.get("duplicates") or [])
        missing = list(stats.get("missing") or [])
        if "added" in stats:
            payload["added_count"] = int(stats.get("added") or 0)
            payload["duplicate_count"] = int(stats.get("skipped_duplicates") or 0)
            payload["invalid_count"] = len(invalid)
            # Words present in the request but not added (invalid + duplicate).
            payload["ignored_hotwords"] = invalid + duplicates
        if "deleted" in stats:
            payload["deleted_count"] = int(stats.get("deleted") or 0)
            payload["missing_count"] = len(missing)
            payload["missing_hotwords"] = missing
    return payload



@app.get("/health")
async def health():
    """Deployment health check for container probes and customer runbooks."""
    cfg = load_config()
    split_enabled = bool(getattr(cfg, "split_asr_enabled", False))
    split_final_only = bool(getattr(cfg, "split_asr_final_only", False))
    partial_backend = str(
        getattr(cfg, "streaming_partial_backend", "vllm") or "vllm"
    ).strip().lower()
    vllm_required = (
        not split_enabled
        or not split_final_only
        or partial_backend != "sherpa"
    )
    split_base_url = (
        str(getattr(cfg, "split_asr_base_url", "") or "").strip()
        or cfg.vllm_base_url
    )
    vllm_url = (
        f"{cfg.vllm_base_url.rstrip('/')}/v1/models"
        if split_final_only or not split_enabled
        else f"{cfg.vllm_base_url.rstrip('/')}/health"
    )
    split_url = f"{split_base_url.rstrip('/')}/health"
    vllm: dict[str, object] = {
        "ready": False,
        "url": vllm_url,
        "split_asr": split_enabled,
        "required": vllm_required,
    }
    if vllm_required:
        try:
            resp = await get_client().get(vllm_url, timeout=3.0)
            vllm["status_code"] = resp.status_code
            vllm["ready"] = 200 <= resp.status_code < 300
            if vllm["ready"]:
                data = resp.json()
                vllm["served_models"] = [
                    item.get("id")
                    for item in data.get("data", [])
                    if isinstance(item, dict) and item.get("id")
                ]
        except Exception as exc:
            vllm["error"] = str(exc)
    else:
        vllm["ready"] = True
        vllm["skipped_reason"] = "streaming partials use sherpa and finals use split ASR"
    split_asr: dict[str, object] | None = None
    if split_enabled:
        split_asr = {
            "ready": False,
            "url": split_url,
            "final_only": split_final_only,
        }
        try:
            resp = await get_client().get(split_url, timeout=3.0)
            split_asr["status_code"] = resp.status_code
            split_asr["ready"] = 200 <= resp.status_code < 300
        except Exception as exc:
            split_asr["error"] = str(exc)

    return {
        "service": "AudioLLM ASR API Service",
        "asr": {
            "base_url": cfg.vllm_base_url,
            "split_base_url": split_base_url if split_enabled else "",
            "model": cfg.vllm_model_name,
            "prompt_template": cfg.vllm_prompt_template,
            "split_enabled": split_enabled,
            "split_final_only": split_final_only,
            "streaming_partial_backend": partial_backend,
            "secondary_enabled": cfg.enable_secondary_asr,
        },
        "vllm": vllm,
        "split_asr": split_asr,
    }


@app.websocket("/ws/audio")
async def audio_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected (/ws/audio)")
    session = AudioSession(websocket)
    try:
        await session.run()
    finally:
        await session.cleanup()


@app.websocket("/transcribe-streaming")
async def transcribe_streaming_ws(websocket: WebSocket, language: str = ""):
    await websocket.accept()
    logger.info("Transcribe-streaming connected (language=%s)", language)
    cfg = load_config()
    session = StreamingSession(
        websocket,
        stream=_make_asr_stream(cfg),
        engine=AsrTaskEngine(),
        language=language,
    )
    try:
        await session.run()
    finally:
        await session.cleanup()


@app.websocket("/tuling/ast/v3")
async def tuling_ast_v3_ws(websocket: WebSocket):
    """iFlytek Tuling AST v3 streaming ASR.

    Same VAD-segmented dual-ASR pipeline as ``/transcribe-streaming``, but the
    on-the-wire framing is the AST v3 ``header/parameter/payload`` envelope:
    audio arrives base64-encoded inside JSON frames, ``header.status`` (0/1/2)
    drives start/stop, and results are repackaged into the ``payload.result``
    lattice. ``AstV3Protocol`` owns all of that translation; the session,
    stream, and engine are the shared ones. ``emit_timing`` lets the engine
    surface segment ``bg``/``ed`` to the protocol.

    See ``docs/tuling-ast-v3-protocol.md``.
    """
    route_enter_at = time.monotonic()
    client = getattr(websocket, "client", None)
    client_addr = f"{client.host}:{client.port}" if client else "-"
    logger.info("WS_SESSION event=route_enter path=/tuling/ast/v3 client=%s", client_addr)
    await websocket.accept()
    logger.info(
        "WS_SESSION event=accepted path=/tuling/ast/v3 client=%s accept_ms=%.1f",
        client_addr,
        (time.monotonic() - route_enter_at) * 1000.0,
    )
    logger.info("Tuling AST v3 connected (/tuling/ast/v3)")
    # Endpoint policy: primary-only (no secondary / no local Qwen / no fusion),
    # with the primary pinned to the AST v3-specific upstream when configured
    # (empty astv3_vllm_* falls back to the global primary). These are forced
    # overrides (see StreamingSession._config_overrides), re-applied after the
    # client's start.config so a client cannot re-enable secondary via
    # parameter.asr_config.
    cfg = load_config()
    astv3_overrides: dict[str, object] = {"enable_secondary_asr": False}
    if cfg.astv3_vllm_base_url:
        astv3_overrides["vllm_base_url"] = cfg.astv3_vllm_base_url
    if cfg.astv3_vllm_model_name:
        astv3_overrides["vllm_model_name"] = cfg.astv3_vllm_model_name
    if cfg.astv3_vllm_prompt_template:
        astv3_overrides["vllm_prompt_template"] = cfg.astv3_vllm_prompt_template
    session = StreamingSession(
        websocket,
        stream=_make_asr_stream(cfg),
        engine=AsrTaskEngine(emit_timing=True),
        protocol=AstV3Protocol(),
        config_overrides=astv3_overrides,
    )
    try:
        await session.run()
    finally:
        cleanup_start = time.monotonic()
        await session.cleanup()
        logger.info(
            "WS_SESSION event=cleanup_done path=/tuling/ast/v3 client=%s cleanup_ms=%.1f",
            client_addr,
            (time.monotonic() - cleanup_start) * 1000.0,
        )


# Hard-coded remote AST v3 backend that the "实时语音识别（测试用）" page targets.
# It speaks plaintext ws://, so an HTTPS-served frontend (playground.amphion.top)
# cannot open it directly — the browser's mixed-content policy forbids ws:// from
# an https:// page. The same-origin proxy below bridges the browser's wss://
# connection to it. Temporary test scaffolding: the address is intentionally
# pinned here, not exposed as config.
ASTV3_TEST_PROXY_UPSTREAM = "ws://159.138.9.106:18082/tuling/ast/v3"


async def _astv3_proxy_pump_to_upstream(client: WebSocket, upstream) -> None:
    """Relay browser -> upstream frames verbatim (text or binary)."""
    try:
        while True:
            message = await client.receive()
            if message["type"] == "websocket.disconnect":
                break
            text = message.get("text")
            if text is not None:
                await upstream.send(text)
                continue
            data = message.get("bytes")
            if data is not None:
                await upstream.send(data)
    except (WebSocketDisconnect, websockets.ConnectionClosed, RuntimeError):
        pass
    finally:
        # Closing the upstream unblocks the opposite pump's async-for so the
        # whole proxy tears down once either side goes away.
        await upstream.close()


async def _astv3_proxy_pump_to_client(upstream, client: WebSocket) -> None:
    """Relay upstream -> browser frames verbatim (text or binary)."""
    try:
        async for message in upstream:
            if isinstance(message, (bytes, bytearray)):
                await client.send_bytes(bytes(message))
            else:
                await client.send_text(message)
    except (websockets.ConnectionClosed, WebSocketDisconnect, RuntimeError):
        pass
    finally:
        try:
            await client.close()
        except RuntimeError:
            # client transport already closed; nothing to do
            pass


@app.websocket("/astv3-test-proxy")
async def astv3_test_proxy_ws(websocket: WebSocket):
    """Same-origin WebSocket proxy for the AST v3 test page.

    The "实时语音识别（测试用）" page is served over HTTPS but its target backend
    speaks plaintext ws:// (``ASTV3_TEST_PROXY_UPSTREAM``), which the browser's
    mixed-content policy forbids opening from an HTTPS page. This endpoint accepts
    the browser's same-origin (wss://) connection and relays every frame, in both
    directions, to/from that upstream without inspecting the AST v3 envelope. It
    is a transparent byte pump, so the on-the-wire contract is identical to
    ``/tuling/ast/v3`` (see ``docs/tuling-ast-v3-protocol.md``).

    Temporary test scaffolding: the upstream address is hard-coded.
    """
    await websocket.accept()
    logger.info("AST v3 test proxy connected -> %s", ASTV3_TEST_PROXY_UPSTREAM)
    try:
        async with websockets.connect(
            ASTV3_TEST_PROXY_UPSTREAM, max_size=None, open_timeout=10
        ) as upstream:
            await asyncio.gather(
                _astv3_proxy_pump_to_upstream(websocket, upstream),
                _astv3_proxy_pump_to_client(upstream, websocket),
            )
    except WebSocketDisconnect:
        # Browser hung up before/while the upstream was being dialed.
        pass
    except Exception as exc:  # upstream connect / handshake failure
        logger.warning("AST v3 test proxy upstream error: %s", exc)
        try:
            await websocket.close(code=1011)
        except RuntimeError:
            pass


@app.websocket("/emotion-segmented-streaming")
async def emotion_segmented_streaming_ws(websocket: WebSocket, language: str = ""):
    await websocket.accept()
    logger.info(
        "Emotion-segmented-streaming connected (language=%s)", language
    )
    session = StreamingSession(
        websocket,
        # Emotion has no partial output, so disable VAD's snapshot bookkeeping
        # regardless of the global pseudo-stream toggle.
        stream=VadSegmentedStream(enable_partial=False),
        engine=EmotionTaskEngine(streaming=True),
        language=language,
    )
    try:
        await session.run()
    finally:
        await session.cleanup()


# ---------------------------------------------------------------------------
# One-shot upload endpoints
# ---------------------------------------------------------------------------
# The /api/* routes power the "Upload audio file" buttons in the demos.
# They deliberately bypass the WebSocket/VAD pipeline: the frontend hands us
# a fully-decoded 16 kHz mono WAV (produced via the browser's Web Audio API),
# and we forward the bytes to the same vLLM endpoints the streaming engines
# call. This keeps the upload flow as "send the whole clip, get one final
# result" — no chunking, no VAD segmentation, no partials.
#
# All caps that the streaming pipeline normally enforces server-side
# (emotion 20s tail, ASR 60s tail) are still applied here so a malicious or
# buggy client can't bypass them by switching from WS to REST.

# Hard cap on multipart upload bytes. ~16-bit / 16 kHz mono WAV at 60 s is
# ~1.9 MB; this 25 MB ceiling is generous for any clip the model would
# realistically be asked to handle.
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# Server-side trim caps (mirror the streaming pipeline's behaviour so REST
# and WS produce identical model inputs for the same recording).
_ASR_MAX_SECONDS = 60.0


def _parse_csv(raw: str | None) -> list[str]:
    """Parse a ``"a,b ,c"`` form field into a clean string list."""
    if not raw:
        return []
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


async def _read_audio_bytes(
    audio: UploadFile, max_bytes: int = _MAX_UPLOAD_BYTES
) -> bytes:
    """Read a multipart audio upload, enforcing the byte cap.

    UploadFile.read with no argument loads into memory; the size check is
    primarily a guard against accidental huge uploads, not a streaming
    safeguard (we need the full payload for vLLM anyway). Long-audio
    transcription passes its own (much larger) cap.
    """
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="audio file is empty")
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"audio file exceeds {max_bytes} bytes",
        )
    return raw


def _wav_to_pcm_capped(raw: bytes, max_seconds: float) -> tuple[bytes, float]:
    """Decode a WAV blob to 16 kHz mono and tail-trim to ``max_seconds``.

    Returns (re_encoded_wav_bytes, duration_sec). When no trim is needed the
    re-encoded WAV is byte-equivalent to ``pcm_to_wav_base64(pcm)``.
    """
    import io
    import wave

    import numpy as np

    from .audio.utils import pcm_to_wav_base64

    wav_b64 = base64.b64encode(raw).decode("ascii")
    try:
        pcm = wav_base64_to_pcm_16k_mono(wav_b64)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"could not decode audio: {exc}"
        ) from exc
    if pcm.size == 0:
        raise HTTPException(status_code=400, detail="audio decoded to empty PCM")
    duration = pcm.size / SAMPLE_RATE
    if max_seconds > 0 and duration > max_seconds:
        # Match streaming engines: keep the trailing window. Emotion picks
        # the tail because the most recent emotion is what users care about;
        # we use the same convention for ASR for consistency.
        keep = int(SAMPLE_RATE * max_seconds)
        pcm = pcm[-keep:]
        duration = pcm.size / SAMPLE_RATE
    new_b64 = pcm_to_wav_base64(pcm.astype(np.float32, copy=False))
    new_bytes = base64.b64decode(new_b64)
    # Sanity: re-encoded WAV should still parse.
    with wave.open(io.BytesIO(new_bytes), "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE
    return new_bytes, duration


def _emotion_result_payload(
    result: object,
    *,
    mode: str,
    duration_sec: float,
    language: str,
) -> dict:
    if isinstance(result, Exception):
        return {
            "type": "error",
            "mode": mode,
            "message": str(result),
            "error_type": result.__class__.__name__,
        }
    payload = {
        "type": "final_emotion",
        "mode": mode,
        "label": result.get("label", ""),
        "text": result.get("text", ""),
        "duration_sec": round(duration_sec, 3),
    }
    if language:
        payload["language"] = language
    return payload


def _public_asr_payload(result: dict) -> dict:
    return {
        "text": result.get("text", ""),
        "language": result.get("language", ""),
    }


def _public_cleanup_payload(result: dict) -> dict:
    return {
        "text": result.get("text", ""),
    }


def _resolve_enrollment_b64(enrollment_id: str | None) -> str | None:
    """Look up an enrollment id, refreshing its TTL. Missing/expired ids
    return ``None`` (caller decides to error or silently fall back)."""
    entry = _resolve_enrollment_entry(enrollment_id)
    return entry.wav_base64 if entry is not None else None


def _resolve_enrollment_entry(enrollment_id: str | None):
    """Return the enrollment entry so callers can preserve the id for embeddings."""
    if not enrollment_id:
        return None
    return get_enrollment_store().get(enrollment_id)


async def _run_dual_asr_upload(
    wav_b64: str,
    *,
    cfg,
    hotwords: list[str],
    language: str,
    enrollment_b64: str | None = None,
    enrollment_id: str | None = None,
) -> dict:
    """Route-facing wrapper over :func:`run_oneshot_asr`.

    Translates the framework-free :class:`OneshotAsrError` into the 502 the
    REST contract documents for "every configured model failed".
    """
    try:
        return await run_oneshot_asr(
            wav_b64,
            cfg=cfg,
            hotwords=hotwords,
            language=language,
            enrollment_b64=enrollment_b64,
            enrollment_id=enrollment_id,
        )
    except OneshotAsrError as exc:
        raise HTTPException(status_code=502, detail=exc.to_detail()) from exc


@app.post("/api/asr/enrollment")
async def asr_enrollment_create(audio: UploadFile = File(...)):
    """Cache a target-speaker enrollment clip and return its opaque id.

    The frontend uploads a 1–8 s clip once (file or mic recording) and
    then passes the returned ``enrollment_id`` on every ``/api/asr/upload``
    call and the WS ``start`` payload. The server validates duration,
    canonicalises to 16 kHz mono WAV, and stores the base64 result so
    final ASR can inject its projector embeddings before target utterance
    embeddings. Embedding precompute is best-effort and never blocks successful
    registration.
    """
    raw = await _read_audio_bytes(audio)
    wav_b64 = base64.b64encode(raw).decode("ascii")
    cfg = load_config()
    try:
        canonical_b64, duration_sec = decode_and_validate(
            wav_b64,
            min_sec=cfg.asr_enrollment_min_sec,
            max_sec=cfg.asr_enrollment_max_sec,
        )
    except EnrollmentError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc

    store = get_enrollment_store()
    store.configure(
        ttl_sec=cfg.asr_enrollment_ttl_sec,
        max_entries=cfg.asr_enrollment_max_entries,
        store_dir=str(getattr(cfg, "asr_enrollment_store_dir", "") or ""),
        scope=str(getattr(cfg, "asr_enrollment_scope", "default") or "default"),
        touch_interval_sec=float(
            getattr(cfg, "asr_enrollment_metadata_touch_interval_sec", 60.0)
        ),
    )
    entry = store.put(canonical_b64, duration_sec)
    if bool(getattr(cfg, "split_asr_enabled", False)):
        split_base_url = (
            str(getattr(cfg, "split_asr_base_url", "") or "").strip()
            or cfg.vllm_base_url
        )
        precompute_timeout = min(float(getattr(cfg, "asr_request_timeout", 30.0)), 5.0)

        async def _background_precompute() -> None:
            try:
                await asyncio.to_thread(
                    precompute_enrollment_embedding_sync,
                    entry.enrollment_id,
                    canonical_b64,
                    base_url=split_base_url,
                    timeout=precompute_timeout,
                    trace_id=f"enrollment-upload-{entry.enrollment_id[:8]}",
                )
            except Exception:  # noqa: BLE001
                logger.debug("enrollment embedding precompute failed", exc_info=True)

        asyncio.create_task(_background_precompute())
    return {
        "enrollment_id": entry.enrollment_id,
        "duration_sec": round(duration_sec, 3),
    }


@app.get("/api/asr/enrollment/{enrollment_id}")
async def asr_enrollment_status(enrollment_id: str):
    """Report whether an enrollment id is currently usable.

    Response is strictly ``{enrollment_id, available, reason}`` and never
    leaks the stored audio/embedding. Reasons: ``ok`` (present + compatible),
    ``not_found`` (never registered or clip missing), ``deleted`` (explicitly
    removed), ``incompatible`` (registered fingerprint differs from the
    current model/adapter), ``upstream_unavailable`` (durable store I/O error).
    """
    status = get_enrollment_store().status(enrollment_id)
    return {
        "enrollment_id": status.enrollment_id,
        "available": status.available,
        "reason": status.reason,
    }


@app.delete("/api/asr/enrollment/{enrollment_id}")
async def asr_enrollment_delete(enrollment_id: str):
    """Drop a previously uploaded enrollment clip.

    Returning 204 on missing ids keeps the frontend's "clear" button
    idempotent — repeated clears never error out.
    """
    get_enrollment_store().delete(enrollment_id)
    return JSONResponse(status_code=204, content=None)


@app.get("/api/asr/hotword-pool")
async def asr_hotword_pool_list(
    query: str = "",
    limit: int = 100,
    offset: int = 0,
    hotword_pool_id: str = "",
):
    """List the RAG-ASR hotword pool for ``hotword_pool_id`` (default when empty)."""
    pool_id = _normalize_pool_id_or_400(hotword_pool_id)
    try:
        result = await manage_hotword_pool(
            "list",
            query=query or None,
            limit=max(0, min(int(limit), 1000)),
            offset=max(0, int(offset)),
            hotword_pool_id=pool_id,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return _hotword_pool_payload(result)


@app.post("/api/asr/hotword-pool")
async def asr_hotword_pool_add(req: HotwordPoolUpdateRequest):
    """Add words to a RAG-ASR hotword pool and update its embeddings."""
    pool_id = _normalize_pool_id_or_400(req.hotword_pool_id)
    try:
        result = await manage_hotword_pool(
            "add", hotwords=req.hotwords, hotword_pool_id=pool_id
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return _hotword_pool_payload(result)


@app.delete("/api/asr/hotword-pool")
async def asr_hotword_pool_delete(req: HotwordPoolUpdateRequest):
    """Delete words from a RAG-ASR hotword pool."""
    pool_id = _normalize_pool_id_or_400(req.hotword_pool_id)
    try:
        result = await manage_hotword_pool(
            "delete", hotwords=req.hotwords, hotword_pool_id=pool_id
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return _hotword_pool_payload(result)


@app.post("/api/asr/hotword-pool/delete")
async def asr_hotword_pool_delete_post(req: HotwordPoolUpdateRequest):
    """Delete words from clients that cannot send a JSON body with DELETE."""
    return await asr_hotword_pool_delete(req)


@app.post("/api/asr/hotword-pool/clear")
async def asr_hotword_pool_clear(
    body: HotwordPoolScopeRequest | None = Body(default=None),
    hotword_pool_id: str = "",
):
    """Clear all hotwords in one pool and refresh its runtime words + embeddings.

    Accepts ``hotword_pool_id`` from JSON body or query string; a conflicting
    pair returns 400. Only the named pool is affected.
    """
    pool_id = _resolve_pool_id_or_400(
        body.hotword_pool_id if body else None, hotword_pool_id
    )
    try:
        result = await manage_hotword_pool("clear", hotword_pool_id=pool_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return _hotword_pool_payload(result)


@app.post("/api/asr/hotword-pool/reload")
async def asr_hotword_pool_reload(
    body: HotwordPoolScopeRequest | None = Body(default=None),
    hotword_pool_id: str = "",
):
    """Reload one hotword pool file and rebuild only its embedding cache.

    Accepts ``hotword_pool_id`` from JSON body or query string; a conflicting
    pair returns 400. Reload never implicitly touches other pools.
    """
    pool_id = _resolve_pool_id_or_400(
        body.hotword_pool_id if body else None, hotword_pool_id
    )
    try:
        result = await manage_hotword_pool("reload", hotword_pool_id=pool_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return _hotword_pool_payload(result)


@app.post("/api/asr/upload")
async def asr_upload(
    audio: UploadFile = File(...),
    language: str = Form(""),
    hotwords: str = Form(""),
    enrollment_id: str = Form(""),
):
    """One-shot ASR over an uploaded clip.

    Mirrors :class:`AsrTaskEngine.handle_segment` but operates on the entire
    clip in a single dual-model call (no VAD segmentation, no partials).
    Returns the same fields the streaming ``final`` event carries.

    When ``enrollment_id`` resolves to a cached enrollment clip the primary
    model is prompted with the dual-audio TS-ASR template (task 5/6 of the
    v4 prompt spec). Unknown / expired ids fall back to plain ASR so a
    stale id from a long-running tab does not break uploads.
    """
    raw = await _read_audio_bytes(audio)
    wav_bytes, duration_sec = _wav_to_pcm_capped(raw, _ASR_MAX_SECONDS)
    wav_b64 = base64.b64encode(wav_bytes).decode("ascii")
    cfg = load_config()
    hw_list = _parse_csv(hotwords)
    enrollment_entry = _resolve_enrollment_entry(enrollment_id)
    enrollment_b64 = enrollment_entry.wav_base64 if enrollment_entry is not None else None
    asr_result = await _run_dual_asr_upload(
        wav_b64,
        cfg=cfg,
        hotwords=hw_list,
        language=language,
        enrollment_b64=enrollment_b64,
        enrollment_id=enrollment_entry.enrollment_id if enrollment_entry is not None else None,
    )

    return {
        "type": "final",
        "text": asr_result["text"],
        "language": asr_result["language"],
        "duration_sec": round(duration_sec, 3),
        "enrollment_used": enrollment_b64 is not None,
    }


@app.post("/api/asr/transcriptions", status_code=202)
async def asr_transcription_create(
    audio: UploadFile = File(...),
    language: str = Form(""),
    hotwords: str = Form(""),
):
    """Enqueue offline transcription of a long recording (meeting minutes).

    Unlike ``/api/asr/upload`` (single clip, 60 s tail-trim, synchronous),
    this accepts whole recordings up to ``transcribe_max_audio_sec``: the
    audio is VAD-segmented server-side with the same parameters as the
    streaming endpoints and each segment is transcribed via the shared
    one-shot dual-ASR path. Recordings longer than the cap are REJECTED
    (400) rather than trimmed — silently losing meeting content is worse
    than asking the client to split the file.

    Poll ``GET /api/asr/transcriptions/{job_id}`` for progress and the final
    segment list (``start_ms``/``end_ms`` per segment + ``full_text``).

    No ``enrollment_id`` here on purpose: target-speaker filtering keeps a
    single speaker and drops everyone else, which is the opposite of what a
    multi-speaker meeting transcript needs.
    """
    # Transcription-specific view: rest.routes.transcribe bindings (model
    # choice, fusion switch) layered over the shared REST defaults.
    req_start = time.monotonic()
    cfg = load_transcribe_config()
    read_start = time.monotonic()
    raw = await _read_audio_bytes(audio, max_bytes=cfg.transcribe_max_upload_bytes)
    read_ms = (time.monotonic() - read_start) * 1000.0
    decode_start = time.monotonic()
    try:
        pcm = wav_bytes_to_pcm_16k_mono(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"could not decode audio: {exc}"
        ) from exc
    decode_ms = (time.monotonic() - decode_start) * 1000.0
    del raw
    if pcm.size == 0:
        raise HTTPException(status_code=400, detail="audio decoded to empty PCM")

    duration_sec = pcm.size / SAMPLE_RATE
    max_sec = float(cfg.transcribe_max_audio_sec)
    if max_sec > 0 and duration_sec > max_sec:
        raise HTTPException(
            status_code=400,
            detail=(
                f"audio duration {duration_sec:.1f}s exceeds the "
                f"{max_sec:.0f}s transcription cap; split the recording"
            ),
        )

    convert_start = time.monotonic()
    pcm_i16 = float_pcm_to_i16_bytes(pcm)
    convert_ms = (time.monotonic() - convert_start) * 1000.0
    del pcm

    store = get_transcription_job_store()
    store.configure(cfg)
    submit_start = time.monotonic()
    try:
        job = await store.submit(
            pcm_i16,
            duration_sec=duration_sec,
            language=language,
            hotwords=_parse_csv(hotwords),
            cfg=cfg,
        )
    except JobQueueFullError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": "5"},
        ) from exc
    submit_ms = (time.monotonic() - submit_start) * 1000.0
    logger.info(
        "TRANSCRIBE_TIMING stage=create job=%s duration_sec=%.3f bytes_i16=%d "
        "read_ms=%.1f decode_ms=%.1f convert_ms=%.1f submit_ms=%.1f total_ms=%.1f",
        job.job_id,
        duration_sec,
        len(pcm_i16),
        read_ms,
        decode_ms,
        convert_ms,
        submit_ms,
        (time.monotonic() - req_start) * 1000.0,
    )

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job.job_id,
            "status": job.status,
            "poll_url": f"/api/asr/transcriptions/{job.job_id}",
            "duration_sec": round(duration_sec, 3),
        },
    )


@app.get("/api/asr/transcriptions/{job_id}")
async def asr_transcription_get(job_id: str):
    """Poll offline transcription job status, progress, and result."""
    store = get_transcription_job_store()
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_poll_dict()


@app.post("/api/emotion/jobs", status_code=202)
async def emotion_create_job(
    audio: UploadFile = File(...),
    mode: str = Form(""),
    language: str = Form(""),
):
    """Enqueue whole-utterance emotion inference; poll GET /api/emotion/jobs/{id}."""
    raw = await _read_audio_bytes(audio)
    cfg = load_config()
    cap = float(getattr(cfg, "emotion_max_audio_seconds", 0.0))
    try:
        decode_wav_capped(raw, cap)
    except EmotionDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store = get_emotion_job_store()
    store.configure(cfg)
    try:
        job = await store.submit(
            raw,
            mode=mode,
            language=language,
            cfg=cfg,
        )
    except JobQueueFullError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": "5"},
        ) from exc

    poll_url = f"/api/emotion/jobs/{job.job_id}"
    return JSONResponse(
        status_code=202,
        content={
            "job_id": job.job_id,
            "status": job.status,
            "poll_url": poll_url,
        },
    )


@app.get("/api/emotion/jobs/{job_id}")
async def emotion_get_job(job_id: str):
    """Poll async emotion job status and result."""
    store = get_emotion_job_store()
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_poll_dict()


@app.post("/api/emotion-spec/jobs", status_code=202)
async def emotion_spec_create_job(
    audio: UploadFile = File(...),
    mode: str = Form(""),
    language: str = Form(""),
):
    """Enqueue whole-utterance AmphionSPEC inference; poll GET /api/emotion-spec/jobs/{id}.

    Independent of ``/api/emotion/jobs`` — separate queue, separate
    concurrency budget, separate vLLM endpoint (cfg.emotion_spec_vllm_*).
    ``mode`` accepts ``ser`` or ``sepc`` (alias ``spec`` is normalized to
    ``sepc``); empty falls back to ``cfg.emotion_spec_task_mode``.
    """
    raw = await _read_audio_bytes(audio)
    cfg = load_config()
    cap = float(getattr(cfg, "emotion_spec_max_audio_seconds", 0.0))
    try:
        decode_wav_capped(raw, cap)
    except EmotionDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store = get_emotion_spec_job_store()
    store.configure(cfg)
    try:
        job = await store.submit(
            raw,
            mode=mode,
            language=language,
            cfg=cfg,
        )
    except JobQueueFullError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": "5"},
        ) from exc

    poll_url = f"/api/emotion-spec/jobs/{job.job_id}"
    return JSONResponse(
        status_code=202,
        content={
            "job_id": job.job_id,
            "status": job.status,
            "poll_url": poll_url,
        },
    )


@app.get("/api/emotion-spec/jobs/{job_id}")
async def emotion_spec_get_job(job_id: str):
    """Poll async AmphionSPEC job status and result."""
    store = get_emotion_spec_job_store()
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_poll_dict()


@app.post("/api/audio/analyze")
async def audio_analyze(
    audio: UploadFile = File(...),
    language: str = Form(""),
    hotwords: str = Form(""),
    emotion_mode: str = Form("both"),
    enrollment_id: str = Form(""),
):
    """One-shot audio analysis: ASR raw output + cleaned text + emotion."""
    raw = await _read_audio_bytes(audio)
    cfg = load_config()
    hw_list = _parse_csv(hotwords)
    enrollment_entry = _resolve_enrollment_entry(enrollment_id)
    enrollment_b64 = enrollment_entry.wav_base64 if enrollment_entry is not None else None

    asr_wav_bytes, duration_sec = _wav_to_pcm_capped(raw, _ASR_MAX_SECONDS)
    asr_wav_b64 = base64.b64encode(asr_wav_bytes).decode("ascii")

    emotion_cap = float(getattr(cfg, "emotion_max_audio_seconds", 0.0))
    emotion_wav_bytes, emotion_duration_sec = _wav_to_pcm_capped(raw, emotion_cap)
    emotion_wav_b64 = base64.b64encode(emotion_wav_bytes).decode("ascii")

    asr_task = asyncio.create_task(
        _run_dual_asr_upload(
            asr_wav_b64,
            cfg=cfg,
            hotwords=hw_list,
            language=language,
            enrollment_b64=enrollment_b64,
            enrollment_id=enrollment_entry.enrollment_id if enrollment_entry is not None else None,
        )
    )
    emotion_ser_task = asyncio.create_task(
        query_emotion_model(
            emotion_wav_b64,
            mode="ser",
            base_url=cfg.emotion_vllm_base_url,
            model_name=cfg.emotion_vllm_model_name,
            timeout=cfg.emotion_request_timeout,
        )
    )
    emotion_sec_task = asyncio.create_task(
        query_emotion_model(
            emotion_wav_b64,
            mode="sec",
            base_url=cfg.emotion_vllm_base_url,
            model_name=cfg.emotion_vllm_model_name,
            timeout=cfg.emotion_request_timeout,
            max_tokens=256,
        )
    )

    asr_out, emotion_ser_out, emotion_sec_out = await asyncio.gather(
        asr_task,
        emotion_ser_task,
        emotion_sec_task,
        return_exceptions=True,
    )
    if isinstance(asr_out, HTTPException):
        raise asr_out
    if isinstance(asr_out, Exception):
        logger.error("Audio analyze ASR failed: %s", asr_out)
        raise HTTPException(status_code=502, detail=str(asr_out)) from asr_out
    asr_result = asr_out

    if isinstance(emotion_ser_out, Exception):
        logger.error("Audio analyze SER inference failed: %s", emotion_ser_out)
    if isinstance(emotion_sec_out, Exception):
        logger.error("Audio analyze SEC inference failed: %s", emotion_sec_out)
    emotion_ser = _emotion_result_payload(
        emotion_ser_out,
        mode="ser",
        duration_sec=emotion_duration_sec,
        language=language,
    )
    emotion_sec = _emotion_result_payload(
        emotion_sec_out,
        mode="sec",
        duration_sec=emotion_duration_sec,
        language=language,
    )
    emotion_payload = {
        "type": "final_emotion_pair",
        "mode": "both",
        "ser": emotion_ser,
        "sec": emotion_sec,
    }

    try:
        cleaned = await clean_asr_text(
            str(asr_result.get("text") or ""),
            hotwords=[],
            language=str(asr_result.get("language") or language or ""),
            emotion=emotion_payload,
            cfg=cfg,
        )
    except TextCleanupConfigError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Audio analyze text cleanup failed")
        raise HTTPException(
            status_code=502,
            detail=f"text cleanup model failed: {exc}",
        ) from exc

    return {
        "type": "audio_analysis",
        "duration_sec": round(duration_sec, 3),
        "language": asr_result.get("language") or language or "",
        "hotwords": hw_list,
        "asr": _public_asr_payload(asr_result),
        "cleaned_asr": _public_cleanup_payload(cleaned),
        "emotion": emotion_payload,
    }


class _RevalidateStaticFiles(StaticFiles):
    """Static files with tiered caching.

    Browsers were aggressively caching ``app.js`` / ``style.css`` because
    starlette's default ``StaticFiles`` ships no explicit ``Cache-Control``
    header, leaving the heuristic up to the client. That made shipping
    frontend fixes during a session unreliable — users had to hard-reload
    to pick up changes. We inject ``no-cache`` so the browser still uses
    its disk copy, but always revalidates with the server's ``ETag`` (set
    by starlette from mtime+size); unchanged files come back as 304 so
    bandwidth stays cheap. Cache-Control is omitted on non-200 responses
    to avoid pinning errors.

    For the assets the demo loads on every page (CSS, JS) we additionally
    grant a short ``max-age`` so the browser can serve repeat requests
    from disk cache without a conditional round-trip. Ten seconds is long
    enough to cover all the navigations of a single user session yet
    short enough that any real frontend change still appears within a
    blink — and the ``ETag`` revalidation kicks back in once the window
    expires. HTML is intentionally left at plain ``no-cache`` so users
    always see the latest markup.
    """

    _CACHEABLE_EXTS = (".css", ".js")

    @staticmethod
    def _cache_header_for(path: str) -> bytes:
        if path.endswith(_RevalidateStaticFiles._CACHEABLE_EXTS):
            return b"no-cache, max-age=10"
        return b"no-cache"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        cache_header = self._cache_header_for((scope.get("path") or "").lower())

        async def send_with_cache(message: dict) -> None:
            if message["type"] == "http.response.start":
                status = message.get("status", 0)
                if 200 <= status < 300:
                    headers = list(message.get("headers", []))
                    headers = [
                        (k, v) for (k, v) in headers if k.lower() != b"cache-control"
                    ]
                    headers.append((b"cache-control", cache_header))
                    message["headers"] = headers
            await send(message)

        await super().__call__(scope, receive, send_with_cache)


# Static mount comes last so it doesn't shadow the /api routes above.
app.mount(
    "/", _RevalidateStaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend"
)
