"""Generic WebSocket session that wires an AudioStream to a TaskEngine.

The session owns:

- WebSocket lifecycle (ready, receive loop, error/close)
- Parsing of common control messages
  (start/stop/update_hotwords/extract_hotwords)
- Per-session config override (Config.override)
- Dispatching ``SegmentReady`` events serially through a work queue
- Throttled, non-overlapping dispatch of ``PartialSnapshot`` events

It does NOT know what "ASR" or "emotion" means; that lives in the engine.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from ..asr.enrollment import get_enrollment_store
from ..asr.hotword import query_text_hotwords, sanitize_hotwords
from ..config import Config, SAMPLE_RATE, load_config
from .audio_stream import AudioStream, _pcm_bytes_to_float32
from .backpressure import final_pressure
from .events import PartialSnapshot, SegmentReady, SpeechDropped, SpeechStarted
from .protocol import ControlAction, NativeProtocol, PcmAction, WireProtocol

if TYPE_CHECKING:
    from ..tasks.base import TaskEngine

logger = logging.getLogger(__name__)
_STREAM_FEED_EXECUTOR_WORKERS = max(
    1,
    int(
        os.getenv(
            "STREAM_FEED_EXECUTOR_WORKERS",
            str(min(32, max(8, os.cpu_count() or 8))),
        )
    ),
)
_STREAM_FEED_EXECUTOR = ThreadPoolExecutor(
    max_workers=_STREAM_FEED_EXECUTOR_WORKERS,
    thread_name_prefix="stream-feed",
)


def _clip_log_text(value: object, limit: int = 200) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _summarize_outbound_wire(wire: dict[str, Any]) -> dict[str, Any]:
    """Return a compact, log-safe summary of a response sent to the client."""
    header = wire.get("header") if isinstance(wire.get("header"), dict) else {}
    payload = wire.get("payload") if isinstance(wire.get("payload"), dict) else {}
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}

    text_parts: list[str] = []
    for item in result.get("ws", []) or []:
        if not isinstance(item, dict):
            continue
        for cw in item.get("cw", []) or []:
            if isinstance(cw, dict):
                text_parts.append(str(cw.get("w", "")))

    summary: dict[str, Any] = {
        "type": wire.get("type"),
        "traceId": header.get("traceId"),
        "status": header.get("status"),
        "code": header.get("code"),
        "msgtype": result.get("msgtype"),
        "text": _clip_log_text(
            "".join(text_parts) if text_parts else result.get("text", wire.get("text"))
        ),
    }
    return {k: v for k, v in summary.items() if v is not None}

LANG_CODE_MAP: dict[str, str] = {
    "zh": "Chinese",
    "cn": "Chinese",
    "en": "English",
    "id": "Indonesian",
    "th": "Thai",
}


def map_language(lang_query: str) -> str:
    """Map a language code or full name to the canonical model-side string."""
    code = (lang_query or "").strip().lower()
    if not code:
        return "N/A"
    if code in LANG_CODE_MAP:
        return LANG_CODE_MAP[code]
    for full_name in ("Chinese", "English", "Indonesian", "Thai"):
        if code == full_name.lower():
            return full_name
    return "N/A"


@dataclass
class SessionContext:
    """Snapshot of common session state passed to engine callbacks.

    The session passes a *frozen* snapshot to per-segment / per-partial calls
    so concurrent updates (e.g. ``update_hotwords``) don't race with in-flight
    inference.
    """

    cfg: Config
    language: str = ""
    src_lang: str = "N/A"
    trace_id: str = ""
    last_speech_started_at: float | None = None
    hotwords: list[str] = field(default_factory=list)
    # Optional cached target-speaker enrollment (base64 WAV). The session
    # resolves the opaque ``enrollment_id`` once at start / on every
    # ``update_hotwords`` and stores the WAV inline so per-segment
    # inference doesn't re-hit the in-memory store on every call.
    enrollment_id: str | None = None
    enrollment_b64: str | None = None
    # RAG-ASR hotword pool selector for this session (empty = default pool).
    # Threaded to the recall path so per-pool isolation can route correctly.
    hotword_pool_id: str = ""
    runtime_state: dict[str, Any] = field(default_factory=dict)
    send_json: Callable[[dict[str, Any]], Awaitable[bool]] = None  # type: ignore[assignment]

    def snapshot(self) -> "SessionContext":
        return replace(self, hotwords=list(self.hotwords))


_SENTINEL = object()


class StreamingSession:
    """Run one client connection by composing an AudioStream and a TaskEngine."""

    def __init__(
        self,
        websocket: WebSocket,
        *,
        stream: AudioStream,
        engine: "TaskEngine",
        language: str = "",
        protocol: WireProtocol | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.ws = websocket
        self.stream = stream
        self.engine = engine
        # The wire protocol owns on-the-wire framing; the rest of the session
        # only ever deals in control dicts + PCM bytes. NativeProtocol is the
        # historical 1:1 framing so existing endpoints are byte-for-byte
        # unchanged when no protocol is supplied.
        self.protocol: WireProtocol = protocol or NativeProtocol()
        self._config_overrides = config_overrides

        self.cfg: Config = load_config()
        if self._config_overrides:
            self.cfg = self.cfg.override(**self._config_overrides)
        self.stream.configure(self.cfg)

        self.ctx = SessionContext(
            cfg=self.cfg,
            language=language,
            src_lang=map_language(language),
            hotwords=[],
            send_json=self._send_json,
        )

        self._work_queue: asyncio.Queue = asyncio.Queue(maxsize=40)
        self._feed_queue: asyncio.Queue[tuple[bytes, float] | None] = asyncio.Queue(
            maxsize=8192
        )
        self._feed_task: asyncio.Task | None = None
        self._partial_task: asyncio.Task | None = None
        self._k2_result_poll_task: asyncio.Task | None = None
        self._ctc_direct_callback_installed = False
        # Per-session semaphore for concurrent final-segment dispatch.
        # Limits how many handle_segment tasks this session may have inflight
        # simultaneously, preventing a single session from monopolising the
        # global final semaphore while still allowing encode/decode overlap.
        _parallel = int(getattr(self.cfg, "asr_final_session_parallel", 2))
        self._session_final_sem: asyncio.Semaphore = asyncio.Semaphore(
            max(1, _parallel)
        )
        self._pending_final_tasks: set[asyncio.Task] = set()
        self._ctc_direct_send_tasks: set[asyncio.Task] = set()
        self._partial_snapshot: PartialSnapshot | None = None
        self._last_first_partial_speech_started_at: float | None = None
        self._last_partial_refresh_launched_at: float = 0.0
        self._launched_first_partial = False
        # Long-text hotword extraction runs out-of-band so the receive
        # loop never blocks on an LLM round-trip; outstanding tasks
        # are tracked here and cancelled in cleanup.
        self._extract_tasks: set[asyncio.Task] = set()

        self._started = False
        self._stopped = False
        self._abort_session = False
        self._sent_any_response = False
        self._sent_first_partial_text = False
        self._last_partial_wire_sent_at = 0.0
        self._ws_closed = False
        self._pcm_frames_seen = 0
        self._pcm_bytes_seen = 0
        self._last_pcm_received_at = 0.0
        self._last_pcm_processed_at = 0.0
        self._feed_draining = False
        self._bulk_flush_extra: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._feed_task = asyncio.create_task(self._feed_loop())
        sent_ready = await self._send_json({"type": "ready"})
        if sent_ready:
            logger.info(
                "%s ready (language=%s)",
                self.engine.name,
                self.ctx.language,
            )
        try:
            await asyncio.gather(self._receive_loop(), self._work_loop())
        except Exception:
            logger.exception("StreamingSession[%s] error", self.engine.name)
        finally:
            # Protocols that frame an explicit end-of-session marker (e.g. AST
            # v3's header.status=2) emit it here, after every queued segment
            # has drained. Native framing returns None and sends nothing.
            terminal = None if self._abort_session else self.protocol.encode_terminal()
            if terminal is not None and not self._ws_closed:
                await self._send_wire(terminal)

    async def cleanup(self) -> None:
        if self._feed_task and not self._feed_task.done():
            self._feed_task.cancel()
        if self._partial_task and not self._partial_task.done():
            self._partial_task.cancel()
        if self._k2_result_poll_task and not self._k2_result_poll_task.done():
            self._k2_result_poll_task.cancel()
        if self._ctc_direct_send_tasks:
            for task in self._ctc_direct_send_tasks:
                task.cancel()
            await asyncio.gather(*self._ctc_direct_send_tasks, return_exceptions=True)
            self._ctc_direct_send_tasks.clear()
        # Cancel any inflight concurrent final-dispatch tasks so they don't
        # hold the per-session semaphore or fire send_json after WS is closed.
        if self._pending_final_tasks:
            for task in list(self._pending_final_tasks):
                task.cancel()
            await asyncio.gather(*self._pending_final_tasks, return_exceptions=True)
            self._pending_final_tasks.clear()
        # CTC partial sessions register native runtime slots before the first
        # final. Client disconnects or killed benchmark clients can otherwise
        # leave inactive streams in the fixed BS52 scheduler, poisoning later
        # sessions until the process restarts.
        await self._close_partial_for_final()
        if self._extract_tasks:
            for task in self._extract_tasks:
                task.cancel()
            await asyncio.gather(*self._extract_tasks, return_exceptions=True)
        logger.info("StreamingSession[%s] ended", self.engine.name)

    def _drain_queue_nowait(self, queue: asyncio.Queue) -> int:
        dropped = 0
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return dropped
            queue.task_done()
            dropped += 1

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------

    async def _send_wire(self, wire: dict[str, Any]) -> bool:
        if self._ws_closed:
            return False
        try:
            # Only build the (structure-walking) summary when it will actually
            # be logged. The important-message flag can be read straight from the
            # header, so partials/finals (the common debug-suppressed case) skip
            # _summarize_outbound_wire entirely — that walk ran on the event-loop
            # thread for every message × 52 streams and fed the server real-time
            # deficit that accumulates vad_lag (design F12).
            header = wire.get("header") if isinstance(wire.get("header"), dict) else {}
            important = header.get("status") == 2 or header.get("code") not in (None, 0)
            if important:
                logger.info("Response to client: %s", _summarize_outbound_wire(wire))
            elif logger.isEnabledFor(logging.DEBUG):
                logger.debug("Response to client: %s", _summarize_outbound_wire(wire))
            await self.ws.send_json(wire)
            return True
        except (WebSocketDisconnect, RuntimeError):
            self._ws_closed = True
            return False

    async def _send_json(self, payload: dict[str, Any]) -> bool:
        """Encode an internal message via the active protocol and send it.

        A protocol may suppress a message (return ``None``) when it has no
        wire representation (e.g. AST v3 has no ``ready``); that is reported
        as success so callers don't treat the no-op as a send failure.
        """
        if (
            payload.get("type") == "partial"
            and str(payload.get("text") or "").strip()
        ):
            ctc_after_first = (
                str(getattr(self.ctx.cfg, "streaming_partial_backend", "vllm") or "vllm")
                .strip()
                .lower()
                == "ctc_om"
                and self._sent_first_partial_text
            )
            if ctc_after_first and self._ctc_final_tail_pause_active():
                self._log_ctc_tail_pause("partial_send_suppressed")
                return not self._ws_closed
            if ctc_after_first:
                interval_ms = max(
                    0,
                    int(
                        getattr(
                            self.ctx.cfg,
                            "ctc_partial_post_first_interval_ms",
                            1500,
                        )
                        or 0
                    ),
                )
                if (
                    interval_ms > 0
                    and self._last_partial_wire_sent_at > 0
                    and (time.monotonic() - self._last_partial_wire_sent_at) * 1000.0
                    < interval_ms
                ):
                    return not self._ws_closed
            self._last_partial_wire_sent_at = time.monotonic()
            self._sent_first_partial_text = True
        wire = self.protocol.encode_outbound(payload)
        if wire is None:
            return not self._ws_closed
        return await self._send_wire(wire)

    def _ctc_final_tail_pause_active(self) -> bool:
        if not bool(getattr(self.ctx.cfg, "ctc_final_tail_pause_enabled", False)):
            return False
        if not self._sent_first_partial_text:
            return False
        backend = (
            str(getattr(self.ctx.cfg, "streaming_partial_backend", "vllm") or "vllm")
            .strip()
            .lower()
        )
        if backend != "ctc_om":
            return False
        pause_fn = getattr(self.stream, "ctc_final_tail_pause_active", None)
        if pause_fn is None:
            return False
        try:
            return bool(pause_fn())
        except Exception:
            logger.debug("ctc_final_tail_pause_active failed", exc_info=True)
            return False

    def _log_ctc_tail_pause(self, action: str) -> None:
        key = f"ctc_tail_pause_logged:{action}"
        if self.ctx.runtime_state.get(key):
            return
        self.ctx.runtime_state[key] = True
        logger.info(
            "CTC_TAIL_PAUSE action=%s traceId=%s sent_first_text=%s",
            action,
            self.ctx.trace_id or "-",
            self._sent_first_partial_text,
        )

    # ------------------------------------------------------------------
    # Receive loop: control messages + binary PCM
    # ------------------------------------------------------------------

    async def _receive_loop(self) -> None:
        try:
            while True:
                msg = await self.ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    self._ws_closed = True
                    if not self._stopped:
                        self._abort_session = True
                    now = time.monotonic()
                    logger.info(
                        "STREAM_INPUT_TIMING event=websocket_disconnect "
                        "traceId=%s stopped=%s pcm_frames=%s pcm_audio_ms=%.1f "
                        "since_last_pcm_received_ms=%.1f since_last_pcm_processed_ms=%.1f",
                        self.ctx.trace_id or "-",
                        self._stopped,
                        self._pcm_frames_seen,
                        self._pcm_bytes_seen / 2 / SAMPLE_RATE * 1000.0,
                        (
                            (now - self._last_pcm_received_at) * 1000.0
                            if self._last_pcm_received_at > 0
                            else -1.0
                        ),
                        (
                            (now - self._last_pcm_processed_at) * 1000.0
                            if self._last_pcm_processed_at > 0
                            else -1.0
                        ),
                    )
                    break

                should_stop = False
                for action in self.protocol.decode_inbound(msg):
                    if isinstance(action, ControlAction):
                        if await self._handle_control(action.ctrl):
                            should_stop = True
                            break
                    elif isinstance(action, PcmAction):
                        # PCM is only meaningful between start and stop; frames
                        # outside that window (or before the protocol has
                        # synthesized a start) are silently dropped.
                        if not self._started or self._stopped:
                            continue
                        await self._handle_pcm(action.data)
                if should_stop:
                    break
        except WebSocketDisconnect:
            self._ws_closed = True
            if not self._stopped:
                self._abort_session = True
            logger.info("WebSocket disconnected (%s)", self.engine.name)
        finally:
            flush_start = time.monotonic()
            if self._abort_session:
                dropped = self._drain_queue_nowait(self._feed_queue)
                logger.info(
                    "STREAM_INPUT_TIMING event=feed_aborted traceId=%s "
                    "reason=client_disconnect dropped_feed_items=%s stopped=%s",
                    self.ctx.trace_id or "-",
                    dropped,
                    self._stopped,
                )
            else:
                drain_timeout = float(
                    getattr(self.ctx.cfg, "stream_feed_drain_timeout_sec", 8.0)
                )
                self._feed_draining = True
                try:
                    await asyncio.wait_for(
                        self._feed_queue.join(), timeout=max(0.1, drain_timeout)
                    )
                except asyncio.TimeoutError:
                    # Collect remaining PCM for bulk_flush (AscendK2Stream path)
                    # before discarding, so the dropped audio can still reach
                    # the final ASR engine instead of being silently lost.
                    _bulk_flush_chunks: list[np.ndarray] = []
                    dropped = 0
                    while True:
                        try:
                            _item = self._feed_queue.get_nowait()
                            self._feed_queue.task_done()
                            dropped += 1
                            if _item is not None:
                                _pb, _ = _item
                                _bulk_flush_chunks.append(_pcm_bytes_to_float32(_pb))
                        except asyncio.QueueEmpty:
                            break
                    self._bulk_flush_extra = (
                        np.concatenate(_bulk_flush_chunks)
                        if _bulk_flush_chunks
                        else None
                    )
                    logger.warning(
                        "STREAM_INPUT_TIMING event=feed_drain_timeout traceId=%s "
                        "timeout_sec=%.1f dropped_feed_items=%s "
                        "pcm_audio_ms=%.1f stream_consumed_ms=%.1f "
                        "bulk_flush_extra_ms=%.1f",
                        self.ctx.trace_id or "-",
                        drain_timeout,
                        dropped,
                        self._pcm_bytes_seen / 2 / SAMPLE_RATE * 1000.0,
                        getattr(self.stream, "_consumed_samples", 0)
                        * 1000.0
                        / SAMPLE_RATE,
                        (len(self._bulk_flush_extra) * 1000.0 / SAMPLE_RATE)
                        if self._bulk_flush_extra is not None
                        else 0.0,
                    )
            await self._feed_queue.put(None)
            if self._feed_task:
                await asyncio.gather(self._feed_task, return_exceptions=True)
            logger.info(
                "STREAM_INPUT_TIMING event=receive_loop_finally traceId=%s "
                "stopped=%s pcm_frames=%s pcm_audio_ms=%.1f "
                "stream_consumed_ms=%.1f",
                self.ctx.trace_id or "-",
                self._stopped,
                self._pcm_frames_seen,
                self._pcm_bytes_seen / 2 / SAMPLE_RATE * 1000.0,
                getattr(self.stream, "_consumed_samples", 0) * 1000.0 / SAMPLE_RATE,
            )
            # Flush exactly once after all accepted PCM has been consumed. If
            # the session was aborted, skip flush so overload does not create
            # stale final ASR work for a disconnected client.
            if not self._abort_session:
                extra = getattr(self, "_bulk_flush_extra", None)
                if extra is not None and hasattr(self.stream, "bulk_flush"):
                    for ev in self.stream.bulk_flush(extra):
                        await self._dispatch_stream_event(ev)
                else:
                    for ev in self.stream.flush(force=True):
                        await self._dispatch_stream_event(ev)
            logger.info(
                "STREAM_INPUT_TIMING event=flush_done traceId=%s "
                "flush_ms=%.1f stream_consumed_ms=%.1f aborted=%s",
                self.ctx.trace_id or "-",
                (time.monotonic() - flush_start) * 1000.0,
                getattr(self.stream, "_consumed_samples", 0) * 1000.0 / SAMPLE_RATE,
                self._abort_session,
            )
            await self._work_queue.put(_SENTINEL)

    async def _handle_control(self, ctrl: dict) -> bool:
        """Dispatch one already-parsed control message; return True to stop."""
        msg_type = ctrl.get("type", "")
        if msg_type == "start":
            await self._handle_start(ctrl)
            return False
        if msg_type == "stop":
            await self._handle_stop()
            return True
        if msg_type == "protocol_error":
            # A protocol-level parameter error (e.g. AST v3 enrollment_enable
            # with an empty enrollment_id). Return an explicit error frame and
            # end the session instead of silently falling back to plain ASR.
            await self._send_json(
                {
                    "type": "error",
                    "message": str(ctrl.get("message") or "parameter error"),
                }
            )
            self._stopped = True
            self._abort_session = True
            return True
        if msg_type == "update_hotwords":
            self._handle_update_hotwords(ctrl)
            return False
        if msg_type == "extract_hotwords":
            self._handle_extract_hotwords(ctrl)
            return False
        # Delegate unknown control messages to engine (returns truthy if handled).
        try:
            await self.engine.on_control(ctrl, self.ctx)
        except Exception:
            logger.exception("engine.on_control failed for %s", msg_type)
        return False

    async def _handle_start(self, ctrl: dict) -> None:
        if self._started:
            logger.warning("Duplicate start message, ignoring")
            return
        self._started = True

        client_config = ctrl.get("config")
        if isinstance(client_config, dict) and client_config:
            # Untrusted per-connection override: only whitelisted tuning knobs
            # are honored; infra/secret fields are dropped (see override_client).
            self.cfg = self.cfg.override_client(**client_config)
            self.ctx.cfg = self.cfg
            self.stream.configure(self.cfg)
            logger.info("Config overridden by client: %s", list(client_config.keys()))
            if self._config_overrides:
                self.cfg = self.cfg.override(**self._config_overrides)
                self.ctx.cfg = self.cfg
                self.stream.configure(self.cfg)

        lang_val = str(ctrl.get("language", "")).strip()
        if lang_val:
            self.ctx.language = lang_val
            self.ctx.src_lang = map_language(lang_val)

        trace_id = str(ctrl.get("trace_id") or "").strip()
        if trace_id:
            self.ctx.trace_id = trace_id

        hw_raw = ctrl.get("hotwords")
        if isinstance(hw_raw, list):
            self.ctx.hotwords = sanitize_hotwords(hw_raw)
            logger.info(
                "Hotwords from start: count=%d items=%s",
                len(self.ctx.hotwords),
                self.ctx.hotwords,
            )

        pool_id = str(ctrl.get("hotword_pool_id") or "").strip()
        if pool_id:
            self.ctx.hotword_pool_id = pool_id

        if "enrollment_id" in ctrl:
            self._apply_enrollment(ctrl.get("enrollment_id"))

        # AST v3: report the real enrollment outcome back to the protocol so
        # sentence frames carry a truthful enrollment_used. When no enrollment
        # id was routed (role separation / plain ASR) or the id was
        # unknown/expired, enrollment_b64 is None -> enrollment_used=False.
        set_enrollment_used = getattr(self.protocol, "set_enrollment_used", None)
        if callable(set_enrollment_used):
            set_enrollment_used(self.ctx.enrollment_b64 is not None)

        fmt = ctrl.get("format", "pcm_s16le")
        sr = ctrl.get("sample_rate_hz", 16000)
        ch = ctrl.get("channels", 1)
        logger.info(
            "Start[%s] mode=%s format=%s sr=%s ch=%s language=%s",
            self.engine.name, ctrl.get("mode"), fmt, sr, ch, self.ctx.language,
        )

        try:
            await self.engine.on_start(ctrl, self.ctx)
        except Exception:
            logger.exception("engine.on_start failed")

    def _handle_update_hotwords(self, ctrl: dict) -> None:
        self.ctx.hotwords = sanitize_hotwords(ctrl.get("hotwords", []))
        if "src_lang" in ctrl:
            lang_val = str(ctrl.get("src_lang", "")).strip()
            if lang_val:
                self.ctx.language = lang_val
                self.ctx.src_lang = map_language(lang_val)
        if "enrollment_id" in ctrl:
            self._apply_enrollment(ctrl.get("enrollment_id"))
        logger.info(
            "Hotwords updated: %s (src_lang=%s, enrollment=%s)",
            self.ctx.hotwords, self.ctx.src_lang, self.ctx.enrollment_id,
        )

    def _apply_enrollment(self, raw_id: object) -> None:
        """Resolve an enrollment id (or ``None``/empty to clear) into a
        cached WAV. Unknown / expired ids are treated as "no enrollment"
        so a stale id from a long-lived tab degrades to plain ASR
        instead of breaking the WS session."""
        if raw_id is None or not isinstance(raw_id, str) or not raw_id.strip():
            self.ctx.enrollment_id = None
            self.ctx.enrollment_b64 = None
            return
        ident = raw_id.strip()
        entry = get_enrollment_store().get(ident)
        if entry is None:
            logger.warning("Enrollment id %s not found / expired", ident)
            self.ctx.enrollment_id = None
            self.ctx.enrollment_b64 = None
            return
        self.ctx.enrollment_id = ident
        self.ctx.enrollment_b64 = entry.wav_base64

    def _handle_extract_hotwords(self, ctrl: dict) -> None:
        """Schedule a long-text hotword extraction in the background.

        The receive loop returns immediately so further audio frames /
        control messages are not blocked by the LLM round-trip; the
        eventual ``extract_hotwords_result`` (or ``..._error``) is
        sent through the same WebSocket from the spawned task.
        """
        request_id = str(ctrl.get("request_id", "")).strip()
        source_text = str(ctrl.get("text", ""))
        task = asyncio.create_task(
            self._run_extract_hotwords(request_id, source_text)
        )
        self._extract_tasks.add(task)
        task.add_done_callback(self._extract_tasks.discard)

    async def _run_extract_hotwords(
        self, request_id: str, source_text: str
    ) -> None:
        try:
            extracted = await query_text_hotwords(source_text)
            await self._send_json(
                {
                    "type": "extract_hotwords_result",
                    "request_id": request_id,
                    "hotwords": extracted,
                }
            )
        except WebSocketDisconnect:
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "extract_hotwords failed (request_id=%s)", request_id or "n/a"
            )
            await self._send_json(
                {
                    "type": "extract_hotwords_error",
                    "request_id": request_id,
                    "message": str(exc),
                }
            )

    async def _handle_stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        logger.info(
            "STREAM_INPUT_TIMING event=stop_received traceId=%s task=%s "
            "pcm_frames=%s pcm_audio_ms=%.1f stream_consumed_ms=%.1f",
            self.ctx.trace_id or "-",
            self.engine.name,
            self._pcm_frames_seen,
            self._pcm_bytes_seen / 2 / SAMPLE_RATE * 1000.0,
            getattr(self.stream, "_consumed_samples", 0) * 1000.0 / SAMPLE_RATE,
        )

    # ------------------------------------------------------------------
    # PCM dispatch
    # ------------------------------------------------------------------

    async def _handle_pcm(self, pcm_bytes: bytes) -> None:
        if self._abort_session or self._ws_closed:
            return
        received_at = time.monotonic()
        self._pcm_frames_seen += 1
        self._pcm_bytes_seen += len(pcm_bytes)
        self._last_pcm_received_at = received_at
        try:
            self._feed_queue.put_nowait((pcm_bytes, received_at))
        except asyncio.QueueFull:
            self._abort_session = True
            logger.warning(
                "STREAM_INPUT_TIMING event=feed_queue_full traceId=%s "
                "pcm_frames=%s pcm_audio_ms=%.1f feed_queue_depth=%s",
                self.ctx.trace_id or "-",
                self._pcm_frames_seen,
                self._pcm_bytes_seen / 2 / SAMPLE_RATE * 1000.0,
                self._feed_queue.qsize(),
            )
            await self._send_json({"type": "error", "message": "server overloaded"})
            return
        queue_depth = self._feed_queue.qsize()
        if queue_depth > 256 or self._pcm_frames_seen % 1000 == 0:
            logger.info(
                "STREAM_INPUT_TIMING event=pcm_received traceId=%s "
                "pcm_frames=%s pcm_audio_ms=%.1f feed_queue_depth=%s",
                self.ctx.trace_id or "-",
                self._pcm_frames_seen,
                self._pcm_bytes_seen / 2 / SAMPLE_RATE * 1000.0,
                queue_depth,
            )

    async def _feed_loop(self) -> None:
        loop = asyncio.get_running_loop()
        processed_frames = 0
        ctc_backend = (
            str(getattr(self.ctx.cfg, "streaming_partial_backend", "vllm") or "vllm")
            .strip()
            .lower()
            == "ctc_om"
        )
        ascend_k2_stream = hasattr(self.stream, "poll_partial_events")
        steady_batch_max = max(
            1, int(getattr(self.ctx.cfg, "stream_feed_batch_max_frames", 16))
        )
        initial_batch_max = max(
            1,
            int(
                getattr(
                    self.ctx.cfg,
                    "stream_feed_initial_batch_max_frames",
                    steady_batch_max,
                )
            ),
        )
        drain_batch_max = max(
            steady_batch_max,
            int(
                getattr(
                    self.ctx.cfg,
                    "stream_feed_drain_batch_max_frames",
                    steady_batch_max,
                )
            ),
        )
        # Duration-quantum coalescing (>0 to enable). The legacy batch caps are
        # in *chunk count*, which couples server feed granularity to the client
        # chunk-ms: 16 chunks is 640ms at 40ms chunks but 3.2s at 200ms chunks,
        # so a fixed chunk cap either fails to coalesce tiny chunks or over-
        # coalesces large ones (measured: initial=16 helped BS52@40ms first-word
        # -29% but hurt BS52@200ms +26%). Target-ms coalescing instead merges
        # only ALREADY-QUEUED backlog (get_nowait, never waits -> no added
        # latency) up to ~target_ms of audio, so tiny client chunks get merged
        # into ~target_ms feeds under load while chunks already >= target_ms
        # stay 1-per-feed. This makes the server hold a stable processing rate
        # independent of client send rate. Bytes/sample = 2 (int16 @ 16kHz).
        feed_target_ms = float(getattr(self.ctx.cfg, "stream_feed_target_ms", 0) or 0)
        feed_target_bytes = int(SAMPLE_RATE * 2 * feed_target_ms / 1000.0)
        while True:
            item = await self._feed_queue.get()
            task_done_count = 1
            try:
                if item is None:
                    return
                if self._abort_session:
                    continue
                pcm_bytes, received_at = item
                batch = [pcm_bytes]
                if self._feed_draining:
                    batch_max = drain_batch_max
                else:
                    batch_max = (
                        steady_batch_max
                        if self._launched_first_partial
                        else initial_batch_max
                    )
                saw_sentinel = False
                if feed_target_bytes > 0 and not self._feed_draining:
                    # Duration-quantum drain: merge queued backlog until the
                    # batch holds ~target_ms of audio (or the queue empties).
                    # A generous chunk cap only guards against pathological
                    # runaway; target_ms is the real limiter.
                    acc_bytes = len(pcm_bytes)
                    hard_cap = max(batch_max, drain_batch_max)
                    while acc_bytes < feed_target_bytes and len(batch) < hard_cap:
                        try:
                            extra = self._feed_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        task_done_count += 1
                        if extra is None:
                            saw_sentinel = True
                            break
                        extra_pcm, _ = extra
                        batch.append(extra_pcm)
                        acc_bytes += len(extra_pcm)
                else:
                    while len(batch) < batch_max:
                        try:
                            extra = self._feed_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        task_done_count += 1
                        if extra is None:
                            saw_sentinel = True
                            break
                        extra_pcm, _ = extra
                        batch.append(extra_pcm)
                pcm_batch = b"".join(batch) if len(batch) > 1 else pcm_bytes
                ctc_prefeed_enabled = (
                    not self._abort_session
                    and not self._feed_draining
                    and ctc_backend
                    and hasattr(self.engine, "prefeed_streaming_partial")
                )
                events = await loop.run_in_executor(
                    _STREAM_FEED_EXECUTOR,
                    self.stream.feed,
                    pcm_batch,
                )
                processed_at = time.monotonic()
                processed_frames += len(batch)
                self._last_pcm_processed_at = processed_at
                process_ms = (processed_at - received_at) * 1000.0
                queue_depth = self._feed_queue.qsize()
                if process_ms > 1000.0 or processed_frames % 1000 < len(batch):
                    logger.info(
                        "STREAM_INPUT_TIMING event=pcm_processed traceId=%s "
                        "processed_frames=%s pcm_frames=%s pcm_audio_ms=%.1f "
                        "process_ms=%.1f feed_batch_frames=%s feed_queue_depth=%s "
                        "stream_consumed_ms=%.1f emitted_events=%s",
                        self.ctx.trace_id or "-",
                        processed_frames,
                        self._pcm_frames_seen,
                        self._pcm_bytes_seen / 2 / SAMPLE_RATE * 1000.0,
                        process_ms,
                        len(batch),
                        queue_depth,
                        getattr(self.stream, "_consumed_samples", 0)
                        * 1000.0
                        / SAMPLE_RATE,
                        len(events),
                    )
                if (
                    ctc_prefeed_enabled
                    and hasattr(self.stream, "prefeed_partial_snapshot")
                ):
                    if self._ctc_final_tail_pause_active():
                        self._log_ctc_tail_pause("prefeed_paused")
                    else:
                        prefeed_snap = self.stream.prefeed_partial_snapshot()
                        if prefeed_snap is not None:
                            await self.engine.prefeed_streaming_partial(
                                prefeed_snap, self.ctx.snapshot()
                            )
                if not self._abort_session:
                    for ev in events:
                        await self._dispatch_stream_event(ev)
                # AscendK2Stream: poll CTC callback queue for new partial texts
                # and schedule direct sends without blocking the feed loop.
                if ascend_k2_stream and not self._abort_session:
                    self._dispatch_ascend_k2_partials()
                if saw_sentinel:
                    return
            finally:
                for _ in range(task_done_count):
                    self._feed_queue.task_done()

    def _dispatch_ascend_k2_partials(self) -> None:
        """Poll AscendK2Stream partial queue and schedule async sends.

        Called from _feed_loop after each stream.feed() batch.  The CTC
        callback already throttles by deduplication; we add a configurable
        minimum interval here so a flood of new tokens doesn't overwhelm the
        WebSocket send path.
        """
        poll_fn = getattr(self.stream, "poll_partial_events", None)
        if poll_fn is None:
            return
        partial_texts: list[str] = poll_fn()
        if not partial_texts:
            return
        if self._ctc_final_tail_pause_active():
            self._log_ctc_tail_pause("direct_poll_paused")
            return
        loop = asyncio.get_running_loop()
        interval_ms = max(
            0,
            int(
                getattr(
                    self.ctx.cfg, "ctc_partial_post_first_interval_ms", 1500
                )
                or 0
            ),
        )
        now = time.monotonic()
        last_emit_at = float(
            self.ctx.runtime_state.get("ctc_partial_last_emit_at") or 0.0
        )
        for text in partial_texts:
            if not text or self._ws_closed or self._stopped or self._abort_session:
                break
            if self.ctx.runtime_state.get("ctc_prefeed_last_text") == text:
                continue
            if (
                interval_ms > 0
                and last_emit_at > 0
                and (now - last_emit_at) * 1000.0 < interval_ms
            ):
                continue
            last_emit_at = now
            self.ctx.runtime_state["ctc_partial_last_emit_at"] = now
            self.ctx.runtime_state["ctc_prefeed_last_text"] = text
            task = asyncio.create_task(self._send_ctc_direct_partial(text, {}))
            self._ctc_direct_send_tasks.add(task)
            task.add_done_callback(self._ctc_direct_send_tasks.discard)

    def _ensure_ctc_direct_result_callback(self) -> None:
        if self._ctc_direct_callback_installed:
            return
        if (
            str(getattr(self.ctx.cfg, "streaming_partial_backend", "vllm") or "vllm")
            .strip()
            .lower()
            != "ctc_om"
        ):
            return
        if not hasattr(self.engine, "install_streaming_result_callback"):
            return
        loop = asyncio.get_running_loop()

        def on_result(text: str, stats: dict[str, Any]) -> None:
            if not text or self._ws_closed or self._stopped or self._abort_session:
                return

            def schedule_send() -> None:
                if self._ws_closed or self._stopped or self._abort_session:
                    return
                if self._ctc_final_tail_pause_active():
                    self._log_ctc_tail_pause("direct_callback_paused")
                    return
                if self.ctx.runtime_state.get("ctc_prefeed_last_text") == text:
                    return
                now = time.monotonic()
                interval_ms = max(
                    0,
                    int(
                        getattr(
                            self.ctx.cfg,
                            "ctc_partial_post_first_interval_ms",
                            1500,
                        )
                        or 0
                    ),
                )
                last_emit_at = float(
                    self.ctx.runtime_state.get("ctc_partial_last_emit_at") or 0.0
                )
                if (
                    interval_ms > 0
                    and last_emit_at > 0
                    and (now - last_emit_at) * 1000.0 < interval_ms
                ):
                    return
                self.ctx.runtime_state["ctc_partial_last_emit_at"] = now
                self.ctx.runtime_state["ctc_prefeed_last_text"] = text
                task = asyncio.create_task(self._send_ctc_direct_partial(text, stats))
                self._ctc_direct_send_tasks.add(task)
                task.add_done_callback(self._ctc_direct_send_tasks.discard)

            loop.call_soon_threadsafe(schedule_send)

        if self.engine.install_streaming_result_callback(self.ctx, on_result):
            self._ctc_direct_callback_installed = True
            logger.info(
                "CTC_DIRECT_RESULT action=installed traceId=%s",
                self.ctx.trace_id or "-",
            )

    async def _send_ctc_direct_partial(
        self, text: str, stats: dict[str, Any]
    ) -> None:
        send_start = time.monotonic()
        try:
            sent = await self._send_json(
                {
                    "type": "partial",
                    "text": text,
                    "language": self.ctx.language,
                }
            )
        except WebSocketDisconnect:
            self._ws_closed = True
            return
        logger.info(
            "ASR_TIMING type=ctc_direct_send traceId=%s sent=%s text_chars=%s "
            "send_ms=%.1f batch_size=%s decode_ms=%.1f total_ms=%.1f",
            self.ctx.trace_id or "-",
            sent,
            len(text),
            (time.monotonic() - send_start) * 1000.0,
            int(stats.get("batch_size", 0)),
            float(stats.get("decode_ms", 0.0)),
            float(stats.get("ctc_total_tick_ms", stats.get("total_ms", 0.0))),
        )

    async def _dispatch_stream_event(self, ev) -> None:
        # Heavy work (full-segment inference) goes through the queue so
        # it stays serialized; lightweight notifications (speech start /
        # dropped, partial snapshot) fan out directly without queuing
        # so the placeholder UI shows up before the segment finishes.
        if isinstance(ev, SegmentReady):
            await self._close_partial_for_final()
            self._enqueue_segment(ev)
        elif isinstance(ev, PartialSnapshot):
            self._maybe_launch_partial(ev)
        elif isinstance(ev, SpeechStarted):
            await self._safe_speech_start(ev)
        elif isinstance(ev, SpeechDropped):
            await self._safe_speech_dropped()

    async def _close_partial_for_final(self) -> None:
        ctc_backend = (
            str(getattr(self.ctx.cfg, "streaming_partial_backend", "vllm") or "vllm")
            .strip()
            .lower()
            == "ctc_om"
        )
        ascend_k2_stream = hasattr(self.stream, "poll_partial_events")
        if not ctc_backend and not ascend_k2_stream:
            return
        # Cancel any in-flight CTC direct send tasks before emitting final.
        # This applies to both the old ctc_om engine-callback path and the
        # new AscendK2Stream poll path.
        for task in list(self._ctc_direct_send_tasks):
            if not task.done():
                task.cancel()
        self._ctc_direct_send_tasks.clear()
        # Old ctc_om engine-level callback teardown (not used by AscendK2Stream)
        if ctc_backend and hasattr(self.engine, "close_streaming_partial"):
            try:
                await self.engine.close_streaming_partial(self.ctx.snapshot())
                self._ctc_direct_callback_installed = False
            except Exception:
                logger.debug("close_streaming_partial failed", exc_info=True)
        logger.info(
            "CTC_DIRECT_RESULT action=closed_for_final traceId=%s",
            self.ctx.trace_id or "-",
        )

    def _enqueue_segment(self, ev: SegmentReady) -> None:
        snapshot = self.ctx.snapshot()
        ev.queued_at = time.monotonic()
        ev.queue_depth_at_enqueue = self._work_queue.qsize()
        try:
            self._work_queue.put_nowait((ev, snapshot))
            logger.info(
                "STREAM_QUEUE_TIMING event=segment_enqueued traceId=%s "
                "queue_depth=%s start_ms=%s end_ms=%s audio_sec=%.3f",
                snapshot.trace_id or "-",
                ev.queue_depth_at_enqueue,
                ev.start_ms,
                ev.end_ms,
                len(ev.pcm) / SAMPLE_RATE,
            )
        except asyncio.QueueFull:
            logger.warning("Work queue full, dropping segment")

    def _maybe_launch_partial(self, snap: PartialSnapshot) -> None:
        if snap.is_first:
            speech_key = snap.speech_started_at or snap.snapshot_at
            if (
                self._last_first_partial_speech_started_at is not None
                and speech_key
                and abs(speech_key - self._last_first_partial_speech_started_at) < 1e-6
            ):
                logger.info(
                    "K2_PARTIAL_SCHEDULE action=drop_duplicate_first traceId=%s "
                    "audio_ms=%.1f",
                    self.ctx.trace_id or "-",
                    len(snap.pcm) * 1000.0 / SAMPLE_RATE,
                )
                return
            self._last_first_partial_speech_started_at = speech_key
            self._launched_first_partial = True
        elif not self._should_launch_partial_refresh(snap):
            return

        if self._partial_task is not None and not self._partial_task.done():
            if self.engine._use_sherpa_partial(self.ctx.cfg):
                # K2/sherpa decode runs in a native executor and cannot be
                # cancelled once submitted. Dropping new refreshes while one is
                # in flight prevents stale cumulative decodes from piling up
                # and delaying first text for other sessions.
                logger.info(
                    "K2_PARTIAL_SCHEDULE action=drop_inflight traceId=%s "
                    "first=%s audio_ms=%.1f",
                    self.ctx.trace_id or "-",
                    snap.is_first,
                    len(snap.pcm) * 1000.0 / SAMPLE_RATE,
                )
                return
            current = self._partial_snapshot
            can_replace_queued_refresh = (
                current is not None
                and not current.is_first
                and current.model_started_at <= 0
                and not snap.is_first
            )
            if not can_replace_queued_refresh:
                return
            logger.info(
                "Replacing queued partial refresh before vLLM: traceId=%s",
                self.ctx.trace_id or "-",
            )
            self._partial_task.cancel()

        snapshot_ctx = self.ctx.snapshot()
        self._partial_snapshot = snap
        if not snap.is_first:
            self._last_partial_refresh_launched_at = snap.snapshot_at or time.monotonic()
        logger.info(
            "K2_PARTIAL_SCHEDULE action=launch traceId=%s first=%s "
            "audio_ms=%.1f sent_first_text=%s",
            self.ctx.trace_id or "-",
            snap.is_first,
            len(snap.pcm) * 1000.0 / SAMPLE_RATE,
            self._sent_first_partial_text,
        )
        self._partial_task = asyncio.create_task(self._safe_partial(snap, snapshot_ctx))
        self._partial_task.add_done_callback(
            lambda task: self._clear_partial_task(task, snap)
        )
        if (
            self.engine._use_sherpa_partial(self.ctx.cfg)
            and not self._sent_first_partial_text
        ):
            self._ensure_k2_result_poll(snap)

    def _ensure_k2_result_poll(self, snap: PartialSnapshot) -> None:
        if self._k2_result_poll_task is not None and not self._k2_result_poll_task.done():
            return
        snapshot_ctx = self.ctx.snapshot()
        poll_snap = PartialSnapshot(
            pcm=snap.pcm,
            speech_started_at=snap.speech_started_at,
            snapshot_at=snap.snapshot_at or time.monotonic(),
            is_first=False,
        )
        self._k2_result_poll_task = asyncio.create_task(
            self._poll_k2_first_result(poll_snap, snapshot_ctx)
        )

    async def _poll_k2_first_result(
        self, snap: PartialSnapshot, ctx: SessionContext
    ) -> None:
        deadline = time.monotonic() + 3.0
        ctc_backend = (
            str(getattr(ctx.cfg, "streaming_partial_backend", "vllm") or "vllm")
            .strip()
            .lower()
            == "ctc_om"
        )
        sleep_sec = 0.02 if ctc_backend else 0.05
        while (
            not self._sent_first_partial_text
            and not self._ws_closed
            and not self._stopped
            and time.monotonic() < deadline
        ):
            await asyncio.sleep(sleep_sec)
            if (
                not ctc_backend
                and self._partial_task is not None
                and not self._partial_task.done()
            ):
                continue
            try:
                if ctc_backend and hasattr(self.engine, "poll_cached_streaming_partial"):
                    sent = await self.engine.poll_cached_streaming_partial(ctx)
                    if sent:
                        return
                else:
                    snap.snapshot_at = time.monotonic()
                    await self.engine.handle_partial(snap, ctx)
            except asyncio.CancelledError:
                raise
            except WebSocketDisconnect:
                self._ws_closed = True
                return
            except Exception:
                logger.debug("K2 cached result poll failed", exc_info=True)
                return

    def _should_launch_partial_refresh(self, snap: PartialSnapshot) -> bool:
        cfg = self.ctx.cfg
        if not getattr(cfg, "asr_partial_refresh_adaptive_enabled", True):
            return True

        now = snap.snapshot_at or time.monotonic()
        if self._last_partial_refresh_launched_at <= 0:
            return True

        if self.engine._use_sherpa_partial(cfg):
            if not self._sent_first_partial_text:
                interval_ms = int(
                    getattr(cfg, "k2_partial_first_text_interval_ms", 500) or 500
                )
            else:
                interval_ms = int(
                    getattr(cfg, "k2_partial_post_text_interval_ms", 3000) or 3000
                )
            elapsed_ms = (now - self._last_partial_refresh_launched_at) * 1000.0
            if elapsed_ms >= max(0, interval_ms):
                return True
            logger.info(
                "K2_PARTIAL_SCHEDULE action=drop_interval traceId=%s "
                "audio_ms=%.1f elapsed_ms=%.1f interval_ms=%s sent_first_text=%s",
                self.ctx.trace_id or "-",
                len(snap.pcm) * 1000.0 / SAMPLE_RATE,
                elapsed_ms,
                interval_ms,
                self._sent_first_partial_text,
            )
            return False

        pressure = final_pressure()
        backlog = pressure.backlog
        high_backlog = int(
            getattr(cfg, "asr_partial_refresh_high_pressure_final_backlog", 24)
        )
        pressure_backlog = int(
            getattr(cfg, "asr_partial_refresh_pressure_final_backlog", 8)
        )
        feed_depth = self._feed_queue.qsize()
        high_feed_depth = int(
            getattr(cfg, "asr_partial_refresh_high_pressure_feed_queue_depth", 1024)
        )
        pressure_feed_depth = int(
            getattr(cfg, "asr_partial_refresh_pressure_feed_queue_depth", 256)
        )
        if feed_depth >= high_feed_depth:
            logger.info(
                "Skipping partial refresh under high feed pressure: traceId=%s "
                "feed_queue_depth=%s threshold=%s",
                self.ctx.trace_id or "-",
                feed_depth,
                high_feed_depth,
            )
            return False
        if backlog >= high_backlog:
            interval_ms = int(
                getattr(cfg, "asr_partial_refresh_high_pressure_interval_ms", 5000)
            )
        elif (
            backlog >= pressure_backlog
            or pressure.active >= pressure.limit
            or feed_depth >= pressure_feed_depth
        ):
            interval_ms = int(
                getattr(cfg, "asr_partial_refresh_pressure_interval_ms", 2000)
            )
        else:
            interval_ms = int(getattr(cfg, "pseudo_stream_interval_ms", 1000))

        elapsed_ms = (now - self._last_partial_refresh_launched_at) * 1000.0
        if elapsed_ms >= max(0, interval_ms):
            return True

        logger.debug(
            "Skipping partial refresh by adaptive interval: traceId=%s "
            "elapsed_ms=%.1f interval_ms=%s final_waiting=%s final_active=%s "
            "final_limit=%s feed_queue_depth=%s",
            self.ctx.trace_id or "-",
            elapsed_ms,
            interval_ms,
            pressure.waiting,
            pressure.active,
            pressure.limit,
            feed_depth,
        )
        return False

    def _clear_partial_task(
        self, task: asyncio.Task, snap: PartialSnapshot
    ) -> None:
        if self._partial_task is task:
            self._partial_task = None
        if self._partial_snapshot is snap:
            self._partial_snapshot = None

    async def _safe_partial(self, snap: PartialSnapshot, ctx: SessionContext) -> None:
        try:
            await self.engine.handle_partial(snap, ctx)
        except asyncio.CancelledError:
            raise
        except WebSocketDisconnect:
            self._ws_closed = True
        except Exception:
            logger.debug("engine.handle_partial failed", exc_info=True)

    async def _safe_speech_start(self, ev: SpeechStarted) -> None:
        try:
            self.ctx.last_speech_started_at = ev.started_at
            for key in list(self.ctx.runtime_state):
                if str(key).startswith("ctc_tail_pause_logged:"):
                    self.ctx.runtime_state.pop(key, None)
            await self.engine.handle_speech_start(self.ctx.snapshot())
        except WebSocketDisconnect:
            self._ws_closed = True
        except Exception:
            logger.debug("engine.handle_speech_start failed", exc_info=True)

    async def _safe_speech_dropped(self) -> None:
        try:
            await self.engine.handle_speech_dropped(self.ctx.snapshot())
        except WebSocketDisconnect:
            self._ws_closed = True
        except Exception:
            logger.debug("engine.handle_speech_dropped failed", exc_info=True)

    # ------------------------------------------------------------------
    # Work loop: dispatch final segments with per-session concurrency
    # ------------------------------------------------------------------

    async def _dispatch_segment(self, seg: object, ctx: object) -> None:
        """Run one final segment under the per-session semaphore.

        This coroutine is wrapped in asyncio.create_task(). When cleanup()
        cancels the task, asyncio.CancelledError is raised here (or wrapped
        as TimeoutError by inner wait_for calls). We must not log that as an
        error — just let the cancellation propagate cleanly.
        """
        async with self._session_final_sem:
            seg.dequeued_at = time.monotonic()  # type: ignore[attr-defined]
            session_queue_ms = (
                (seg.dequeued_at - seg.queued_at) * 1000.0  # type: ignore[attr-defined]
                if seg.queued_at > 0  # type: ignore[attr-defined]
                else 0.0
            )
            logger.info(
                "STREAM_QUEUE_TIMING event=segment_dequeued traceId=%s "
                "session_queue_ms=%.1f queue_depth_at_enqueue=%s "
                "start_ms=%s end_ms=%s audio_sec=%.3f",
                ctx.trace_id or "-",  # type: ignore[attr-defined]
                session_queue_ms,
                seg.queue_depth_at_enqueue,  # type: ignore[attr-defined]
                seg.start_ms,  # type: ignore[attr-defined]
                seg.end_ms,  # type: ignore[attr-defined]
                len(seg.pcm) / SAMPLE_RATE,  # type: ignore[attr-defined]
            )
            # Spread concurrent stop-flush requests over a random jitter window
            # so that BS52 streams ending simultaneously do not all slam vLLM
            # with long audio segments at the exact same instant.
            if getattr(seg, "is_stop_flush", False):  # type: ignore[attr-defined]
                jitter_ms = int(getattr(self.ctx.cfg, "asr_stop_flush_jitter_ms", 0))
                if jitter_ms > 0:
                    jitter_sec = random.uniform(0.0, jitter_ms / 1000.0)
                    logger.debug(
                        "STOP_FLUSH_JITTER traceId=%s jitter_ms=%.1f",
                        ctx.trace_id or "-",  # type: ignore[attr-defined]
                        jitter_sec * 1000.0,
                    )
                    await asyncio.sleep(jitter_sec)
            try:
                sent = await self.engine.handle_segment(seg, ctx)
                if sent:
                    self._sent_any_response = True
                if self._ws_closed:
                    dropped = self._drain_queue_nowait(self._work_queue)
                    logger.info(
                        "STREAM_QUEUE_TIMING event=work_send_closed traceId=%s "
                        "dropped_work_items=%s",
                        ctx.trace_id or "-",  # type: ignore[attr-defined]
                        dropped,
                    )
            except asyncio.CancelledError:
                # Task cancelled by cleanup() — propagate cleanly, don't log as error.
                raise
            except (TimeoutError, asyncio.TimeoutError) as e:
                # asyncio.wait_for wraps CancelledError as TimeoutError when the
                # outer task is cancelled (Python 3.11+). Treat this as clean
                # cancellation so it doesn't surface as a user-visible error.
                cause = getattr(e, "__cause__", None)
                if isinstance(cause, asyncio.CancelledError):
                    raise asyncio.CancelledError(
                        "segment dispatch cancelled during vLLM wait"
                    ) from e
                logger.exception("engine.handle_segment timed out")
                if not await self._send_json(
                    {"type": "error", "message": str(e)}
                ):
                    self._ws_closed = True
                    self._drain_queue_nowait(self._work_queue)
            except WebSocketDisconnect:
                self._ws_closed = True
                self._drain_queue_nowait(self._work_queue)
            except Exception as e:
                logger.exception("engine.handle_segment failed")
                if not await self._send_json(
                    {"type": "error", "message": str(e)}
                ):
                    self._ws_closed = True
                    self._drain_queue_nowait(self._work_queue)

    async def _work_loop(self) -> None:
        while True:
            item = await self._work_queue.get()
            if item is _SENTINEL:
                break
            if self._abort_session or self._ws_closed:
                dropped = self._drain_queue_nowait(self._work_queue)
                logger.info(
                    "STREAM_QUEUE_TIMING event=work_aborted traceId=%s "
                    "dropped_work_items=%s ws_closed=%s aborted=%s",
                    self.ctx.trace_id or "-",
                    dropped,
                    self._ws_closed,
                    self._abort_session,
                )
                break
            seg, ctx = item
            # Dispatch concurrently under per-session semaphore so that the
            # next segment's encode can begin while the current one's vLLM
            # decode is still in flight. The semaphore (asr_final_session_parallel)
            # prevents a single session from monopolising the global final pool.
            task = asyncio.create_task(self._dispatch_segment(seg, ctx))
            self._pending_final_tasks.add(task)
            task.add_done_callback(self._pending_final_tasks.discard)

        # Drain remaining queue items on sentinel/abort
        self._drain_queue_nowait(self._work_queue)

        # Wait for all inflight dispatch tasks to finish before on_stop
        if self._pending_final_tasks:
            await asyncio.gather(*self._pending_final_tasks, return_exceptions=True)
            self._pending_final_tasks.clear()

        if not self._abort_session:
            try:
                await self.engine.on_stop(
                    self.ctx.snapshot(),
                    sent_any_response=self._sent_any_response,
                    stopped=self._stopped,
                )
            except Exception:
                logger.exception("engine.on_stop failed")
