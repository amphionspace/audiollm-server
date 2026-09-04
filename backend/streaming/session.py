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
import hashlib
import json
import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from fastapi import WebSocket, WebSocketDisconnect

from ..asr.enrollment import get_enrollment_store
from ..asr.hotword import query_text_hotwords, sanitize_hotwords
from ..config import Config, load_config
from ..debug_dump import SessionDumper, new_session_id
from ..diarization.client import DiarizationSession, DiarizationUnavailableError
from ..recall_user import HotwordPoolIdError, normalize_hotword_pool_id
from .audio_stream import AudioStream, VadSegmentedStream
from .events import (
    PartialSnapshot,
    PartialText,
    SegmentReady,
    SpeechDropped,
    SpeechStarted,
)
from .protocol import ControlAction, NativeProtocol, PcmAction, WireProtocol

if TYPE_CHECKING:
    from ..tasks.base import TaskEngine

logger = logging.getLogger(__name__)

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


def _hotwords_log_fields(hotwords: list[str]) -> dict[str, Any]:
    count = len(hotwords)
    if count <= 50:
        return {"hotwords_count": count, "hotwords": list(hotwords)}
    encoded = json.dumps(hotwords, ensure_ascii=False, separators=(",", ":"))
    return {
        "hotwords_count": count,
        "hotwords_preview": hotwords[:20],
        "hotwords_hash": hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16],
    }


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
    hotwords: list[str] = field(default_factory=list)
    hotword_pool_id: str = "default"
    gateway_trace_id: str = ""
    # Optional cached target-speaker enrollment (base64 WAV). The session
    # resolves the opaque ``enrollment_id`` once at start / on every
    # ``update_hotwords`` and stores the WAV inline so per-segment
    # inference doesn't re-hit the in-memory store on every call.
    enrollment_id: str | None = None
    enrollment_b64: str | None = None
    enrollment_fallback_reason: str = ""
    # Mutable per-session sidecar timeline. Snapshots intentionally retain the
    # same object so queued final segments can wait for newly finalized turns.
    diarization: DiarizationSession | None = None
    # Per-connection debug context. ``session_id`` is unique per WebSocket;
    # ``dumper`` is non-None only when ``debug_dump_enabled`` so engines can
    # cheaply skip the dump path otherwise.
    session_id: str = ""
    dumper: SessionDumper | None = None
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
        allow_diarization: bool = False,
    ) -> None:
        self.ws = websocket
        self.stream = stream
        self.engine = engine
        # The wire protocol owns on-the-wire framing; the rest of the session
        # only ever deals in control dicts + PCM bytes. NativeProtocol is the
        # historical 1:1 framing so existing endpoints are byte-for-byte
        # unchanged when no protocol is supplied.
        self.protocol: WireProtocol = protocol or NativeProtocol()
        self._allow_diarization = allow_diarization

        # Endpoint-level forced overrides: highest precedence. Applied here and
        # again right after the client's start.config override (see
        # _handle_start) so an untrusted client cannot undo an endpoint binding
        # via parameter.asr_config / start.config. Used by /tuling/ast/v3 to
        # pin a per-endpoint primary upstream and force primary-only
        # (enable_secondary_asr=False -> the local secondary is never queried).
        self._config_overrides = config_overrides

        self.cfg: Config = load_config()
        if config_overrides:
            self.cfg = self.cfg.override(**config_overrides)
        self.stream.configure(self.cfg)

        # Debug dump is an operator-only knob (not client-overridable), so it
        # is resolved once here from the endpoint-effective config and never
        # changes for the life of the connection.
        self.session_id = new_session_id()
        self._dumper: SessionDumper | None = (
            SessionDumper(
                self.cfg.debug_dump_dir,
                self.session_id,
                engine=self.engine.name,
            )
            if self.cfg.debug_dump_enabled
            else None
        )

        self.ctx = SessionContext(
            cfg=self.cfg,
            language=language,
            src_lang=map_language(language),
            hotwords=[],
            hotword_pool_id=normalize_hotword_pool_id(
                None,
                default=self.cfg.hotword_pool_id,
            ),
            gateway_trace_id="",
            session_id=self.session_id,
            dumper=self._dumper,
            send_json=self._send_json,
        )

        self._work_queue: asyncio.Queue = asyncio.Queue(maxsize=40)
        self._partial_task: asyncio.Task | None = None
        # Long-text hotword extraction runs out-of-band so the receive
        # loop never blocks on an LLM round-trip; outstanding tasks
        # are tracked here and cancelled in cleanup.
        self._extract_tasks: set[asyncio.Task] = set()

        self._started = False
        self._stopped = False
        self._sent_any_response = False
        self._ws_closed = False
        self._suppress_terminal = False
        self._diarization: DiarizationSession | None = None

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        # ``session_id`` / ``dump_dir`` ride the ``ready`` frame only when
        # dumping is on, so default deployments keep the bare {type:"ready"}
        # contract. AST v3 suppresses ``ready`` entirely, so this is native-only.
        ready: dict[str, Any] = {"type": "ready"}
        if self._dumper is not None:
            ready["session_id"] = self.session_id
            ready["dump_dir"] = self._dumper.base_dir
        sent_ready = await self._send_json(ready)
        if sent_ready:
            logger.info(
                "%s ready (language=%s)",
                self.engine.name,
                self.ctx.language,
            )
        try:
            if await self._start_async_stream_or_fallback():
                receive_task = asyncio.create_task(self._receive_loop())
                event_task = asyncio.create_task(self._stream_event_loop())
                work_task = asyncio.create_task(self._work_loop())
                await receive_task
                await event_task
                await self._work_queue.put(_SENTINEL)
                await work_task
            else:
                await asyncio.gather(self._receive_loop(), self._work_loop())
        except Exception:
            logger.exception("StreamingSession[%s] error", self.engine.name)
        finally:
            # Protocols that frame an explicit end-of-session marker (e.g. AST
            # v3's header.status=2) emit it here, after every queued segment
            # has drained. Native framing returns None and sends nothing.
            if not self._suppress_terminal:
                terminal = self.protocol.encode_terminal()
                if terminal is not None:
                    await self._send_wire(terminal)

    async def cleanup(self) -> None:
        if self._partial_task and not self._partial_task.done():
            self._partial_task.cancel()
        if self._extract_tasks:
            for task in self._extract_tasks:
                task.cancel()
            await asyncio.gather(*self._extract_tasks, return_exceptions=True)
        aclose = getattr(self.stream, "aclose", None)
        if callable(aclose):
            await aclose()
        if self._diarization is not None:
            await self._diarization.aclose()
        logger.info("StreamingSession[%s] ended", self.engine.name)

    async def _start_async_stream_or_fallback(self) -> bool:
        start = getattr(self.stream, "start", None)
        if not callable(start):
            return False
        try:
            await start()
            return True
        except Exception as exc:
            if getattr(self.cfg, "k2_fallback_to_local", True):
                logger.warning(
                    "Async stream start failed; falling back to local VAD: %s",
                    exc,
                )
                self.stream = VadSegmentedStream()
                self.stream.configure(self.cfg)
                return False
            raise

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------

    async def _send_wire(self, wire: dict[str, Any]) -> bool:
        if self._ws_closed:
            return False
        try:
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
        wire = self.protocol.encode_outbound(payload)
        if wire is None:
            return not self._ws_closed
        return await self._send_wire(wire)

    # ------------------------------------------------------------------
    # Receive loop: control messages + binary PCM
    # ------------------------------------------------------------------

    async def _receive_loop(self) -> None:
        try:
            while True:
                msg = await self.ws.receive()
                if msg.get("type") == "websocket.disconnect":
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
            logger.info("WebSocket disconnected (%s)", self.engine.name)
        finally:
            await self._finish_diarization()
            # Always flush remaining audio so engine sees the tail.
            for ev in await self._flush_stream(force=True):
                await self._dispatch_stream_event(ev)
            if not callable(getattr(self.stream, "events", None)):
                await self._work_queue.put(_SENTINEL)

    async def _handle_control(self, ctrl: dict) -> bool:
        """Dispatch one already-parsed control message; return True to stop."""
        msg_type = ctrl.get("type", "")
        if msg_type == "start":
            await self._handle_start(ctrl)
            return False
        if msg_type == "error":
            await self._send_json(
                {
                    "type": "error",
                    "code": str(ctrl.get("code") or "invalid_request"),
                    "message": str(ctrl.get("message") or "invalid request"),
                }
            )
            self._suppress_terminal = True
            return True
        if msg_type == "stop":
            await self._handle_stop()
            return True
        if msg_type == "update_hotwords":
            await self._handle_update_hotwords(ctrl)
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

        client_config = ctrl.get("config")
        if isinstance(client_config, dict) and client_config:
            # Untrusted per-connection override: only whitelisted tuning knobs
            # are honored; infra/secret fields are dropped (see override_client).
            self.cfg = self.cfg.override_client(**client_config)
            self.ctx.cfg = self.cfg
            self.stream.configure(self.cfg)
            logger.info("Config overridden by client: %s", list(client_config.keys()))
            # Endpoint-level overrides win over the client: re-apply them so a
            # client cannot re-enable a force-disabled knob (e.g. AST v3's
            # enable_secondary_asr=False) through start.config / asr_config.
            if self._config_overrides:
                self.cfg = self.cfg.override(**self._config_overrides)
                self.ctx.cfg = self.cfg
                self.stream.configure(self.cfg)

        if "user_id" in ctrl and "hotword_pool_id" not in ctrl:
            await self._send_json(
                {
                    "type": "error",
                    "code": "invalid_hotword_pool_id",
                    "message": "user_id is no longer supported; use hotword_pool_id",
                }
            )
            return
        hotword_pool_raw = ctrl.get("hotword_pool_id", _SENTINEL)
        if hotword_pool_raw is not _SENTINEL:
            try:
                self.ctx.hotword_pool_id = normalize_hotword_pool_id(
                    hotword_pool_raw,
                    default=self.cfg.hotword_pool_id,
                )
            except HotwordPoolIdError as exc:
                await self._send_json(
                    {
                        "type": "error",
                        "code": "invalid_hotword_pool_id",
                        "message": str(exc),
                    }
                )
                return
        else:
            self.ctx.hotword_pool_id = normalize_hotword_pool_id(
                None,
                default=self.cfg.hotword_pool_id,
            )

        for key in (
            "gateway_trace_id",
            "gateway_session_id",
            "session_id",
            "trace_id",
            "traceId",
        ):
            value = str(ctrl.get(key, "") or "").strip()
            if value:
                self.ctx.gateway_trace_id = value
                break

        lang_val = str(ctrl.get("language", "")).strip()
        if lang_val:
            self.ctx.language = lang_val
            self.ctx.src_lang = map_language(lang_val)

        hw_raw = ctrl.get("hotwords")
        if isinstance(hw_raw, list):
            self.ctx.hotwords = sanitize_hotwords(hw_raw)

        if "enrollment_id" in ctrl:
            self._apply_enrollment(ctrl.get("enrollment_id"))

        await self._start_diarization_if_requested()

        self._started = True
        fmt = ctrl.get("format", "pcm_s16le")
        sr = ctrl.get("sample_rate_hz", 16000)
        ch = ctrl.get("channels", 1)
        logger.info(
            "Start[%s] backend_session_id=%s gateway_trace_id=%s mode=%s "
            "format=%s sr=%s ch=%s language=%s hotword_pool_id=%s hotwords=%s",
            self.engine.name,
            self.session_id,
            self.ctx.gateway_trace_id or "n/a",
            ctrl.get("mode"), fmt, sr, ch, self.ctx.language,
            self.ctx.hotword_pool_id, _hotwords_log_fields(self.ctx.hotwords),
        )

        try:
            await self.engine.on_start(ctrl, self.ctx)
        except ValueError as exc:
            self._started = False
            logger.warning("engine.on_start rejected request: %s", exc)
            await self._send_json(
                {
                    "type": "error",
                    "code": "invalid_start",
                    "message": str(exc),
                }
            )
        except Exception:
            logger.exception("engine.on_start failed")

    async def _handle_update_hotwords(self, ctrl: dict) -> None:
        self.ctx.hotwords = sanitize_hotwords(ctrl.get("hotwords", []))
        if "user_id" in ctrl and "hotword_pool_id" not in ctrl:
            await self._send_json(
                {
                    "type": "error",
                    "code": "invalid_hotword_pool_id",
                    "message": "user_id is no longer supported; use hotword_pool_id",
                }
            )
            return
        hotword_pool_raw = ctrl.get("hotword_pool_id", _SENTINEL)
        if hotword_pool_raw is not _SENTINEL:
            try:
                self.ctx.hotword_pool_id = normalize_hotword_pool_id(
                    hotword_pool_raw,
                    default=self.cfg.hotword_pool_id,
                )
            except HotwordPoolIdError as exc:
                await self._send_json(
                    {
                        "type": "error",
                        "code": "invalid_hotword_pool_id",
                        "message": str(exc),
                    }
                )
                return
        if "src_lang" in ctrl:
            lang_val = str(ctrl.get("src_lang", "")).strip()
            if lang_val:
                self.ctx.language = lang_val
                self.ctx.src_lang = map_language(lang_val)
        if "enrollment_id" in ctrl:
            self._apply_enrollment(ctrl.get("enrollment_id"))
        logger.info(
            "Hotwords updated: backend_session_id=%s gateway_trace_id=%s "
            "src_lang=%s enrollment=%s hotword_pool_id=%s hotwords=%s",
            self.session_id,
            self.ctx.gateway_trace_id or "n/a",
            self.ctx.src_lang,
            self.ctx.enrollment_id,
            self.ctx.hotword_pool_id,
            _hotwords_log_fields(self.ctx.hotwords),
        )

    def _apply_enrollment(self, raw_id: object) -> None:
        """Resolve an enrollment id (or ``None``/empty to clear) into a
        cached WAV. Unknown / expired ids are treated as "no enrollment"
        so a stale id from a long-lived tab degrades to plain ASR
        instead of breaking the WS session."""
        if raw_id is None or not isinstance(raw_id, str) or not raw_id.strip():
            self.ctx.enrollment_id = None
            self.ctx.enrollment_b64 = None
            self.ctx.enrollment_fallback_reason = "disabled"
            return
        ident = raw_id.strip()
        if self.cfg.enable_triton_enrollment_store:
            self.ctx.enrollment_id = ident
            self.ctx.enrollment_b64 = None
            self.ctx.enrollment_fallback_reason = ""
            return
        entry = get_enrollment_store().get(ident)
        if entry is None:
            logger.warning("Enrollment id %s not found / expired", ident)
            self.ctx.enrollment_id = None
            self.ctx.enrollment_b64 = None
            self.ctx.enrollment_fallback_reason = "not_found"
            return
        self.ctx.enrollment_id = ident
        self.ctx.enrollment_b64 = entry.wav_base64
        self.ctx.enrollment_fallback_reason = ""

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
        logger.info("Stop received (%s), flushing", self.engine.name)
        await self._finish_diarization()
        for ev in await self._flush_stream(force=True):
            await self._dispatch_stream_event(ev)

    # ------------------------------------------------------------------
    # PCM dispatch
    # ------------------------------------------------------------------

    async def _handle_pcm(self, pcm_bytes: bytes) -> None:
        if self._diarization is not None:
            await self._diarization.feed(pcm_bytes)
        result = self.stream.feed(pcm_bytes)
        if hasattr(result, "__await__"):
            result = await result
        for ev in result:
            await self._dispatch_stream_event(ev)

    async def _dispatch_stream_event(self, ev) -> None:
        # Heavy work (full-segment inference) goes through the queue so
        # it stays serialized; lightweight notifications (speech start /
        # dropped, partial snapshot) fan out directly without queuing
        # so the placeholder UI shows up before the segment finishes.
        if isinstance(ev, SegmentReady):
            self._enqueue_segment(ev)
        elif isinstance(ev, PartialSnapshot):
            self._maybe_launch_partial(ev)
        elif isinstance(ev, PartialText):
            await self._safe_partial_text(ev)
        elif isinstance(ev, SpeechStarted):
            await self._safe_speech_start()
        elif isinstance(ev, SpeechDropped):
            await self._safe_speech_dropped()

    def _enqueue_segment(self, ev: SegmentReady) -> None:
        snapshot = self.ctx.snapshot()
        try:
            self._work_queue.put_nowait((ev, snapshot))
        except asyncio.QueueFull:
            logger.warning("Work queue full, dropping segment")

    def _maybe_launch_partial(self, snap: PartialSnapshot) -> None:
        if self._partial_task is not None and not self._partial_task.done():
            return
        snapshot_ctx = self.ctx.snapshot()
        self._partial_task = asyncio.create_task(self._safe_partial(snap, snapshot_ctx))

    async def _safe_partial(self, snap: PartialSnapshot, ctx: SessionContext) -> None:
        try:
            await self.engine.handle_partial(snap, ctx)
        except WebSocketDisconnect:
            self._ws_closed = True
        except Exception:
            logger.debug("engine.handle_partial failed", exc_info=True)

    async def _safe_partial_text(self, part: PartialText) -> None:
        text = str(part.text or "").strip()
        if not text:
            return
        try:
            payload = {
                "type": "partial",
                "text": text,
                "language": part.language or self.ctx.language,
            }
            if part.id:
                payload["id"] = part.id
            await self._send_json(payload)
        except WebSocketDisconnect:
            self._ws_closed = True
        except Exception:
            logger.debug("partial text send failed", exc_info=True)

    async def _safe_speech_start(self) -> None:
        try:
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
    # Work loop: drain final segments serially
    # ------------------------------------------------------------------

    async def _work_loop(self) -> None:
        while True:
            item = await self._work_queue.get()
            if item is _SENTINEL:
                break
            seg, ctx = item
            try:
                sent = await self.engine.handle_segment(seg, ctx)
                if sent:
                    self._sent_any_response = True
            except WebSocketDisconnect:
                self._ws_closed = True
                break
            except Exception as e:
                logger.exception("engine.handle_segment failed")
                if not await self._send_json(
                    {"type": "error", "message": str(e)}
                ):
                    break

        try:
            await self.engine.on_stop(
                self.ctx.snapshot(),
                sent_any_response=self._sent_any_response,
                stopped=self._stopped,
            )
        except Exception:
            logger.exception("engine.on_stop failed")

    async def _flush_stream(self, *, force: bool) -> list:
        result = self.stream.flush(force=force)
        if hasattr(result, "__await__"):
            result = await result
        return list(result or [])

    async def _start_diarization_if_requested(self) -> None:
        if not (
            self._allow_diarization
            and self.cfg.enable_role_separation
            and self.cfg.diarization_enabled
        ):
            return
        client = DiarizationSession(
            self.cfg,
            session_id=self.session_id,
            trace_id=self.ctx.gateway_trace_id,
        )
        try:
            await client.start()
        except DiarizationUnavailableError as exc:
            logger.warning(
                "Diarization unavailable; AST session will use rl=0: "
                "session_id=%s trace_id=%s reason=%s",
                self.session_id,
                self.ctx.gateway_trace_id or "n/a",
                exc,
            )
            await client.aclose()
            return
        self._diarization = client
        self.ctx.diarization = client

    async def _finish_diarization(self) -> None:
        if self._diarization is not None:
            await self._diarization.finish()

    async def _stream_event_loop(self) -> None:
        events = getattr(self.stream, "events", None)
        if not callable(events):
            return
        try:
            async for ev in events():
                await self._dispatch_stream_event(ev)
        except WebSocketDisconnect:
            self._ws_closed = True
        except Exception as exc:
            logger.exception("Async stream event loop failed")
            await self._send_json({"type": "error", "message": str(exc)})
