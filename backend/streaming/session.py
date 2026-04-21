"""Generic WebSocket session that wires an AudioStream to a TaskEngine.

The session owns:

- WebSocket lifecycle (ready, receive loop, error/close)
- Parsing of common control messages (start/stop/update_hotwords)
- Per-session config override (Config.override)
- Dispatching ``SegmentReady`` events serially through a work queue
- Throttled, non-overlapping dispatch of ``PartialSnapshot`` events

It does NOT know what "ASR" or "emotion" means; that lives in the engine.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from fastapi import WebSocket, WebSocketDisconnect

from ..asr.hotword import sanitize_hotwords
from ..config import Config, load_config
from .audio_stream import AudioStream
from .events import PartialSnapshot, SegmentReady

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
    ) -> None:
        self.ws = websocket
        self.stream = stream
        self.engine = engine

        self.cfg: Config = load_config()
        self.stream.configure(self.cfg)

        self.ctx = SessionContext(
            cfg=self.cfg,
            language=language,
            src_lang=map_language(language),
            hotwords=[],
            send_json=self._send_json,
        )

        self._work_queue: asyncio.Queue = asyncio.Queue(maxsize=40)
        self._partial_task: asyncio.Task | None = None

        self._started = False
        self._stopped = False
        self._sent_any_response = False
        self._ws_closed = False

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
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

    async def cleanup(self) -> None:
        if self._partial_task and not self._partial_task.done():
            self._partial_task.cancel()
        logger.info("StreamingSession[%s] ended", self.engine.name)

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------

    async def _send_json(self, payload: dict[str, Any]) -> bool:
        if self._ws_closed:
            return False
        try:
            await self.ws.send_json(payload)
            return True
        except (WebSocketDisconnect, RuntimeError):
            self._ws_closed = True
            return False

    # ------------------------------------------------------------------
    # Receive loop: control messages + binary PCM
    # ------------------------------------------------------------------

    async def _receive_loop(self) -> None:
        try:
            while True:
                msg = await self.ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break

                if "text" in msg and msg["text"]:
                    stop_after = await self._handle_text(msg["text"])
                    if stop_after:
                        break
                elif "bytes" in msg and msg["bytes"]:
                    if not self._started or self._stopped:
                        continue
                    await self._handle_pcm(msg["bytes"])
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected (%s)", self.engine.name)
        finally:
            # Always flush remaining audio so engine sees the tail.
            for ev in self.stream.flush(force=True):
                self._enqueue_segment(ev)
            await self._work_queue.put(_SENTINEL)

    async def _handle_text(self, text: str) -> bool:
        ctrl = self._parse_json(text)
        if ctrl is None:
            return False
        msg_type = ctrl.get("type", "")
        if msg_type == "start":
            await self._handle_start(ctrl)
            return False
        if msg_type == "stop":
            await self._handle_stop()
            return True
        if msg_type == "update_hotwords":
            self._handle_update_hotwords(ctrl)
            return False
        # Delegate unknown control messages to engine (returns truthy if handled).
        try:
            await self.engine.on_control(ctrl, self.ctx)
        except Exception:
            logger.exception("engine.on_control failed for %s", msg_type)
        return False

    def _parse_json(self, text: str) -> dict | None:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON from client: %.200s", text)
            return None

    async def _handle_start(self, ctrl: dict) -> None:
        if self._started:
            logger.warning("Duplicate start message, ignoring")
            return
        self._started = True

        client_config = ctrl.get("config")
        if isinstance(client_config, dict) and client_config:
            self.cfg = self.cfg.override(**client_config)
            self.ctx.cfg = self.cfg
            self.stream.configure(self.cfg)
            logger.info("Config overridden by client: %s", list(client_config.keys()))

        lang_val = str(ctrl.get("language", "")).strip()
        if lang_val:
            self.ctx.language = lang_val
            self.ctx.src_lang = map_language(lang_val)

        hw_raw = ctrl.get("hotwords")
        if isinstance(hw_raw, list):
            self.ctx.hotwords = sanitize_hotwords(hw_raw)
            logger.info("Hotwords from start: %d items", len(self.ctx.hotwords))

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
        logger.info(
            "Hotwords updated: %s (src_lang=%s)",
            self.ctx.hotwords, self.ctx.src_lang,
        )

    async def _handle_stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        logger.info("Stop received (%s), flushing", self.engine.name)
        for ev in self.stream.flush(force=True):
            self._enqueue_segment(ev)

    # ------------------------------------------------------------------
    # PCM dispatch
    # ------------------------------------------------------------------

    async def _handle_pcm(self, pcm_bytes: bytes) -> None:
        for ev in self.stream.feed(pcm_bytes):
            if isinstance(ev, SegmentReady):
                self._enqueue_segment(ev)
            elif isinstance(ev, PartialSnapshot):
                self._maybe_launch_partial(ev)

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
