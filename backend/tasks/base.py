"""TaskEngine protocol and reusable defaults.

A TaskEngine encapsulates everything specific to one audio task (ASR,
emotion recognition, ...). It is intentionally agnostic to the WebSocket
protocol; the :class:`StreamingSession` calls the engine's hooks at well
defined points in the session lifecycle and provides a snapshot of the
session context (config, language, hotwords, send_json) on every call.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..streaming.events import PartialSnapshot, SegmentReady
from ..streaming.session import SessionContext


@runtime_checkable
class TaskEngine(Protocol):
    """Strategy interface for one audio task."""

    name: str

    async def on_start(self, ctrl: dict, ctx: SessionContext) -> None:
        """Hook for engine-specific parsing of the ``start`` control message.

        Common fields (``config``, ``language``, ``hotwords``) have already
        been applied to ``ctx`` and the AudioStream by the time this is
        called.
        """

    async def on_control(self, ctrl: dict, ctx: SessionContext) -> None:
        """Handle control messages other than start/stop/update_hotwords."""

    async def handle_segment(self, seg: SegmentReady, ctx: SessionContext) -> bool:
        """Run inference on a final segment and emit response message(s).

        Returns True iff at least one response message was sent for this
        segment; the session uses this to decide whether to send a fallback
        empty result on stop (see ``on_stop``).
        """

    async def handle_partial(self, snap: PartialSnapshot, ctx: SessionContext) -> None:
        """Optional: emit incremental ("partial") result mid-utterance."""

    async def on_stop(
        self,
        ctx: SessionContext,
        *,
        sent_any_response: bool,
        stopped: bool,
    ) -> None:
        """Called once after all queued segments are processed.

        ``stopped`` is True when the client sent a ``stop`` (vs. socket close).
        ``sent_any_response`` reflects whether any segment produced output.
        Engines may use this to send a guaranteed terminating message.
        """


class BaseTaskEngine:
    """Reasonable default that engines can subclass to skip unused hooks."""

    name: str = "base"

    async def on_start(self, ctrl: dict, ctx: SessionContext) -> None:
        return None

    async def on_control(self, ctrl: dict, ctx: SessionContext) -> None:
        return None

    async def handle_partial(self, snap: PartialSnapshot, ctx: SessionContext) -> None:
        return None

    async def on_stop(
        self,
        ctx: SessionContext,
        *,
        sent_any_response: bool,
        stopped: bool,
    ) -> None:
        return None
