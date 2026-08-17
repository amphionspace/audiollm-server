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

    async def handle_speech_start(self, ctx: SessionContext) -> None:
        """Optional: react to VAD's silent->speaking transition.

        Fires once per utterance, ahead of ``handle_partial`` /
        ``handle_segment``. Useful for engines that want to paint a
        placeholder UI ("识别中…") the moment the user opens their
        mouth instead of waiting for the segment to finalize.
        """

    async def handle_speech_dropped(self, ctx: SessionContext) -> None:
        """Optional: react when an announced utterance is abandoned.

        Pairs with ``handle_speech_start``: fires when the in-flight
        utterance never produces a usable segment (too short, force
        flush mid-speech, etc.). Engines that announced a placeholder
        on speech-start should retract it here.
        """

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

    def _use_sherpa_partial(self, cfg) -> bool:
        """Whether partial decoding runs on the native sherpa/K2/CTC path.

        The session calls this generically on ``self.engine`` to pick the
        partial-refresh strategy. Non-ASR engines default to the vLLM path.
        """
        return False

    async def on_start(self, ctrl: dict, ctx: SessionContext) -> None:
        return None

    async def on_control(self, ctrl: dict, ctx: SessionContext) -> None:
        return None

    async def handle_partial(self, snap: PartialSnapshot, ctx: SessionContext) -> None:
        return None

    async def handle_speech_start(self, ctx: SessionContext) -> None:
        return None

    async def handle_speech_dropped(self, ctx: SessionContext) -> None:
        return None

    async def on_stop(
        self,
        ctx: SessionContext,
        *,
        sent_any_response: bool,
        stopped: bool,
    ) -> None:
        return None
