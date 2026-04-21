"""Emotion task engine (SER / SEC), aligned with AmphionASR.

Designed to be paired with :class:`backend.streaming.WholeUtteranceStream`:
the stream accumulates the entire utterance and emits exactly one
``SegmentReady`` on stop / disconnect, which this engine turns into a single
``final_emotion`` message.

The task variant (``ser`` for classification, ``sec`` for free-form caption)
is selected per session via the ``mode`` field on the ``start`` control
message; if absent, ``Config.emotion_task_mode`` (default ``"ser"``) is used.
"""

from __future__ import annotations

import logging
import time

from ..audio.utils import pcm_to_wav_base64
from ..config import SAMPLE_RATE
from ..emotion.client import query_emotion_model
from ..emotion.prompt import DEFAULT_MODE, EmotionMode, normalize_mode
from ..streaming.events import SegmentReady
from ..streaming.session import SessionContext
from .base import BaseTaskEngine

logger = logging.getLogger(__name__)


class EmotionTaskEngine(BaseTaskEngine):
    """Run a single emotion inference per session, on the full utterance."""

    name = "emotion"

    def __init__(self) -> None:
        self._mode: EmotionMode = DEFAULT_MODE

    async def on_start(self, ctrl: dict, ctx: SessionContext) -> None:
        cfg_default = getattr(ctx.cfg, "emotion_task_mode", DEFAULT_MODE)
        chosen = ctrl.get("mode", cfg_default)
        self._mode = normalize_mode(chosen)
        logger.info("Emotion session mode=%s", self._mode)

    async def handle_segment(
        self, seg: SegmentReady, ctx: SessionContext
    ) -> bool:
        cfg = ctx.cfg
        segment = seg.pcm
        audio_duration = len(segment) / SAMPLE_RATE

        max_seconds = float(getattr(cfg, "emotion_max_audio_seconds", 0.0))
        if max_seconds > 0 and audio_duration > max_seconds:
            max_samples = int(SAMPLE_RATE * max_seconds)
            logger.info(
                "Trimming emotion audio %.1fs -> %.1fs (cap)",
                audio_duration, max_seconds,
            )
            segment = segment[-max_samples:]
            audio_duration = len(segment) / SAMPLE_RATE

        t0 = time.monotonic()
        wav_b64 = pcm_to_wav_base64(segment)

        result = await query_emotion_model(
            wav_b64,
            mode=self._mode,
            base_url=cfg.emotion_vllm_base_url,
            model_name=cfg.emotion_vllm_model_name,
            timeout=cfg.emotion_request_timeout,
        )

        elapsed = time.monotonic() - t0
        rtf = elapsed / audio_duration if audio_duration > 0 else 0.0
        logger.info(
            "Final emotion[%s]: audio=%.2fs infer=%.3fs RTF=%.3f label=%r",
            self._mode, audio_duration, elapsed, rtf, result.get("label"),
        )

        return await ctx.send_json(self._build_payload(result, audio_duration, ctx))

    async def on_stop(
        self,
        ctx: SessionContext,
        *,
        sent_any_response: bool,
        stopped: bool,
    ) -> None:
        if stopped and not sent_any_response:
            empty: dict = {
                "type": "final_emotion",
                "mode": self._mode,
                "label": "",
                "text": "",
                "duration_sec": 0.0,
            }
            if ctx.language:
                empty["language"] = ctx.language
            await ctx.send_json(empty)

    def _build_payload(
        self, result: dict, audio_duration: float, ctx: SessionContext
    ) -> dict:
        payload: dict = {
            "type": "final_emotion",
            "mode": self._mode,
            "label": result.get("label", ""),
            "text": result.get("text", ""),
            "duration_sec": round(audio_duration, 3),
        }
        raw_text = result.get("raw_text", "")
        if raw_text and raw_text != payload["text"]:
            payload["raw_text"] = raw_text
        if ctx.language:
            payload["language"] = ctx.language
        return payload
