"""Target-Speaker ASR task engine.

Paired with ``VadSegmentedStream`` (to match the primary ASR pipeline so the
demo feels identical from the user side), this engine:

* decodes and caches the enrollment (reference) audio during ``on_start``;
* submits ``build_tsasr_content(enrollment, mixed)`` to the TS-capable vLLM
  endpoint on every segment;
* skips the secondary-ASR dual / fusion path entirely (Qwen3-ASR-1.7B does
  not accept dual-audio inputs and would just re-decode the noisy mixture);
* disables the pseudo-streaming partial by default — dual-audio encoding
  doubles the per-call RTF and we don't have a silence gate without the
  secondary model.

Design intent: this file is a *thin orchestrator*. All prompt / network
details belong in :mod:`backend.tsasr` so the engine can remain stable as
the prompt template evolves.
"""

from __future__ import annotations

import logging
import time
import uuid

from ..audio.utils import pcm_to_wav_base64
from ..config import SAMPLE_RATE
from ..streaming.events import PartialSnapshot, SegmentReady
from ..streaming.session import SessionContext
from ..tsasr.client import query_tsasr_model
from ..tsasr.enrollment import EnrollmentAudio, EnrollmentError, decode_enrollment
from .base import BaseTaskEngine

logger = logging.getLogger(__name__)


def _resolve(cfg, tsasr_field: str, fallback_field: str) -> str:
    """Return ``cfg.tsasr_*`` when set, else fall back to ``cfg.vllm_*``."""
    value = getattr(cfg, tsasr_field, "") or ""
    if value:
        return value
    return getattr(cfg, fallback_field, "") or ""


class TsAsrTaskEngine(BaseTaskEngine):
    """Runs TS-ASR inference against per-segment mixed audio."""

    name = "tsasr"

    def __init__(self) -> None:
        self._enrollment: EnrollmentAudio | None = None
        self._voice_traits: str | None = None
        # Cache the resolved hotword-enable flag so ``on_start`` is the single
        # source of truth regardless of later hotword updates.
        self._hotwords_enabled: bool = False
        # Per-engine segment-ID source. The frontend keys its replayable WAV
        # cache by this id, so the value MUST be unique across every engine
        # instance the browser has ever talked to on this page — otherwise a
        # new session's ``tsasr-1`` would collide with a previous session's
        # cached blob and the replay button would play the wrong clip. We
        # mint a fresh 8-hex-char prefix on construction (one engine = one
        # WebSocket) and append a monotonic counter, giving readable ids
        # like ``tsasr-3f9a1c2e-1`` that stay ordered per session.
        self._segment_prefix: str = uuid.uuid4().hex[:8]
        self._segment_counter: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_start(self, ctrl: dict, ctx: SessionContext) -> None:
        cfg = ctx.cfg
        min_sec = float(getattr(cfg, "tsasr_enrollment_min_sec", 1.0))
        max_sec = float(getattr(cfg, "tsasr_enrollment_max_sec", 30.0))

        audio_b64 = (
            ctrl.get("enrollment_audio")
            or ctrl.get("enrollment_wav_base64")
            or ""
        )
        audio_fmt = ctrl.get("enrollment_format", "wav")

        try:
            self._enrollment = decode_enrollment(
                audio_b64,
                min_sec=min_sec,
                max_sec=max_sec,
                audio_format=audio_fmt,
            )
        except EnrollmentError as err:
            logger.warning("Enrollment rejected [%s]: %s", err.code, err)
            # Don't re-raise: the session wraps on_start in try/except and
            # would otherwise continue accepting PCM silently. Instead we
            # leave ``_enrollment`` as None so subsequent segments are
            # short-circuited and the client sees this error + no finals.
            await ctx.send_json(
                {
                    "type": "error",
                    "code": f"enrollment_{err.code}",
                    "message": str(err),
                }
            )
            return

        voice_traits = ctrl.get("voice_traits")
        if isinstance(voice_traits, str) and voice_traits.strip():
            self._voice_traits = voice_traits.strip()
        else:
            self._voice_traits = None

        self._hotwords_enabled = bool(
            getattr(cfg, "tsasr_enable_hotwords", False)
        )

        logger.info(
            "TS-ASR session ready: enrollment=%.2fs traits=%r hotwords=%s",
            self._enrollment.duration_sec,
            self._voice_traits,
            self._hotwords_enabled,
        )
        await ctx.send_json(
            {
                "type": "enrollment_ok",
                "duration_sec": round(self._enrollment.duration_sec, 3),
                "sample_rate_hz": SAMPLE_RATE,
            }
        )

    # ------------------------------------------------------------------
    # Per-segment inference
    # ------------------------------------------------------------------

    async def handle_segment(
        self, seg: SegmentReady, ctx: SessionContext
    ) -> bool:
        if self._enrollment is None:
            logger.warning("TS-ASR segment dropped: no enrollment cached")
            return False

        cfg = ctx.cfg
        segment = seg.pcm
        audio_duration = len(segment) / SAMPLE_RATE

        max_seconds = float(getattr(cfg, "tsasr_max_audio_seconds", 0.0))
        if max_seconds > 0 and audio_duration > max_seconds:
            max_samples = int(SAMPLE_RATE * max_seconds)
            logger.info(
                "Trimming TS-ASR segment %.1fs -> %.1fs (cap)",
                audio_duration, max_seconds,
            )
            segment = segment[-max_samples:]
            audio_duration = len(segment) / SAMPLE_RATE

        t0 = time.monotonic()
        mixed_b64 = pcm_to_wav_base64(segment)
        hotwords = list(ctx.hotwords) if self._hotwords_enabled else None

        base_url = _resolve(cfg, "tsasr_base_url", "vllm_base_url")
        model_name = _resolve(cfg, "tsasr_model_name", "vllm_model_name")
        timeout = float(
            getattr(cfg, "tsasr_request_timeout", 0)
            or getattr(cfg, "asr_request_timeout", 30.0)
        )

        try:
            result = await query_tsasr_model(
                mixed_b64,
                self._enrollment.wav_base64,
                hotwords=hotwords,
                voice_traits=self._voice_traits,
                base_url=base_url,
                model_name=model_name,
                timeout=timeout,
                enrollment_duration_sec=self._enrollment.duration_sec,
            )
        except Exception as exc:
            logger.exception("TS-ASR inference failed: %s", exc)
            raise

        text = str(result.get("transcription") or "").strip()
        detected_lang = result.get("detected_language") or ctx.language

        elapsed = time.monotonic() - t0
        rtf = elapsed / audio_duration if audio_duration > 0 else 0.0
        logger.info(
            "Final TS-ASR: audio=%.2fs infer=%.3fs RTF=%.3f text=%r",
            audio_duration, elapsed, rtf, text[:80],
        )

        if not text:
            return False

        self._segment_counter += 1
        seg_id = f"tsasr-{self._segment_prefix}-{self._segment_counter}"
        payload: dict = {
            "type": "final",
            "id": seg_id,
            "text": text,
            "language": detected_lang,
            "task": "tsasr",
            # The mixed audio that was actually fed to the model. The client
            # uses it to wire up a replay button next to the transcript so
            # users can sanity-check what the target-speaker extraction
            # heard. Size is bounded by tsasr_max_audio_seconds.
            "audio_b64": mixed_b64,
            "duration_sec": round(audio_duration, 2),
        }
        return await ctx.send_json(payload)

    # ------------------------------------------------------------------
    # Pseudo-streaming partial (disabled by default)
    # ------------------------------------------------------------------

    async def handle_partial(
        self, snap: PartialSnapshot, ctx: SessionContext
    ) -> None:
        cfg = ctx.cfg
        if not bool(getattr(cfg, "tsasr_enable_partial", False)):
            return
        if self._enrollment is None:
            return

        snapshot = snap.pcm
        audio_duration = len(snapshot) / SAMPLE_RATE
        t0 = time.monotonic()
        mixed_b64 = pcm_to_wav_base64(snapshot)
        hotwords = list(ctx.hotwords) if self._hotwords_enabled else None

        base_url = _resolve(cfg, "tsasr_base_url", "vllm_base_url")
        model_name = _resolve(cfg, "tsasr_model_name", "vllm_model_name")
        timeout = float(
            getattr(cfg, "tsasr_request_timeout", 0)
            or getattr(cfg, "asr_request_timeout", 30.0)
        )

        try:
            result = await query_tsasr_model(
                mixed_b64,
                self._enrollment.wav_base64,
                hotwords=hotwords,
                voice_traits=self._voice_traits,
                base_url=base_url,
                model_name=model_name,
                timeout=timeout,
            )
        except Exception as exc:
            logger.debug("TS-ASR partial skipped: %s", exc)
            return

        text = str(result.get("transcription") or "").strip()
        elapsed = time.monotonic() - t0
        rtf = elapsed / audio_duration if audio_duration > 0 else 0.0
        logger.info(
            "Partial TS-ASR: audio=%.2fs infer=%.3fs RTF=%.3f text=%r",
            audio_duration, elapsed, rtf, text[:80],
        )
        if not text:
            return

        await ctx.send_json(
            {
                "type": "partial",
                "text": text,
                "language": ctx.language,
                "task": "tsasr",
            }
        )

    # ------------------------------------------------------------------
    # Stop guarantee
    # ------------------------------------------------------------------

    async def on_stop(
        self,
        ctx: SessionContext,
        *,
        sent_any_response: bool,
        stopped: bool,
    ) -> None:
        if stopped and not sent_any_response:
            await ctx.send_json(
                {
                    "type": "final",
                    "text": "",
                    "language": ctx.language,
                    "task": "tsasr",
                }
            )
