"""AscendK2Stream: event-driven streaming ASR for Ascend 910B3/B4.

Architecture (aligned with NVIDIA remote K2 design):
- TEN VAD: voice gate only — detects speech onset, opens gate.
  Segment boundaries are NOT determined by VAD silence detection.
- CTC OM: event-driven partials via set_result_callback → SimpleQueue.
  Acoustic endpoint detection: 1.2 s of no new CTC tokens → SegmentReady.
- Bulk drain on flush(force=True): no feed_queue timeout wait needed.

Replaces VadSegmentedStream when ``ascend_k2_enabled=True``.
"""

from __future__ import annotations

import logging
import queue
import time
from typing import Iterable

import numpy as np

from ..asr.ctc_streaming import CtcStreamingConfig, CtcStreamingSession
from ..audio.vad import VADProcessor
from ..config import SAMPLE_RATE, Config
from .audio_stream import StreamEvent, _pcm_bytes_to_float32
from .events import PartialSnapshot, SegmentReady, SpeechDropped, SpeechStarted

logger = logging.getLogger(__name__)


class AscendK2Stream:
    """AudioStream using CTC OM callbacks for event-driven partials.

    TEN VAD acts solely as a voice gate (speech onset detection).
    Segment boundaries are determined by CTC acoustic silence (no new tokens
    for ``k2_endpoint_silence_sec``), not by VAD silence detection.
    """

    def __init__(self, ctc_cfg: CtcStreamingConfig) -> None:
        self._ctc_cfg = ctc_cfg
        self._vad = VADProcessor()
        self._pcm_carry: np.ndarray = np.empty(0, dtype=np.float32)
        self._consumed_samples: int = 0

        # Voice gate state
        self._gate_open: bool = False
        self._announced_speech: bool = False
        self._speech_started_at: float = 0.0
        self._gate_opened_at: float = 0.0
        self._speech_buf: list[np.ndarray] = []

        # Rolling pre-speech buffer: keeps last pre_speech_ms of audio so we
        # can prepend it to speech_buf when the gate opens, capturing the
        # portion of speech that occurs before TEN VAD fires.
        # Size is set during configure(); default 500ms at 16kHz = 8000 samples.
        self._pre_speech_max_samples: int = 8000
        self._pre_speech_ring: list[np.ndarray] = []
        self._pre_speech_total: int = 0

        # CTC session (created per utterance on gate open)
        self._ctc_session: CtcStreamingSession | None = None

        # Pre-fill CTC session: created immediately on first audio, fed
        # incrementally with all incoming audio (including pre-gate silence).
        # Adopted as _ctc_session on gate-open so the CTC is already
        # past chunk_length=45 frames → first partial arrives ~85 ms after
        # gate-open instead of ~535 ms.
        self._ctc_prefill_session: CtcStreamingSession | None = None
        # Flag: True when the current _ctc_session was adopted from prefill
        # and must be fed via accept_incremental (not accept_cumulative).
        self._ctc_incremental_mode: bool = False
        # Prefill warmup cap (samples). 0 = legacy unbounded feeding. When >0,
        # the gate-closed prefill session is fed only until it holds this many
        # samples of audio, then parked (not fed -> not decode-ready -> dropped
        # from the CTC batch) to stop wasting decode capacity on discarded
        # silence results (design F7). Pre-roll continuity is restored by
        # feeding the pre-speech ring to the adopted session on gate-open.
        self._prefill_warmup_samples: int = 0

        # Thread-safe partial text queue (CTC callback → asyncio poll)
        self._partial_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._last_ctc_text: str = ""

        # CTC acoustic endpoint tracking
        self._last_token_at: float = 0.0

        # VAD-silence endpoint tracking (design F9-A). Audio-timeline position
        # (absolute sample count) of the end of the most recent speech frame
        # while the gate is open. Endpoint fires when the gap between the
        # current consumed position and this reaches the threshold, so segment
        # close references the *true* speech end and is decoupled from the CTC
        # decode backlog (which lags under BS52 and inflates vad_lag).
        self._last_speech_sample: int = 0
        self._vad_endpoint_enabled: bool = False
        self._vad_endpoint_silence_sec: float = 0.0

        # Per-session tunables (set via configure())
        # 2.0s default: Chinese tokens typically appear within 0.6s after gate
        # opens; English tokens may take up to 1.6s before the bilingual model
        # emits a first non-blank token. 1.2s caused premature endpoint on EN.
        self._endpoint_silence_sec: float = 1.5
        self._max_segment_sec: float = 30.0
        self._min_segment_ms: int = 200
        self._cfg: Config | None = None

    # ------------------------------------------------------------------
    # AudioStream protocol
    # ------------------------------------------------------------------

    def configure(self, cfg: Config) -> None:
        """Apply per-session Config knobs. Must be called before feed()."""
        self._cfg = cfg
        self._vad.apply_config(cfg)
        self._endpoint_silence_sec = float(
            getattr(cfg, "k2_endpoint_silence_sec", 1.5)
        )
        self._max_segment_sec = float(getattr(cfg, "k2_max_segment_sec", 30.0))
        self._min_segment_ms = int(getattr(cfg, "min_segment_duration_ms", 200))
        pre_speech_ms = int(getattr(cfg, "vad_pre_speech_ms", 500))
        self._pre_speech_max_samples = max(0, pre_speech_ms * SAMPLE_RATE // 1000)
        warmup_ms = int(getattr(cfg, "ctc_prefill_warmup_ms", 0) or 0)
        self._prefill_warmup_samples = max(0, warmup_ms * SAMPLE_RATE // 1000)
        self._vad_endpoint_enabled = bool(
            getattr(cfg, "k2_vad_endpoint_enabled", False)
        )
        vad_ep = float(getattr(cfg, "k2_vad_endpoint_silence_sec", 0.0) or 0.0)
        # 0 = reuse the CTC endpoint silence threshold.
        self._vad_endpoint_silence_sec = (
            vad_ep if vad_ep > 0 else self._endpoint_silence_sec
        )

    def feed(self, pcm_bytes: bytes) -> list[StreamEvent]:
        """Push int16 LE PCM bytes; return zero or more stream events."""
        events: list[StreamEvent] = []
        pcm = _pcm_bytes_to_float32(pcm_bytes)

        if self._pcm_carry.size > 0:
            pcm = np.concatenate([self._pcm_carry, pcm])

        hop = self._vad.hop_size
        used = (len(pcm) // hop) * hop
        self._pcm_carry = (
            pcm[used:].copy() if used < len(pcm) else np.empty(0, dtype=np.float32)
        )

        if used == 0:
            return events

        aligned = pcm[:used]

        # Lazily create the pre-fill CTC session on the very first audio chunk.
        # This warms up the CTC pipeline so it has accumulated chunk_length
        # frames by the time VAD fires, reducing TTFT by ~450 ms.
        if self._ctc_prefill_session is None and not self._gate_open:
            self._ctc_prefill_session = CtcStreamingSession(self._ctc_cfg)
            self._ctc_prefill_session.set_result_callback(self._on_ctc_result)

        # Run VAD for voice gate onset detection only.
        # VAD-emitted segments (silence-based) are intentionally ignored, so
        # request the detection-only path: it skips the per-hop frame copies /
        # audio_buffer maintenance / segment concatenation that would otherwise
        # run on the single-process GIL for every 10 ms hop across all 52
        # streams (design F12 — a share of the server real-time deficit).
        vad_results = self._vad.process_chunk(aligned, detection_only=True)

        new_speech_frames: list[np.ndarray] = []

        for idx, (_, was_speaking, now_speaking) in enumerate(vad_results):
            frame = aligned[idx * hop : (idx + 1) * hop]

            # Silent → speaking: open voice gate (once per utterance).
            # Prepend the rolling pre-speech buffer so early speech content
            # arriving before TEN VAD fires is not lost.
            if not was_speaking and now_speaking and not self._gate_open:
                now = time.monotonic()
                events.append(self._open_gate(now))

            # Maintain rolling pre-speech ring when gate is closed.
            if not self._gate_open and self._pre_speech_max_samples > 0:
                self._pre_speech_ring.append(frame.copy())
                self._pre_speech_total += len(frame)
                # Trim to keep only the last pre_speech_max_samples
                while (
                    self._pre_speech_total > self._pre_speech_max_samples
                    and self._pre_speech_ring
                ):
                    oldest = self._pre_speech_ring.pop(0)
                    self._pre_speech_total -= len(oldest)

            # Accumulate raw PCM when gate is open.
            if self._gate_open:
                # Append the view (not a copy): `aligned` is a fresh per-call
                # array that is never mutated after this, and both buffers are
                # only ever np.concatenate'd. new_speech_frames already relies
                # on this. Saves one small allocation per 10 ms hop × 52 streams
                # on the GIL-bound feed path (design F12).
                self._speech_buf.append(frame)
                new_speech_frames.append(frame)
                # Track the true speech end on the audio timeline for the
                # VAD-silence endpoint (F9-A). self._consumed_samples is still
                # the pre-batch value here (updated after this loop), so add the
                # in-batch frame offset.
                if now_speaking:
                    self._last_speech_sample = (
                        self._consumed_samples + (idx + 1) * hop
                    )

        self._consumed_samples += used

        # --- CTC feeding (done AFTER the VAD loop to avoid lock contention) ---

        if not self._gate_open and self._ctc_prefill_session is not None and aligned.size > 0:
            # Batch-feed all frames from this call to the pre-fill session
            # in a single accept_incremental call.  Batching avoids per-frame
            # lock contention with the CTC background thread, which would
            # otherwise delay VAD gate detection.
            #
            # Warmup cap (design F7): once the prefill session holds enough
            # audio to warm the encoder state, stop feeding it so the CTC decode
            # loop no longer decodes inter-segment silence (whose results are
            # discarded because the gate is closed). The parked session stays
            # registered but falls out of the decode-ready set, shrinking the
            # per-tick state I/O batch. 0 = legacy unbounded feeding.
            warm_cap = self._prefill_warmup_samples
            if warm_cap <= 0 or (
                int(getattr(self._ctc_prefill_session, "_fed_samples", 0)) < warm_cap
            ):
                self._ctc_prefill_session.accept_incremental(aligned[:used])

        if self._gate_open and new_speech_frames:
            # Feed only the new frames from this call.
            # When the session was adopted from prefill, use incremental to
            # avoid position mismatch with _fed_samples.  Otherwise use the
            # original cumulative path.
            if self._ctc_incremental_mode:
                combined = np.concatenate(new_speech_frames)
                logger.debug(
                    "K2_PREFILL_FEED incremental_samples=%d session_id=%s",
                    len(combined),
                    id(self._ctc_session),
                )
                self._ctc_session.accept_incremental(combined)
            else:
                self._feed_ctc_nonblocking()

        # Endpoint check runs on EVERY feed() while the gate is open, not only
        # on batches that carry new speech (design F12). Previously the check
        # was nested under `new_speech_frames`, so during the inter-utterance
        # silence gap no endpoint could fire — a segment stayed open until the
        # NEXT utterance's speech arrived, which both delayed segment close by
        # up to the whole silence gap and coupled the close time to next-speech
        # onset detection (which inherits the feed real-time deficit → inflates
        # and accumulates vad_lag). _check_ctc_endpoint is self-guarding: it
        # returns False until a token exists and never fires during active
        # speech (tokens keep _last_token_at fresh), so this only makes genuine
        # endpoints fire promptly at the true 1.5 s silence mark.
        if self._gate_open:
            now = time.monotonic()
            if self._check_max_duration(now):
                logger.warning(
                    "K2_VOICE_GATE forced cut at max_segment_sec=%.1f",
                    self._max_segment_sec,
                )
                events.append(self._close_gate(reason="max_duration"))
            elif self._vad_endpoint_enabled and self._check_vad_endpoint():
                # VAD silence references the true speech end (audio timeline),
                # so it fires ~CTC-backlog seconds earlier than the CTC endpoint
                # under load (F9-A). Hybrid: CTC endpoint below stays as backstop.
                events.append(self._close_gate(reason="vad_endpoint"))
            elif self._check_ctc_endpoint(now):
                events.append(self._close_gate(reason="ctc_endpoint"))

        return events

    def flush(self, *, force: bool) -> list[StreamEvent]:
        """Drain any remaining buffered speech (called on stop / disconnect)."""
        # Always clean up pre-fill session on stream end.
        if self._ctc_prefill_session is not None:
            self._ctc_prefill_session.close()
            self._ctc_prefill_session = None
        if not self._gate_open:
            if self._announced_speech:
                self._reset_gate()
                return [SpeechDropped()]
            return []
        return [self._close_gate(is_stop_flush=force, reason="flush")]

    # ------------------------------------------------------------------
    # Ascend-specific extension used by session.py
    # ------------------------------------------------------------------

    def poll_partial_events(self) -> list[str]:
        """Non-blocking drain of CTC partial text queue.

        Returns new partial texts since the last call. Called from
        session.py._feed_loop() after each stream.feed() to dispatch
        partials to the client without blocking on CTC timing.
        """
        results: list[str] = []
        while True:
            try:
                results.append(self._partial_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def bulk_flush(self, extra_pcm: np.ndarray | None) -> list[StreamEvent]:
        """Accept remaining drain-queue PCM and emit SegmentReady directly.

        Called by session.py on the status=2 bulk drain path instead of
        waiting for feed_queue timeout. The extra_pcm (all remaining queued
        PCM frames merged by the caller) is appended to the speech buffer
        before flushing.
        """
        if extra_pcm is not None and extra_pcm.size > 0:
            if self._gate_open:
                self._speech_buf.append(
                    extra_pcm.astype(np.float32, copy=False)
                )
            else:
                # Gate not open yet — treat as a brand-new gate-open so the
                # accumulated PCM reaches the final ASR path.
                now = time.monotonic()
                self._open_gate(now)
                self._speech_buf.append(
                    extra_pcm.astype(np.float32, copy=False)
                )
        return self.flush(force=True)

    def prefeed_partial_snapshot(self) -> PartialSnapshot | None:
        """Not used in ascend_k2 mode; present for interface compatibility."""
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_ctc_result(
        self, text: str, stats: dict[str, float | int | str]
    ) -> None:
        """CTC background-thread callback. Must be lightweight and thread-safe."""
        if not self._gate_open:
            return
        if text and text != self._last_ctc_text:
            self._last_ctc_text = text
            self._last_token_at = time.monotonic()
            self._partial_queue.put(text)

    def _open_gate(self, now: float) -> SpeechStarted:
        # Close any stale session from a prior utterance
        if self._ctc_session is not None:
            self._ctc_session.close()
            self._ctc_session = None
        # Drain stale partials from any prior utterance
        self._drain_partial_queue()

        self._gate_open = True
        self._announced_speech = True
        self._speech_started_at = now
        self._gate_opened_at = now
        # Grace period: treat gate-open time as latest token to avoid
        # immediate endpoint trigger before any audio arrives.
        self._last_token_at = now
        # Seed the VAD-endpoint tracker at the gate-open audio position; the
        # onset frame(s) update it as speech continues (F9-A).
        self._last_speech_sample = self._consumed_samples
        self._last_ctc_text = ""
        # Prepend pre-speech buffer (captures early speech before VAD fires)
        self._speech_buf.clear()
        if self._pre_speech_ring:
            self._speech_buf.extend(self._pre_speech_ring)
            pre_ms = self._pre_speech_total * 1000.0 / SAMPLE_RATE
            logger.debug("K2_VOICE_GATE prepend_pre_speech_ms=%.1f", pre_ms)
        self._pre_speech_ring.clear()
        self._pre_speech_total = 0

        if self._ctc_prefill_session is not None:
            # Adopt the pre-warmed session.  It has already processed
            # chunk_length+ frames so first CTC output fires ~85 ms from now.
            # Switch to incremental feed mode so post-gate audio is appended
            # correctly regardless of how many samples the pre-fill saw.
            self._ctc_session = self._ctc_prefill_session
            self._ctc_prefill_session = None
            self._ctc_incremental_mode = True
            # If the prefill was warmup-capped and parked (F7), it stopped
            # ingesting audio before onset, so it never saw the immediate
            # pre-speech pre-roll (captured in _pre_speech_ring, now in
            # speech_buf). Feed that pre-roll now so early phonemes are not
            # dropped. Only when parked, to avoid double-feeding audio the
            # (still-feeding) prefill already saw on short silence gaps.
            warm_cap = self._prefill_warmup_samples
            parked = warm_cap > 0 and int(
                getattr(self._ctc_session, "_fed_samples", 0)
            ) >= warm_cap
            if parked and self._speech_buf:
                preroll = np.concatenate(self._speech_buf)
                if preroll.size > 0:
                    self._ctc_session.accept_incremental(preroll)
            logger.info(
                "K2_VOICE_GATE adopted prefill CTC session id=%s fed_samples=%s",
                id(self._ctc_session),
                getattr(self._ctc_session, "_fed_samples", "?"),
            )
        else:
            # Fallback: no pre-fill available (first ever call or after error).
            self._ctc_session = CtcStreamingSession(self._ctc_cfg)
            self._ctc_session.set_result_callback(self._on_ctc_result)
            self._ctc_incremental_mode = False

        logger.info(
            "K2_VOICE_GATE event=open timeline_ms=%.1f",
            self._consumed_samples * 1000.0 / SAMPLE_RATE,
        )
        return SpeechStarted(started_at=now)

    def _close_gate(
        self, *, is_stop_flush: bool = False, reason: str = "ctc_endpoint"
    ) -> SegmentReady | SpeechDropped:
        announced = self._announced_speech
        speech_buf = self._speech_buf[:]
        self._reset_gate()

        if not speech_buf:
            logger.info("K2_VOICE_GATE event=close reason=%s no_audio", reason)
            return SpeechDropped()

        pcm = np.concatenate(speech_buf)
        min_samples = int(SAMPLE_RATE * self._min_segment_ms / 1000)
        dur_ms = len(pcm) * 1000.0 / SAMPLE_RATE

        if not is_stop_flush and len(pcm) < min_samples:
            logger.info(
                "K2_VOICE_GATE event=close reason=%s drop_short audio_ms=%.1f",
                reason,
                dur_ms,
            )
            return SpeechDropped()

        logger.info(
            "K2_VOICE_GATE event=close reason=%s audio_ms=%.1f announced=%s",
            reason,
            dur_ms,
            announced,
        )
        end_ms = self._consumed_samples * 1000.0 / SAMPLE_RATE
        start_ms = max(0.0, end_ms - dur_ms)
        return SegmentReady(
            pcm=pcm,
            is_stop_flush=is_stop_flush,
            start_ms=start_ms,
            end_ms=end_ms,
        )

    def _reset_gate(self) -> None:
        self._gate_open = False
        self._announced_speech = False
        self._speech_started_at = 0.0
        self._gate_opened_at = 0.0
        self._last_token_at = 0.0
        self._last_speech_sample = 0
        self._last_ctc_text = ""
        self._ctc_incremental_mode = False
        self._speech_buf.clear()
        self._pre_speech_ring.clear()
        self._pre_speech_total = 0
        self._drain_partial_queue()
        if self._ctc_session is not None:
            self._ctc_session.close()
            self._ctc_session = None
        # Start a fresh pre-fill session immediately so the next utterance
        # arrives to a warm CTC pipeline (inter-utterance silence pre-fills it).
        if self._ctc_prefill_session is not None:
            self._ctc_prefill_session.close()
        self._ctc_prefill_session = CtcStreamingSession(self._ctc_cfg)
        self._ctc_prefill_session.set_result_callback(self._on_ctc_result)

    def _drain_partial_queue(self) -> None:
        while True:
            try:
                self._partial_queue.get_nowait()
            except queue.Empty:
                break

    def _feed_ctc_nonblocking(self) -> None:
        if self._ctc_session is None or not self._speech_buf:
            return
        pcm = np.concatenate(self._speech_buf)
        # max_decode_steps=-1 → wait_for_result=False (non-blocking)
        self._ctc_session.accept_cumulative(pcm, max_decode_steps=-1)

    def _check_ctc_endpoint(self, now: float) -> bool:
        """True when CTC output has been silent for endpoint_silence_sec."""
        if self._last_token_at <= 0:
            return False
        # Avoid triggering during the grace period right after gate opens
        if (now - self._gate_opened_at) < self._endpoint_silence_sec:
            return False
        return (now - self._last_token_at) >= self._endpoint_silence_sec

    def _check_vad_endpoint(self) -> bool:
        """True when VAD has seen silence for vad_endpoint_silence_sec on the
        audio timeline since the last speech frame (design F9-A).

        Unlike the CTC endpoint, this references the audio position of the true
        speech end rather than the (backlog-lagged) last CTC token, so it does
        not inflate/accumulate vad_lag under BS52 decode pressure.
        """
        if self._last_speech_sample <= 0:
            return False
        silence_samples = self._consumed_samples - self._last_speech_sample
        if silence_samples <= 0:
            return False
        return (
            silence_samples / SAMPLE_RATE
        ) >= self._vad_endpoint_silence_sec

    def _check_max_duration(self, now: float) -> bool:
        if not self._gate_open or self._gate_opened_at <= 0:
            return False
        return (now - self._gate_opened_at) >= self._max_segment_sec
