"""Enrollment (speaker reference) decoding and validation for TS-ASR.

Keeps the base64 → PCM plumbing and the duration-range guard out of the task
engine so that the engine stays a thin orchestrator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..audio.utils import pcm_to_wav_base64, wav_base64_to_pcm_16k_mono
from ..audio.vad import vad_trim_audio
from ..config import SAMPLE_RATE

logger = logging.getLogger(__name__)


class EnrollmentError(ValueError):
    """Raised when enrollment audio is missing or fails validation.

    The ``code`` attribute is a short machine-readable tag ("missing" /
    "empty" / "too_short" / "decode_failed" / "unsupported_format")
    suitable for forwarding to the client via structured error messages.
    Over-long clips are handled silently by VAD-trimming rather than
    raising; see :func:`decode_enrollment`.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class EnrollmentAudio:
    """Decoded + validated enrollment reference."""

    pcm: np.ndarray            # float32 mono @ 16 kHz
    duration_sec: float
    wav_base64: str            # re-encoded canonical WAV base64

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE


def decode_enrollment(
    audio_b64: str | None,
    *,
    min_sec: float,
    max_sec: float,
    audio_format: str = "wav",
) -> EnrollmentAudio:
    """Decode the base64 WAV and validate its duration.

    Only ``"wav"`` is supported for ``audio_format`` in the v1 protocol; other
    formats raise :class:`EnrollmentError` with code ``"unsupported_format"``
    so the caller can surface a clear error to the client.
    """
    if not audio_b64:
        raise EnrollmentError(
            "missing", "start.enrollment_audio is required for TS-ASR"
        )

    fmt = (audio_format or "wav").strip().lower()
    if fmt not in {"wav"}:
        raise EnrollmentError(
            "unsupported_format",
            f"Unsupported enrollment_format {fmt!r}; only 'wav' is accepted",
        )

    try:
        pcm = wav_base64_to_pcm_16k_mono(audio_b64)
    except ValueError as exc:
        raise EnrollmentError("decode_failed", str(exc)) from exc

    if pcm.size == 0:
        raise EnrollmentError("empty", "enrollment audio decoded to empty PCM")

    duration_sec = pcm.size / float(SAMPLE_RATE)
    if duration_sec < min_sec:
        raise EnrollmentError(
            "too_short",
            f"enrollment audio is {duration_sec:.2f}s, "
            f"must be >= {min_sec:.2f}s",
        )
    if duration_sec > max_sec:
        # Users often paste in longer takes (e.g. a 20s clean read) even
        # though TS-ASR only needs a few seconds of the target voice. Rather
        # than rejecting the upload, run a VAD pass and keep the first
        # ``max_sec`` of voiced audio. This matches the browser UI, which
        # auto-stops recording at the same cap.
        trimmed = vad_trim_audio(pcm, target_sec=max_sec)
        trimmed_duration = trimmed.size / float(SAMPLE_RATE)
        logger.info(
            "VAD-trimmed enrollment %.2fs -> %.2fs (cap=%.2fs)",
            duration_sec, trimmed_duration, max_sec,
        )
        pcm = trimmed
        duration_sec = trimmed_duration
        if duration_sec < min_sec:
            # VAD couldn't find enough voiced material inside the long clip
            # (e.g. mostly silence with a whisper at the end). Surface the
            # same ``too_short`` code so the client can re-prompt the user.
            raise EnrollmentError(
                "too_short",
                f"VAD trim produced {duration_sec:.2f}s of voiced audio, "
                f"must be >= {min_sec:.2f}s",
            )

    # Re-encode to a canonical 16 kHz mono WAV so every downstream inference
    # call uses the same bytes regardless of the client's upload format.
    canonical_b64 = pcm_to_wav_base64(pcm, sample_rate=SAMPLE_RATE)

    return EnrollmentAudio(
        pcm=pcm, duration_sec=duration_sec, wav_base64=canonical_b64
    )
