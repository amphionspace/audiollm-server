"""In-process enrollment cache for target-speaker ASR.

The frontend uploads a 5–10 second enrollment clip (file or mic-recorded)
once via ``POST /api/asr/enrollment`` and gets back an opaque
``enrollment_id``. The realtime WS sessions and the REST upload endpoint
then dereference that id to fetch the base64 WAV that needs to be
prepended into the dual-audio prompt for every primary-ASR call. Storing
the WAV server-side (instead of having the client retransmit it on every
VAD segment) keeps WS messages cheap and means the backend can validate
the clip exactly once at upload time.

Design notes (first principles):

* **Scope** — single-process in-memory dict. The audiollm demo runs as a
  single ASGI worker (see ``start.sh`` / systemd unit); we explicitly do
  not want a Redis dependency here. If we ever scale horizontally,
  swap the ``_Store`` implementation for a shared cache without
  changing call sites.
* **Lifetime** — last-used timestamp, evicted via TTL. Entries are
  *not* deleted on WS disconnect because users can navigate between
  pages within a single session and we want the enrollment to survive
  reconnects. ``asr_enrollment_ttl_sec`` (default 1h) is generous.
* **Capacity** — bounded by ``asr_enrollment_max_entries`` (default
  256). When full we evict the LRU entry. The cap is a memory safety
  rail (each entry is ~320 KB for 10s @ 16 kHz / 16-bit), not a
  business rule.
* **Duration validation** — done at upload time, so by the time a WS
  session resolves the id we already know the clip is in [min, max] s.
  Clips longer than ``max`` are tail-trimmed to ``max`` (not rejected)
  to match the existing emotion / ASR upload convention.
* **Format** — stored as a base64-encoded 16 kHz mono WAV string, which
  is exactly the shape vLLM's ``input_audio`` field consumes; this
  saves a round-trip through ``base64.b64encode`` on every primary
  inference call.
"""

from __future__ import annotations

import base64
import secrets
import threading
import time
from dataclasses import dataclass

import numpy as np

from ..audio.utils import (
    mp3_bytes_to_pcm_16k_mono,
    pcm_s16le_bytes_to_pcm_16k_mono,
    pcm_to_wav_base64,
    wav_base64_to_pcm_16k_mono,
)
from ..config import SAMPLE_RATE


class EnrollmentError(Exception):
    """Structured error: the upload was rejected at validation time."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class EnrollmentEntry:
    enrollment_id: str
    wav_base64: str
    duration_sec: float
    created_at: float
    last_used_at: float


def _now() -> float:
    return time.monotonic()


def decode_and_validate(
    wav_base64: str,
    *,
    min_sec: float,
    max_sec: float,
) -> tuple[str, float]:
    """Decode a base64-encoded WAV upload to canonical 16 kHz mono.

    Returns the canonicalised ``(wav_base64, duration_sec)``. Raises
    :class:`EnrollmentError` with a stable ``code`` on invalid input so
    the HTTP layer can map it to a structured ``detail.code`` field.
    """
    if not isinstance(wav_base64, str) or not wav_base64.strip():
        raise EnrollmentError("empty", "enrollment audio is empty")
    try:
        pcm = wav_base64_to_pcm_16k_mono(wav_base64)
    except ValueError as exc:
        raise EnrollmentError("decode_failed", str(exc)) from exc
    if pcm.size == 0:
        raise EnrollmentError("empty", "enrollment audio decoded to empty PCM")
    duration = pcm.size / SAMPLE_RATE
    if duration < float(min_sec):
        raise EnrollmentError(
            "too_short",
            f"enrollment audio is {duration:.2f}s, need at least {min_sec:.2f}s",
        )
    if duration > float(max_sec):
        keep = int(SAMPLE_RATE * float(max_sec))
        # Match the upload convention used elsewhere (tail-trim): the most
        # informative speech tends to sit late in the clip after the user
        # cleared their throat / leading silence.
        pcm = pcm[-keep:]
        duration = pcm.size / SAMPLE_RATE
    canonical_b64 = pcm_to_wav_base64(pcm.astype(np.float32, copy=False))
    return canonical_b64, duration


def decode_and_validate_audio_bytes(
    audio_bytes: bytes,
    *,
    audio_format: str,
    min_sec: float,
    max_sec: float,
) -> tuple[str, float]:
    """Decode uploaded enrollment audio and return canonical 16 kHz mono WAV.

    Supported input formats are WAV, MP3, and raw 16 kHz mono s16le PCM. The
    stored representation stays unchanged: base64 WAV for local cache and
    float32 PCM for Triton enrollment upsert.
    """
    fmt = (audio_format or "").strip().lower()
    if fmt in {"wav", "wave"}:
        wav_base64 = base64.b64encode(audio_bytes).decode("ascii")
        return decode_and_validate(wav_base64, min_sec=min_sec, max_sec=max_sec)
    try:
        if fmt == "mp3":
            pcm = mp3_bytes_to_pcm_16k_mono(audio_bytes)
        elif fmt in {"pcm", "raw", "s16le"}:
            pcm = pcm_s16le_bytes_to_pcm_16k_mono(audio_bytes)
        else:
            raise EnrollmentError(
                "unsupported_format",
                "unsupported enrollment audio format; supported: wav, mp3, pcm",
            )
    except ValueError as exc:
        raise EnrollmentError("decode_failed", str(exc)) from exc

    if pcm.size == 0:
        raise EnrollmentError("empty", "enrollment audio decoded to empty PCM")
    duration = pcm.size / SAMPLE_RATE
    if duration < float(min_sec):
        raise EnrollmentError(
            "too_short",
            f"enrollment audio is {duration:.2f}s, need at least {min_sec:.2f}s",
        )
    if duration > float(max_sec):
        keep = int(SAMPLE_RATE * float(max_sec))
        pcm = pcm[-keep:]
        duration = pcm.size / SAMPLE_RATE
    canonical_b64 = pcm_to_wav_base64(pcm.astype(np.float32, copy=False))
    return canonical_b64, duration


class _Store:
    """LRU-ish in-memory enrollment cache with TTL eviction."""

    def __init__(
        self,
        *,
        ttl_sec: float,
        max_entries: int,
    ) -> None:
        self._ttl = float(ttl_sec)
        self._max_entries = int(max_entries)
        self._entries: dict[str, EnrollmentEntry] = {}
        self._lock = threading.Lock()

    def configure(self, *, ttl_sec: float, max_entries: int) -> None:
        with self._lock:
            self._ttl = float(ttl_sec)
            self._max_entries = int(max_entries)

    def put(self, wav_base64: str, duration_sec: float) -> EnrollmentEntry:
        now = _now()
        enrollment_id = secrets.token_urlsafe(16)
        entry = EnrollmentEntry(
            enrollment_id=enrollment_id,
            wav_base64=wav_base64,
            duration_sec=duration_sec,
            created_at=now,
            last_used_at=now,
        )
        with self._lock:
            self._evict_expired_locked(now)
            self._evict_overflow_locked()
            self._entries[enrollment_id] = entry
        return entry

    def get(self, enrollment_id: str) -> EnrollmentEntry | None:
        """Return the entry and refresh its last-used timestamp."""
        if not enrollment_id:
            return None
        now = _now()
        with self._lock:
            entry = self._entries.get(enrollment_id)
            if entry is None:
                return None
            if now - entry.last_used_at > self._ttl:
                # Lazily evict instead of running a sweeper thread.
                self._entries.pop(enrollment_id, None)
                return None
            refreshed = EnrollmentEntry(
                enrollment_id=entry.enrollment_id,
                wav_base64=entry.wav_base64,
                duration_sec=entry.duration_sec,
                created_at=entry.created_at,
                last_used_at=now,
            )
            self._entries[enrollment_id] = refreshed
            return refreshed

    def delete(self, enrollment_id: str) -> bool:
        with self._lock:
            return self._entries.pop(enrollment_id, None) is not None

    def _evict_expired_locked(self, now: float) -> None:
        cutoff = now - self._ttl
        stale = [k for k, v in self._entries.items() if v.last_used_at < cutoff]
        for k in stale:
            self._entries.pop(k, None)

    def _evict_overflow_locked(self) -> None:
        # Approximate LRU: when we're at the cap, drop the oldest
        # last_used_at. The store is small enough (≤ 256 by default)
        # that an O(n) scan beats maintaining a separate priority queue.
        while len(self._entries) >= self._max_entries:
            oldest_id = min(
                self._entries,
                key=lambda k: self._entries[k].last_used_at,
            )
            self._entries.pop(oldest_id, None)


_STORE: _Store | None = None


def get_enrollment_store() -> _Store:
    """Lazy singleton — instantiated on first use to avoid touching
    config at import time."""
    global _STORE
    if _STORE is None:
        from ..config import default_config

        _STORE = _Store(
            ttl_sec=default_config.asr_enrollment_ttl_sec,
            max_entries=default_config.asr_enrollment_max_entries,
        )
    return _STORE


def reset_enrollment_store_for_tests() -> None:
    """Reset the singleton in unit tests so they don't bleed state."""
    global _STORE
    _STORE = None


__all__ = [
    "EnrollmentError",
    "EnrollmentEntry",
    "decode_and_validate",
    "decode_and_validate_audio_bytes",
    "get_enrollment_store",
    "reset_enrollment_store_for_tests",
]
