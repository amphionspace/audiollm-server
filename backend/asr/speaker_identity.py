"""Short-lived speaker embeddings used by the browser meeting mode."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpeakerIdentityEntry:
    enrollment_id: str
    embedding: np.ndarray
    created_at: float
    last_used_at: float


class SpeakerIdentityStore:
    def __init__(self, *, ttl_sec: float, max_entries: int) -> None:
        self._ttl = max(1.0, float(ttl_sec))
        self._max_entries = max(1, int(max_entries))
        self._entries: dict[str, SpeakerIdentityEntry] = {}
        self._lock = threading.Lock()

    def configure(self, *, ttl_sec: float, max_entries: int) -> None:
        with self._lock:
            self._ttl = max(1.0, float(ttl_sec))
            self._max_entries = max(1, int(max_entries))
            self._evict_locked(time.monotonic())

    def put(self, enrollment_id: str, embedding: np.ndarray) -> None:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1).copy()
        norm = float(np.linalg.norm(vector))
        if not enrollment_id or not np.isfinite(norm) or norm <= 0:
            raise ValueError("speaker identity embedding is invalid")
        vector /= norm
        vector.setflags(write=False)
        now = time.monotonic()
        with self._lock:
            self._evict_locked(now)
            self._entries[enrollment_id] = SpeakerIdentityEntry(
                enrollment_id=enrollment_id,
                embedding=vector,
                created_at=now,
                last_used_at=now,
            )
            self._evict_locked(now)

    def get(self, enrollment_id: str) -> SpeakerIdentityEntry | None:
        now = time.monotonic()
        with self._lock:
            self._evict_locked(now)
            entry = self._entries.get(enrollment_id)
            if entry is None:
                return None
            refreshed = SpeakerIdentityEntry(
                enrollment_id=entry.enrollment_id,
                embedding=entry.embedding,
                created_at=entry.created_at,
                last_used_at=now,
            )
            self._entries[enrollment_id] = refreshed
            return refreshed

    def delete(self, enrollment_id: str) -> bool:
        with self._lock:
            return self._entries.pop(enrollment_id, None) is not None

    def _evict_locked(self, now: float) -> None:
        cutoff = now - self._ttl
        for enrollment_id in [
            key for key, value in self._entries.items() if value.last_used_at < cutoff
        ]:
            self._entries.pop(enrollment_id, None)
        while len(self._entries) > self._max_entries:
            oldest_id = min(
                self._entries,
                key=lambda key: self._entries[key].last_used_at,
            )
            self._entries.pop(oldest_id, None)


_STORE: SpeakerIdentityStore | None = None


def get_speaker_identity_store() -> SpeakerIdentityStore:
    global _STORE
    if _STORE is None:
        from ..config import default_config

        _STORE = SpeakerIdentityStore(
            ttl_sec=default_config.asr_enrollment_ttl_sec,
            max_entries=default_config.asr_enrollment_max_entries,
        )
    return _STORE


def reset_speaker_identity_store_for_tests() -> None:
    global _STORE
    _STORE = None


def match_speaker_embedding(
    query: np.ndarray,
    candidates: list[SpeakerIdentityEntry],
    *,
    threshold: float,
    margin: float,
) -> tuple[SpeakerIdentityEntry | None, float | None, str]:
    vector = np.asarray(query, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        return None, None, "incompatible"
    vector = vector / norm
    if not candidates:
        return None, None, "no_candidates"
    if any(candidate.embedding.size != vector.size for candidate in candidates):
        return None, None, "incompatible"
    scores = sorted(
        ((float(np.dot(vector, candidate.embedding)), candidate) for candidate in candidates),
        key=lambda item: item[0],
        reverse=True,
    )
    best_score, best = scores[0]
    if not np.isfinite(best_score) or best_score < threshold:
        return None, best_score if np.isfinite(best_score) else None, "below_threshold"
    if len(scores) > 1 and best_score - scores[1][0] < margin:
        return None, best_score, "ambiguous"
    return best, best_score, "matched"


__all__ = [
    "SpeakerIdentityEntry",
    "SpeakerIdentityStore",
    "get_speaker_identity_store",
    "match_speaker_embedding",
    "reset_speaker_identity_store_for_tests",
]
