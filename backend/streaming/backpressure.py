"""Lightweight process-local pressure signals for streaming scheduling."""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class FinalPressure:
    waiting: int
    active: int
    limit: int

    @property
    def backlog(self) -> int:
        return self.waiting + self.active


_LOCK = threading.Lock()
_FINAL_WAITING = 0
_FINAL_ACTIVE = 0
_FINAL_LIMIT = 1


def note_final_queued(limit: int) -> None:
    global _FINAL_WAITING, _FINAL_LIMIT
    with _LOCK:
        _FINAL_LIMIT = max(1, int(limit))
        _FINAL_WAITING += 1


def note_final_started(limit: int) -> None:
    global _FINAL_WAITING, _FINAL_ACTIVE, _FINAL_LIMIT
    with _LOCK:
        _FINAL_LIMIT = max(1, int(limit))
        _FINAL_WAITING = max(0, _FINAL_WAITING - 1)
        _FINAL_ACTIVE += 1


def note_final_finished() -> None:
    global _FINAL_ACTIVE
    with _LOCK:
        _FINAL_ACTIVE = max(0, _FINAL_ACTIVE - 1)


def final_pressure() -> FinalPressure:
    with _LOCK:
        return FinalPressure(
            waiting=_FINAL_WAITING,
            active=_FINAL_ACTIVE,
            limit=max(1, _FINAL_LIMIT),
        )

