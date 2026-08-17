"""In-process async job queue for offline long-audio transcription.

Thin shell over the generic :class:`backend.jobstore.JobStore`, same shape
as the emotion job stores. Differences that are transcription-specific:

- the job holds the decoded recording as int16 PCM bytes (not the original
  upload) so the pipeline can replay it straight into the VAD stream;
- a ``progress`` block (segments planned / done) is exposed in the poll
  payload because a long meeting takes minutes, not seconds, to transcribe;
- vLLM pressure is bounded by ``transcribe_max_concurrent_jobs`` because each
  job transcribes its own VAD segments serially in file order.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import Config, load_transcribe_config
from ..jobstore import (
    JOB_STATUS_FAILED,
    JOB_STATUS_QUEUED,
    JOB_STATUS_RUNNING,
    JOB_STATUS_SUCCEEDED,
    BaseJob,
    JobQueueFullError,
    JobStore,
)
from .transcribe import transcribe_pcm_i16

logger = logging.getLogger(__name__)

__all__ = [
    "TranscriptionJob",
    "TranscriptionJobStore",
    "JobQueueFullError",
    "JOB_STATUS_FAILED",
    "JOB_STATUS_QUEUED",
    "JOB_STATUS_RUNNING",
    "JOB_STATUS_SUCCEEDED",
    "get_transcription_job_store",
]


@dataclass(kw_only=True)
class TranscriptionJob(BaseJob):
    language: str = ""
    hotwords: list[str] = field(default_factory=list)
    pcm_i16: bytes = field(default=b"", repr=False)
    duration_sec: float = 0.0
    segments_total: int | None = None
    segments_done: int = 0

    def release_payload(self) -> None:
        self.pcm_i16 = b""

    def to_poll_dict(self) -> dict[str, Any]:
        payload = super().to_poll_dict()
        payload["progress"] = {
            "segments_total": self.segments_total,
            "segments_done": self.segments_done,
        }
        return payload


class TranscriptionJobStore(JobStore[TranscriptionJob]):
    """Process-local transcription job store (semaphore-limited)."""

    id_prefix = "tr"
    label = "Transcription"

    def __init__(self) -> None:
        super().__init__()
        # Transcription view of the config: rest.routes.transcribe model
        # bindings and fusion switch applied on top of the global REST defaults.
        self._cfg: Config = load_transcribe_config()

    def configure(self, cfg: Config | None = None) -> None:
        if cfg is not None:
            self._cfg = cfg
        self.configure_limits(
            max_concurrent=getattr(self._cfg, "transcribe_max_concurrent_jobs", 2),
            queue_max=getattr(self._cfg, "transcribe_job_queue_max", 8),
            ttl_sec=getattr(self._cfg, "transcribe_job_ttl_sec", 3600),
        )

    async def submit(
        self,
        pcm_i16: bytes,
        *,
        duration_sec: float,
        language: str = "",
        hotwords: list[str] | None = None,
        cfg: Config | None = None,
    ) -> TranscriptionJob:
        if cfg is not None:
            self.configure(cfg)
        elif not self.is_configured:
            self.configure()

        now = time.time()
        job = TranscriptionJob(
            job_id=self.new_job_id(),
            status=JOB_STATUS_QUEUED,
            created_at=now,
            updated_at=now,
            language=language,
            hotwords=list(hotwords or []),
            pcm_i16=pcm_i16,
            duration_sec=duration_sec,
        )
        return await self.enqueue(job)

    async def run_job_payload(self, job: TranscriptionJob) -> dict[str, Any]:
        # Progress writers run on the event loop (no awaits in between), so
        # plain attribute assignment is race-free here.
        def on_planned(total: int) -> None:
            job.segments_total = total
            job.updated_at = time.time()

        def on_done(done: int) -> None:
            job.segments_done = done
            job.updated_at = time.time()

        return await transcribe_pcm_i16(
            job.pcm_i16,
            cfg=self._cfg,
            language=job.language,
            hotwords=job.hotwords,
            on_segments_planned=on_planned,
            on_segment_done=on_done,
            # Segments hold their own PCM copies once cut; drop the full
            # recording before minutes of inference start.
            release_input=job.release_payload,
        )


_store: TranscriptionJobStore | None = None


def get_transcription_job_store() -> TranscriptionJobStore:
    global _store
    if _store is None:
        _store = TranscriptionJobStore()
        _store.configure()
    return _store
