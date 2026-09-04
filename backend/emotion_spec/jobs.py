"""In-process async job queue for the AmphionSPEC HTTP API.

Thin shell over the generic :class:`backend.jobstore.JobStore`, independent
of :mod:`backend.emotion.jobs` at runtime: its own semaphore (sized by
``emotion_spec_max_concurrent_jobs``) so SPEC traffic cannot starve the
baseline emotion store (and vice versa). :class:`JobQueueFullError` is
re-exported from the shared skeleton to keep route-level ``except`` clauses
single-line.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import Config, load_config
from ..jobstore import (
    JOB_STATUS_FAILED,
    JOB_STATUS_QUEUED,
    JOB_STATUS_RUNNING,
    JOB_STATUS_SUCCEEDED,
    BaseJob,
    JobExecutionError,
    JobQueueFullError,
    JobStore,
)
from .service import EmotionDecodeError, infer_emotion_spec_from_wav

logger = logging.getLogger(__name__)

__all__ = [
    "EmotionSpecJob",
    "EmotionSpecJobStore",
    "JobQueueFullError",
    "JOB_STATUS_FAILED",
    "JOB_STATUS_QUEUED",
    "JOB_STATUS_RUNNING",
    "JOB_STATUS_SUCCEEDED",
    "get_emotion_spec_job_store",
]


@dataclass(kw_only=True)
class EmotionSpecJob(BaseJob):
    mode: str = ""
    language: str = ""
    wav_bytes: bytes = field(default=b"", repr=False)

    def release_payload(self) -> None:
        self.wav_bytes = b""


class EmotionSpecJobStore(JobStore[EmotionSpecJob]):
    """Process-local SPEC job store with semaphore-limited vLLM concurrency."""

    id_prefix = "es"
    label = "Emotion-SPEC"

    def __init__(self) -> None:
        super().__init__()
        self._cfg: Config = load_config()

    def configure(self, cfg: Config | None = None) -> None:
        if cfg is not None:
            self._cfg = cfg
        self.configure_limits(
            max_concurrent=getattr(self._cfg, "emotion_spec_max_concurrent_jobs", 8),
            queue_max=getattr(self._cfg, "emotion_spec_job_queue_max", 64),
            ttl_sec=getattr(self._cfg, "emotion_spec_job_ttl_sec", 3600),
        )

    async def submit(
        self,
        wav_bytes: bytes,
        *,
        mode: str = "",
        language: str = "",
        cfg: Config | None = None,
    ) -> EmotionSpecJob:
        if cfg is not None:
            self.configure(cfg)
        elif not self.is_configured:
            self.configure()

        now = time.time()
        job = EmotionSpecJob(
            job_id=self.new_job_id(),
            status=JOB_STATUS_QUEUED,
            created_at=now,
            updated_at=now,
            mode=mode,
            language=language,
            wav_bytes=wav_bytes,
        )
        return await self.enqueue(job)

    async def run_job_payload(self, job: EmotionSpecJob) -> dict[str, Any]:
        chosen_mode = job.mode or getattr(self._cfg, "emotion_spec_task_mode", "sec")
        try:
            return await infer_emotion_spec_from_wav(
                job.wav_bytes,
                mode=chosen_mode,
                language=job.language,
                cfg=self._cfg,
            )
        except EmotionDecodeError as exc:
            raise JobExecutionError(str(exc), code="decode_error") from exc
        except asyncio.TimeoutError as exc:
            raise JobExecutionError(
                "emotion-spec model request timed out", code="inference_timeout"
            ) from exc


_spec_store: EmotionSpecJobStore | None = None


def get_emotion_spec_job_store() -> EmotionSpecJobStore:
    global _spec_store
    if _spec_store is None:
        _spec_store = EmotionSpecJobStore()
        _spec_store.configure()
    return _spec_store
