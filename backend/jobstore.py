"""Generic in-process async job store shared by the REST job APIs.

Extracted from the previously copy-pasted twins ``backend/emotion/jobs.py``
and ``backend/emotion_spec/jobs.py`` (and now also used by the offline ASR
transcription jobs). The skeleton owns everything that is task-agnostic:

- job id generation (``{id_prefix}_{token_hex}``)
- the queued/running/succeeded/failed state machine
- semaphore-limited concurrency + active-job queue cap
- TTL purge of terminal jobs
- payload release on terminal states (``BaseJob.release_payload``)

Subclasses implement :meth:`JobStore.run_job_payload` and translate their
domain failures into :class:`JobExecutionError` so the error ``code`` stays a
stable part of the HTTP contract. Any other exception is recorded as
``inference_failed`` (matching the historical emotion-job behaviour).
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_SUCCEEDED = "succeeded"
JOB_STATUS_FAILED = "failed"

ACTIVE_STATUSES = frozenset({JOB_STATUS_QUEUED, JOB_STATUS_RUNNING})


class JobQueueFullError(Exception):
    """Raised when the waiting queue exceeds configured capacity."""


class JobExecutionError(Exception):
    """Domain failure with a stable error ``code`` for the poll payload."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(kw_only=True)
class BaseJob:
    job_id: str
    status: str
    created_at: float
    updated_at: float
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    def to_poll_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.status == JOB_STATUS_SUCCEEDED and self.result is not None:
            payload["result"] = self.result
        if self.status == JOB_STATUS_FAILED and self.error is not None:
            payload["error"] = self.error
        return payload

    def release_payload(self) -> None:
        """Drop large in-memory inputs once the job reaches a terminal state."""


TJob = TypeVar("TJob", bound=BaseJob)


class JobStore(Generic[TJob]):
    """Process-local job store with semaphore-limited execution.

    Subclasses set ``id_prefix`` / ``label`` and implement
    :meth:`run_job_payload`. They typically also keep their own ``submit()``
    facade that builds the concrete job dataclass and calls :meth:`enqueue`.
    """

    id_prefix = "job"
    label = "job"

    def __init__(self) -> None:
        self._jobs: dict[str, TJob] = {}
        self._lock = asyncio.Lock()
        self._semaphore: asyncio.Semaphore | None = None
        self._max_concurrent = 8
        self._queue_max = 64
        self._ttl_sec = 3600.0

    # -- configuration ------------------------------------------------------

    def configure_limits(
        self, *, max_concurrent: int, queue_max: int, ttl_sec: float
    ) -> None:
        self._max_concurrent = max(1, int(max_concurrent))
        self._queue_max = max(1, int(queue_max))
        self._ttl_sec = max(60.0, float(ttl_sec))
        self._semaphore = asyncio.Semaphore(self._max_concurrent)

    @property
    def is_configured(self) -> bool:
        return self._semaphore is not None

    # -- task-specific hook --------------------------------------------------

    async def run_job_payload(self, job: TJob) -> dict[str, Any]:
        """Execute the job and return its ``result`` payload.

        Raise :class:`JobExecutionError` for domain failures with a stable
        error code; any other exception becomes ``inference_failed``.
        """
        raise NotImplementedError

    # -- public API -----------------------------------------------------------

    def new_job_id(self) -> str:
        return f"{self.id_prefix}_{secrets.token_hex(12)}"

    async def enqueue(self, job: TJob) -> TJob:
        if self._semaphore is None:
            raise RuntimeError(f"{type(self).__name__} is not configured")

        async with self._lock:
            await self._purge_expired_locked()
            active = sum(
                1 for j in self._jobs.values() if j.status in ACTIVE_STATUSES
            )
            if active >= self._queue_max:
                raise JobQueueFullError(
                    f"{self.label} job queue full ({self._queue_max} active jobs)"
                )
            self._jobs[job.job_id] = job

        asyncio.create_task(self._run_job(job.job_id))
        logger.info("%s job %s queued (active=%s)", self.label, job.job_id, active + 1)
        return job

    async def get(self, job_id: str) -> TJob | None:
        async with self._lock:
            await self._purge_expired_locked()
            return self._jobs.get(job_id)

    # -- internals -------------------------------------------------------------

    async def _purge_expired_locked(self) -> None:
        if self._ttl_sec <= 0:
            return
        cutoff = time.time() - self._ttl_sec
        expired = [
            jid
            for jid, job in self._jobs.items()
            if job.updated_at < cutoff and job.status not in ACTIVE_STATUSES
        ]
        for jid in expired:
            del self._jobs[jid]

    async def _run_job(self, job_id: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return

        assert self._semaphore is not None
        try:
            wait_start = time.time()
            async with self._semaphore:
                wait_ms = (time.time() - wait_start) * 1000.0
                logger.info(
                    "%s job %s starting after queue_wait_ms=%.1f",
                    self.label,
                    job_id,
                    wait_ms,
                )
                await self._set_status(job_id, JOB_STATUS_RUNNING)
                await self._execute(job_id)
        except Exception:
            logger.exception("Unexpected error running %s job %s", self.label, job_id)
            await self._set_failed(
                job_id,
                message="internal job runner error",
                code="internal_error",
            )

    async def _execute(self, job_id: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return

        try:
            result = await self.run_job_payload(job)
        except JobExecutionError as exc:
            await self._set_failed(job_id, message=str(exc), code=exc.code)
            return
        except Exception as exc:
            logger.exception("%s job %s inference failed", self.label, job_id)
            await self._set_failed(job_id, message=str(exc), code="inference_failed")
            return

        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JOB_STATUS_SUCCEEDED
            job.result = result
            job.updated_at = time.time()
            job.release_payload()

    async def _set_status(self, job_id: str, status: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = status
            job.updated_at = time.time()

    async def _set_failed(self, job_id: str, *, message: str, code: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JOB_STATUS_FAILED
            job.error = {"message": message, "code": code}
            job.updated_at = time.time()
            job.release_payload()
