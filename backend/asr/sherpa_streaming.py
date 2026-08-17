"""Streaming partial ASR backend based on sherpa-onnx.

This module is intentionally a thin synchronous wrapper. The session/task layer
runs calls in a dedicated executor so importing or decoding with sherpa-onnx
does not block the event loop.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, replace
from hashlib import blake2b
from pathlib import Path

import numpy as np

from ..config import SAMPLE_RATE, Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SherpaStreamingConfig:
    model_dir: str
    provider: str = "cpu"
    num_threads: int = 2
    blank_penalty: float = 0.0
    use_int8: bool = True
    runtime_id: int = 0
    decode_batch_wait_ms: int = 0
    decode_batch_max_size: int = 16
    online_scheduler_enabled: bool = False
    online_result_wait_ms: int = 0
    online_result_wait_min_audio_ms: int = 800
    online_ready_coalesce_ms: int = 0


_RECOGNIZER_LOCK = threading.Lock()
_RECOGNIZERS: dict[SherpaStreamingConfig, object] = {}
_RECOGNIZER_RUNTIME_LOCKS: dict[SherpaStreamingConfig, threading.Lock] = {}
_DECODE_BATCHERS: dict[SherpaStreamingConfig, "_DecodeBatcher"] = {}
_ONLINE_SCHEDULERS: dict[SherpaStreamingConfig, "_OnlineDecodeScheduler"] = {}


class _PendingDecode:
    def __init__(
        self, stream: object, *, max_steps: int = 0, trace_id: str = ""
    ) -> None:
        self.stream = stream
        self.trace_id = trace_id
        self.max_steps = max(0, int(max_steps))
        self.steps = 0
        self.enqueued_at = time.monotonic()
        self.batch_size = 0
        self.pending_depth = 0
        self.queue_ms = 0.0
        self.batch_wait_ms = 0.0
        self.lock_wait_ms = 0.0
        self.is_ready_ms = 0.0
        self.decode_ms = 0.0
        self.decode_loops = 0
        self.ready_checks = 0
        self.done = False
        self.exc: BaseException | None = None


class _DecodeBatcher:
    """Small synchronous coalescer for sherpa-onnx ``decode_streams``.

    Each WebSocket session still owns its OnlineStream, but concurrent ready
    streams share one recognizer. Coalescing for a few milliseconds lets the
    Ascend OM backend run a batched encoder/joiner pass instead of 52 serialized
    single-stream calls under load.
    """

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.pending: list[_PendingDecode] = []
        self.owner_active = False

    def decode(
        self,
        *,
        recognizer: object,
        stream: object,
        runtime_lock: threading.Lock,
        wait_ms: int,
        max_batch_size: int,
        max_steps: int,
        trace_id: str = "",
    ) -> dict[str, float | int | str]:
        item = _PendingDecode(stream, max_steps=max_steps, trace_id=trace_id)
        with self.cond:
            self.pending.append(item)
            if not self.owner_active:
                self.owner_active = True
                owner = True
            else:
                owner = False
                self.cond.notify()

        if owner:
            self._run_owner(
                recognizer=recognizer,
                runtime_lock=runtime_lock,
                wait_ms=max(0, int(wait_ms)),
                max_batch_size=max(1, int(max_batch_size)),
            )

        with self.cond:
            while not item.done and item.exc is None:
                self.cond.wait()
            if item.exc is not None:
                raise item.exc
            return {
                "queue_ms": item.queue_ms,
                "batch_wait_ms": item.batch_wait_ms,
                "lock_wait_ms": item.lock_wait_ms,
                "is_ready_ms": item.is_ready_ms,
                "decode_ms": item.decode_ms,
                "decode_loops": item.decode_loops,
                "ready_checks": item.ready_checks,
                "batch_size": item.batch_size,
                "pending_depth": item.pending_depth,
            }

    def _run_owner(
        self,
        *,
        recognizer: object,
        runtime_lock: threading.Lock,
        wait_ms: int,
        max_batch_size: int,
    ) -> None:
        try:
            while True:
                if wait_ms > 0:
                    deadline = time.monotonic() + wait_ms / 1000.0
                    wait_start = time.monotonic()
                    with self.cond:
                        while self.pending and time.monotonic() < deadline:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                break
                            self.cond.wait(timeout=remaining)
                    batch_wait_ms = (time.monotonic() - wait_start) * 1000.0
                else:
                    batch_wait_ms = 0.0

                with self.cond:
                    pending_depth = len(self.pending)
                    batch = self.pending[:max_batch_size]
                    self.pending = self.pending[max_batch_size:]
                    if not batch:
                        self.owner_active = False
                        self.cond.notify_all()
                        return
                    now = time.monotonic()
                    for item in batch:
                        item.batch_size = len(batch)
                        item.pending_depth = pending_depth
                        item.queue_ms = (now - item.enqueued_at) * 1000.0
                        item.batch_wait_ms = batch_wait_ms

                try:
                    lock_start = time.monotonic()
                    with runtime_lock:
                        lock_wait_ms = (time.monotonic() - lock_start) * 1000.0
                        for item in batch:
                            item.lock_wait_ms = lock_wait_ms
                        while True:
                            ready_start = time.monotonic()
                            ready = [
                                item.stream
                                for item in batch
                                if recognizer.is_ready(item.stream)
                                and (item.max_steps <= 0 or item.steps < item.max_steps)
                            ]
                            ready_elapsed_ms = (time.monotonic() - ready_start) * 1000.0
                            for item in batch:
                                item.is_ready_ms += ready_elapsed_ms
                                item.ready_checks += 1
                            if not ready:
                                break
                            decode_start = time.monotonic()
                            recognizer.decode_streams(ready)
                            decode_elapsed_ms = (time.monotonic() - decode_start) * 1000.0
                            for item in batch:
                                if item.stream in ready:
                                    item.steps += 1
                                    item.decode_ms += decode_elapsed_ms
                                    item.decode_loops += 1
                    logger.info(
                        "K2_BATCH_TIMING batch_size=%s pending_depth=%s "
                        "batch_wait_ms=%.1f lock_wait_ms=%.1f "
                        "decode_loops_max=%s decode_ms_max=%.1f "
                        "is_ready_ms_max=%.1f traces=%s",
                        len(batch),
                        pending_depth,
                        batch_wait_ms,
                        max((item.lock_wait_ms for item in batch), default=0.0),
                        max((item.decode_loops for item in batch), default=0),
                        max((item.decode_ms for item in batch), default=0.0),
                        max((item.is_ready_ms for item in batch), default=0.0),
                        ",".join(item.trace_id or "-" for item in batch[:8]),
                    )
                except BaseException as exc:
                    for item in batch:
                        item.exc = exc
                finally:
                    with self.cond:
                        for item in batch:
                            item.done = True
                        self.cond.notify_all()
        except BaseException as exc:
            with self.cond:
                for item in self.pending:
                    item.exc = exc
                    item.done = True
                self.pending = []
                self.owner_active = False
                self.cond.notify_all()
            raise


class _OnlineStreamState:
    def __init__(self, stream: object, *, trace_id: str = "") -> None:
        self.stream = stream
        self.trace_id = trace_id
        self.active = True
        self.text = ""
        self.version = 0
        self.registered_at = time.monotonic()
        self.decode_ticks = 0
        self.first_nonempty_logged = False
        self.total_samples = 0
        self.last_stats: dict[str, float | int | str] = {
            "queue_ms": 0.0,
            "batch_wait_ms": 0.0,
            "lock_wait_ms": 0.0,
            "is_ready_ms": 0.0,
            "decode_ms": 0.0,
            "decode_loops": 0,
            "ready_checks": 0,
            "batch_size": 0,
            "pending_depth": 0,
            "scheduler": "online",
        }


class _OnlineDecodeScheduler:
    """Long-lived online batch scheduler for K2 streams.

    Python only feeds audio deltas and reads cached results. This scheduler owns
    the active stream set and runs native ``decode_streams`` on ready streams in
    the background, so batching is based on runtime readiness rather than on
    partial request arrival timing.
    """

    def __init__(
        self,
        *,
        recognizer: object,
        runtime_lock: threading.Lock,
        wait_ms: int,
        max_batch_size: int,
        ready_coalesce_ms: int = 0,
    ) -> None:
        self.recognizer = recognizer
        self.runtime_lock = runtime_lock
        self.wait_ms = max(1, int(wait_ms))
        self.max_batch_size = max(1, int(max_batch_size))
        self.ready_coalesce_ms = max(0, int(ready_coalesce_ms))
        self.cond = threading.Condition()
        self.states: dict[int, _OnlineStreamState] = {}
        self.thread = threading.Thread(
            target=self._run,
            name="k2-online-decode-scheduler",
            daemon=True,
        )
        self.thread.start()

    def register(self, stream: object, *, trace_id: str = "") -> _OnlineStreamState:
        key = id(stream)
        with self.cond:
            state = self.states.get(key)
            if state is None:
                state = _OnlineStreamState(stream, trace_id=trace_id)
                self.states[key] = state
            else:
                state.active = True
                state.trace_id = trace_id or state.trace_id
            self.cond.notify_all()
            return state

    def unregister(self, stream: object | None) -> None:
        if stream is None:
            return
        with self.cond:
            state = self.states.pop(id(stream), None)
            if state is not None:
                state.active = False
                state.version += 1
            self.cond.notify_all()

    def feed_and_get(
        self,
        *,
        stream: object,
        chunk: np.ndarray | None,
        total_samples: int,
        trace_id: str,
        result_wait_ms: int,
        result_wait_min_audio_ms: int,
    ) -> tuple[str, dict[str, float | int | str]]:
        total_start = time.monotonic()
        state = self.register(stream, trace_id=trace_id)
        start_version = state.version

        accept_start = time.monotonic()
        if chunk is not None and chunk.size > 0:
            # Serialize feature updates with is_ready/decode for the same native
            # stream. This is conservative but avoids races while the scheduler
            # runs concurrently with WebSocket feed loops.
            with self.runtime_lock:
                stream.accept_waveform(SAMPLE_RATE, chunk)
        accept_ms = (time.monotonic() - accept_start) * 1000.0

        with self.cond:
            # Result-only polls may pass an older cumulative snapshot while a
            # feed call has already advanced the native stream. Keep the
            # scheduler's accounting monotonic so logs and wait thresholds do
            # not move backwards.
            state.total_samples = max(state.total_samples, total_samples)
            self.cond.notify_all()
            total_audio_ms = state.total_samples * 1000.0 / SAMPLE_RATE
            wait_s = (
                max(0, int(result_wait_ms)) / 1000.0
                if total_audio_ms >= max(0, int(result_wait_min_audio_ms))
                else 0.0
            )
            if wait_s > 0 and state.version == start_version:
                self.cond.wait(timeout=wait_s)
            text = state.text
            stats = dict(state.last_stats)

        stats.update(
            {
                "runtime_id": 0,
                "total_samples": total_samples,
                "accept_ms": accept_ms,
                "result_lock_wait_ms": 0.0,
                "result_ms": 0.0,
                "total_ms": (time.monotonic() - total_start) * 1000.0,
            }
        )
        return text, stats

    def _run(self) -> None:
        while True:
            with self.cond:
                self.cond.wait(timeout=self.wait_ms / 1000.0)
                states = [s for s in self.states.values() if s.active]

            if not states:
                continue

            coalesced_ready = False
            lock_start = time.monotonic()
            with self.runtime_lock:
                lock_wait_ms = (time.monotonic() - lock_start) * 1000.0
                ready_start = time.monotonic()
                ready_states = [
                    state
                    for state in states
                    if state.active and self.recognizer.is_ready(state.stream)
                ]
                is_ready_ms = (time.monotonic() - ready_start) * 1000.0
                if not ready_states:
                    continue
                coalesced_ready = (
                    self.ready_coalesce_ms > 0
                    and len(ready_states) < self.max_batch_size
                )

            if coalesced_ready:
                with self.cond:
                    self.cond.wait(timeout=self.ready_coalesce_ms / 1000.0)
                    states = [s for s in self.states.values() if s.active]

            lock_start = time.monotonic()
            with self.runtime_lock:
                lock_wait_ms = (time.monotonic() - lock_start) * 1000.0
                ready_start = time.monotonic()
                ready_states = [
                    state
                    for state in states
                    if state.active and self.recognizer.is_ready(state.stream)
                ]
                is_ready_ms = (time.monotonic() - ready_start) * 1000.0
                if not ready_states:
                    continue

                batch = ready_states[: self.max_batch_size]
                decode_start = time.monotonic()
                self.recognizer.decode_streams([state.stream for state in batch])
                decode_ms = (time.monotonic() - decode_start) * 1000.0

                texts = [
                    str(self.recognizer.get_result(state.stream)).strip()
                    for state in batch
                ]

            now = time.monotonic()
            with self.cond:
                for state, text in zip(batch, texts):
                    if not state.active:
                        continue
                    state.decode_ticks += 1
                    first_nonempty = (
                        bool(text)
                        and not state.first_nonempty_logged
                        and not state.text
                    )
                    state.text = text
                    state.version += 1
                    state.last_stats = {
                        "queue_ms": 0.0,
                        "batch_wait_ms": float(self.wait_ms),
                        "lock_wait_ms": lock_wait_ms,
                        "is_ready_ms": is_ready_ms,
                        "decode_ms": decode_ms,
                        "decode_loops": 1,
                        "ready_checks": 1,
                        "batch_size": len(batch),
                        "pending_depth": len(ready_states),
                        "scheduler": "online",
                        "updated_at": now,
                        "decode_ticks": state.decode_ticks,
                    }
                    if first_nonempty:
                        state.first_nonempty_logged = True
                        logger.info(
                            "K2_ONLINE_FIRST_TEXT traceId=%s audio_ms=%.1f "
                            "since_register_ms=%.1f decode_ticks=%s "
                            "batch_size=%s ready_depth=%s decode_ms=%.1f "
                            "text_chars=%s text=%r",
                            state.trace_id or "-",
                            state.total_samples * 1000.0 / SAMPLE_RATE,
                            (now - state.registered_at) * 1000.0,
                            state.decode_ticks,
                            len(batch),
                            len(ready_states),
                            decode_ms,
                            len(text),
                            text[:80],
                        )
                self.cond.notify_all()
            logger.info(
                "K2_ONLINE_BATCH_TIMING batch_size=%s ready_depth=%s "
                "wait_ms=%s lock_wait_ms=%.1f decode_ms=%.1f "
                "is_ready_ms=%.1f traces=%s",
                len(batch),
                len(ready_states),
                self.wait_ms,
                lock_wait_ms,
                decode_ms,
                is_ready_ms,
                ",".join(state.trace_id or "-" for state in batch[:8]),
            )


def _model_path(model_dir: Path, name: str, *, use_int8: bool) -> str:
    if name == "encoder.onnx":
        encoder_om = model_dir / "encoder_linux_aarch64.om"
        if encoder_om.is_file():
            return str(encoder_om)
    else:
        om = model_dir / name.replace(".onnx", ".om")
        if om.is_file():
            return str(om)
        linux_om = model_dir / name.replace(".onnx", "_linux_aarch64.om")
        if linux_om.is_file():
            return str(linux_om)
    if use_int8:
        int8 = model_dir / name.replace(".onnx", ".int8.onnx")
        if int8.is_file():
            return str(int8)
    return str(model_dir / name)


def _build_recognizer(cfg: SherpaStreamingConfig) -> object:
    import sherpa_onnx  # type: ignore[import-not-found]

    model_dir = Path(cfg.model_dir)
    encoder = _model_path(model_dir, "encoder.onnx", use_int8=cfg.use_int8)
    decoder = _model_path(model_dir, "decoder.onnx", use_int8=cfg.use_int8)
    joiner = _model_path(model_dir, "joiner.onnx", use_int8=cfg.use_int8)
    tokens_path = model_dir / "tokens.txt"
    if not tokens_path.is_file() and model_dir.parent != model_dir:
        parent_tokens = model_dir.parent / "tokens.txt"
        if parent_tokens.is_file():
            tokens_path = parent_tokens
    tokens = str(tokens_path)

    missing = [
        path
        for path in (encoder, decoder, joiner, tokens)
        if not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"sherpa model files missing: {missing}")

    start = time.monotonic()
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        tokens=tokens,
        num_threads=max(1, int(cfg.num_threads)),
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        decoding_method="greedy_search",
        provider=cfg.provider,
        blank_penalty=float(cfg.blank_penalty),
        model_type="zipformer2" if cfg.provider == "ascend" else "",
    )
    logger.info(
        "Loaded sherpa streaming recognizer provider=%s runtime_id=%s int8=%s "
        "threads=%s blank_penalty=%.2f model_dir=%s load_ms=%.1f",
        cfg.provider,
        cfg.runtime_id,
        cfg.use_int8,
        cfg.num_threads,
        cfg.blank_penalty,
        cfg.model_dir,
        (time.monotonic() - start) * 1000.0,
    )
    return recognizer


def get_recognizer(cfg: SherpaStreamingConfig) -> object:
    recognizer = _RECOGNIZERS.get(cfg)
    if recognizer is not None:
        return recognizer
    with _RECOGNIZER_LOCK:
        recognizer = _RECOGNIZERS.get(cfg)
        if recognizer is None:
            recognizer = _build_recognizer(cfg)
            _RECOGNIZERS[cfg] = recognizer
        return recognizer


def _get_runtime_lock(cfg: SherpaStreamingConfig) -> threading.Lock:
    with _RECOGNIZER_LOCK:
        lock = _RECOGNIZER_RUNTIME_LOCKS.get(cfg)
        if lock is None:
            lock = threading.Lock()
            _RECOGNIZER_RUNTIME_LOCKS[cfg] = lock
        return lock


def _get_decode_batcher(cfg: SherpaStreamingConfig) -> _DecodeBatcher:
    with _RECOGNIZER_LOCK:
        batcher = _DECODE_BATCHERS.get(cfg)
        if batcher is None:
            batcher = _DecodeBatcher()
            _DECODE_BATCHERS[cfg] = batcher
        return batcher


def _get_online_scheduler(cfg: SherpaStreamingConfig) -> _OnlineDecodeScheduler:
    recognizer = get_recognizer(cfg)
    runtime_lock = _get_runtime_lock(cfg)
    with _RECOGNIZER_LOCK:
        scheduler = _ONLINE_SCHEDULERS.get(cfg)
        if scheduler is None:
            scheduler = _OnlineDecodeScheduler(
                recognizer=recognizer,
                runtime_lock=runtime_lock,
                wait_ms=cfg.decode_batch_wait_ms or 10,
                max_batch_size=cfg.decode_batch_max_size,
                ready_coalesce_ms=cfg.online_ready_coalesce_ms,
            )
            _ONLINE_SCHEDULERS[cfg] = scheduler
        return scheduler


def _runtime_slot(session_key: str | None, pool_size: int) -> int:
    pool_size = max(1, int(pool_size))
    if pool_size == 1:
        return 0
    if not session_key:
        return int(time.monotonic_ns() % pool_size)
    digest = blake2b(session_key.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % pool_size


def config_from_app(cfg: Config, *, session_key: str | None = None) -> SherpaStreamingConfig:
    backend = str(getattr(cfg, "streaming_partial_backend", "vllm") or "vllm")
    if backend.strip().lower() == "k2_om":
        pool_size = int(getattr(cfg, "k2_om_runtime_pool_size", 1) or 1)
        return SherpaStreamingConfig(
            model_dir=str(getattr(cfg, "k2_om_model_dir", "") or "").strip(),
            provider="ascend",
            num_threads=1,
            blank_penalty=float(getattr(cfg, "sherpa_blank_penalty", 0.0)),
            use_int8=False,
            runtime_id=_runtime_slot(session_key, pool_size),
            decode_batch_wait_ms=int(
                getattr(cfg, "k2_decode_batch_wait_ms", 0) or 0
            ),
            decode_batch_max_size=int(
                getattr(cfg, "k2_decode_batch_max_size", 16) or 16
            ),
            online_scheduler_enabled=bool(
                getattr(cfg, "k2_online_scheduler_enabled", False)
            ),
            online_result_wait_ms=int(
                getattr(cfg, "k2_online_result_wait_ms", 0) or 0
            ),
            online_result_wait_min_audio_ms=int(
                getattr(cfg, "k2_online_result_wait_min_audio_ms", 800) or 800
            ),
            online_ready_coalesce_ms=int(
                getattr(cfg, "k2_online_ready_coalesce_ms", 0) or 0
            ),
        )
    pool_size = int(getattr(cfg, "sherpa_runtime_pool_size", 1) or 1)
    return SherpaStreamingConfig(
        model_dir=str(getattr(cfg, "sherpa_model_dir", "") or "").strip(),
        provider=str(getattr(cfg, "sherpa_provider", "cpu") or "cpu").strip(),
        num_threads=int(getattr(cfg, "sherpa_num_threads", 2)),
        blank_penalty=float(getattr(cfg, "sherpa_blank_penalty", 0.0)),
        use_int8=bool(getattr(cfg, "sherpa_use_int8", True)),
        runtime_id=_runtime_slot(session_key, pool_size),
        decode_batch_wait_ms=0,
    )


def maybe_warmup_from_app(cfg: Config) -> None:
    backend = str(getattr(cfg, "streaming_partial_backend", "vllm") or "vllm")
    if backend.strip().lower() not in {"sherpa", "k2_om"}:
        return
    base = config_from_app(cfg)
    pool_size = int(
        getattr(
            cfg,
            "k2_om_runtime_pool_size"
            if backend.strip().lower() == "k2_om"
            else "sherpa_runtime_pool_size",
            1,
        )
        or 1
    )
    for runtime_id in range(max(1, pool_size)):
        get_recognizer(replace(base, runtime_id=runtime_id))


class SherpaStreamingSession:
    """Per-WebSocket sherpa stream fed from cumulative PartialSnapshot buffers."""

    def __init__(self, cfg: SherpaStreamingConfig) -> None:
        if not cfg.model_dir:
            raise ValueError("sherpa_model_dir is required when sherpa partials are enabled")
        self._cfg = cfg
        self._recognizer = get_recognizer(cfg)
        self._runtime_lock = _get_runtime_lock(cfg)
        self._decode_batcher = _get_decode_batcher(cfg)
        self._online_scheduler = (
            _get_online_scheduler(cfg) if cfg.online_scheduler_enabled else None
        )
        self._stream = None
        self._fed_samples = 0
        self.reset()
        logger.info(
            "Created sherpa streaming session provider=%s runtime_id=%s model_dir=%s",
            cfg.provider,
            cfg.runtime_id,
            cfg.model_dir,
        )

    def reset(self) -> None:
        if self._online_scheduler is not None:
            self._online_scheduler.unregister(self._stream)
        self._stream = self._recognizer.create_stream()
        self._fed_samples = 0

    def close(self) -> None:
        if self._online_scheduler is not None:
            self._online_scheduler.unregister(self._stream)
        self._stream = None
        self._fed_samples = 0

    def accept_cumulative(
        self, pcm: np.ndarray, *, max_decode_steps: int = 0, trace_id: str = ""
    ) -> tuple[str, dict[str, float | int | str]]:
        """Feed only the delta from a cumulative utterance snapshot."""
        total_start = time.monotonic()
        if self._stream is None:
            self.reset()
        if pcm.ndim != 1:
            pcm = pcm.reshape(-1)
        total = int(len(pcm))
        start = min(self._fed_samples, total)
        accept_start = time.monotonic()
        if total > start:
            chunk = np.asarray(pcm[start:total], dtype=np.float32)
            self._fed_samples = total
        else:
            chunk = None
        accept_ms = (time.monotonic() - accept_start) * 1000.0
        if self._online_scheduler is not None:
            text, decode_stats = self._online_scheduler.feed_and_get(
                stream=self._stream,
                chunk=chunk,
                total_samples=total,
                trace_id=trace_id,
                result_wait_ms=self._cfg.online_result_wait_ms,
                result_wait_min_audio_ms=self._cfg.online_result_wait_min_audio_ms,
            )
            decode_stats["delta_samples"] = max(0, total - start)
        elif int(self._cfg.decode_batch_wait_ms) > 0:
            if chunk is not None and chunk.size > 0:
                # OnlineStream owns its feature state; only shared recognizer
                # decode needs the global runtime lock on the legacy path.
                self._stream.accept_waveform(SAMPLE_RATE, chunk)
            decode_stats = self._decode_batcher.decode(
                recognizer=self._recognizer,
                stream=self._stream,
                runtime_lock=self._runtime_lock,
                wait_ms=self._cfg.decode_batch_wait_ms,
                max_batch_size=self._cfg.decode_batch_max_size,
                max_steps=max_decode_steps,
                trace_id=trace_id,
            )
        else:
            if chunk is not None and chunk.size > 0:
                self._stream.accept_waveform(SAMPLE_RATE, chunk)
            steps = 0
            decode_ms = 0.0
            is_ready_ms = 0.0
            lock_start = time.monotonic()
            with self._runtime_lock:
                lock_wait_ms = (time.monotonic() - lock_start) * 1000.0
                while self._recognizer.is_ready(self._stream):
                    ready_start = time.monotonic()
                    ready = self._recognizer.is_ready(self._stream)
                    is_ready_ms += (time.monotonic() - ready_start) * 1000.0
                    if not ready:
                        break
                    decode_start = time.monotonic()
                    self._recognizer.decode_stream(self._stream)
                    decode_ms += (time.monotonic() - decode_start) * 1000.0
                    steps += 1
                    if max_decode_steps > 0 and steps >= max_decode_steps:
                        break
            decode_stats = {
                "queue_ms": 0.0,
                "batch_wait_ms": 0.0,
                "lock_wait_ms": lock_wait_ms,
                "is_ready_ms": is_ready_ms,
                "decode_ms": decode_ms,
                "decode_loops": steps,
                "ready_checks": steps,
                "batch_size": 1,
                "pending_depth": 1,
            }
        if self._online_scheduler is None:
            result_lock_start = time.monotonic()
            with self._runtime_lock:
                result_lock_wait_ms = (time.monotonic() - result_lock_start) * 1000.0
                result_start = time.monotonic()
                text = str(self._recognizer.get_result(self._stream)).strip()
                result_ms = (time.monotonic() - result_start) * 1000.0
        else:
            result_lock_wait_ms = float(decode_stats.get("result_lock_wait_ms", 0.0))
            result_ms = float(decode_stats.get("result_ms", 0.0))
        stats: dict[str, float | int | str] = dict(decode_stats)
        effective_accept_ms = (
            float(stats.get("accept_ms", 0.0))
            if self._online_scheduler is not None
            else accept_ms
        )
        stats.update(
            {
                "runtime_id": self._cfg.runtime_id,
                "total_samples": total,
                "delta_samples": max(0, total - start),
                "accept_ms": effective_accept_ms,
                "result_lock_wait_ms": result_lock_wait_ms,
                "result_ms": result_ms,
                "total_ms": (time.monotonic() - total_start) * 1000.0,
            }
        )
        logger.info(
            "K2_SESSION_TIMING traceId=%s runtime_id=%s total_audio_ms=%.1f "
            "delta_audio_ms=%.1f accept_ms=%.1f queue_ms=%.1f "
            "batch_wait_ms=%.1f lock_wait_ms=%.1f decode_ms=%.1f "
            "decode_loops=%s ready_checks=%s result_ms=%.1f "
            "result_lock_wait_ms=%.1f total_ms=%.1f text_chars=%s",
            trace_id or "-",
            self._cfg.runtime_id,
            total * 1000.0 / SAMPLE_RATE,
            max(0, total - start) * 1000.0 / SAMPLE_RATE,
            effective_accept_ms,
            float(stats.get("queue_ms", 0.0)),
            float(stats.get("batch_wait_ms", 0.0)),
            float(stats.get("lock_wait_ms", 0.0)),
            float(stats.get("decode_ms", 0.0)),
            int(stats.get("decode_loops", 0)),
            int(stats.get("ready_checks", 0)),
            result_ms,
            result_lock_wait_ms,
            float(stats.get("total_ms", 0.0)),
            len(text),
        )
        return text, stats

