"""In-process enrollment cache for target-speaker ASR.

The frontend uploads a 1–8 second enrollment clip (file or mic-recorded)
once via ``POST /api/asr/enrollment`` and gets back an opaque
``enrollment_id``. The realtime WS sessions and the REST upload endpoint
then dereference that id to fetch the canonical WAV and reusable projector
embeddings. Storing the WAV server-side (instead of having the client
retransmit it on every VAD segment) keeps WS messages cheap and means the
backend can validate the clip exactly once at upload time; persisted embeddings
avoid repeatedly calling the split encoder for the same enrollment.

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
  rail (each entry is ~256 KB for 8s @ 16 kHz / 16-bit), not a
  business rule.
* **Duration validation** — done at upload time, so by the time a WS
  session resolves the id we already know the clip is in [min, max] s.
  Clips longer than ``max`` are tail-trimmed to ``max`` (not rejected)
  to match the existing emotion / ASR upload convention.
* **Format** — the canonical audio is stored as a base64-encoded 16 kHz mono
  WAV string. When the split encoder is available, a derived projector embedding
  is also stored on disk; the WAV remains the source of truth for re-encoding.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..audio.utils import pcm_to_wav_base64, wav_base64_to_pcm_16k_mono
from ..config import SAMPLE_RATE

logger = logging.getLogger(__name__)
EMBEDDING_FORMAT_VERSION = "split-projector-torch-float16-v1"


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


@dataclass(frozen=True)
class EnrollmentStatus:
    """Diagnostic status for ``GET /api/asr/enrollment/{id}`` (no sensitive data)."""

    enrollment_id: str
    available: bool
    reason: str


def _now() -> float:
    return time.monotonic()


def _wall_now() -> float:
    return time.time()


def current_model_fingerprint() -> str:
    """A stable string identifying the model/adapter that consumes enrollments.

    A change here (model or prompt template swap) flips previously registered
    enrollments to ``incompatible`` on status query, matching the doc's intent
    that a stored enrollment must line up with the current model/adapter.
    """
    from ..config import default_config

    override = str(getattr(default_config, "asr_enrollment_model_fingerprint", "") or "")
    if override:
        return override
    parts = [
        str(getattr(default_config, "vllm_model_name", "")),
        str(getattr(default_config, "vllm_prompt_template", "")),
        str(getattr(default_config, "astv3_vllm_model_name", "")),
        str(getattr(default_config, "astv3_vllm_prompt_template", "")),
    ]
    return "|".join(parts)


def current_embedding_fingerprint() -> str:
    """Fingerprint for persisted enrollment projector embeddings."""
    return f"{current_model_fingerprint()}|{EMBEDDING_FORMAT_VERSION}"


def wav_digest(wav_base64: str) -> str:
    """SHA-256 digest of canonical WAV bytes."""
    raw = base64.b64decode(wav_base64.encode("ascii"))
    return hashlib.sha256(raw).hexdigest()


def _looks_like_supported_audio(wav_base64: str) -> bool:
    """Best-effort magic-byte sniff for WAV / MP3 / plausible raw PCM."""
    try:
        raw = base64.b64decode(wav_base64, validate=False)
    except Exception:
        return False
    if len(raw) < 4:
        return False
    head = raw[:4]
    if head == b"RIFF":  # WAV
        return True
    if head[:3] == b"ID3":  # MP3 with ID3 tag
        return True
    if head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:  # MP3 frame sync
        return True
    # 16 kHz mono s16le raw PCM has no header; treat an even-length blob as a
    # plausible PCM candidate (decode_failed) rather than unsupported_format.
    if len(raw) % 2 == 0:
        return True
    return False


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
    except Exception as exc:  # noqa: BLE001 - any decode failure is classified below
        # Distinguish an unrecognized container (unsupported_format) from a
        # payload that cannot be decoded at all or is a recognized-but-corrupt
        # container (decode_failed). Base64 that will not decode is a broken
        # payload, not a wrong file type.
        try:
            base64.b64decode(wav_base64, validate=False)
            base64_ok = True
        except Exception:
            base64_ok = False
        if not base64_ok:
            code = "decode_failed"
        else:
            code = (
                "decode_failed"
                if _looks_like_supported_audio(wav_base64)
                else "unsupported_format"
            )
        raise EnrollmentError(code, str(exc)) from exc
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


class _Store:
    """Enrollment cache with an in-memory hot tier over a durable disk tier.

    * In-memory: LRU-ish with TTL eviction, keeps hot clips cheap to splice.
    * Disk (when ``store_dir`` set): source of truth. Each id persists as
      ``<store_dir>/<scope>/<id>.json`` (metadata + model fingerprint + status)
      and ``<id>.wav`` (canonical clip). Disk entries do NOT TTL-expire; a
      TTL-evicted memory copy is rehydrated from disk. Delete leaves a
      ``status=deleted`` tombstone so status can report ``deleted`` vs
      ``not_found``.
    """

    def __init__(
        self,
        *,
        ttl_sec: float,
        max_entries: int,
        store_dir: str = "",
        scope: str = "default",
        touch_interval_sec: float = 60.0,
    ) -> None:
        self._ttl = float(ttl_sec)
        self._max_entries = int(max_entries)
        self._entries: dict[str, EnrollmentEntry] = {}
        self._lock = threading.RLock()
        self._store_dir = Path(store_dir) if store_dir else None
        self._scope = scope or "default"
        self._touch_interval = float(touch_interval_sec)

    def configure(
        self,
        *,
        ttl_sec: float,
        max_entries: int,
        store_dir: str | None = None,
        scope: str | None = None,
        touch_interval_sec: float | None = None,
    ) -> None:
        with self._lock:
            self._ttl = float(ttl_sec)
            self._max_entries = int(max_entries)
            if store_dir is not None:
                self._store_dir = Path(store_dir) if store_dir else None
            if scope:
                self._scope = scope
            if touch_interval_sec is not None:
                self._touch_interval = float(touch_interval_sec)

    # ---- disk helpers ---------------------------------------------------
    @property
    def persistent(self) -> bool:
        return self._store_dir is not None

    def _scope_dir(self) -> Path:
        return self._store_dir / self._scope  # type: ignore[operator]

    def _paths(self, enrollment_id: str) -> tuple[Path, Path]:
        d = self._scope_dir()
        return d / f"{enrollment_id}.json", d / f"{enrollment_id}.wav"

    def _embedding_path(self, enrollment_id: str) -> Path:
        return self._scope_dir() / f"{enrollment_id}.embeds.json"

    def _read_meta(self, enrollment_id: str) -> dict | None:
        meta_path, _ = self._paths(enrollment_id)
        if not meta_path.exists():
            return None
        with meta_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write_meta(self, enrollment_id: str, meta: dict) -> None:
        meta_path, _ = self._paths(enrollment_id)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = meta_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False)
        os.replace(tmp, meta_path)

    def _persist_entry(self, entry: EnrollmentEntry, wall_now: float) -> None:
        _, wav_path = self._paths(entry.enrollment_id)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = wav_path.with_suffix(".wav.tmp")
        with tmp.open("wb") as fh:
            fh.write(base64.b64decode(entry.wav_base64))
        os.replace(tmp, wav_path)
        self._write_meta(
            entry.enrollment_id,
            {
                "enrollment_id": entry.enrollment_id,
                "created_at": wall_now,
                "last_used_at": wall_now,
                "duration_sec": entry.duration_sec,
                "model_fingerprint": current_model_fingerprint(),
                "source_wav_sha256": wav_digest(entry.wav_base64),
                "embedding_status": "pending",
                "embedding": None,
                "status": "active",
            },
        )

    def _load_from_disk(self, enrollment_id: str) -> EnrollmentEntry | None:
        meta = self._read_meta(enrollment_id)
        if not meta or meta.get("status") != "active":
            return None
        _, wav_path = self._paths(enrollment_id)
        if not wav_path.exists():
            return None
        wav_b64 = base64.b64encode(wav_path.read_bytes()).decode("ascii")
        now = _now()
        return EnrollmentEntry(
            enrollment_id=enrollment_id,
            wav_base64=wav_b64,
            duration_sec=float(meta.get("duration_sec") or 0.0),
            created_at=now,
            last_used_at=now,
        )

    def _touch_last_used(self, enrollment_id: str) -> None:
        """Throttled persist of last_used_at (no TTL expiry on disk)."""
        try:
            meta = self._read_meta(enrollment_id)
            if not meta or meta.get("status") != "active":
                return
            now = _wall_now()
            if now - float(meta.get("last_used_at") or 0.0) < self._touch_interval:
                return
            meta["last_used_at"] = now
            self._write_meta(enrollment_id, meta)
        except OSError:
            pass

    # ---- public API -----------------------------------------------------
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
            if self.persistent:
                try:
                    self._persist_entry(entry, _wall_now())
                except OSError as exc:  # pragma: no cover - disk failure
                    logger.warning("enrollment persist failed id=%s: %s", enrollment_id, exc)
        return entry

    def get(self, enrollment_id: str) -> EnrollmentEntry | None:
        """Return the entry (memory hot tier, else disk) and refresh last-used."""
        if not enrollment_id:
            return None
        now = _now()
        with self._lock:
            entry = self._entries.get(enrollment_id)
            if entry is not None and now - entry.last_used_at <= self._ttl:
                refreshed = EnrollmentEntry(
                    enrollment_id=entry.enrollment_id,
                    wav_base64=entry.wav_base64,
                    duration_sec=entry.duration_sec,
                    created_at=entry.created_at,
                    last_used_at=now,
                )
                self._entries[enrollment_id] = refreshed
                if self.persistent:
                    self._touch_last_used(enrollment_id)
                return refreshed
            # Memory miss or TTL-evicted: rehydrate from disk if persistent.
            self._entries.pop(enrollment_id, None)
            if not self.persistent:
                return None
            try:
                disk_entry = self._load_from_disk(enrollment_id)
            except OSError:
                return None
            if disk_entry is None:
                return None
            self._entries[enrollment_id] = disk_entry
            self._touch_last_used(enrollment_id)
            return disk_entry

    def delete(self, enrollment_id: str) -> bool:
        with self._lock:
            existed_mem = self._entries.pop(enrollment_id, None) is not None
            if not self.persistent:
                return existed_mem
            existed_disk = False
            try:
                meta = self._read_meta(enrollment_id)
                _, wav_path = self._paths(enrollment_id)
                embed_path = self._embedding_path(enrollment_id)
                if meta is not None or wav_path.exists():
                    existed_disk = meta is None or meta.get("status") == "active"
                    if wav_path.exists():
                        wav_path.unlink()
                    if embed_path.exists():
                        embed_path.unlink()
                    # Tombstone so status() can distinguish deleted vs not_found.
                    self._write_meta(
                        enrollment_id,
                        {
                            "enrollment_id": enrollment_id,
                            "status": "deleted",
                            "deleted_at": _wall_now(),
                        },
                    )
            except OSError as exc:  # pragma: no cover - disk failure
                logger.warning("enrollment delete failed id=%s: %s", enrollment_id, exc)
            return existed_mem or existed_disk

    def status(self, enrollment_id: str) -> EnrollmentStatus:
        """Judgment order: upstream_unavailable -> not_found -> deleted ->
        incompatible -> ok (最终版 §查询声纹状态)."""
        if not enrollment_id:
            return EnrollmentStatus(enrollment_id, False, "not_found")
        with self._lock:
            if not self.persistent:
                present = self.get(enrollment_id) is not None
                return EnrollmentStatus(
                    enrollment_id, present, "ok" if present else "not_found"
                )
            try:
                meta = self._read_meta(enrollment_id)
            except OSError:
                return EnrollmentStatus(enrollment_id, False, "upstream_unavailable")
            if meta is None:
                # No metadata on disk; a memory-only entry (persist failure)
                # still counts as usable.
                if enrollment_id in self._entries:
                    return EnrollmentStatus(enrollment_id, True, "ok")
                return EnrollmentStatus(enrollment_id, False, "not_found")
            if meta.get("status") == "deleted":
                return EnrollmentStatus(enrollment_id, False, "deleted")
            _, wav_path = self._paths(enrollment_id)
            try:
                clip_present = wav_path.exists()
            except OSError:
                return EnrollmentStatus(enrollment_id, False, "upstream_unavailable")
            if not clip_present:
                return EnrollmentStatus(enrollment_id, False, "not_found")
            if str(meta.get("model_fingerprint") or "") != current_model_fingerprint():
                return EnrollmentStatus(enrollment_id, False, "incompatible")
            self._touch_last_used(enrollment_id)
            return EnrollmentStatus(enrollment_id, True, "ok")

    # ---- persisted projector embeddings --------------------------------
    def persist_embedding(
        self,
        enrollment_id: str,
        wav_base64: str,
        audio_embeds_base64: str,
        *,
        encode_response: dict | None = None,
    ) -> bool:
        """Persist a derived split projector embedding for an enrollment.

        The canonical WAV remains authoritative; this helper only writes a
        reusable derived asset that can be discarded and regenerated.
        """
        if not enrollment_id or not audio_embeds_base64 or not self.persistent:
            return False
        encode_response = encode_response or {}
        try:
            source_sha = wav_digest(wav_base64)
            shape = list(encode_response.get("shape") or [])
            token_len = int(encode_response.get("token_len") or (shape[0] if shape else 0))
            meta_embed = {
                "fingerprint": current_embedding_fingerprint(),
                "model_fingerprint": current_model_fingerprint(),
                "source_wav_sha256": source_sha,
                "format_version": EMBEDDING_FORMAT_VERSION,
                "dtype": str(encode_response.get("dtype") or "float16"),
                "serialization": str(encode_response.get("serialization") or "torch"),
                "shape": shape,
                "token_len": token_len,
                "feature_len": int(encode_response.get("feature_len") or 0),
                "projector_len": token_len,
                "created_at": _wall_now(),
            }
            payload = {
                **meta_embed,
                "enrollment_id": enrollment_id,
                "audio_embeds_base64": audio_embeds_base64,
            }
            embed_path = self._embedding_path(enrollment_id)
            embed_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = embed_path.with_suffix(".embeds.json.tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            os.replace(tmp, embed_path)

            meta = self._read_meta(enrollment_id) or {"enrollment_id": enrollment_id}
            if meta.get("status") == "deleted":
                return False
            meta["embedding_status"] = "ready"
            meta["embedding"] = meta_embed
            self._write_meta(enrollment_id, meta)
            logger.info(
                "ENROLLMENT_EMBEDDING event=persisted id=%s source_wav_sha256=%s projector_len=%s",
                enrollment_id,
                source_sha[:12],
                token_len,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("enrollment embedding persist failed id=%s: %s", enrollment_id, exc)
            return False

    def load_embedding(self, enrollment_id: str, wav_base64: str) -> str | None:
        """Load a compatible persisted enrollment embedding, if present."""
        if not enrollment_id or not self.persistent:
            return None
        try:
            meta = self._read_meta(enrollment_id)
            if not meta or meta.get("status") != "active":
                return None
            if str(meta.get("model_fingerprint") or "") != current_model_fingerprint():
                return None
            source_sha = wav_digest(wav_base64)
            embed_meta = meta.get("embedding") if isinstance(meta.get("embedding"), dict) else {}
            if embed_meta.get("fingerprint") != current_embedding_fingerprint():
                return None
            if embed_meta.get("source_wav_sha256") != source_sha:
                return None
            embed_path = self._embedding_path(enrollment_id)
            if not embed_path.exists():
                return None
            payload = json.loads(embed_path.read_text(encoding="utf-8"))
            if payload.get("fingerprint") != current_embedding_fingerprint():
                return None
            if payload.get("source_wav_sha256") != source_sha:
                return None
            embeds = str(payload.get("audio_embeds_base64") or "")
            if not embeds:
                return None
            logger.info(
                "ENROLLMENT_EMBEDDING event=loaded_persisted id=%s projector_len=%s",
                enrollment_id,
                payload.get("projector_len"),
            )
            return embeds
        except Exception as exc:  # noqa: BLE001
            logger.warning("enrollment embedding load failed id=%s: %s", enrollment_id, exc)
            return None

    def _evict_expired_locked(self, now: float) -> None:
        cutoff = now - self._ttl
        stale = [k for k, v in self._entries.items() if v.last_used_at < cutoff]
        for k in stale:
            self._entries.pop(k, None)

    def _evict_overflow_locked(self) -> None:
        # Approximate LRU: when we're at the cap, drop the oldest
        # last_used_at from the in-memory hot tier. Disk copies survive.
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
            store_dir=str(getattr(default_config, "asr_enrollment_store_dir", "") or ""),
            scope=str(getattr(default_config, "asr_enrollment_scope", "default") or "default"),
            touch_interval_sec=float(
                getattr(default_config, "asr_enrollment_metadata_touch_interval_sec", 60.0)
            ),
        )
    return _STORE


def reset_enrollment_store_for_tests() -> None:
    """Reset the singleton in unit tests so they don't bleed state."""
    global _STORE
    _STORE = None


__all__ = [
    "EnrollmentError",
    "EnrollmentEntry",
    "EnrollmentStatus",
    "EMBEDDING_FORMAT_VERSION",
    "current_embedding_fingerprint",
    "current_model_fingerprint",
    "decode_and_validate",
    "get_enrollment_store",
    "reset_enrollment_store_for_tests",
    "wav_digest",
]
