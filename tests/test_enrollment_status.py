from __future__ import annotations

import asyncio
import base64
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.asr.enrollment as enr  # noqa: E402
import backend.main as m  # noqa: E402
from backend.asr.enrollment import (  # noqa: E402
    EnrollmentError,
    decode_and_validate,
    get_enrollment_store,
    reset_enrollment_store_for_tests,
)


# A minimal valid 16 kHz mono WAV (0.01s of silence) so put()/persist works.
def _tiny_wav_b64() -> str:
    import numpy as np

    from backend.audio.utils import pcm_to_wav_base64

    return pcm_to_wav_base64(np.zeros(160, dtype=np.float32))


def _store_with_dir(tmp_path: Path, monkeypatch):
    reset_enrollment_store_for_tests()
    store = get_enrollment_store()
    store.configure(
        ttl_sec=3600.0,
        max_entries=256,
        store_dir=str(tmp_path),
        scope="default",
        touch_interval_sec=0.0,
    )
    monkeypatch.setattr(enr, "current_model_fingerprint", lambda: "model-A")
    return store


def _status(enrollment_id: str) -> dict:
    return asyncio.run(m.asr_enrollment_status(enrollment_id))


def test_status_ok_for_registered_id(tmp_path, monkeypatch) -> None:
    store = _store_with_dir(tmp_path, monkeypatch)
    entry = store.put(_tiny_wav_b64(), 2.0)
    out = _status(entry.enrollment_id)
    assert out == {"enrollment_id": entry.enrollment_id, "available": True, "reason": "ok"}


def test_status_not_found_for_unknown_id(tmp_path, monkeypatch) -> None:
    _store_with_dir(tmp_path, monkeypatch)
    out = _status("does-not-exist")
    assert out == {"enrollment_id": "does-not-exist", "available": False, "reason": "not_found"}


def test_status_deleted_after_delete(tmp_path, monkeypatch) -> None:
    store = _store_with_dir(tmp_path, monkeypatch)
    entry = store.put(_tiny_wav_b64(), 2.0)
    assert store.delete(entry.enrollment_id) is True
    out = _status(entry.enrollment_id)
    assert out["available"] is False
    assert out["reason"] == "deleted"


def test_status_incompatible_on_model_fingerprint_change(tmp_path, monkeypatch) -> None:
    store = _store_with_dir(tmp_path, monkeypatch)
    entry = store.put(_tiny_wav_b64(), 2.0)  # persisted with fingerprint model-A
    monkeypatch.setattr(enr, "current_model_fingerprint", lambda: "model-B")
    out = _status(entry.enrollment_id)
    assert out["available"] is False
    assert out["reason"] == "incompatible"


def test_status_upstream_unavailable_on_store_io_error(tmp_path, monkeypatch) -> None:
    store = _store_with_dir(tmp_path, monkeypatch)
    entry = store.put(_tiny_wav_b64(), 2.0)

    def _boom(_eid):
        raise OSError("disk offline")

    monkeypatch.setattr(store, "_read_meta", _boom)
    out = _status(entry.enrollment_id)
    assert out["reason"] == "upstream_unavailable"


def test_persistence_survives_restart(tmp_path, monkeypatch) -> None:
    store = _store_with_dir(tmp_path, monkeypatch)
    entry = store.put(_tiny_wav_b64(), 3.0)
    eid = entry.enrollment_id

    # Simulate a process restart: drop the singleton + memory, reopen same dir.
    reset_enrollment_store_for_tests()
    store2 = get_enrollment_store()
    store2.configure(ttl_sec=3600.0, max_entries=256, store_dir=str(tmp_path), scope="default")
    monkeypatch.setattr(enr, "current_model_fingerprint", lambda: "model-A")

    assert _status(eid)["reason"] == "ok"
    rehydrated = store2.get(eid)
    assert rehydrated is not None
    assert base64.b64decode(rehydrated.wav_base64)  # clip rehydrated from disk


def test_delete_unknown_id_is_204_idempotent(tmp_path, monkeypatch) -> None:
    _store_with_dir(tmp_path, monkeypatch)
    resp = asyncio.run(m.asr_enrollment_delete("never-registered"))
    assert resp.status_code == 204


def test_registration_error_codes() -> None:
    with pytest.raises(EnrollmentError) as empty:
        decode_and_validate("", min_sec=1.0, max_sec=8.0)
    assert empty.value.code == "empty"

    # Odd-length, non-WAV/MP3 blob -> unsupported_format.
    bad_fmt = base64.b64encode(b"\x01\x02\x03").decode("ascii")
    with pytest.raises(EnrollmentError) as unsupported:
        decode_and_validate(bad_fmt, min_sec=1.0, max_sec=8.0)
    assert unsupported.value.code == "unsupported_format"

    # RIFF header but truncated/corrupt -> decode_failed.
    corrupt_wav = base64.b64encode(b"RIFF\x00\x00\x00\x00WAVEjunk").decode("ascii")
    with pytest.raises(EnrollmentError) as decode_failed:
        decode_and_validate(corrupt_wav, min_sec=1.0, max_sec=8.0)
    assert decode_failed.value.code == "decode_failed"
