from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import HTTPException  # noqa: E402

import backend.main as m  # noqa: E402


@dataclass
class _FakeResult:
    action: str
    hotword_pool_id: str
    status: str = "ok"
    message: str = ""
    hotwords: list[str] = field(default_factory=list)
    total_count: int = 0
    stats: dict[str, object] = field(default_factory=dict)


def _patch_manage(monkeypatch):
    calls: dict[str, object] = {}

    async def fake_manage(action, *, hotword_pool_id="", **kwargs):
        calls["action"] = action
        calls["hotword_pool_id"] = hotword_pool_id
        calls["kwargs"] = kwargs
        return _FakeResult(action=action, hotword_pool_id=hotword_pool_id)

    monkeypatch.setattr(m, "manage_hotword_pool", fake_manage)
    return calls


def test_resolve_pool_id_body_query_precedence() -> None:
    assert m._resolve_pool_id_or_400(None, "") == ""
    assert m._resolve_pool_id_or_400("", None) == ""
    assert m._resolve_pool_id_or_400("team-a", "") == "team-a"
    assert m._resolve_pool_id_or_400(None, "team-b") == "team-b"
    assert m._resolve_pool_id_or_400("team-a", "team-a") == "team-a"


def test_resolve_pool_id_conflict_is_400() -> None:
    with pytest.raises(HTTPException) as exc:
        m._resolve_pool_id_or_400("team-a", "team-b")
    assert exc.value.status_code == 400


def test_resolve_pool_id_invalid_is_400() -> None:
    with pytest.raises(HTTPException) as exc:
        m._resolve_pool_id_or_400("../escape", None)
    assert exc.value.status_code == 400


def test_clear_endpoint_routes_to_named_pool(monkeypatch) -> None:
    calls = _patch_manage(monkeypatch)
    body = m.HotwordPoolScopeRequest(hotword_pool_id="team-a")
    out = asyncio.run(m.asr_hotword_pool_clear(body=body, hotword_pool_id=""))
    assert calls["action"] == "clear"
    assert calls["hotword_pool_id"] == "team-a"
    assert out["action"] == "clear"
    assert out["hotword_pool_id"] == "team-a"


def test_clear_endpoint_defaults_to_default_pool(monkeypatch) -> None:
    calls = _patch_manage(monkeypatch)
    asyncio.run(m.asr_hotword_pool_clear(body=None, hotword_pool_id=""))
    assert calls["action"] == "clear"
    assert calls["hotword_pool_id"] == ""


def test_clear_endpoint_conflict_is_400(monkeypatch) -> None:
    _patch_manage(monkeypatch)
    body = m.HotwordPoolScopeRequest(hotword_pool_id="team-a")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(m.asr_hotword_pool_clear(body=body, hotword_pool_id="team-b"))
    assert exc.value.status_code == 400


def test_reload_endpoint_routes_to_named_pool(monkeypatch) -> None:
    calls = _patch_manage(monkeypatch)
    out = asyncio.run(m.asr_hotword_pool_reload(body=None, hotword_pool_id="team-b"))
    assert calls["action"] == "reload"
    assert calls["hotword_pool_id"] == "team-b"
    assert out["action"] == "reload"


def test_empty_delete_is_delete_not_clear(monkeypatch) -> None:
    # An empty hotwords array must be forwarded as a (no-op) delete, never as a
    # clear (最终版 §5: 空数组不得解释为清空).
    calls = _patch_manage(monkeypatch)
    req = m.HotwordPoolUpdateRequest(hotwords=[], hotword_pool_id="team-a")
    asyncio.run(m.asr_hotword_pool_delete(req))
    assert calls["action"] == "delete"
    assert calls["kwargs"].get("hotwords") == []
