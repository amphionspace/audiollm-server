from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.main as main_mod  # noqa: E402


@pytest.mark.asyncio
async def test_hotword_pool_add_proxies_to_recall(monkeypatch):
    seen: dict[str, object] = {}

    async def fake_add(words, *, hotword_pool_id=None):
        seen["words"] = words
        seen["hotword_pool_id"] = hotword_pool_id
        return {"status": "ok", "added": len(words), "total_count": 2}

    monkeypatch.setattr(main_mod, "add_recall_hotwords", fake_add)

    result = await main_mod.asr_hotword_pool_add(
        {"hotword_pool_id": "tenant-a", "hotwords": [" 挚音科技 ", "", "张硕"]}
    )

    assert seen["words"] == ["挚音科技", "张硕"]
    assert seen["hotword_pool_id"] == "tenant-a"
    assert result == {
        "status": "ok",
        "added": 2,
        "total_count": 2,
        "added_count": 2,
        "duplicate_count": 0,
        "invalid_count": 0,
    }


@pytest.mark.asyncio
async def test_hotword_pool_add_includes_review_contract_count_aliases(monkeypatch):
    async def fake_add(words, *, hotword_pool_id=None):
        return {
            "status": "ok",
            "action": "add",
            "added": 2,
            "skipped_duplicates": 1,
            "invalid": ["x"],
            "duplicates": ["挚音科技"],
            "hotwords": words,
            "total_count": 3,
        }

    monkeypatch.setattr(main_mod, "add_recall_hotwords", fake_add)

    result = await main_mod.asr_hotword_pool_add(
        {"hotword_pool_id": "tenant-a", "hotwords": ["挚音科技", "张硕"]}
    )

    assert result["added_count"] == 2
    assert result["duplicate_count"] == 1
    assert result["invalid_count"] == 1
    assert result["ignored_hotwords"] == ["x"]


@pytest.mark.asyncio
async def test_hotword_pool_delete_compat_route_proxies_to_recall(monkeypatch):
    seen: dict[str, object] = {}

    async def fake_delete(words, *, hotword_pool_id=None):
        seen["words"] = words
        seen["hotword_pool_id"] = hotword_pool_id
        return {"status": "ok", "deleted": len(words), "total_count": 0}

    monkeypatch.setattr(main_mod, "delete_recall_hotwords", fake_delete)

    result = await main_mod.asr_hotword_pool_delete(
        {"hotword_pool_id": "tenant-a", "hotwords": ["挚音科技"]}
    )

    assert seen["words"] == ["挚音科技"]
    assert seen["hotword_pool_id"] == "tenant-a"
    assert result == {
        "status": "ok",
        "deleted": 1,
        "total_count": 0,
        "deleted_count": 1,
        "missing_count": 0,
        "invalid_count": 0,
    }


@pytest.mark.asyncio
async def test_hotword_pool_delete_includes_review_contract_count_aliases(monkeypatch):
    async def fake_delete(words, *, hotword_pool_id=None):
        return {
            "status": "ok",
            "action": "delete",
            "deleted": 1,
            "missing": ["张三"],
            "invalid": ["x"],
            "hotwords": words,
            "total_count": 2,
        }

    monkeypatch.setattr(main_mod, "delete_recall_hotwords", fake_delete)

    result = await main_mod.asr_hotword_pool_delete(
        {"hotword_pool_id": "tenant-a", "hotwords": ["挚音科技", "张三"]}
    )

    assert result["deleted_count"] == 1
    assert result["missing_count"] == 1
    assert result["missing_hotwords"] == ["张三"]
    assert result["invalid_count"] == 1


@pytest.mark.asyncio
async def test_hotword_pool_clear_proxies_to_recall(monkeypatch):
    seen: dict[str, object] = {}

    async def fake_clear(*, hotword_pool_id=None):
        seen["hotword_pool_id"] = hotword_pool_id
        return {"status": "ok", "action": "clear", "cleared": 2, "total_count": 0}

    monkeypatch.setattr(main_mod, "clear_hotword_pool", fake_clear)

    result = await main_mod.asr_hotword_pool_clear(
        body={"hotword_pool_id": "tenant-a"},
    )

    assert seen["hotword_pool_id"] == "tenant-a"
    assert result == {"status": "ok", "action": "clear", "cleared": 2, "total_count": 0}


@pytest.mark.asyncio
async def test_hotword_pool_clear_rejects_query_body_mismatch(monkeypatch):
    async def fake_clear(*, hotword_pool_id=None):  # pragma: no cover - must not run
        raise AssertionError("clear should not be called")

    monkeypatch.setattr(main_mod, "clear_hotword_pool", fake_clear)

    with pytest.raises(HTTPException) as exc:
        await main_mod.asr_hotword_pool_clear(
            hotword_pool_id="tenant-a",
            body={"hotword_pool_id": "tenant-b"},
        )

    assert exc.value.status_code == 400
    assert exc.value.detail["code"] == "invalid_hotword_pool_id"


@pytest.mark.asyncio
async def test_hotword_pool_reload_accepts_body_hotword_pool_id(monkeypatch):
    seen: dict[str, object] = {}

    async def fake_reload(*, hotword_pool_id=None):
        seen["hotword_pool_id"] = hotword_pool_id
        return {"status": "ok", "action": "reload", "total_count": 3}

    monkeypatch.setattr(main_mod, "reload_hotword_pool", fake_reload)

    result = await main_mod.asr_hotword_pool_reload(
        body={"hotword_pool_id": "tenant-a"},
    )

    assert seen["hotword_pool_id"] == "tenant-a"
    assert result == {"status": "ok", "action": "reload", "total_count": 3}


@pytest.mark.asyncio
async def test_hotword_pool_reload_accepts_query_hotword_pool_id(monkeypatch):
    seen: dict[str, object] = {}

    async def fake_reload(*, hotword_pool_id=None):
        seen["hotword_pool_id"] = hotword_pool_id
        return {"status": "ok"}

    monkeypatch.setattr(main_mod, "reload_hotword_pool", fake_reload)

    result = await main_mod.asr_hotword_pool_reload(hotword_pool_id="tenant-a")

    assert seen["hotword_pool_id"] == "tenant-a"
    assert result == {"status": "ok"}


@pytest.mark.asyncio
async def test_hotword_pool_reload_rejects_query_body_mismatch(monkeypatch):
    async def fake_reload(*, hotword_pool_id=None):  # pragma: no cover - must not run
        raise AssertionError("reload should not be called")

    monkeypatch.setattr(main_mod, "reload_hotword_pool", fake_reload)

    with pytest.raises(HTTPException) as exc:
        await main_mod.asr_hotword_pool_reload(
            hotword_pool_id="tenant-a",
            body={"hotword_pool_id": "tenant-b"},
        )

    assert exc.value.status_code == 400
    assert exc.value.detail["code"] == "invalid_hotword_pool_id"


@pytest.mark.asyncio
async def test_hotword_pool_rejects_empty_payload():
    with pytest.raises(HTTPException) as exc:
        await main_mod.asr_hotword_pool_add({"hotwords": []})
    assert exc.value.status_code == 400
    assert exc.value.detail["code"] == "empty_hotwords"


@pytest.mark.asyncio
async def test_hotword_pool_rejects_invalid_hotword_pool_id():
    with pytest.raises(HTTPException) as exc:
        await main_mod.asr_hotword_pool_add(
            {"hotword_pool_id": "../escape", "hotwords": ["挚音科技"]}
        )
    assert exc.value.status_code == 400
    assert exc.value.detail["code"] == "invalid_hotword_pool_id"


@pytest.mark.asyncio
async def test_hotword_pool_rejects_legacy_user_id():
    with pytest.raises(HTTPException) as exc:
        await main_mod.asr_hotword_pool_add(
            {"user_id": "tenant-a", "hotwords": ["挚音科技"]}
        )
    assert exc.value.status_code == 400
    assert exc.value.detail["code"] == "invalid_hotword_pool_id"
