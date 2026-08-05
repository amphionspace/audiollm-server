from __future__ import annotations

from fastapi.testclient import TestClient

import backend.main as main_mod
from backend.main import app


def test_healthz_returns_liveness_status():
    client = TestClient(app)

    resp = client.get("/healthz")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readyz_returns_ok_when_configured_upstreams_are_ready(monkeypatch):
    openai_checks: list[str] = []

    async def fake_openai(name, upstream):
        openai_checks.append(name)
        return {
            "name": name,
            "kind": "openai_compatible",
            "target": f"{upstream.base_url}/v1/models",
            "status": "ok",
        }

    async def fake_triton(upstream):
        return {
            "name": "rag_asr_triton",
            "kind": "triton",
            "target": f"{upstream.base_url}/v2/health/ready",
            "status": "ok",
        }

    async def fake_management(upstream):
        return {
            "name": "rag_asr_management",
            "kind": "http",
            "target": "",
            "status": "skipped",
        }

    async def fake_k2(cfg):
        return {
            "name": "k2",
            "kind": "grpc",
            "target": cfg.k2_target,
            "status": "ok",
        }

    async def fake_diarization(cfg):
        return {
            "name": "diarization",
            "kind": "grpc",
            "target": cfg.diarization_target,
            "status": "ok",
            "required": False,
        }

    monkeypatch.setattr(main_mod, "_check_openai_upstream", fake_openai)
    monkeypatch.setattr(main_mod, "_check_triton_recall", fake_triton)
    monkeypatch.setattr(main_mod, "_check_rag_management", fake_management)
    monkeypatch.setattr(main_mod, "_check_k2_ready", fake_k2)
    monkeypatch.setattr(main_mod, "_check_diarization_ready", fake_diarization)

    client = TestClient(app)
    resp = client.get("/readyz")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "rest.primary:amphion_asr" in openai_checks
    assert "rest.secondary:qwen_asr" in openai_checks
    assert "rest.emotion:amphion_emotion" in openai_checks
    assert "rest.emotion_spec:amphion_spec" in openai_checks
    assert {check["name"] for check in body["checks"]} >= {
        "rag_asr_triton",
        "rag_asr_management",
        "k2",
        "diarization",
    }


def test_readyz_returns_503_when_a_configured_upstream_fails(monkeypatch):
    async def fake_openai(name, upstream):
        status = "error" if name == "rest.primary:amphion_asr" else "ok"
        detail = "connection refused" if status == "error" else ""
        check = {
            "name": name,
            "kind": "openai_compatible",
            "target": f"{upstream.base_url}/v1/models",
            "status": status,
        }
        if detail:
            check["detail"] = detail
        return check

    async def fake_ok(name, kind):
        return {"name": name, "kind": kind, "target": "", "status": "ok"}

    async def fake_diarization(_cfg):
        check = await fake_ok("diarization", "grpc")
        check["required"] = False
        return check

    monkeypatch.setattr(main_mod, "_check_openai_upstream", fake_openai)
    monkeypatch.setattr(
        main_mod,
        "_check_triton_recall",
        lambda upstream: fake_ok("rag_asr_triton", "triton"),
    )
    monkeypatch.setattr(
        main_mod,
        "_check_rag_management",
        lambda upstream: fake_ok("rag_asr_management", "http"),
    )
    monkeypatch.setattr(main_mod, "_check_k2_ready", lambda cfg: fake_ok("k2", "grpc"))
    monkeypatch.setattr(
        main_mod,
        "_check_diarization_ready",
        fake_diarization,
    )

    client = TestClient(app)
    resp = client.get("/readyz")

    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "error"
    assert any(
        check["name"] == "rest.primary:amphion_asr"
        and check["status"] == "error"
        and check["detail"] == "connection refused"
        for check in body["checks"]
    )


def test_readyz_reports_optional_diarization_failure_without_503(monkeypatch):
    async def fake_ok(name, kind):
        return {"name": name, "kind": kind, "target": "", "status": "ok"}

    async def fake_diarization(_cfg):
        return {
            "name": "diarization",
            "kind": "grpc",
            "target": "localhost:50052",
            "status": "error",
            "detail": "connection refused",
            "required": False,
        }

    monkeypatch.setattr(
        main_mod,
        "_check_openai_upstream",
        lambda name, _upstream: fake_ok(name, "openai_compatible"),
    )
    monkeypatch.setattr(
        main_mod,
        "_check_triton_recall",
        lambda _upstream: fake_ok("rag_asr_triton", "triton"),
    )
    monkeypatch.setattr(
        main_mod,
        "_check_rag_management",
        lambda _upstream: fake_ok("rag_asr_management", "http"),
    )
    monkeypatch.setattr(
        main_mod, "_check_k2_ready", lambda _cfg: fake_ok("k2", "grpc")
    )
    monkeypatch.setattr(main_mod, "_check_diarization_ready", fake_diarization)

    resp = TestClient(app).get("/readyz")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert any(
        check["name"] == "diarization"
        and check["status"] == "error"
        and check["required"] is False
        for check in body["checks"]
    )


def test_readyz_returns_503_when_config_cannot_load(monkeypatch):
    def fail_load():
        raise ValueError("bad config")

    monkeypatch.setattr(main_mod, "load_parsed", fail_load)

    client = TestClient(app)
    resp = client.get("/readyz")

    assert resp.status_code == 503
    assert resp.json() == {
        "status": "error",
        "checks": [
            {
                "name": "config",
                "kind": "config",
                "target": "config.yaml",
                "status": "error",
                "detail": "bad config",
            }
        ],
    }
