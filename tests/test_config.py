"""Unit tests for the declarative YAML config loader and endpoint projection.

Covers
------
1. config.yaml loads with ${ENV} interpolation (unset -> empty string).
2. upstreams parse: base_url stripped to service root, max_tokens optional.
3. endpoints parse + (protocol, task) whitelist fail-fast; unknown upstream
   references (endpoint / service / rest) fail-fast at parse time.
4. resolve_endpoint projects: global default + per-endpoint upstream bindings +
   policy (soft) + lock (hard); an asr endpoint with no secondary upstream is
   forced secondary-off even if policy tries to enable it.
5. The text_cleanup bridge re-appends /v1 (un-migrated client expects it).
6. __post_init__ invariants (fusion<-secondary, first_partial clamp) still fire
   on every construction path, and the override / override_client whitelist
   contract is unchanged.
7. default_config is exactly load_config() -> single entry point.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import (  # noqa: E402
    CLIENT_OVERRIDABLE_FIELDS,
    ENDPOINTS,
    VALID_ENDPOINT_COMBINATIONS,
    VALID_PRIMARY_PROMPT_TEMPLATES,
    Config,
    Upstream,
    default_config,
    get_service_upstream,
    load_config,
    load_parsed,
    resolve_endpoint,
)


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _minimal() -> dict:
    """A small but fully valid config used as a base for targeted mutations."""
    return {
        "http_pool": {"max_connections": 8, "max_keepalive_connections": 4},
        "upstreams": {
            "p": {"base_url": "http://p:1", "model_name": "P", "timeout": 11},
            "s": {"base_url": "http://s:2", "model_name": "S", "timeout": 12},
            "e": {"base_url": "http://e:3", "model_name": "E", "timeout": 13},
            "clean": {
                "base_url": "http://c:4",
                "model_name": "C",
                "api_key": "k",
                "timeout": 14,
                "max_tokens": 99,
            },
        },
        "defaults": {
            "asr": {"enable_secondary_asr": True, "enable_dual_asr_fusion": False},
            "vad": {"vad_threshold": 0.6, "min_segment_duration_ms": 350},
            "pseudo_stream": {"pseudo_stream_first_partial_ms": 200},
        },
        "services": {"text_cleanup": "clean"},
        "rest": {"upstreams": {"primary": "p", "secondary": "s", "emotion": "e"}},
        "endpoints": [
            {
                "path": "/x",
                "protocol": "native",
                "task": "asr",
                "upstreams": {"primary": "p", "secondary": "s"},
            }
        ],
    }


# --------------------------------------------------------------------------- #
# Loading + ${ENV} interpolation
# --------------------------------------------------------------------------- #


def test_env_interpolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("T_URL", "http://up:9")
    monkeypatch.setenv("T_KEY", "secret")
    data = _minimal()
    data["upstreams"]["p"]["base_url"] = "${T_URL}"
    data["upstreams"]["clean"]["api_key"] = "${T_KEY}"
    parsed = load_parsed(_write_yaml(tmp_path, data))
    assert parsed.upstreams["p"].base_url == "http://up:9"
    assert parsed.upstreams["clean"].api_key == "secret"


def test_env_unset_interpolates_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("T_MISSING", raising=False)
    data = _minimal()
    data["upstreams"]["clean"]["api_key"] = "${T_MISSING}"
    parsed = load_parsed(_write_yaml(tmp_path, data))
    assert parsed.upstreams["clean"].api_key == ""


def test_config_path_env_overrides_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_yaml(tmp_path, _minimal())
    monkeypatch.setenv("CONFIG_PATH", str(path))
    cfg = load_config()  # no explicit path -> reads CONFIG_PATH
    assert cfg.vllm_base_url == "http://p:1"


# --------------------------------------------------------------------------- #
# upstream parsing
# --------------------------------------------------------------------------- #


def test_upstream_base_url_stripped_and_max_tokens(tmp_path: Path) -> None:
    data = _minimal()
    data["upstreams"]["p"]["base_url"] = "http://p:1/"  # trailing slash
    parsed = load_parsed(_write_yaml(tmp_path, data))
    assert parsed.upstreams["p"].base_url == "http://p:1"
    assert parsed.upstreams["p"].prompt_template == "amphion_asr"
    assert parsed.upstreams["clean"].max_tokens == 99
    assert parsed.upstreams["p"].max_tokens is None
    assert isinstance(parsed.upstreams["p"], Upstream)


def test_unknown_primary_prompt_template_fails(tmp_path: Path) -> None:
    data = _minimal()
    data["upstreams"]["p"]["prompt_template"] = "unknown_template"
    with pytest.raises(ValueError, match="prompt_template"):
        load_parsed(_write_yaml(tmp_path, data))


def test_prompt_template_validation_applies_to_config_construction() -> None:
    with pytest.raises(ValueError, match="vllm_prompt_template"):
        Config(vllm_prompt_template="unknown_template")
    with pytest.raises(ValueError, match="astv3_vllm_prompt_template"):
        Config(astv3_vllm_prompt_template="unknown_template")
    with pytest.raises(ValueError, match="astv3_vllm_prompt_template"):
        load_config().override(astv3_vllm_prompt_template="unknown_template")


# --------------------------------------------------------------------------- #
# global (REST-bound) projection
# --------------------------------------------------------------------------- #


def test_rest_projection_maps_roles_to_fields(tmp_path: Path) -> None:
    parsed = load_parsed(_write_yaml(tmp_path, _minimal()))
    cfg = parsed.default_config
    assert cfg.vllm_base_url == "http://p:1"
    assert cfg.vllm_model_name == "P"
    assert cfg.vllm_prompt_template == "amphion_asr"
    assert cfg.asr_request_timeout == 11  # primary upstream timeout
    assert cfg.secondary_vllm_base_url == "http://s:2"
    assert cfg.emotion_vllm_base_url == "http://e:3"
    assert cfg.emotion_request_timeout == 13
    assert cfg.http_max_connections == 8
    assert cfg.http_max_keepalive_connections == 4


def test_text_cleanup_bridge_appends_v1(tmp_path: Path) -> None:
    """The un-migrated text_cleanup client expects base to contain /v1; the
    projection re-appends it onto the service-root upstream base_url."""
    parsed = load_parsed(_write_yaml(tmp_path, _minimal()))
    cfg = parsed.default_config
    assert cfg.text_cleanup_base_url == "http://c:4/v1"
    assert cfg.text_cleanup_model_name == "C"
    assert cfg.text_cleanup_api_key == "k"
    assert cfg.text_cleanup_api_key_env == ""
    assert cfg.text_cleanup_timeout == 14
    assert cfg.text_cleanup_max_tokens == 99


def test_shipped_config_projects_rest_bindings() -> None:
    cfg = load_config()  # reads the shipped config.yaml
    assert cfg.vllm_base_url == "http://localhost:8009"
    assert cfg.vllm_model_name == "AmphionASR-1.7B"
    assert cfg.vllm_prompt_template == "amphion_asr_1.7b"
    assert cfg.vllm_prompt_template in VALID_PRIMARY_PROMPT_TEMPLATES
    assert cfg.secondary_vllm_base_url == "http://localhost:8001"
    assert cfg.emotion_vllm_base_url == "http://localhost:8222"
    assert cfg.emotion_spec_vllm_base_url == "http://localhost:9001"
    assert cfg.asr_request_timeout == 120
    assert cfg.emotion_request_timeout == 30
    assert cfg.vad_threshold == 0.65
    assert cfg.min_segment_duration_ms == 350
    assert cfg.pseudo_stream_first_partial_ms == 200
    assert cfg.enable_dual_asr_fusion is False
    assert cfg.enable_secondary_asr is True
    assert cfg.emotion_task_mode == "ser"
    assert cfg.emotion_spec_task_mode == "sepc"
    assert cfg.enable_asr_repetition_fix is True
    assert cfg.k2_enabled is True
    assert cfg.k2_target == "localhost:50051"
    assert cfg.k2_include_token_timestamps is False
    assert cfg.k2_max_segment_sec == 30.0
    assert cfg.k2_idle_keep_ms == 1500
    assert cfg.k2_voice_gate_enabled is True
    assert cfg.k2_voice_gate_threshold == 0.65
    assert cfg.k2_voice_gate_start_frames == 10
    assert cfg.asr_segment_voice_gate_enabled is True
    assert cfg.asr_segment_voice_gate_threshold == 0.65
    assert cfg.asr_segment_voice_gate_min_ratio == 0.05
    assert cfg.asr_segment_voice_gate_min_ms == 120
    assert cfg.asr_segment_voice_gate_min_rms == 0.001
    assert cfg.debug_dump_enabled is False
    assert cfg.debug_dump_dir == "debug_dumps"
    assert (
        cfg.text_cleanup_base_url
        == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    assert cfg.http_max_connections == 32


# --------------------------------------------------------------------------- #
# endpoint parsing + validation
# --------------------------------------------------------------------------- #


def test_real_endpoints_parsed() -> None:
    paths = {e.path for e in ENDPOINTS}
    assert {
        "/transcribe-streaming",
        "/tuling/ast/v3",
        "/emotion-segmented-streaming",
    } <= paths
    tuling = next(e for e in ENDPOINTS if e.path == "/tuling/ast/v3")
    assert tuling.protocol == "astv3" and tuling.task == "asr"
    assert tuling.lock == {"enable_secondary_asr": False}
    ts = next(e for e in ENDPOINTS if e.path == "/transcribe-streaming")
    assert ts.client_overridable is True


def test_valid_combinations_constant() -> None:
    assert ("native", "asr") in VALID_ENDPOINT_COMBINATIONS
    assert ("astv3", "asr") in VALID_ENDPOINT_COMBINATIONS
    assert ("astv3", "browser_demo") not in VALID_ENDPOINT_COMBINATIONS


def test_invalid_protocol_task_combo_fails(tmp_path: Path) -> None:
    data = _minimal()
    data["endpoints"] = [
        {
            "path": "/bad",
            "protocol": "astv3",
            "task": "browser_demo",
            "upstreams": {"primary": "p"},
        }
    ]
    with pytest.raises(ValueError, match="invalid"):
        load_parsed(_write_yaml(tmp_path, data))


def test_unknown_endpoint_upstream_fails(tmp_path: Path) -> None:
    data = _minimal()
    data["endpoints"] = [
        {
            "path": "/x",
            "protocol": "native",
            "task": "asr",
            "upstreams": {"primary": "ghost"},
        }
    ]
    with pytest.raises(ValueError, match="unknown upstream"):
        load_parsed(_write_yaml(tmp_path, data))


def test_unknown_endpoint_role_fails(tmp_path: Path) -> None:
    data = _minimal()
    data["endpoints"] = [
        {
            "path": "/x",
            "protocol": "native",
            "task": "asr",
            "upstreams": {"tertiary": "p"},
        }
    ]
    with pytest.raises(ValueError, match="unknown upstream role"):
        load_parsed(_write_yaml(tmp_path, data))


def test_unknown_service_upstream_fails(tmp_path: Path) -> None:
    data = _minimal()
    data["services"] = {"text_cleanup": "ghost"}
    with pytest.raises(ValueError, match="unknown upstream"):
        load_parsed(_write_yaml(tmp_path, data))


def test_unknown_rest_upstream_fails(tmp_path: Path) -> None:
    data = _minimal()
    data["rest"]["upstreams"]["primary"] = "ghost"
    with pytest.raises(ValueError, match="unknown upstream"):
        load_parsed(_write_yaml(tmp_path, data))


# --------------------------------------------------------------------------- #
# resolve_endpoint projection
# --------------------------------------------------------------------------- #


def test_resolve_binds_per_endpoint_primary(tmp_path: Path) -> None:
    data = _minimal()
    data["upstreams"]["other"] = {
        "base_url": "http://other:9",
        "model_name": "OTHER",
        "prompt_template": "amphion_asr_1.7b",
        "timeout": 7,
    }
    data["endpoints"] = [
        {
            "path": "/b",
            "protocol": "native",
            "task": "asr",
            "upstreams": {"primary": "other", "secondary": "s"},
        }
    ]
    parsed = load_parsed(_write_yaml(tmp_path, data))
    cfg = resolve_endpoint(parsed.endpoints[0], parsed)
    assert cfg.vllm_base_url == "http://other:9"
    assert cfg.vllm_model_name == "OTHER"
    assert cfg.vllm_prompt_template == "amphion_asr_1.7b"
    assert cfg.asr_request_timeout == 7
    # secondary still bound from this endpoint's own upstreams
    assert cfg.secondary_vllm_base_url == "http://s:2"


def test_resolve_no_secondary_forces_off_even_if_policy_enables(
    tmp_path: Path,
) -> None:
    data = _minimal()
    data["endpoints"] = [
        {
            "path": "/p",
            "protocol": "native",
            "task": "asr",
            "upstreams": {"primary": "p"},
            "policy": {"enable_secondary_asr": True},
        }
    ]
    parsed = load_parsed(_write_yaml(tmp_path, data))
    cfg = resolve_endpoint(parsed.endpoints[0], parsed)
    assert cfg.enable_secondary_asr is False
    assert cfg.enable_dual_asr_fusion is False  # invariant downgrade


def test_resolve_policy_soft_applies(tmp_path: Path) -> None:
    data = _minimal()
    data["endpoints"] = [
        {
            "path": "/p",
            "protocol": "native",
            "task": "asr",
            "upstreams": {"primary": "p", "secondary": "s"},
            "policy": {"vad_threshold": 0.91},
        }
    ]
    parsed = load_parsed(_write_yaml(tmp_path, data))
    cfg = resolve_endpoint(parsed.endpoints[0], parsed)
    assert cfg.vad_threshold == 0.91


def test_resolve_real_tuling_primary_only() -> None:
    spec = next(e for e in ENDPOINTS if e.path == "/tuling/ast/v3")
    cfg = resolve_endpoint(spec)
    assert cfg.vllm_base_url == "http://localhost:8009"  # amphion_asr (primary)
    assert cfg.vllm_prompt_template == "amphion_asr_1.7b"
    assert cfg.enable_secondary_asr is False  # lock + no secondary binding
    assert cfg.enable_dual_asr_fusion is False


def test_get_service_upstream() -> None:
    assert get_service_upstream("text_cleanup").name == "dashscope_cleanup"
    assert get_service_upstream("hotword").name == "hotword_llm"
    assert get_service_upstream("recall").name == "triton_recall"
    assert get_service_upstream("nonexistent") is None


# --------------------------------------------------------------------------- #
# __post_init__ invariants (every construction path)
# --------------------------------------------------------------------------- #


def test_fusion_requires_secondary_invariant() -> None:
    cfg = Config(enable_secondary_asr=False, enable_dual_asr_fusion=True)
    assert cfg.enable_dual_asr_fusion is False


def test_encoder_bypass_requires_recall_invariant() -> None:
    cfg = Config(enable_hotword_recall=False, enable_encoder_bypass=True)
    assert cfg.enable_encoder_bypass is False


def test_triton_enrollment_store_defaults_off_and_server_side() -> None:
    cfg = Config()
    assert cfg.enable_triton_enrollment_store is False
    assert cfg.enable_enrollment_embedding_bypass is True
    assert "enable_triton_enrollment_store" not in CLIENT_OVERRIDABLE_FIELDS
    assert "enable_enrollment_embedding_bypass" not in CLIENT_OVERRIDABLE_FIELDS


def test_recall_top_k_clamps_to_non_negative() -> None:
    cfg = Config(recall_top_k=-1)
    assert cfg.recall_top_k == 0


def test_recall_custom_hotword_limit_is_server_side_and_non_negative() -> None:
    cfg = Config(recall_custom_hotword_limit=-1)
    assert cfg.recall_custom_hotword_limit == 0
    assert "recall_custom_hotword_limit" not in CLIENT_OVERRIDABLE_FIELDS


def test_pseudo_stream_first_partial_dataclass_default_is_neutral() -> None:
    cfg = Config()
    assert cfg.pseudo_stream_first_partial_ms == cfg.min_segment_duration_ms == 350


def test_pseudo_stream_first_partial_clamped_to_min_segment() -> None:
    clamped = Config(pseudo_stream_first_partial_ms=500, min_segment_duration_ms=350)
    assert clamped.pseudo_stream_first_partial_ms == 350
    lower = Config(pseudo_stream_first_partial_ms=200, min_segment_duration_ms=350)
    assert lower.pseudo_stream_first_partial_ms == 200


def test_debug_dump_dataclass_defaults_off() -> None:
    cfg = Config()
    assert cfg.debug_dump_enabled is False
    assert cfg.debug_dump_dir == "debug_dumps"


def test_debug_dump_empty_dir_falls_back_to_default() -> None:
    assert Config(debug_dump_dir="   ").debug_dump_dir == "debug_dumps"


def test_debug_dump_not_client_overridable() -> None:
    # A client must not be able to turn on disk writes; override_client drops it.
    cfg = Config(debug_dump_enabled=False)
    overridden = cfg.override_client(
        debug_dump_enabled=True, debug_dump_dir="/tmp/evil"
    )
    assert overridden.debug_dump_enabled is False
    assert overridden.debug_dump_dir == "debug_dumps"


def test_k2_requires_target_and_clamps_bounds() -> None:
    cfg = Config(
        k2_enabled=True,
        k2_target="",
        k2_sample_rate=0,
        k2_max_segment_sec=-1,
        k2_idle_keep_ms=-20,
        k2_voice_gate_threshold=2.0,
        k2_voice_gate_start_frames=0,
    )
    assert cfg.k2_enabled is False
    assert cfg.k2_sample_rate == 16000
    assert cfg.k2_max_segment_sec == 0.0
    assert cfg.k2_idle_keep_ms == 0
    assert cfg.k2_voice_gate_threshold == 1.0
    assert cfg.k2_voice_gate_start_frames == 1


def test_asr_segment_voice_gate_clamps_bounds() -> None:
    high = Config(
        asr_segment_voice_gate_threshold=2.0,
        asr_segment_voice_gate_min_ratio=2.0,
    )
    low = Config(
        asr_segment_voice_gate_threshold=-1.0,
        asr_segment_voice_gate_min_ratio=-1.0,
        asr_segment_voice_gate_min_ms=-1,
        asr_segment_voice_gate_min_rms=-1.0,
    )
    assert high.asr_segment_voice_gate_threshold == 1.0
    assert high.asr_segment_voice_gate_min_ratio == 1.0
    assert low.asr_segment_voice_gate_threshold == 0.0
    assert low.asr_segment_voice_gate_min_ratio == 0.0
    assert low.asr_segment_voice_gate_min_ms == 0
    assert low.asr_segment_voice_gate_min_rms == 0.0


# --------------------------------------------------------------------------- #
# override / override_client whitelist contract
# --------------------------------------------------------------------------- #


def test_override_uses_flat_keys_only() -> None:
    cfg = load_config()
    overridden = cfg.override(vad_threshold=0.45)
    assert overridden.vad_threshold == 0.45
    untouched = cfg.override(**{"asr.vad.vad_threshold": 0.1})
    assert untouched.vad_threshold == cfg.vad_threshold


def test_override_primary_does_not_touch_secondary() -> None:
    cfg = load_config()
    out = cfg.override(vllm_base_url="http://pub:8000", vllm_model_name="Amphion-4B")
    assert out.vllm_base_url == "http://pub:8000"
    assert out.vllm_model_name == "Amphion-4B"
    assert out.secondary_vllm_base_url == cfg.secondary_vllm_base_url
    assert out.secondary_vllm_model_name == cfg.secondary_vllm_model_name


def test_override_secondary_off_downgrades_fusion() -> None:
    cfg = load_config().override(enable_secondary_asr=True, enable_dual_asr_fusion=True)
    out = cfg.override(enable_secondary_asr=False)
    assert out.enable_secondary_asr is False
    assert out.enable_dual_asr_fusion is False


def test_pseudo_stream_first_partial_client_overridable() -> None:
    assert "pseudo_stream_first_partial_ms" in CLIENT_OVERRIDABLE_FIELDS
    out = load_config().override_client(pseudo_stream_first_partial_ms=200)
    assert out.pseudo_stream_first_partial_ms == 200


def test_recall_knobs_client_overridable() -> None:
    assert {"enable_hotword_recall", "recall_top_k"} <= CLIENT_OVERRIDABLE_FIELDS
    out = load_config().override_client(recall_top_k=3)
    assert out.recall_top_k == 3


def test_role_separation_switch_is_compatibility_only() -> None:
    cfg = load_config()
    assert cfg.enable_role_separation is False
    assert "enable_role_separation" in CLIENT_OVERRIDABLE_FIELDS

    out = cfg.override_client(enable_role_separation=True)
    assert out.enable_role_separation is True


def test_asr_segment_voice_gate_client_overridable() -> None:
    fields = {
        "asr_segment_voice_gate_enabled",
        "asr_segment_voice_gate_threshold",
        "asr_segment_voice_gate_min_ratio",
        "asr_segment_voice_gate_min_ms",
        "asr_segment_voice_gate_min_rms",
    }
    assert fields <= CLIENT_OVERRIDABLE_FIELDS
    out = load_config().override_client(
        asr_segment_voice_gate_enabled=False,
        asr_segment_voice_gate_threshold=0.7,
        asr_segment_voice_gate_min_ratio=0.1,
        asr_segment_voice_gate_min_ms=200,
        asr_segment_voice_gate_min_rms=0.002,
    )
    assert out.asr_segment_voice_gate_enabled is False
    assert out.asr_segment_voice_gate_threshold == 0.7
    assert out.asr_segment_voice_gate_min_ratio == 0.1
    assert out.asr_segment_voice_gate_min_ms == 200
    assert out.asr_segment_voice_gate_min_rms == 0.002


def test_hotword_pool_id_default_and_validation() -> None:
    cfg = load_config()
    assert cfg.hotword_pool_id == "default"
    out = cfg.override(hotword_pool_id="tenant-a")
    assert out.hotword_pool_id == "tenant-a"
    with pytest.raises(ValueError, match="HOTWORD_POOL_ID"):
        cfg.override(hotword_pool_id="../escape")


def test_pseudo_stream_first_partial_clamp_applies_on_override() -> None:
    cfg = load_config().override(min_segment_duration_ms=300)
    out = cfg.override_client(pseudo_stream_first_partial_ms=400)
    assert out.pseudo_stream_first_partial_ms == 300


def test_override_client_allows_whitelisted_fields() -> None:
    cfg = load_config()
    out = cfg.override_client(vad_threshold=0.42, enable_pseudo_stream=False)
    assert out.vad_threshold == 0.42
    assert out.enable_pseudo_stream is False


def test_override_client_drops_non_whitelisted_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = load_config()
    with caplog.at_level(logging.WARNING, logger="backend.config"):
        out = cfg.override_client(
            vllm_base_url="http://evil:1",
            text_cleanup_api_key="leaked",
            vad_threshold=0.42,
        )
    assert out.vad_threshold == 0.42
    assert out.vllm_base_url == cfg.vllm_base_url
    assert out.text_cleanup_api_key == cfg.text_cleanup_api_key
    assert any("non-overridable config field" in r.message for r in caplog.records)


def test_override_client_ignores_invalid_value() -> None:
    cfg = load_config()
    out = cfg.override_client(vad_threshold="not-a-number")
    assert out.vad_threshold == cfg.vad_threshold


def test_override_client_enforces_fusion_invariant() -> None:
    cfg = load_config().override(enable_secondary_asr=True, enable_dual_asr_fusion=True)
    out = cfg.override_client(enable_secondary_asr=False, enable_dual_asr_fusion=True)
    assert out.enable_secondary_asr is False
    assert out.enable_dual_asr_fusion is False


def test_client_overridable_fields_are_real_and_safe() -> None:
    names = {f.name for f in dataclasses.fields(Config)}
    assert CLIENT_OVERRIDABLE_FIELDS <= names
    forbidden = {
        "vllm_base_url",
        "vllm_prompt_template",
        "astv3_vllm_base_url",
        "astv3_vllm_model_name",
        "astv3_vllm_prompt_template",
        "secondary_vllm_base_url",
        "emotion_vllm_base_url",
        "emotion_spec_vllm_base_url",
        "text_cleanup_base_url",
        "text_cleanup_api_key",
        "text_cleanup_api_key_env",
        "http_max_connections",
        "http_max_keepalive_connections",
        "enable_asr_repetition_fix",
        "enable_encoder_bypass",
        "k2_target",
        "k2_enabled",
        "k2_max_segment_sec",
    }
    assert CLIENT_OVERRIDABLE_FIELDS.isdisjoint(forbidden)


def test_vad_end_frames_field_removed() -> None:
    names = {f.name for f in dataclasses.fields(Config)}
    assert "vad_end_frames" not in names
    assert "vad_end_frames" not in CLIENT_OVERRIDABLE_FIELDS


def test_default_config_is_load_config_snapshot() -> None:
    assert dataclasses.asdict(default_config) == dataclasses.asdict(load_config())
