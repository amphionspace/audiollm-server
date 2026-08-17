from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# 配置真源: 项目根 config.yaml; CONFIG_PATH 环境变量可覆盖(测试 / 多部署)。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"

HOP_SIZE = 160  # 10ms at 16kHz, TEN VAD recommended
SAMPLE_RATE = 16000
VALID_PRIMARY_PROMPT_TEMPLATES: frozenset[str] = frozenset(
    {"amphion_asr", "amphion_asr_1.7b"}
)

# ${VAR} 环境变量插值; 未设置 -> 空串(import 期不因缺密钥而崩, 调用时才暴露)。
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _interpolate_env(node: Any) -> Any:
    """Recursively replace ``${VAR}`` references inside parsed YAML strings."""
    if isinstance(node, str):
        return _ENV_REF.sub(lambda m: os.getenv(m.group(1), ""), node)
    if isinstance(node, dict):
        return {k: _interpolate_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_interpolate_env(v) for v in node]
    return node


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        logger.warning("Config file not found: %s, using built-in defaults", path)
        return {}
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Config root must be a mapping, got {type(raw).__name__}: {path}"
        )
    return _interpolate_env(raw)


def _flatten_leaves(node: dict[str, Any]) -> dict[str, Any]:
    """Collect non-dict leaves from the feature-grouped ``defaults`` block.

    Nested groups (vad / asr / fusion / ...) are visual grouping only; their
    leaves become flat ``Config`` field candidates. A leaf key colliding across
    groups is a config error: warn and keep the last value (never silent).
    """
    flat: dict[str, Any] = {}

    def walk(n: dict[str, Any]) -> None:
        for key, value in n.items():
            if isinstance(value, dict):
                walk(value)
                continue
            if key in flat:
                logger.warning("Duplicate config key across groups: %s", key)
            flat[key] = value

    walk(node)
    return flat


@dataclass(frozen=True)
class Upstream:
    """A named downstream service instance (vLLM or external OpenAI-compatible API).

    ``base_url`` is the service root WITHOUT ``/v1``; the unified call layer
    appends ``/v1/chat/completions``. ``api_key`` is already env-interpolated
    plaintext (empty -> no auth header). ``timeout`` is the per-request HTTP
    timeout in seconds. ``max_tokens`` is an optional per-backend response cap.
    ``prompt_template`` is a model-level ASR prompt format and is only projected
    for the primary ASR role.
    """

    name: str
    base_url: str
    model_name: str
    api_key: str = ""
    timeout: float = 120.0
    max_tokens: int | None = None
    prompt_template: str = "amphion_asr"


@dataclass(frozen=True)
class Config:
    # ---- ASR: primary / secondary vLLM endpoints --------------------------
    vllm_base_url: str = "http://localhost:8000"
    vllm_model_name: str = "Amphion/AmphionASR-4.3B"
    vllm_prompt_template: str = "amphion_asr"
    split_asr_enabled: bool = False
    split_asr_base_url: str = ""
    split_asr_final_only: bool = False
    secondary_vllm_base_url: str = "http://localhost:8001"
    secondary_vllm_model_name: str = "Qwen/Qwen3-ASR-1.7B"
    enable_secondary_asr: bool = True
    enable_primary_asr: bool = True
    primary_asr_timeout: float = 4.0
    debug_show_dual_asr: bool = True
    # `enable_secondary_asr` is the resource gate: when False the secondary
    # vLLM is never queried (no partial noise gate, no final fusion).
    # `enable_dual_asr_fusion` controls only the *final-segment* path:
    # True  -> run primary + secondary in parallel and merge via
    #          `choose_fused_result` (requires secondary to be on);
    # False -> final is primary-only even if secondary is reachable,
    #          which saves one vLLM call per segment while keeping the
    #          partial noise gate functional. Inconsistent combinations
    #          (fusion=True, secondary=False) are downgraded at load time.
    enable_dual_asr_fusion: bool = True

    # ---- ASR: per-endpoint primary override (/tuling/ast/v3) -------------
    # The AST v3 endpoint serves a *different* primary model than the global
    # ``vllm_base_url``. Empty string means "fall back to the global primary";
    # a non-empty value routes only the ``/tuling/ast/v3`` endpoint's primary
    # to this upstream (binding wired in ``backend/main.py``). These stay out
    # of ``CLIENT_OVERRIDABLE_FIELDS`` — model URLs are SSRF-sensitive. That
    # endpoint also runs primary-only: secondary is force-disabled at the
    # route, so the local Qwen is never queried there.
    astv3_vllm_base_url: str = ""
    astv3_vllm_model_name: str = ""
    astv3_vllm_prompt_template: str = ""

    # ---- ASR: dual-model fusion knobs ------------------------------------
    fusion_similarity_threshold: float = 0.85
    fusion_min_primary_score: float = 0.55
    fusion_max_repetition_ratio: float = 0.35
    fusion_disagreement_threshold: float = 0.55
    fusion_hotword_boost: float = 0.12
    fusion_primary_score_margin: float = 0.08

    # ---- ASR: Triton hotword recall ---------------------------------------
    # RAG-ASR owns the large process-wide hotword pool and returns top-K words
    # per VAD final segment. The split ASR path receives only recalled words.
    enable_hotword_recall: bool = False
    recall_top_k: int = 50
    recall_custom_hotword_limit: int = 50
    recall_max_concurrent: int = 4
    recall_queue_timeout_ms: int = 200

    # ---- ASR: inverse text normalization (ITN) + license plate -----------
    # The model emits spoken-form text (六五四三八, 二零二四年); for display we
    # normalize finals to written form. Two independent switches (final only —
    # partials stay spoken-form to avoid flicker):
    #   enable_asr_itn             -> general ITN via wetext (Chinese only)
    #   enable_asr_plate_normalize -> zero-dep plate pass: uppercase plate
    #                                 letters, strip in-plate separators, map
    #                                 spoken digits, GB-plate-shape validated.
    # A province abbreviation misheard as a Latin letter (冀->J) is a
    # recognition error and is intentionally NOT recovered here.
    enable_asr_itn: bool = True
    asr_itn_enable_0_to_9: bool = False
    enable_asr_plate_normalize: bool = True
    enable_asr_repetition_fix: bool = True

    # ---- Common: HTTP / pseudo-streaming ---------------------------------
    asr_request_timeout: float = 120
    asr_repetition_penalty: float = 1.0
    enable_pseudo_stream: bool = True
    pseudo_stream_interval_ms: int = 500
    asr_first_partial_max_concurrent: int = 24
    asr_partial_refresh_max_concurrent: int = 4
    asr_partial_refresh_queue_timeout_ms: int = 0
    asr_partial_first_max_tokens: int = 64
    asr_partial_refresh_max_tokens: int = 128
    asr_final_max_tokens: int = 256
    asr_final_max_concurrent: int = 20
    asr_final_executor_workers: int = 20
    asr_stop_flush_max_concurrent: int = 20
    asr_stop_flush_executor_workers: int = 20
    # Random jitter applied before vLLM dispatch for stop-flush segments.
    # When BS52 streams all end simultaneously, 52 stop-flush requests arrive
    # at once. A uniform random delay in [0, jitter_ms] spreads them across a
    # time window, reducing KV-cache burst pressure on vLLM.
    # 0 = disabled (original behavior, fully backward-compatible).
    asr_stop_flush_jitter_ms: int = 0
    # Max number of final-ASR tasks that one session may dispatch in parallel.
    # 1 = legacy serial behaviour; 2 = allows the next segment's encode to
    # overlap with the current segment's vLLM decode, reducing per-session
    # queue wait under BS52 high-segment-density conditions.
    asr_final_session_parallel: int = 2
    asr_first_partial_priority_ms: int = 0
    # Partial backend selection:
    #   vllm   -> existing pseudo-streaming path via primary/secondary ASR
    #   sherpa -> local sherpa-onnx streaming recognizer for partials only;
    #             final/stop_flush remain on the normal ASR path.
    #   k2_om  -> local Ascend OM online transducer runtime for partials.
    streaming_partial_backend: str = "vllm"
    sherpa_model_dir: str = ""
    sherpa_provider: str = "cpu"
    sherpa_num_threads: int = 2
    sherpa_executor_workers: int = 4
    sherpa_blank_penalty: float = 0.0
    sherpa_use_int8: bool = True
    sherpa_runtime_pool_size: int = 1
    k2_om_model_dir: str = ""
    k2_om_soc_version: str = ""
    k2_om_runtime_pool_size: int = 1
    k2_partial_first_text_interval_ms: int = 500
    k2_partial_post_text_interval_ms: int = 3000
    k2_decode_batch_wait_ms: int = 0
    k2_decode_batch_max_size: int = 16
    k2_online_scheduler_enabled: bool = False
    k2_online_result_wait_ms: int = 0
    k2_online_result_wait_min_audio_ms: int = 800
    k2_online_ready_coalesce_ms: int = 0
    k2_first_partial_decode_max_steps: int = 0
    k2_partial_decode_max_steps: int = 0
    ctc_om_model_dir: str = ""
    ctc_om_model_path: str = ""
    ctc_onnx_model_path: str = ""
    ctc_tokens_path: str = ""
    ctc_decode_batch_size: int = 52
    ctc_decode_wait_ms: int = 10
    ctc_ready_coalesce_ms: int = 30
    ctc_result_wait_ms: int = 120
    # Keep the 116 CTC OM state tensors resident on the NPU across decode ticks
    # (fixed per-stream device slot + device-to-device state_b->state_a) instead
    # of the per-tick ~126MB host round-trip that bottlenecks the decode loop
    # under BS52 (design F4/F5). Default false (gated rollout); the legacy host
    # round-trip path remains the fallback.
    ctc_device_resident_state: bool = False
    # Device-resident decode pacing (ms). 0 = auto (~0.75x chunk_shift), <0 =
    # disabled. Batches in-phase streams into fast pure-device ticks.
    ctc_resident_pace_ms: int = 0
    # First-frame priority: skip the ready_coalesce second-wait when a brand-new
    # (never-decoded) stream is already ready, so BS52 burst registration does
    # not inflate first-word latency by up to ready_coalesce_ms per stream.
    ctc_firstframe_bypass_coalesce: bool = True
    # AscendK2Stream prefill warmup cap (ms). The gate-closed prefill CTC
    # session only needs enough audio to warm the encoder state so the first
    # partial fires ~85ms after gate-open. Feeding (and decoding) it every chunk
    # through inter-segment silence keeps up to ~52 discarded-result streams in
    # the CTC decode batch, inflating per-tick state I/O (design F4/F5/F7) and
    # driving the device-resident preserve host-merge storm (F6). When >0, feed
    # the prefill only until it has this many ms of audio, then PARK it (stop
    # feeding -> not is_ready -> dropped from the decode batch) until gate-open,
    # where the pre-speech ring is fed to restore pre-roll continuity.
    # 0 = legacy unbounded feeding (fully backward-compatible).
    ctc_prefill_warmup_ms: int = 0
    ctc_partial_post_first_interval_ms: int = 500
    ctc_prefeed_emit_enabled: bool = True
    # When enabled, CTC partial work is paused during the trailing-silence
    # window after the first partial has already been sent. TEN VAD continues
    # to endpoint the segment; this only prevents non-essential CTC refreshes
    # from competing with the imminent split final request.
    ctc_final_tail_pause_enabled: bool = False
    ctc_final_tail_pause_silence_ms: int = 200
    ctc_device_id: int = 0
    # AscendK2Stream: event-driven CTC partial + acoustic endpoint detection.
    # Replaces VadSegmentedStream on Ascend 910B3/B4 when enabled.
    ascend_k2_enabled: bool = False
    k2_endpoint_silence_sec: float = 1.2
    # VAD-silence endpoint (design F9-A). The default endpoint fires
    # endpoint_silence_sec after the last *CTC token*, so under BS52 the CTC
    # decode backlog (min_ready up to ~3.3s, F7) delays segment close and
    # inflates vad_lag by that backlog, which accumulates through the stream.
    # TEN VAD already runs in feed() in realtime (unaffected by CTC backlog),
    # so tracking silence on the audio timeline lets the endpoint fire relative
    # to the *true* speech end. When enabled, the gate closes on whichever of
    # {VAD silence, CTC token silence} fires first (CTC kept as backstop), which
    # removes the backlog term from vad_lag without changing correctness on
    # natural sentence pauses (>= threshold). Default off (A/B gated rollout).
    k2_vad_endpoint_enabled: bool = False
    # VAD-silence endpoint threshold (sec). 0 = reuse k2_endpoint_silence_sec.
    k2_vad_endpoint_silence_sec: float = 0.0
    k2_max_segment_sec: float = 30.0
    # Delivery target for 10-minute realtime streaming on Ascend 910B:
    # BS52, TTFT around 800 ms, final lag <= 3 s, and visible mid-utterance
    # refresh. Under final pressure, refresh snapshots are lower priority than
    # final segments and self-throttle instead of piling up stale vLLM calls.
    asr_partial_refresh_adaptive_enabled: bool = True
    asr_partial_refresh_pressure_final_backlog: int = 8
    asr_partial_refresh_pressure_interval_ms: int = 3000
    asr_partial_refresh_high_pressure_final_backlog: int = 24
    asr_partial_refresh_high_pressure_interval_ms: int = 8000
    asr_partial_refresh_pressure_feed_queue_depth: int = 256
    asr_partial_refresh_high_pressure_feed_queue_depth: int = 1024
    stream_feed_batch_max_frames: int = 16
    # First-word window now coalesces already-queued backlog (was 1). Measured:
    # under BS52 the feed thread pool (TEN VAD) is the bottleneck, not per-chunk
    # freshness, so draining up to N queued chunks per stream.feed cuts feed-call
    # count and first-word latency (BS52@40ms ttft p50 2393->1288, @200ms
    # 1636->1473) with no accuracy loss. Draining only merges backlog that has
    # ALREADY arrived, so it adds no latency when the server keeps up (BS1/low
    # concurrency see ~1 chunk per feed and are unchanged).
    stream_feed_initial_batch_max_frames: int = 16
    stream_feed_drain_batch_max_frames: int = 1024
    # Optional duration-quantum coalescing (ms per stream.feed). >0 merges queued
    # backlog up to this many ms instead of a chunk count. Kept for clients that
    # send sub-frame chunks, but DEFAULT 0: a fixed ms quantum over-merges large
    # client chunks (600ms quantum regressed BS52@200ms first-word 1473->1870),
    # whereas chunk-count draining self-limits to the actual backlog depth.
    stream_feed_target_ms: int = 0
    stream_feed_drain_timeout_sec: float = 8.0
    asr_post_chat_timing_enabled: bool = True

    # ---- ASR: target speaker enrollment ----------------------------------
    # When a target-speaker enrollment is uploaded the primary ASR prompt
    # switches to a two-audio shape (enrollment first, target second). The
    # actual text placement is model-template specific: Amphion 4B interleaves
    # text and audio in the user turn; Amphion 1.7B puts the instruction in
    # the system turn and keeps the user turn audio-only. The clip is cached
    # server-side for the duration of a session so the WS stream does not have
    # to retransmit it on every VAD segment. Trained distributions use 1–8 s
    # enrollment clips; anything outside that window is silently OOD even when
    # the API accepts it.
    asr_enrollment_min_sec: float = 1.0
    asr_enrollment_max_sec: float = 8.0
    asr_enrollment_ttl_sec: float = 3600.0
    asr_enrollment_max_entries: int = 256
    # Durable enrollment store (最终版 §声纹管理/§查询声纹状态). When set, each
    # enrollment persists as ``<store_dir>/<scope>/<id>.json`` (metadata + model
    # fingerprint) plus ``<id>.wav`` (canonical clip) so ``GET
    # /api/asr/enrollment/{id}`` can report ok/not_found/deleted/incompatible
    # across restarts, and the realtime path can rehydrate the clip after a
    # reload. Mount this dir on a durable volume in the delivery run. Empty
    # disables persistence (pure in-memory, legacy behavior).
    asr_enrollment_store_dir: str = "var/enrollments"
    asr_enrollment_scope: str = "default"
    # Throttle interval for persisting last_used_at touches on status/read.
    asr_enrollment_metadata_touch_interval_sec: float = 60.0
    # Optional explicit model fingerprint override; empty derives it from the
    # primary ASR model + prompt template so a model/adapter swap flips stored
    # enrollments to ``incompatible``.
    asr_enrollment_model_fingerprint: str = ""

    # ---- ASR: VAD segmentation -------------------------------------------
    # These VAD defaults mirror config.yaml's vad block. They are pure
    # tuning thresholds (no per-deployment meaning), so keeping the in-code
    # fallback equal to the shipped values avoids a confusing third number;
    # config.yaml still overrides them at load time.
    vad_threshold: float = 0.65
    silence_duration_ms: int = 1200
    vad_smoothing_alpha: float = 0.3
    vad_start_frames: int = 20
    vad_pre_speech_ms: int = 500
    vad_keep_tail_ms: int = 40
    min_segment_duration_ms: int = 350
    # 每段语音"首个 partial(伪流式中间结果)"的触发门槛:VAD 累积音频达到它才发出
    # 第一个 partial。因一段语音内 snapshot 单调增长(audio_buffer 只增不减),过了
    # 门槛后续 partial 必然更长,该门槛只对每段的首个 partial 真正 binding,故它实质
    # 是首字延迟旋钮 —— 与 vad_start_frames 一起按 max 决定首字(见
    # docs/tuling-ast-v3-protocol.md "首字延迟优化")。从 min_segment_duration_ms 解耦
    # 的原因:后者一参多职(还管 final 段过滤、flush 残余过滤),直接调它会放松短噪声
    # 段过滤。这里的 dataclass 默认 350(= min_segment_duration_ms)是"文件缺字段时的
    # 中性兜底";随附的 config.yaml 显式设 200,即默认部署选择全局低延迟首字
    # (对所有产 partial 的端点生效,不止 tuling)。注意只把它降到 200 而 vad_start_frames
    # 仍 20 时,首字 max 仍由起音确认(约 320ms)主导,需同时调小 vad_start_frames 才到
    # 最优。不变量 first_partial<=min_segment 见 __post_init__。
    pseudo_stream_first_partial_ms: int = 350

    # ---- ASR: offline long-audio transcription jobs ------------------------
    # POST /api/asr/transcriptions decodes the upload, replays it through the
    # same VAD segmentation as the streaming endpoints, and transcribes the
    # segments via the shared one-shot dual-ASR path. Knobs:
    #   transcribe_max_concurrent_jobs   -> jobs running at once; each job runs
    #                                       its own segments serially in file
    #                                       order, so one upload cannot fan out
    #                                       many vLLM requests at once.
    #   transcribe_max_segment_sec       -> force-cut ceiling for uninterrupted
    #                                       speech (VAD only cuts on silence);
    #                                       sized well under the 60 s one-shot
    #                                       REST cap to stay in the range the
    #                                       segment models see in streaming.
    #   transcribe_max_upload_bytes      -> multipart cap; 2 h of 16 kHz mono
    #                                       s16 WAV is ~220 MB, so 512 MB
    #                                       leaves headroom for higher-rate
    #                                       client WAVs.
    #   transcribe_max_audio_sec         -> decoded-duration cap. Unlike the
    #                                       60 s upload tail-trim, exceeding it
    #                                       REJECTS the request (400): silently
    #                                       dropping the head of a meeting
    #                                       recording is never acceptable.
    #   transcribe_silence_duration_ms   -> offline-only override of the VAD
    #                                       cut pause. 0 = follow the global
    #                                       silence_duration_ms. The global one
    #                                       follows deployment VAD tuning;
    #                                       minutes-style transcripts read
    #                                       better with longer pauses (~800 ms)
    #                                       and offline has no latency cost,
    #                                       but raising the GLOBAL knob would
    #                                       also delay every live endpoint's
    #                                       finals — hence the scoped override
    #                                       (same fallback pattern as
    #                                       astv3_vllm_base_url). All other VAD
    #                                       knobs stay shared: there is no
    #                                       offline reason for them to differ.
    transcribe_max_concurrent_jobs: int = 2
    transcribe_segment_concurrency: int = 1
    transcribe_job_queue_max: int = 8
    transcribe_job_ttl_sec: float = 3600.0
    transcribe_max_segment_sec: float = 30.0
    transcribe_max_upload_bytes: int = 512 * 1024 * 1024
    transcribe_max_audio_sec: float = 10800.0
    transcribe_silence_duration_ms: int = 0

    # ---- Emotion recognition: vLLM endpoint ------------------------------
    # The Amphion multi-task model is trained to handle SER/SEC alongside ASR
    # via different text prompts, so by default we point the emotion endpoint
    # at the same backend as the primary ASR. Override
    # ``emotion_vllm_base_url`` if you serve a dedicated emotion model.
    emotion_vllm_base_url: str = "http://localhost:8000"
    emotion_vllm_model_name: str = "Amphion/AmphionASR-4.3B"
    emotion_request_timeout: float = 30.0
    # Amphion SER/SEC training uses 1-20s utterances, so we cap longer audio
    # to the trailing 20 seconds (where the most recent speech lives).
    emotion_max_audio_seconds: float = 20.0
    # Default task variant when the client doesn't specify one in start.mode.
    # "ser" -> single label classification; "sec" -> free-form caption.
    emotion_task_mode: str = "ser"
    # Whole-utterance HTTP job API backpressure (in-process store).
    emotion_max_concurrent_jobs: int = 8
    emotion_job_queue_max: int = 64
    emotion_job_ttl_sec: float = 3600.0

    # ---- Paralinguistic emotion model (AmphionSPEC) ----------------------
    # Independent vLLM endpoint that serves the AmphionSPEC checkpoint. It
    # is trained with two prompts: ``ser`` (same 8-way label set as the
    # baseline emotion model) and ``sepc`` (free-form description of
    # paralinguistic emotion cues — prosody, tempo, voice quality, etc.).
    # Configuration mirrors the emotion knobs so it can scale and back-off
    # independently from the baseline emotion store.
    emotion_spec_vllm_base_url: str = "http://localhost:9001"
    emotion_spec_vllm_model_name: str = "AmphionSPEC"
    emotion_spec_request_timeout: float = 30.0
    emotion_spec_max_audio_seconds: float = 20.0
    # Default mode when the client omits ``mode``; the prompt label
    # ``sepc`` is the literal training token (do not rename to ``spec``).
    emotion_spec_task_mode: str = "sepc"
    emotion_spec_max_concurrent_jobs: int = 8
    emotion_spec_job_queue_max: int = 64
    emotion_spec_job_ttl_sec: float = 3600.0

    # Shared httpx pool ceiling for all vLLM / upstream HTTP calls.
    http_max_connections: int = 128
    http_max_keepalive_connections: int = 64

    # ---- Text cleanup LLM (DashScope OpenAI-compatible) -------------------
    text_cleanup_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    text_cleanup_model_name: str = "qwen2.5-32b-instruct"
    text_cleanup_api_key_env: str = "DASHSCOPE_API_KEY"
    text_cleanup_api_key: str = ""
    text_cleanup_timeout: float = 30.0
    text_cleanup_max_tokens: int = 1024

    def __post_init__(self) -> None:
        if self.vllm_prompt_template not in VALID_PRIMARY_PROMPT_TEMPLATES:
            raise ValueError(
                f"vllm_prompt_template={self.vllm_prompt_template!r} is invalid; "
                f"allowed: {sorted(VALID_PRIMARY_PROMPT_TEMPLATES)}"
            )
        if (
            self.astv3_vllm_prompt_template
            and self.astv3_vllm_prompt_template not in VALID_PRIMARY_PROMPT_TEMPLATES
        ):
            raise ValueError(
                "astv3_vllm_prompt_template="
                f"{self.astv3_vllm_prompt_template!r} is invalid; "
                f"allowed: {sorted(VALID_PRIMARY_PROMPT_TEMPLATES)}"
            )
        # Silently enforce the invariant that fusion requires a secondary
        # decoder. Dataclass-level so any construction path (load, override,
        # direct ctor in tests) yields a consistent Config — downstream
        # code can rely on `enable_dual_asr_fusion` being a clean truth.
        # Loud WARNING happens in `load_config` against the raw input so
        # operators notice misconfigured files; in-process overrides stay
        # silent to avoid log spam.
        if self.enable_dual_asr_fusion and not self.enable_secondary_asr:
            object.__setattr__(self, "enable_dual_asr_fusion", False)
        # 首个 partial 门槛若严于 final 段最小时长,partial 就会永远比 final 晚、失去
        # "中间结果"意义;夹到 <= min_segment_duration_ms。和 fusion 不变量一样下沉
        # 到 dataclass,确保 load/override/直接构造(测试)各路径都一致。
        if self.pseudo_stream_first_partial_ms > self.min_segment_duration_ms:
            object.__setattr__(
                self, "pseudo_stream_first_partial_ms", self.min_segment_duration_ms
            )

    @property
    def resolved_text_cleanup_api_key(self) -> str:
        """Return configured API key, preferring the named environment variable."""
        env_name = self.text_cleanup_api_key_env.strip()
        if env_name:
            value = os.getenv(env_name, "").strip()
            if value:
                return value
        return self.text_cleanup_api_key.strip()

    def override(self, **kwargs: Any) -> Config:
        """Return a new Config with the given fields replaced (unknown keys ignored)."""
        valid_names = {f.name for f in fields(self)}
        accepted: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k not in valid_names:
                continue
            expected = type(getattr(self, k))
            try:
                accepted[k] = expected(v)
            except (TypeError, ValueError):
                logger.warning("Ignoring invalid config override %s=%r", k, v)
        # `__post_init__` silently fixes any inconsistent combinations,
        # which is the right behavior for in-process overrides.
        return replace(self, **accepted) if accepted else self

    def override_client(self, **kwargs: Any) -> Config:
        """Like :meth:`override` but only honors client-overridable fields.

        For untrusted per-connection overrides (WebSocket ``start.config`` and
        AST v3 ``parameter.asr_config``). Infrastructure / secret / process-wide
        fields (model URLs -> SSRF, API keys, HTTP pools, job queues) are not in
        ``CLIENT_OVERRIDABLE_FIELDS`` and are dropped with a WARN: an operator
        can see a client reaching for a restricted knob, while the long-lived
        connection is never broken. Type coercion and dataclass invariants are
        delegated to :meth:`override`.
        """
        allowed: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in CLIENT_OVERRIDABLE_FIELDS:
                allowed[k] = v
            else:
                logger.warning(
                    "Ignoring non-overridable config field from client: %s", k
                )
        return self.override(**allowed)


# Per-connection client overrides are restricted to this whitelist. Guiding
# line: a field is overridable only if it tunes how *this session* processes its
# own audio. Process-wide knobs (HTTP pools, job queues, cache capacity), backend
# routing (``*_vllm_base_url`` -> SSRF, ``*_model_name``) and secrets
# (``text_cleanup_api_key*``) are intentionally excluded so an untrusted client
# cannot reach them via ``start.config`` / ``parameter.asr_config``.
CLIENT_OVERRIDABLE_FIELDS: frozenset[str] = frozenset({
    # VAD / segmentation
    "vad_threshold",
    "silence_duration_ms",
    "vad_smoothing_alpha",
    "vad_start_frames",
    "vad_pre_speech_ms",
    "vad_keep_tail_ms",
    "min_segment_duration_ms",
    # Pseudo-streaming partials
    "enable_pseudo_stream",
    "pseudo_stream_interval_ms",
    "pseudo_stream_first_partial_ms",
    # ASR model combination / timeouts
    "enable_primary_asr",
    "enable_secondary_asr",
    "enable_dual_asr_fusion",
    "primary_asr_timeout",
    "asr_request_timeout",
    "debug_show_dual_asr",
    # Dual-model fusion thresholds
    "fusion_similarity_threshold",
    "fusion_min_primary_score",
    "fusion_max_repetition_ratio",
    "fusion_disagreement_threshold",
    "fusion_hotword_boost",
    "fusion_primary_score_margin",
    # TS-ASR enrollment bounds
    "asr_enrollment_min_sec",
    "asr_enrollment_max_sec",
    "asr_enrollment_ttl_sec",
    # Emotion (baseline + paralinguistic spec) per-request tuning
    "emotion_task_mode",
    "emotion_request_timeout",
    "emotion_max_audio_seconds",
    "emotion_spec_task_mode",
    "emotion_spec_request_timeout",
    "emotion_spec_max_audio_seconds",
})

# Fail-fast: a whitelisted name that is not a real Config field is a typo that
# would silently make a knob un-overridable forever, so reject it at import.
_unknown_overridable = CLIENT_OVERRIDABLE_FIELDS - {f.name for f in fields(Config)}
if _unknown_overridable:
    raise ValueError(
        "CLIENT_OVERRIDABLE_FIELDS has unknown Config fields: "
        f"{sorted(_unknown_overridable)}"
    )


# 合法的 (protocol, task) 组合白名单。其余组合(如 astv3 × browser_demo)在解析期
# fail-fast, 避免注册出无意义的端点。browser_demo 复用 native wire, 不是独立协议。
VALID_ENDPOINT_COMBINATIONS: frozenset[tuple[str, str]] = frozenset({
    ("native", "asr"),
    ("astv3", "asr"),
    ("native", "emotion"),
    ("native", "browser_demo"),
})

# upstream 角色: 决定一个 upstream 投影到哪些扁平 Config 字段。
_UPSTREAM_ROLES: frozenset[str] = frozenset(
    {"primary", "secondary", "emotion", "emotion_spec"}
)


@dataclass(frozen=True)
class EndpointSpec:
    """A declarative WebSocket endpoint parsed from config.yaml ``endpoints``."""

    path: str
    protocol: str
    task: str
    # role -> upstream name (primary / secondary / emotion / emotion_spec)
    upstream_roles: dict[str, str] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)  # 软默认(客户端白名单可覆盖)
    lock: dict[str, Any] = field(default_factory=dict)    # 硬锁定(客户端覆盖后重施)
    client_overridable: bool = False
    input_sample_rate: int = SAMPLE_RATE


@dataclass(frozen=True)
class ParsedConfig:
    """Fully parsed config.yaml: registries + the global (REST-bound) Config."""

    upstreams: dict[str, Upstream]
    defaults: dict[str, Any]          # flattened tuning leaves
    services: dict[str, str]          # service role -> upstream name
    rest_roles: dict[str, str]        # REST role -> upstream name
    endpoints: tuple[EndpointSpec, ...]
    http_pool: dict[str, int]
    default_config: Config
    # POST /api/asr/transcriptions runtime view: default_config plus the
    # optional `rest.routes.transcribe` block (own model bindings + fusion
    # switch); equal to default_config when the block is absent.
    transcribe_config: Config


def _parse_upstream(name: str, raw: dict[str, Any]) -> Upstream:
    if not isinstance(raw, dict):
        raise ValueError(f"upstreams.{name} must be a mapping")
    max_tokens = raw.get("max_tokens")
    prompt_template = str(raw.get("prompt_template", "amphion_asr"))
    if prompt_template not in VALID_PRIMARY_PROMPT_TEMPLATES:
        raise ValueError(
            f"upstreams.{name}.prompt_template={prompt_template!r} is invalid; "
            f"allowed: {sorted(VALID_PRIMARY_PROMPT_TEMPLATES)}"
        )
    return Upstream(
        name=name,
        base_url=str(raw.get("base_url", "")).rstrip("/"),
        model_name=str(raw.get("model_name", "")),
        api_key=str(raw.get("api_key", "")),
        timeout=float(raw.get("timeout", 120.0)),
        max_tokens=int(max_tokens) if max_tokens is not None else None,
        prompt_template=prompt_template,
    )


def _role_fields(role: str, up: Upstream) -> dict[str, Any]:
    """Project an upstream bound to a role onto the flat Config fields it feeds."""
    if role == "primary":
        return {
            "vllm_base_url": up.base_url,
            "vllm_model_name": up.model_name,
            "vllm_prompt_template": up.prompt_template,
            "asr_request_timeout": up.timeout,
        }
    if role == "secondary":
        # Secondary reuses the single asr_request_timeout (primary's) by design.
        return {
            "secondary_vllm_base_url": up.base_url,
            "secondary_vllm_model_name": up.model_name,
        }
    if role == "emotion":
        return {
            "emotion_vllm_base_url": up.base_url,
            "emotion_vllm_model_name": up.model_name,
            "emotion_request_timeout": up.timeout,
        }
    if role == "emotion_spec":
        return {
            "emotion_spec_vllm_base_url": up.base_url,
            "emotion_spec_vllm_model_name": up.model_name,
            "emotion_spec_request_timeout": up.timeout,
        }
    logger.warning("Unknown upstream role ignored: %s", role)
    return {}


def _text_cleanup_fields(up: Upstream) -> dict[str, Any]:
    """Project the text-cleanup service upstream onto the legacy text_cleanup_* fields.

    BRIDGE: ``text_cleanup/client.py`` currently builds ``{base}/chat/completions``
    expecting ``base`` to already contain ``/v1``, while ``Upstream.base_url`` is
    normalized to the service root (no ``/v1``). We re-append ``/v1`` here so the
    un-migrated client keeps working. Phase 4 routes cleanup through the unified
    ``query_chat_completion`` and removes both this bridge and the text_cleanup_*
    fields.
    """
    return {
        "text_cleanup_base_url": up.base_url + "/v1",
        "text_cleanup_model_name": up.model_name,
        "text_cleanup_api_key": up.api_key,
        "text_cleanup_api_key_env": "",  # already interpolated to plaintext on load
        "text_cleanup_timeout": up.timeout,
        "text_cleanup_max_tokens": up.max_tokens if up.max_tokens is not None else 1024,
    }


def _project_base(
    *,
    upstreams: dict[str, Upstream],
    rest_roles: dict[str, str],
    defaults: dict[str, Any],
    services: dict[str, str],
    http_pool: dict[str, int],
) -> Config:
    """Project the global (REST-bound) Config view: defaults + http_pool +
    text_cleanup service + REST role bindings. This is what ``load_config()`` and
    ``default_config`` expose; per-endpoint configs override on top of it.
    """
    values: dict[str, Any] = dict(defaults)
    values["http_max_connections"] = int(http_pool.get("max_connections", 32))
    values["http_max_keepalive_connections"] = int(
        http_pool.get("max_keepalive_connections", 16)
    )
    cleanup_name = services.get("text_cleanup")
    if cleanup_name and cleanup_name in upstreams:
        values.update(_text_cleanup_fields(upstreams[cleanup_name]))
    for role, up_name in rest_roles.items():
        if up_name in upstreams:
            values.update(_role_fields(role, upstreams[up_name]))
    valid = {f.name for f in fields(Config)}
    unknown = set(defaults) - valid
    if unknown:
        logger.warning("Unknown defaults keys ignored: %s", sorted(unknown))
    filtered = {k: v for k, v in values.items() if k in valid}
    return Config(**filtered)


def _parse_endpoint(raw: dict[str, Any], upstreams: dict[str, Upstream]) -> EndpointSpec:
    path = str(raw.get("path", "")).strip()
    protocol = str(raw.get("protocol", "")).strip()
    task = str(raw.get("task", "")).strip()
    if not path:
        raise ValueError("endpoint entry missing 'path'")
    if (protocol, task) not in VALID_ENDPOINT_COMBINATIONS:
        raise ValueError(
            f"endpoint {path}: invalid (protocol, task)=({protocol!r}, {task!r}); "
            f"allowed: {sorted(VALID_ENDPOINT_COMBINATIONS)}"
        )
    roles = dict(raw.get("upstreams", {}) or {})
    for role, up_name in roles.items():
        if role not in _UPSTREAM_ROLES:
            raise ValueError(f"endpoint {path}: unknown upstream role {role!r}")
        if up_name not in upstreams:
            raise ValueError(
                f"endpoint {path}: unknown upstream {up_name!r} for role {role!r}"
            )
    return EndpointSpec(
        path=path,
        protocol=protocol,
        task=task,
        upstream_roles=roles,
        policy=dict(raw.get("policy", {}) or {}),
        lock=dict(raw.get("lock", {}) or {}),
        client_overridable=bool(raw.get("client_overridable", False)),
        input_sample_rate=int(raw.get("input_sample_rate", SAMPLE_RATE)),
    )


def _project_transcribe(
    base: Config,
    raw_transcribe: dict[str, Any],
    upstreams: dict[str, Upstream],
) -> Config:
    """Project the `rest.routes.transcribe` block onto the transcription Config.

    `rest.upstreams` is the shared role->upstream binding table for every
    REST route; `rest.routes.<name>` holds per-route overrides so operators
    can see and change, in one place, which model(s) one endpoint runs on.
    Recognized keys for the transcribe route:

    - ``upstreams``: role -> upstream name, roles limited to primary /
      secondary (an emotion model makes no sense for transcription);
    - ``enable_dual_asr_fusion``: route-scoped override of the global switch.

    Unknown keys raise: a typo silently falling back to the shared bindings
    is exactly the "can't see what this endpoint uses" problem again.
    """
    allowed = {"upstreams", "enable_dual_asr_fusion"}
    unknown = set(raw_transcribe) - allowed
    if unknown:
        raise ValueError(
            f"rest.routes.transcribe: unknown keys {sorted(unknown)}; "
            f"allowed: {sorted(allowed)}"
        )

    overrides: dict[str, Any] = {}
    roles = dict(raw_transcribe.get("upstreams", {}) or {})
    for role, up_name in roles.items():
        if role not in ("primary", "secondary"):
            raise ValueError(
                f"rest.routes.transcribe.upstreams: unknown role {role!r}; "
                "allowed: ['primary', 'secondary']"
            )
        if up_name not in upstreams:
            raise ValueError(
                f"rest.routes.transcribe.upstreams.{role}: unknown upstream {up_name!r}"
            )
        overrides.update(_role_fields(role, upstreams[up_name]))

    if "enable_dual_asr_fusion" in raw_transcribe:
        fusion = bool(raw_transcribe["enable_dual_asr_fusion"])
        overrides["enable_dual_asr_fusion"] = fusion
        if fusion and not base.enable_secondary_asr:
            logger.warning(
                "rest.routes.transcribe.enable_dual_asr_fusion=true requires "
                "enable_secondary_asr=true (defaults.asr); downgrading to false"
            )

    return base.override(**overrides) if overrides else base


def _parse(raw: dict[str, Any]) -> ParsedConfig:
    upstreams = {
        name: _parse_upstream(name, spec)
        for name, spec in (raw.get("upstreams", {}) or {}).items()
    }
    defaults = _flatten_leaves(raw.get("defaults", {}) or {})

    services = dict(raw.get("services", {}) or {})
    for svc, up_name in services.items():
        if up_name not in upstreams:
            raise ValueError(f"services.{svc}: unknown upstream {up_name!r}")

    rest_raw = dict(raw.get("rest", {}) or {})
    unknown_rest = set(rest_raw) - {"upstreams", "routes"}
    if unknown_rest:
        raise ValueError(
            f"rest: unknown keys {sorted(unknown_rest)}; allowed: ['upstreams', "
            "'routes'] (per-route overrides moved under rest.routes.<name>)"
        )
    rest_roles = dict(rest_raw.get("upstreams", {}) or {})
    for role, up_name in rest_roles.items():
        if role not in _UPSTREAM_ROLES:
            raise ValueError(f"rest.upstreams: unknown role {role!r}")
        if up_name not in upstreams:
            raise ValueError(f"rest.upstreams.{role}: unknown upstream {up_name!r}")

    http_pool = dict(raw.get("http_pool", {}) or {})
    endpoints = tuple(
        _parse_endpoint(ep, upstreams) for ep in (raw.get("endpoints", []) or [])
    )
    rest_routes = dict(rest_raw.get("routes", {}) or {})
    unknown_routes = set(rest_routes) - {"transcribe"}
    if unknown_routes:
        raise ValueError(
            f"rest.routes: unknown routes {sorted(unknown_routes)}; "
            "known: ['transcribe']"
        )

    # Loud WARN on an impossible global combo (silent downgrade still happens in
    # __post_init__; this is just the operator-facing log line).
    if defaults.get("enable_dual_asr_fusion") and not defaults.get(
        "enable_secondary_asr", True
    ):
        logger.warning(
            "enable_dual_asr_fusion=true requires enable_secondary_asr=true; "
            "downgrading fusion to false"
        )

    base = _project_base(
        upstreams=upstreams,
        rest_roles=rest_roles,
        defaults=defaults,
        services=services,
        http_pool=http_pool,
    )
    transcribe_cfg = _project_transcribe(
        base, dict(rest_routes.get("transcribe", {}) or {}), upstreams
    )
    return ParsedConfig(
        upstreams=upstreams,
        defaults=defaults,
        services=services,
        rest_roles=rest_roles,
        endpoints=endpoints,
        http_pool=http_pool,
        default_config=base,
        transcribe_config=transcribe_cfg,
    )


def load_parsed(path: Path | None = None) -> ParsedConfig:
    """Parse the full config.yaml (registries + global Config).

    ``CONFIG_PATH`` env overrides the default path when no explicit path is given.
    """
    if path is None:
        env_path = os.getenv("CONFIG_PATH", "").strip()
        path = Path(env_path) if env_path else _DEFAULT_CONFIG_PATH
    return _parse(_load_yaml(path))


def load_config(path: Path | None = None) -> Config:
    """Backward-compatible entry point: the global (REST-bound) default Config.

    Per-endpoint runtime configs come from :func:`resolve_endpoint`.
    """
    return load_parsed(path).default_config


def load_transcribe_config(path: Path | None = None) -> Config:
    """Runtime Config for POST /api/asr/transcriptions.

    Equals :func:`load_config` unless config.yaml declares a
    `rest.routes.transcribe` block (own primary/secondary bindings and/or
    fusion switch).
    """
    return load_parsed(path).transcribe_config


def resolve_endpoint(
    spec: EndpointSpec, parsed: ParsedConfig | None = None
) -> Config:
    """Project an endpoint's runtime Config.

    global default (REST-bound) + this endpoint's explicit upstream bindings +
    ``policy`` (soft) + ``lock`` (hard). The route layer re-applies ``spec.lock``
    after any client override so locks can't be undone by ``start.config``.
    """
    parsed = parsed or _PARSED
    overrides: dict[str, Any] = {}
    for role, up_name in spec.upstream_roles.items():
        if up_name in parsed.upstreams:
            overrides.update(_role_fields(role, parsed.upstreams[up_name]))
    overrides.update(spec.policy)
    overrides.update(spec.lock)
    # 端点未绑 secondary 上游 => 物理上没有副模型, 强制关闭(放在最后, 不可被
    # policy/lock 误开成一个并不存在的副模型)。
    if spec.task == "asr" and "secondary" not in spec.upstream_roles:
        overrides["enable_secondary_asr"] = False
    return parsed.default_config.override(**overrides)


def get_service_upstream(
    service: str, parsed: ParsedConfig | None = None
) -> Upstream | None:
    """Resolve a global auxiliary service (hotword / text_cleanup) to its Upstream."""
    parsed = parsed or _PARSED
    name = parsed.services.get(service)
    return parsed.upstreams.get(name) if name else None


# 进程级解析结果(基于默认路径)。下面的注册表都从此派生, 全进程一份。
_PARSED: ParsedConfig = load_parsed()
UPSTREAMS: dict[str, Upstream] = _PARSED.upstreams
ENDPOINTS: tuple[EndpointSpec, ...] = _PARSED.endpoints
SERVICES: dict[str, str] = _PARSED.services
REST_ROLES: dict[str, str] = _PARSED.rest_roles

# Process-wide default Config singleton (REST-bound projection). Modules without a
# per-session Config (module-level helpers, ``value or <default>`` fallbacks) read
# fields off this. Single entry point preserved: every reader reaches config via a
# Config object.
default_config: Config = _PARSED.default_config
