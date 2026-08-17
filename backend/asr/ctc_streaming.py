"""Streaming CTC Ascend OM partial backend.

This is a service-oriented wrapper around the validated BS52 streaming
Zipformer CTC OM. The runtime owns fixed batch slots and uses ping-pong cached
state buffers so each tick avoids copying 74 state tensors.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import acl
import numpy as np
import sherpa_onnx

from ..config import SAMPLE_RATE, Config

logger = logging.getLogger(__name__)

ACL_ERROR_NONE = 0
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3


def _check(name: str, ret: int) -> None:
    if ret != ACL_ERROR_NONE:
        raise RuntimeError(f"{name} failed ret={ret}")


# Optional batcher profiling hook. When set (e.g. by
# benchmark_ctc_om_standalone.py), the scheduler reports one record per OM
# execute so TTFT bottleneck attribution (single-execute queueing vs chunk
# accumulation) can be measured. Default is None → zero overhead in delivery.
_BATCH_PROFILE_HOOK: "Callable[[dict[str, object]], None] | None" = None


def set_batch_profile_hook(hook: "Callable[[dict[str, object]], None] | None") -> None:
    """Install (or clear with None) the per-execute batcher profiling hook."""
    global _BATCH_PROFILE_HOOK
    _BATCH_PROFILE_HOOK = hook


_tokens_cache: dict[str, tuple[dict[int, str], str]] = {}


def _load_tokens(tokens_path: str) -> tuple[dict[int, str], str]:
    """Load a CTC tokens file and detect its encoding scheme.

    Returns (id_to_token, mode) where mode is one of:

    ``"byte_bpe"``
        Current Chinese sherpa-onnx format (``<token> <int_id>`` per line).
        Token text uses a byte-level Unicode encoding: every byte 0–255 maps
        to a unique printable character.  ``▁`` is a word-boundary placeholder
        that is *skipped* during decoding (Chinese output needs no spaces).
        Detection: integer ids AND no ASCII-letter-only tokens after the first
        three special entries.

    ``"hybrid_bpe"``
        Bilingual k2/icefall ``tokens.txt`` (also ``<token> <int_id>`` per
        line).  The vocabulary contains *both* standard English-word tokens
        (e.g. ``▁THE``, ``S``) and byte-level Chinese tokens (e.g. ``▁ƎĽĥ``).
        ``▁`` is decoded as a space so English words are separated; all other
        characters go through the byte-BPE mapping as usual.
        Detection: integer ids AND the file contains tokens whose text (after
        stripping ``▁``) is pure ASCII letters (indicating readable words).

    ``"sentencepiece_byte_fallback_bpe"``
        SentencePiece byte-fallback format.  The vocabulary contains readable
        BPE pieces (``▁the``), literal CJK characters, and byte fallback tokens
        written as ``<0xNN>``.  ``▁`` is a word-boundary placeholder, and
        ``<0xNN>`` tokens are decoded as raw UTF-8 bytes.

    ``"standard_bpe"``
        k2/icefall ``bbpe.vocab`` format (``<token> <float_score>``; line index
        is the token id).  All tokens are readable Unicode text; ``▁`` → space.
    """
    if tokens_path in _tokens_cache:
        return _tokens_cache[tokens_path]
    tokens: dict[int, str] = {}
    has_float_score = False
    with open(tokens_path, encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            parts = line.strip().split()
            if not parts:
                continue
            token_text = parts[0]
            if len(parts) >= 2:
                try:
                    token_id = int(parts[-1])
                    tokens[token_id] = token_text
                except ValueError:
                    has_float_score = True
                    tokens[line_idx] = token_text
            else:
                tokens[line_idx] = token_text

    if has_float_score:
        mode = "standard_bpe"
    else:
        # Integer-id format.  Distinguish byte_bpe from hybrid_bpe by checking
        # whether any non-special token is composed of pure ASCII letters, which
        # indicates real English subwords (not byte-level encoded characters).
        special = frozenset({"<blk>", "<sos>", "<eos>", "<sos/eos>", "<pad>", "<unk>"})
        has_byte_fallback = any(
            len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">")
            for tok in tokens.values()
        )
        has_ascii_word = any(
            all(ord(c) < 128 and c.isalpha() for c in tok.lstrip("▁"))
            for tok_id, tok in tokens.items()
            if tok_id >= 3 and tok not in special and len(tok.lstrip("▁")) >= 2
        )
        if has_byte_fallback:
            mode = "sentencepiece_byte_fallback_bpe"
        else:
            mode = "hybrid_bpe" if has_ascii_word else "byte_bpe"

    _tokens_cache[tokens_path] = (tokens, mode)
    return tokens, mode


_BYTE_BPE_UNK = chr(8263)
_BYTE_BPE_PRINTABLE = (
    list(range(256, 287 + 1))
    + list(range(32, 126 + 1))
    + list(range(288, 305 + 1))
    + list(range(308, 318 + 1))
    + list(range(321, 328 + 1))
    + list(range(330, 382 + 1))
    + list(range(384, 422 + 1))
)
_BCHAR_TO_BYTE: dict[str, int] = {
    chr(ch): b for b, ch in enumerate(_BYTE_BPE_PRINTABLE)
}
_BCHAR_TO_BYTE[_BYTE_BPE_UNK] = 32

_STANDARD_BPE_SKIP = frozenset({"<blk>", "<sos>", "<eos>", "<sos/eos>", "<pad>", "<unk>", ""})


def _decode_tokens(tokens_path: str, token_ids: list[int]) -> str:
    """Decode a list of CTC token ids to text.

    Supports encoding modes detected automatically from the tokens file:

    * ``byte_bpe``   – current Chinese model: ``▁`` is skipped, every other
                       character maps to a byte via ``_BCHAR_TO_BYTE``.
    * ``hybrid_bpe`` – bilingual k2 ``tokens.txt``: ``▁`` → space byte, all
                       other characters still go through ``_BCHAR_TO_BYTE``.
                       This produces correct English spacing while keeping
                       Chinese byte sequences intact.
    * ``standard_bpe`` – k2 ``bbpe.vocab``: ``▁`` → space, remaining text is
                         literal Unicode (no byte mapping needed).
    * ``sentencepiece_byte_fallback_bpe`` – readable SentencePiece tokens plus
                         ``<0xNN>`` byte fallback tokens.
    """
    id_to_token, mode = _load_tokens(tokens_path)

    if mode in ("byte_bpe", "hybrid_bpe"):
        data = b""
        for token_id in token_ids:
            token = id_to_token.get(int(token_id), "")
            for ch in token:
                if ch == "▁":
                    if mode == "hybrid_bpe":
                        data += b" "   # word boundary → space
                    # byte_bpe: skip ▁ (Chinese output needs no spaces)
                else:
                    if ch in _BCHAR_TO_BYTE:
                        data += bytes([_BCHAR_TO_BYTE[ch]])
        return data.decode("utf-8", errors="ignore").strip()

    if mode == "sentencepiece_byte_fallback_bpe":
        parts: list[str | bytes] = []
        pending_bytes = bytearray()

        def flush_bytes() -> None:
            if pending_bytes:
                parts.append(bytes(pending_bytes))
                pending_bytes.clear()

        for token_id in token_ids:
            token = id_to_token.get(int(token_id), "")
            if token in _STANDARD_BPE_SKIP:
                continue
            if len(token) == 6 and token.startswith("<0x") and token.endswith(">"):
                try:
                    pending_bytes.append(int(token[3:5], 16))
                    continue
                except ValueError:
                    pass
            flush_bytes()
            if token.startswith("▁"):
                parts.append(" " + token[1:])
            else:
                parts.append(token)
        flush_bytes()
        text_parts = [
            part.decode("utf-8", errors="ignore") if isinstance(part, bytes) else part
            for part in parts
        ]
        return "".join(text_parts).lstrip()

    # standard_bpe: ▁ = space, rest is literal Unicode text.
    parts: list[str] = []
    for token_id in token_ids:
        token = id_to_token.get(int(token_id), "")
        if token in _STANDARD_BPE_SKIP:
            continue
        if token.startswith("▁"):
            parts.append(" " + token[1:])
        else:
            parts.append(token)
    return "".join(parts).lstrip()


def _decode_byte_tokens(tokens_path: str, token_ids: list[int]) -> str:
    """Legacy alias kept for call-sites that haven't been updated yet."""
    return _decode_tokens(tokens_path, token_ids)


@dataclass(frozen=True)
class CtcStreamingConfig:
    om_model_path: str
    onnx_model_path: str
    tokens_path: str
    batch_size: int = 52
    wait_ms: int = 10
    ready_coalesce_ms: int = 30
    result_wait_ms: int = 120
    device_id: int = 0
    device_resident_state: bool = False
    # Device-resident pacing: minimum ms between decode ticks once at least one
    # stream is already initialised, UNLESS all initialised-active streams are
    # ready (then decode now) or a brand-new (uninitialised) stream is ready
    # (first-frame priority).  Batches in-phase streams into one fast pure-device
    # tick and avoids per-tick host-merge preservation of not-yet-ready slots.
    # 0 disables pacing.  A value near chunk_shift duration (~320ms) is ideal.
    resident_pace_ms: int = 0
    # First-frame priority: when a brand-new (never-decoded, decode_ticks==0)
    # stream is already ready, skip the ready_coalesce_ms second-wait so its
    # first partial is not delayed by up to ready_coalesce_ms under BS52 burst
    # registration.  Steady state (no brand-new stream ready) keeps coalescing
    # to preserve batch efficiency.
    firstframe_bypass_coalesce: bool = True


@dataclass(frozen=True)
class _StateSpec:
    index: int
    dims: tuple[int, ...]
    batch_dim: int
    dtype: np.dtype
    name: str


class _CtcStreamState:
    def __init__(self, *, slot: int, feature_stream: object, trace_id: str = "") -> None:
        self.slot = slot
        self.feature_stream = feature_stream
        self.trace_id = trace_id
        self.active = True
        self.total_samples = 0
        self.processed_frames = 0
        self.prev_token = 0
        self.emitted_tokens: list[int] = []
        self.cached_states: list[np.ndarray] | None = None
        self.text = ""
        self.version = 0
        self.decode_ticks = 0
        # Device-resident path: whether this stream's fixed device slot has been
        # zero-initialised for a real decode yet.  Until then the slot may hold a
        # previous occupant's leftover state, so its first ready tick must zero it.
        self.resident_initialized = False
        # Wall-clock when the stream first became decode-ready for the pending
        # tick; used to measure ready->execute queue time. Reset after decode.
        self.ready_since = 0.0
        self.first_nonempty_logged = False
        self.last_callback_text = ""
        self.result_callback: Callable[[str, dict[str, float | int | str]], None] | None = None
        self.registered_at = time.monotonic()
        self.last_stats: dict[str, float | int | str] = {
            "scheduler": "ctc_om",
            "batch_size": 0,
            "decode_ms": 0.0,
            "ready_checks": 0,
            "decode_loops": 0,
        }

    @property
    def frames_ready(self) -> int:
        # Matches sherpa fbank with snip_edges=false closely enough while
        # staying conservative to avoid GetFrames boundary aborts.
        return max(0, int(self.total_samples * 100 / SAMPLE_RATE) - 1)

    def is_ready(self, chunk_length: int) -> bool:
        return self.processed_frames + chunk_length <= self.frames_ready


class _CtcFeatureFactory:
    def __init__(self, cfg: CtcStreamingConfig) -> None:
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
            tokens=cfg.tokens_path,
            model=cfg.onnx_model_path,
            num_threads=1,
            provider="cpu",
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
        )

    def create_stream(self) -> object:
        return self.recognizer.create_stream()


class _CtcOmRuntime:
    def __init__(self, cfg: CtcStreamingConfig) -> None:
        self.cfg = cfg
        self.batch_size = int(cfg.batch_size)
        # chunk_length and vocab_size are inferred from the OM after loading;
        # set provisional values here that will be overwritten in _init_acl().
        self.chunk_length = 0
        self.chunk_shift = 0
        self.feat_dim = 80
        self.vocab_size = 0
        self.cond = threading.Condition()
        self.states: dict[int, _CtcStreamState] = {}
        self._used_slots: set[int] = set()
        self._feature_factory = _CtcFeatureFactory(cfg)
        self._init_acl()
        self._tokens_path = cfg.tokens_path
        self._last_status_log_at = 0.0
        self._last_status: tuple[int, int, int] | None = None
        self.thread = threading.Thread(
            target=self._run, name="ctc-om-online-scheduler", daemon=True
        )
        self.thread.start()

    def _init_acl(self) -> None:
        _check("acl.init", acl.init())
        _check("acl.rt.set_device", acl.rt.set_device(int(self.cfg.device_id)))
        self.context, ret = acl.rt.create_context(int(self.cfg.device_id))
        _check("acl.rt.create_context", ret)
        self.model_id, ret = acl.mdl.load_from_file(self.cfg.om_model_path)
        _check("acl.mdl.load_from_file", ret)
        self.desc = acl.mdl.create_desc()
        _check("acl.mdl.get_desc", acl.mdl.get_desc(self.desc, self.model_id))
        self.input_sizes = [
            int(acl.mdl.get_input_size_by_index(self.desc, i))
            for i in range(int(acl.mdl.get_num_inputs(self.desc)))
        ]
        self.output_sizes = [
            int(acl.mdl.get_output_size_by_index(self.desc, i))
            for i in range(int(acl.mdl.get_num_outputs(self.desc)))
        ]
        n_in, n_out = len(self.input_sizes), len(self.output_sizes)
        if n_in != n_out or n_in < 2:
            raise RuntimeError(
                f"unexpected CTC OM IO count: inputs={n_in} outputs={n_out}"
            )
        self.state_specs = self._build_state_specs()

        # Infer chunk_length from OM input[0] size: [batch, chunk_length, feat_dim] × fp32
        self.chunk_length = self.input_sizes[0] // (self.batch_size * self.feat_dim * 4)
        if self.chunk_length == 0:
            raise RuntimeError(
                f"Cannot infer chunk_length from input[0] size={self.input_sizes[0]}, "
                f"batch={self.batch_size} feat_dim={self.feat_dim}"
            )
        # Verify batch dimension matches config
        inferred_batch = self.input_sizes[0] // (self.chunk_length * self.feat_dim * 4)
        if inferred_batch != self.batch_size:
            raise RuntimeError(
                f"CTC OM batch {inferred_batch} != config batch {self.batch_size}"
            )

        # Infer vocab_size from OM output[0] size:
        # [batch, num_output_frames, vocab_size] × fp32. Prefer the actual
        # token file length so new CTC candidates do not need hard-coded vocab
        # constants in the runtime.
        tokens, token_mode = _load_tokens(self.cfg.tokens_path)
        token_count = len(tokens)
        candidate_vocabs = [token_count, 8000, 8003, 8001, 1000, 500]
        seen_vocabs: set[int] = set()
        for candidate_vocab in candidate_vocabs:
            if candidate_vocab <= 0 or candidate_vocab in seen_vocabs:
                continue
            seen_vocabs.add(candidate_vocab)
            frames = self.output_sizes[0] // (self.batch_size * candidate_vocab * 4)
            if frames > 0 and self.output_sizes[0] == self.batch_size * frames * candidate_vocab * 4:
                self.vocab_size = candidate_vocab
                self.num_output_frames = frames
                break
        else:
            raise RuntimeError(
                f"Cannot infer vocab_size from output[0] size={self.output_sizes[0]}, "
                f"batch={self.batch_size}"
            )

        # chunk_shift: how many feature frames advance per OM call.
        #
        # For streaming Zipformer:
        #   x input shape: [batch, chunk_length, feat_dim]
        #   chunk_length = chunk_size + right_context_frames
        #   chunk_size = num_output_frames × subsampling_factor (typically 4)
        #   chunk_shift = chunk_size  (advance by one new-chunk worth of frames)
        #
        # Derive from OM shapes:
        #   num_output_frames → from output[0] shape already inferred
        #   subsampling_factor → 4 for standard Zipformer
        #
        # Example: old Chinese model: chunk_length=77, output_frames=16, shift=16×4=64 ✓
        #          bilingual model:  chunk_length=45, output_frames=8,  shift=8×4=32
        ZIPFORMER_SUBSAMPLING = 4
        derived_shift = self.num_output_frames * ZIPFORMER_SUBSAMPLING
        recognizer = self._feature_factory.recognizer
        if hasattr(recognizer, "chunk_shift"):
            self.chunk_shift = int(recognizer.chunk_shift)
        elif hasattr(recognizer, "model") and hasattr(recognizer.model, "chunk_shift"):
            self.chunk_shift = int(recognizer.model.chunk_shift)
        elif derived_shift > 0 and derived_shift < self.chunk_length:
            self.chunk_shift = derived_shift
        else:
            # Last fallback
            self.chunk_shift = max(1, self.chunk_length // 2)
        self.x_host = np.zeros(
            (self.batch_size, self.chunk_length, self.feat_dim), dtype=np.float32
        )
        self.log_probs_host = np.empty(
            (self.batch_size, self.num_output_frames, self.vocab_size),
            dtype=np.float32,
        )
        self.state_input_hosts = [
            np.zeros(spec.dims, dtype=spec.dtype) for spec in self.state_specs
        ]
        self.state_output_hosts = [
            np.empty(spec.dims, dtype=spec.dtype) for spec in self.state_specs
        ]
        self.x_ptr = self._malloc_zero(self.input_sizes[0], "x")
        self.log_probs_ptr = self._malloc_zero(self.output_sizes[0], "log_probs")
        self.state_a = [
            self._malloc_zero(size, f"state_a_{i}")
            for i, size in enumerate(self.input_sizes[1:], 1)
        ]
        self.state_b = [
            self._malloc_zero(size, f"state_b_{i}")
            for i, size in enumerate(self.input_sizes[1:], 1)
        ]
        self.input_dataset_a = acl.mdl.create_dataset()
        self.input_dataset_b = acl.mdl.create_dataset()
        self.output_dataset_a_to_b = acl.mdl.create_dataset()
        self.output_dataset_b_to_a = acl.mdl.create_dataset()
        self._add_buffer(self.input_dataset_a, self.x_ptr, self.input_sizes[0], "in_a_x")
        self._add_buffer(self.input_dataset_b, self.x_ptr, self.input_sizes[0], "in_b_x")
        self._add_buffer(
            self.output_dataset_a_to_b,
            self.log_probs_ptr,
            self.output_sizes[0],
            "out_ab_log_probs",
        )
        self._add_buffer(
            self.output_dataset_b_to_a,
            self.log_probs_ptr,
            self.output_sizes[0],
            "out_ba_log_probs",
        )
        for i, (ptr_a, ptr_b, size) in enumerate(
            zip(self.state_a, self.state_b, self.input_sizes[1:]), 1
        ):
            self._add_buffer(self.input_dataset_a, ptr_a, size, f"in_a_state_{i}")
            self._add_buffer(self.input_dataset_b, ptr_b, size, f"in_b_state_{i}")
            self._add_buffer(
                self.output_dataset_a_to_b, ptr_b, size, f"out_ab_state_{i}"
            )
            self._add_buffer(
                self.output_dataset_b_to_a, ptr_a, size, f"out_ba_state_{i}"
            )
        self._use_a_as_input = True
        logger.info(
            "Loaded CTC OM runtime model=%s onnx=%s tokens=%s token_mode=%s batch=%s "
            "chunk_length=%s output_frames=%s chunk_shift=%s vocab_size=%s token_count=%s "
            "input_mb=%.2f output_mb=%.2f compact_state_host_mb=%.2f num_state_tensors=%s",
            self.cfg.om_model_path,
            self.cfg.onnx_model_path,
            self.cfg.tokens_path,
            token_mode,
            self.batch_size,
            self.chunk_length,
            self.num_output_frames,
            self.chunk_shift,
            self.vocab_size,
            token_count,
            sum(self.input_sizes) / 1024 / 1024,
            sum(self.output_sizes) / 1024 / 1024,
            sum(arr.nbytes for arr in self.state_input_hosts)
            / 1024
            / 1024,
            len(self.state_specs),
        )

    def _build_state_specs(self) -> list[_StateSpec]:
        specs: list[_StateSpec] = []
        for input_idx, size in enumerate(self.input_sizes[1:], 1):
            dims_info, ret = acl.mdl.get_input_dims(self.desc, input_idx)
            _check(f"acl.mdl.get_input_dims {input_idx}", ret)
            dims = tuple(int(v) for v in dims_info.get("dims", []))
            batch_dims = [i for i, value in enumerate(dims) if value == self.batch_size]
            if not batch_dims:
                raise RuntimeError(
                    f"CTC state input {input_idx} has no batch dim: {dims_info}"
                )
            elements = 1
            for dim in dims:
                elements *= dim
            if elements <= 0 or int(size) % elements != 0:
                raise RuntimeError(
                    f"CTC state input {input_idx} bad size/dims: size={size} dims={dims}"
                )
            elem_size = int(size) // elements
            if elem_size == 4:
                dtype = np.dtype(np.float32)
            elif elem_size == 8:
                dtype = np.dtype(np.int64)
            else:
                raise RuntimeError(
                    f"CTC state input {input_idx} unsupported elem_size={elem_size}"
                )
            specs.append(
                _StateSpec(
                    index=input_idx,
                    dims=dims,
                    batch_dim=batch_dims[0],
                    dtype=dtype,
                    name=str(dims_info.get("name", f"state_{input_idx}")),
                )
            )
        return specs

    def _malloc_zero(self, size: int, label: str) -> int:
        ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
        _check(f"acl.rt.malloc {label}", ret)
        _check(f"acl.rt.memset {label}", acl.rt.memset(ptr, size, 0, size))
        return ptr

    def _add_buffer(self, dataset: int, ptr: int, size: int, label: str) -> None:
        data_buf = acl.create_data_buffer(ptr, size)
        _, ret = acl.mdl.add_dataset_buffer(dataset, data_buf)
        _check(f"acl.mdl.add_dataset_buffer {label}", ret)

    def register(self, trace_id: str = "") -> _CtcStreamState:
        with self.cond:
            if len(self._used_slots) >= self.batch_size:
                raise RuntimeError("CTC OM runtime has no free BS52 slot")
            slot = min(i for i in range(self.batch_size) if i not in self._used_slots)
            self._used_slots.add(slot)
            state = _CtcStreamState(
                slot=slot,
                feature_stream=self._feature_factory.create_stream(),
                trace_id=trace_id,
            )
            self.states[id(state)] = state
            self.cond.notify_all()
            return state

    def unregister(self, state: _CtcStreamState | None) -> None:
        if state is None:
            return
        with self.cond:
            self.states.pop(id(state), None)
            state.active = False
            # Free the fixed slot so it can be reused.  A new stream reusing this
            # slot starts with resident_initialized=False, so the device-resident
            # path zero-inits the slot on its first ready tick (the leftover state
            # here is never read as authoritative).
            self._used_slots.discard(state.slot)
            if not self.states:
                _check("acl.rt.set_context unregister", acl.rt.set_context(self.context))
                self._used_slots.clear()
                for ptr, size in zip(self.state_a, self.input_sizes[1:]):
                    _check("acl.rt.memset state_a reset", acl.rt.memset(ptr, size, 0, size))
                for ptr, size in zip(self.state_b, self.input_sizes[1:]):
                    _check("acl.rt.memset state_b reset", acl.rt.memset(ptr, size, 0, size))
            self.cond.notify_all()

    def feed_and_get(
        self,
        state: _CtcStreamState,
        *,
        chunk: np.ndarray | None,
        total_samples: int,
        trace_id: str,
        wait_for_result: bool = True,
    ) -> tuple[str, dict[str, float | int | str]]:
        total_start = time.monotonic()
        if trace_id:
            state.trace_id = trace_id
        accept_start = time.monotonic()
        if chunk is not None and chunk.size > 0:
            state.feature_stream.accept_waveform(SAMPLE_RATE, chunk)
        accept_ms = (time.monotonic() - accept_start) * 1000.0
        with self.cond:
            state.total_samples = max(state.total_samples, total_samples)
            initial_version = state.version
            should_wait = (
                wait_for_result
                and self.cfg.result_wait_ms > 0
                and state.active
                and state.is_ready(self.chunk_length)
            )
            stats = dict(state.last_stats)
            text = state.text
            self.cond.notify_all()
            if should_wait:
                deadline = time.monotonic() + self.cfg.result_wait_ms / 1000.0
                while state.version == initial_version and time.monotonic() < deadline:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self.cond.wait(timeout=remaining)
                stats = dict(state.last_stats)
                text = state.text
        stats.update(
            {
                "scheduler": "ctc_om",
                "runtime_id": 0,
                "total_samples": total_samples,
                "accept_ms": accept_ms,
                "result_ms": 0.0,
                "result_lock_wait_ms": 0.0,
                "total_ms": (time.monotonic() - total_start) * 1000.0,
            }
        )
        return text, stats

    def _resident_should_wait(
        self, ready: list[_CtcStreamState], active: list[_CtcStreamState]
    ) -> bool:
        """Pace steady-state re-decodes so in-phase streams batch into one fast
        pure-device tick.  Do NOT wait if (a) pacing disabled, (b) a brand-new
        (uninitialised) stream is ready — first-frame priority, or (c) every
        initialised-active stream is already ready.  Otherwise wait until
        resident_pace_ms has elapsed since the last decode."""
        pace_ms = self.cfg.resident_pace_ms
        if pace_ms < 0:
            return False
        if pace_ms == 0:
            # Auto-derive ~0.75x chunk_shift duration (100 frames/sec).
            pace_ms = int(self.chunk_shift * 10 * 0.75)
        ready_set = set(id(s) for s in ready)
        if any(not s.resident_initialized for s in ready):
            return False
        initialized_active = [s for s in active if s.resident_initialized]
        if initialized_active and all(
            id(s) in ready_set or s.is_ready(self.chunk_length)
            for s in initialized_active
        ):
            return False
        since = (time.monotonic() - self._last_resident_decode) * 1000.0
        return since < pace_ms

    def _run(self) -> None:
        _check("acl.rt.set_context", acl.rt.set_context(self.context))
        self._last_resident_decode = 0.0
        while True:
            with self.cond:
                self.cond.wait(timeout=max(1, int(self.cfg.wait_ms)) / 1000.0)
                active = [s for s in self.states.values() if s.active]
            if not active:
                continue
            now = time.monotonic()
            ready = [s for s in active if s.is_ready(self.chunk_length)]
            for s in ready:
                if s.ready_since == 0.0:
                    s.ready_since = now
            min_frames = min((s.frames_ready - s.processed_frames for s in active), default=0)
            self._log_status(
                active=len(active), ready=len(ready), min_frames=min_frames, now=now
            )
            if not ready:
                continue
            # First-frame priority: a brand-new (never-decoded) stream that is
            # already ready must not wait out the coalesce window, otherwise
            # BS52 burst registration inflates first-word latency by up to
            # ready_coalesce_ms per stream.  Steady state (no brand-new ready
            # stream) still coalesces to keep batches large.
            _bypass_coalesce = self.cfg.firstframe_bypass_coalesce and any(
                s.decode_ticks == 0 for s in ready
            )
            if self.cfg.ready_coalesce_ms > 0 and not _bypass_coalesce:
                with self.cond:
                    self.cond.wait(timeout=self.cfg.ready_coalesce_ms / 1000.0)
                    active = [s for s in self.states.values() if s.active]
                coalesce_now = time.monotonic()
                ready = [s for s in active if s.is_ready(self.chunk_length)]
                for s in ready:
                    if s.ready_since == 0.0:
                        s.ready_since = coalesce_now
                if not ready:
                    continue
            if self.cfg.device_resident_state:
                if self._resident_should_wait(ready, active):
                    continue
                self._decode_ready_resident(ready[: self.batch_size], active)
                self._last_resident_decode = time.monotonic()
            else:
                self._decode_ready(ready[: self.batch_size])

    def _log_status(self, *, active: int, ready: int, min_frames: int, now: float) -> None:
        status = (active, ready, min_frames)
        if status == self._last_status and now - self._last_status_log_at < 1.0:
            return
        self._last_status = status
        self._last_status_log_at = now
        logger.info(
            "CTC_ONLINE_STATUS active=%s ready=%s min_ready_frames=%s batch_size=%s",
            active,
            ready,
            min_frames,
            self.batch_size,
        )

    @staticmethod
    def _state_row_view(arr: np.ndarray, spec: _StateSpec, row: int) -> np.ndarray:
        selectors: list[object] = [slice(None)] * arr.ndim
        selectors[spec.batch_dim] = row
        return arr[tuple(selectors)]

    def _pack_state_row(
        self,
        *,
        state: _CtcStreamState,
        compact_row: int,
    ) -> None:
        if state.cached_states is None:
            return
        for host_arr, spec, cached in zip(
            self.state_input_hosts, self.state_specs, state.cached_states
        ):
            view = self._state_row_view(host_arr, spec, compact_row)
            if isinstance(view, np.ndarray):
                view[...] = cached
            else:
                host_arr[compact_row] = cached

    def _save_state_row(self, *, state: _CtcStreamState, compact_row: int) -> None:
        if state.cached_states is None:
            # First save for this stream: allocate its persistent per-row buffers
            # once, then reuse them on every subsequent tick.
            state.cached_states = [
                np.array(self._state_row_view(host_arr, spec, compact_row), copy=True)
                for host_arr, spec in zip(self.state_output_hosts, self.state_specs)
            ]
            return
        # Reuse the pre-allocated buffers (np.copyto, no new allocation) to avoid
        # churning ~126MB of state arrays every tick (see design F4).
        for buf, host_arr, spec in zip(
            state.cached_states, self.state_output_hosts, self.state_specs
        ):
            np.copyto(buf, self._state_row_view(host_arr, spec, compact_row))

    def _decode_ready(self, ready: list[_CtcStreamState]) -> None:
        ready = ready[: self.batch_size]
        pack_start = time.monotonic()
        self.x_host.fill(0.0)
        for host_arr in self.state_input_hosts:
            host_arr.fill(0)
        fill_ms = (time.monotonic() - pack_start) * 1000.0
        _gf0 = time.monotonic()
        for compact_row, state in enumerate(ready):
            frames = state.feature_stream.get_frames(
                state.processed_frames, self.chunk_length
            )
            self.x_host[compact_row, :, :] = np.asarray(
                frames, dtype=np.float32
            ).reshape(self.chunk_length, self.feat_dim)
        _pr0 = time.monotonic()
        gf_ms = (_pr0 - _gf0) * 1000.0
        for compact_row, state in enumerate(ready):
            self._pack_state_row(state=state, compact_row=compact_row)
        pr_ms = (time.monotonic() - _pr0) * 1000.0
        pack_ms = (time.monotonic() - pack_start) * 1000.0

        total_start = time.monotonic()
        _check(
            "acl.rt.memcpy x",
            acl.rt.memcpy(
                self.x_ptr,
                self.input_sizes[0],
                acl.util.numpy_to_ptr(self.x_host),
                self.x_host.nbytes,
                ACL_MEMCPY_HOST_TO_DEVICE,
            ),
        )
        state_h2d_start = time.monotonic()
        for ptr, size, host_arr in zip(
            self.state_a, self.input_sizes[1:], self.state_input_hosts
        ):
            _check(
                "acl.rt.memcpy ctc_state_h2d",
                acl.rt.memcpy(
                    ptr,
                    size,
                    acl.util.numpy_to_ptr(host_arr),
                    host_arr.nbytes,
                    ACL_MEMCPY_HOST_TO_DEVICE,
                ),
            )
        state_h2d_ms = (time.monotonic() - state_h2d_start) * 1000.0
        input_dataset = self.input_dataset_a
        output_dataset = self.output_dataset_a_to_b
        execute_start = time.monotonic()
        _check("acl.mdl.execute", acl.mdl.execute(self.model_id, input_dataset, output_dataset))
        execute_ms = (time.monotonic() - execute_start) * 1000.0
        state_d2h_start = time.monotonic()
        for ptr, size, host_arr in zip(
            self.state_b, self.output_sizes[1:], self.state_output_hosts
        ):
            _check(
                "acl.rt.memcpy ctc_state_d2h",
                acl.rt.memcpy(
                    acl.util.numpy_to_ptr(host_arr),
                    host_arr.nbytes,
                    ptr,
                    size,
                    ACL_MEMCPY_DEVICE_TO_HOST,
                ),
            )
        state_d2h_ms = (time.monotonic() - state_d2h_start) * 1000.0
        _check(
            "acl.rt.memcpy log_probs",
            acl.rt.memcpy(
                acl.util.numpy_to_ptr(self.log_probs_host),
                self.log_probs_host.nbytes,
                self.log_probs_ptr,
                self.output_sizes[0],
                ACL_MEMCPY_DEVICE_TO_HOST,
            ),
        )
        total_ms = (time.monotonic() - total_start) * 1000.0
        now = time.monotonic()
        callbacks: list[
            tuple[
                Callable[[str, dict[str, float | int | str]], None],
                str,
                dict[str, float | int | str],
            ]
        ] = []
        # Collect first-text log entries to emit AFTER releasing the lock so
        # np.argmax / string formatting / I/O do not block concurrent feed_and_get.
        first_text_log_entries: list[tuple] = []
        profile_hook = _BATCH_PROFILE_HOOK
        queue_ms_list: list[float] = []
        post_start = time.monotonic()
        with self.cond:
            for compact_row, state in enumerate(ready):
                if profile_hook is not None:
                    ready_since = state.ready_since
                    queue_ms_list.append(
                        (now - ready_since) * 1000.0 if ready_since > 0.0 else 0.0
                    )
                state.ready_since = 0.0
                token_ids = np.argmax(self.log_probs_host[compact_row], axis=1).tolist()
                for token_id in token_ids:
                    token_id = int(token_id)
                    if token_id != 0 and token_id != state.prev_token and token_id > 2:
                        state.emitted_tokens.append(token_id)
                    state.prev_token = token_id
                self._save_state_row(state=state, compact_row=compact_row)
                state.processed_frames += self.chunk_shift
                state.decode_ticks += 1
                state.text = _decode_tokens(self._tokens_path, state.emitted_tokens).strip()
                state.version += 1
                if state.text and not state.first_nonempty_logged:
                    state.first_nonempty_logged = True
                    # Collect data for out-of-lock logging below
                    first_text_log_entries.append((
                        state.trace_id or "-",
                        state.slot,
                        state.total_samples * 1000.0 / SAMPLE_RATE,
                        (now - state.registered_at) * 1000.0,
                        state.decode_ticks,
                        len(ready),
                        execute_ms,
                        total_ms,
                        state.text[:80],
                    ))
                state.last_stats = {
                    "scheduler": "ctc_om",
                    "batch_size": len(ready),
                    "pending_depth": len(ready),
                    "decode_ms": execute_ms,
                    "decode_loops": 1,
                    "ready_checks": 1,
                    "batch_wait_ms": float(self.cfg.wait_ms),
                    "lock_wait_ms": 0.0,
                    "queue_ms": 0.0,
                    "updated_at": now,
                    "ctc_total_tick_ms": total_ms,
                    "ctc_state_h2d_ms": state_h2d_ms,
                    "ctc_state_d2h_ms": state_d2h_ms,
                    "ctc_decode_ticks": state.decode_ticks,
                }
                if (
                    state.text
                    and state.result_callback is not None
                    and state.text != state.last_callback_text
                ):
                    state.last_callback_text = state.text
                    callbacks.append(
                        (state.result_callback, state.text, dict(state.last_stats))
                    )
            self.cond.notify_all()
        post_ms = (time.monotonic() - post_start) * 1000.0
        cb_start = time.monotonic()
        # Log first-text events outside the lock to avoid blocking feed_and_get callers.
        for entry in first_text_log_entries:
            logger.info(
                "CTC_ONLINE_FIRST_TEXT traceId=%s slot=%s audio_ms=%.1f "
                "since_register_ms=%.1f decode_ticks=%s batch_size=%s "
                "execute_ms=%.1f total_ms=%.1f text=%r",
                *entry,
            )
        for callback, text, stats in callbacks:
            try:
                callback(text, stats)
            except Exception:
                logger.debug("CTC result callback failed", exc_info=True)
        cb_ms = (time.monotonic() - cb_start) * 1000.0
        logger.info(
            "CTC_ONLINE_BATCH_TIMING batch_size=%s execute_ms=%.1f "
            "state_h2d_ms=%.1f state_d2h_ms=%.1f total_ms=%.1f "
            "pack_ms=%.1f fill_ms=%.1f gf_ms=%.1f pr_ms=%.1f post_ms=%.1f cb_ms=%.1f traces=%s",
            len(ready),
            execute_ms,
            state_h2d_ms,
            state_d2h_ms,
            total_ms,
            pack_ms,
            fill_ms,
            gf_ms,
            pr_ms,
            post_ms,
            cb_ms,
            ",".join(s.trace_id or "-" for s in ready[:8]),
        )
        if profile_hook is not None:
            try:
                profile_hook(
                    {
                        "t": now,
                        "batch_size": len(ready),
                        "execute_ms": execute_ms,
                        "total_ms": total_ms,
                        "state_h2d_ms": state_h2d_ms,
                        "state_d2h_ms": state_d2h_ms,
                        "queue_ms": queue_ms_list,
                    }
                )
            except Exception:
                logger.debug("CTC batch profile hook failed", exc_info=True)

    def _decode_ready_resident(
        self, ready: list[_CtcStreamState], active: list[_CtcStreamState]
    ) -> None:
        """Device-resident state decode path (config ctc_device_resident_state).

        Streams stay pinned to their fixed register() slot; the 116 state tensors
        stay resident in device buffers across ticks.  The common (fast) tick
        does NO host state transfer at all: fill x, execute A->B, then a single
        contiguous device-to-device copy state_b->state_a per tensor.  A tick is
        promoted to a slower host-merge only when it involves a first-time-ready
        slot (needs zero-init input state) or an initialised non-ready active
        slot (its state must be preserved, since the OM advances all 52 rows).
        See design-device-resident-state.md."""
        ready = ready[: self.batch_size]
        if not ready:
            return
        ready_slots = {s.slot for s in ready}
        init_states = [s for s in ready if not s.resident_initialized]
        init_slots = [s.slot for s in init_states]
        preserve_slots = [
            s.slot
            for s in active
            if s.slot not in ready_slots and s.resident_initialized
        ]
        special = bool(init_slots) or bool(preserve_slots)

        self.x_host.fill(0.0)
        gf_start = time.monotonic()
        for state in ready:
            frames = state.feature_stream.get_frames(
                state.processed_frames, self.chunk_length
            )
            self.x_host[state.slot, :, :] = np.asarray(
                frames, dtype=np.float32
            ).reshape(self.chunk_length, self.feat_dim)
        gf_ms = (time.monotonic() - gf_start) * 1000.0

        total_start = time.monotonic()
        state_h2d_ms = 0.0
        state_d2h_ms = 0.0
        host_pre = None
        if special:
            _t = time.monotonic()
            for host_arr, ptr, size in zip(
                self.state_output_hosts, self.state_a, self.input_sizes[1:]
            ):
                _check(
                    "acl.rt.memcpy resident state_a d2h",
                    acl.rt.memcpy(
                        acl.util.numpy_to_ptr(host_arr),
                        host_arr.nbytes,
                        ptr,
                        size,
                        ACL_MEMCPY_DEVICE_TO_HOST,
                    ),
                )
            state_d2h_ms += (time.monotonic() - _t) * 1000.0
            host_pre = self.state_output_hosts
            if init_slots:
                for host_arr, spec in zip(host_pre, self.state_specs):
                    sl = [slice(None)] * host_arr.ndim
                    for slot in init_slots:
                        sl[spec.batch_dim] = slot
                        host_arr[tuple(sl)] = 0
                _t = time.monotonic()
                for host_arr, ptr, size in zip(
                    host_pre, self.state_a, self.input_sizes[1:]
                ):
                    _check(
                        "acl.rt.memcpy resident state_a h2d init",
                        acl.rt.memcpy(
                            ptr,
                            size,
                            acl.util.numpy_to_ptr(host_arr),
                            host_arr.nbytes,
                            ACL_MEMCPY_HOST_TO_DEVICE,
                        ),
                    )
                state_h2d_ms += (time.monotonic() - _t) * 1000.0

        _check(
            "acl.rt.memcpy resident x",
            acl.rt.memcpy(
                self.x_ptr,
                self.input_sizes[0],
                acl.util.numpy_to_ptr(self.x_host),
                self.x_host.nbytes,
                ACL_MEMCPY_HOST_TO_DEVICE,
            ),
        )
        execute_start = time.monotonic()
        _check(
            "acl.mdl.execute resident",
            acl.mdl.execute(
                self.model_id, self.input_dataset_a, self.output_dataset_a_to_b
            ),
        )
        execute_ms = (time.monotonic() - execute_start) * 1000.0
        _check(
            "acl.rt.memcpy resident log_probs",
            acl.rt.memcpy(
                acl.util.numpy_to_ptr(self.log_probs_host),
                self.log_probs_host.nbytes,
                self.log_probs_ptr,
                self.output_sizes[0],
                ACL_MEMCPY_DEVICE_TO_HOST,
            ),
        )

        d2d_ms = 0.0
        if not special:
            _t = time.monotonic()
            for ptr_a, ptr_b, size in zip(
                self.state_a, self.state_b, self.input_sizes[1:]
            ):
                _check(
                    "acl.rt.memcpy resident b2a d2d",
                    acl.rt.memcpy(
                        ptr_a, size, ptr_b, size, ACL_MEMCPY_DEVICE_TO_DEVICE
                    ),
                )
            d2d_ms = (time.monotonic() - _t) * 1000.0
        else:
            _t = time.monotonic()
            for host_arr, ptr, size in zip(
                self.state_input_hosts, self.state_b, self.input_sizes[1:]
            ):
                _check(
                    "acl.rt.memcpy resident state_b d2h",
                    acl.rt.memcpy(
                        acl.util.numpy_to_ptr(host_arr),
                        host_arr.nbytes,
                        ptr,
                        size,
                        ACL_MEMCPY_DEVICE_TO_HOST,
                    ),
                )
            state_d2h_ms += (time.monotonic() - _t) * 1000.0
            if preserve_slots and host_pre is not None:
                for host_post, host_pre_arr, spec in zip(
                    self.state_input_hosts, host_pre, self.state_specs
                ):
                    sl = [slice(None)] * host_post.ndim
                    for slot in preserve_slots:
                        sl[spec.batch_dim] = slot
                        host_post[tuple(sl)] = host_pre_arr[tuple(sl)]
            _t = time.monotonic()
            for host_arr, ptr, size in zip(
                self.state_input_hosts, self.state_a, self.input_sizes[1:]
            ):
                _check(
                    "acl.rt.memcpy resident state_a h2d final",
                    acl.rt.memcpy(
                        ptr,
                        size,
                        acl.util.numpy_to_ptr(host_arr),
                        host_arr.nbytes,
                        ACL_MEMCPY_HOST_TO_DEVICE,
                    ),
                )
            state_h2d_ms += (time.monotonic() - _t) * 1000.0

        total_ms = (time.monotonic() - total_start) * 1000.0
        for state in init_states:
            state.resident_initialized = True

        now = time.monotonic()
        callbacks: list[
            tuple[
                Callable[[str, dict[str, float | int | str]], None],
                str,
                dict[str, float | int | str],
            ]
        ] = []
        first_text_log_entries: list[tuple] = []
        profile_hook = _BATCH_PROFILE_HOOK
        queue_ms_list: list[float] = []
        post_start = time.monotonic()
        with self.cond:
            for state in ready:
                row = state.slot
                if profile_hook is not None:
                    ready_since = state.ready_since
                    queue_ms_list.append(
                        (now - ready_since) * 1000.0 if ready_since > 0.0 else 0.0
                    )
                state.ready_since = 0.0
                token_ids = np.argmax(self.log_probs_host[row], axis=1).tolist()
                for token_id in token_ids:
                    token_id = int(token_id)
                    if token_id != 0 and token_id != state.prev_token and token_id > 2:
                        state.emitted_tokens.append(token_id)
                    state.prev_token = token_id
                state.processed_frames += self.chunk_shift
                state.decode_ticks += 1
                state.text = _decode_tokens(
                    self._tokens_path, state.emitted_tokens
                ).strip()
                state.version += 1
                if state.text and not state.first_nonempty_logged:
                    state.first_nonempty_logged = True
                    first_text_log_entries.append(
                        (
                            state.trace_id or "-",
                            state.slot,
                            state.total_samples * 1000.0 / SAMPLE_RATE,
                            (now - state.registered_at) * 1000.0,
                            state.decode_ticks,
                            len(ready),
                            execute_ms,
                            total_ms,
                            state.text[:80],
                        )
                    )
                state.last_stats = {
                    "scheduler": "ctc_om",
                    "batch_size": len(ready),
                    "pending_depth": len(ready),
                    "decode_ms": execute_ms,
                    "decode_loops": 1,
                    "ready_checks": 1,
                    "batch_wait_ms": float(self.cfg.wait_ms),
                    "lock_wait_ms": 0.0,
                    "queue_ms": 0.0,
                    "updated_at": now,
                    "ctc_total_tick_ms": total_ms,
                    "ctc_decode_ticks": state.decode_ticks,
                }
                if (
                    state.text
                    and state.result_callback is not None
                    and state.text != state.last_callback_text
                ):
                    state.last_callback_text = state.text
                    callbacks.append(
                        (state.result_callback, state.text, dict(state.last_stats))
                    )
            self.cond.notify_all()
        post_ms = (time.monotonic() - post_start) * 1000.0
        for entry in first_text_log_entries:
            logger.info(
                "CTC_ONLINE_FIRST_TEXT traceId=%s slot=%s audio_ms=%.1f "
                "since_register_ms=%.1f decode_ticks=%s batch_size=%s "
                "execute_ms=%.1f total_ms=%.1f text=%r",
                *entry,
            )
        for callback, text, stats in callbacks:
            try:
                callback(text, stats)
            except Exception:
                logger.debug("CTC result callback failed", exc_info=True)
        logger.info(
            "CTC_ONLINE_BATCH_TIMING path=resident batch_size=%s special=%s "
            "execute_ms=%.1f total_ms=%.1f gf_ms=%.1f d2d_ms=%.1f "
            "state_h2d_ms=%.1f state_d2h_ms=%.1f post_ms=%.1f init=%s preserve=%s",
            len(ready),
            int(special),
            execute_ms,
            total_ms,
            gf_ms,
            d2d_ms,
            state_h2d_ms,
            state_d2h_ms,
            post_ms,
            len(init_slots),
            len(preserve_slots),
        )
        if profile_hook is not None:
            try:
                profile_hook(
                    {
                        "t": now,
                        "batch_size": len(ready),
                        "execute_ms": execute_ms,
                        "total_ms": total_ms,
                        "state_h2d_ms": state_h2d_ms,
                        "state_d2h_ms": state_d2h_ms,
                        "queue_ms": queue_ms_list,
                    }
                )
            except Exception:
                logger.debug("CTC batch profile hook failed", exc_info=True)


_RUNTIMES: dict[CtcStreamingConfig, _CtcOmRuntime] = {}
_RUNTIMES_LOCK = threading.Lock()


def _get_runtime(cfg: CtcStreamingConfig) -> _CtcOmRuntime:
    with _RUNTIMES_LOCK:
        runtime = _RUNTIMES.get(cfg)
        if runtime is None:
            runtime = _CtcOmRuntime(cfg)
            _RUNTIMES[cfg] = runtime
        return runtime


def config_from_app(cfg: Config) -> CtcStreamingConfig:
    model_dir = Path(str(getattr(cfg, "ctc_om_model_dir", "") or ""))
    om_model_path = str(getattr(cfg, "ctc_om_model_path", "") or "").strip()
    if not om_model_path and model_dir:
        candidate = model_dir / "model_linux_aarch64.om"
        om_model_path = str(candidate)
    onnx_model_path = str(getattr(cfg, "ctc_onnx_model_path", "") or "").strip()
    tokens_path = str(getattr(cfg, "ctc_tokens_path", "") or "").strip()
    return CtcStreamingConfig(
        om_model_path=om_model_path,
        onnx_model_path=onnx_model_path,
        tokens_path=tokens_path,
        batch_size=int(getattr(cfg, "ctc_decode_batch_size", 52) or 52),
        wait_ms=int(getattr(cfg, "ctc_decode_wait_ms", 10) or 10),
        ready_coalesce_ms=int(getattr(cfg, "ctc_ready_coalesce_ms", 30) or 30),
        result_wait_ms=int(getattr(cfg, "ctc_result_wait_ms", 120) or 120),
        device_id=int(getattr(cfg, "ctc_device_id", 0) or 0),
        device_resident_state=bool(
            getattr(cfg, "ctc_device_resident_state", False)
        ),
        resident_pace_ms=int(getattr(cfg, "ctc_resident_pace_ms", 0) or 0),
        firstframe_bypass_coalesce=bool(
            getattr(cfg, "ctc_firstframe_bypass_coalesce", True)
        ),
    )


def maybe_warmup_from_app(cfg: Config) -> None:
    backend = str(getattr(cfg, "streaming_partial_backend", "vllm") or "vllm")
    if backend.strip().lower() != "ctc_om":
        return
    _get_runtime(config_from_app(cfg))


class CtcStreamingSession:
    def __init__(self, cfg: CtcStreamingConfig) -> None:
        if not cfg.om_model_path or not cfg.onnx_model_path or not cfg.tokens_path:
            raise ValueError("ctc_om requires OM, ONNX, and tokens paths")
        self._cfg = cfg
        self._runtime = _get_runtime(cfg)
        self._state: _CtcStreamState | None = None
        self._fed_samples = 0
        # Serializes waveform feeding for THIS stream. The feature extractor
        # (sherpa OnlineStream) is not thread-safe, and the _fed_samples
        # check-and-advance must be atomic; the CTC light prefeed submits from a
        # shared ThreadPoolExecutor so multiple frames of the same stream can run
        # concurrently. Per-stream (not global) so cross-stream concurrency is
        # preserved.
        self._feed_lock = threading.Lock()
        self._result_callback: Callable[[str, dict[str, float | int | str]], None] | None = None

    def reset(self) -> None:
        self.close()

    def close(self) -> None:
        if self._state is not None:
            self._runtime.unregister(self._state)
        self._state = None
        self._fed_samples = 0

    def cached_result(self) -> tuple[str, dict[str, float | int | str]]:
        if self._state is None:
            return "", {"scheduler": "ctc_om", "batch_size": 0, "decode_ms": 0.0}
        with self._runtime.cond:
            return self._state.text, dict(self._state.last_stats)

    def set_result_callback(
        self, callback: Callable[[str, dict[str, float | int | str]], None] | None
    ) -> None:
        self._result_callback = callback
        if self._state is None:
            return
        with self._runtime.cond:
            self._state.result_callback = callback

    def accept_incremental(self, chunk: np.ndarray, *, trace_id: str = "") -> None:
        """Feed a single new audio chunk without cumulative-buffer tracking.

        Unlike accept_cumulative, this method does NOT assume the caller passes
        a growing buffer starting from sample 0.  It is safe to call after
        adopting a pre-fill session (where _fed_samples already reflects
        pre-gate audio) because we advance _fed_samples by the new chunk size
        and feed only the new chunk to the runtime.
        """
        with self._feed_lock:
            if self._state is None:
                self._state = self._runtime.register(trace_id=trace_id)
                if self._result_callback is not None:
                    with self._runtime.cond:
                        self._state.result_callback = self._result_callback
            if chunk is None or chunk.size == 0:
                return
            chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
            self._fed_samples += len(chunk)
            self._runtime.feed_and_get(
                self._state,
                chunk=chunk,
                total_samples=self._fed_samples,
                trace_id=trace_id,
                wait_for_result=False,
            )

    def accept_cumulative(
        self, pcm: np.ndarray, *, max_decode_steps: int = 0, trace_id: str = ""
    ) -> tuple[str, dict[str, float | int | str]]:
        wait_for_result = max_decode_steps >= 0
        with self._feed_lock:
            if self._state is None:
                self._state = self._runtime.register(trace_id=trace_id)
                if self._result_callback is not None:
                    with self._runtime.cond:
                        self._state.result_callback = self._result_callback
            if pcm.ndim != 1:
                pcm = pcm.reshape(-1)
            total = int(len(pcm))
            start = min(self._fed_samples, total)
            if total > start:
                # Copy out of the (possibly shared/growing) source buffer so the
                # async batcher never aliases audio the producer keeps mutating.
                chunk = np.array(pcm[start:total], dtype=np.float32)
                self._fed_samples = total
            else:
                chunk = None
            text, stats = self._runtime.feed_and_get(
                self._state,
                chunk=chunk,
                total_samples=total,
                trace_id=trace_id,
                wait_for_result=wait_for_result,
            )
            stats["delta_samples"] = max(0, total - start)
        return text, stats
