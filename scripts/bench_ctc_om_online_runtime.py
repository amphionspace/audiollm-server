#!/usr/bin/env python3
"""Benchmark a streaming Zipformer CTC Ascend OM as an online stateful runtime.

The streaming CTC OM has one feature input, followed by cached state tensors.
It returns log_probs plus the next cached states. This benchmark preallocates
all device buffers, repeatedly runs acl.mdl.execute(), and copies output states
back to the corresponding input state buffers on device.
"""

from __future__ import annotations

import argparse
import os
import statistics as stats
import sys
import time
import wave

import acl
import numpy as np


ACL_ERROR_NONE = 0
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_DEVICE_TO_DEVICE = 3
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def _check(name: str, ret: int) -> None:
    if ret != ACL_ERROR_NONE:
        raise RuntimeError(f"{name} failed ret={ret}")


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(q * len(ordered)) - 1))
    return ordered[idx]


def _summarize(name: str, values: list[float]) -> str:
    return (
        f"{name} min={min(values):.2f} p50={stats.median(values):.2f} "
        f"mean={sum(values) / len(values):.2f} p90={_pct(values, 0.90):.2f} "
        f"p99={_pct(values, 0.99):.2f} max={max(values):.2f} n={len(values)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to static BS52 CTC OM")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--wav", help="Optional WAV file for a greedy decode smoke test")
    parser.add_argument("--tokens", help="tokens.txt, used only with --wav")
    parser.add_argument("--bbpe-model", help="Unused compatibility arg for older commands")
    parser.add_argument(
        "--onnx-model-for-features",
        help="ONNX model used only to create a sherpa stream for feature extraction",
    )
    parser.add_argument(
        "--state-mode",
        choices=("copy", "pingpong", "none"),
        default="copy",
        help=(
            "State handling mode. copy copies output states back to input states; "
            "pingpong alternates two state buffer sets so outputs become next "
            "inputs; none measures execute only."
        ),
    )
    args = parser.parse_args()

    _check("acl.init", acl.init())
    _check("acl.rt.set_device", acl.rt.set_device(args.device_id))
    _, ret = acl.rt.create_context(args.device_id)
    _check("acl.rt.create_context", ret)

    model_id, ret = acl.mdl.load_from_file(args.model)
    _check("acl.mdl.load_from_file", ret)
    desc = acl.mdl.create_desc()
    _check("acl.mdl.get_desc", acl.mdl.get_desc(desc, model_id))

    num_inputs = int(acl.mdl.get_num_inputs(desc))
    num_outputs = int(acl.mdl.get_num_outputs(desc))
    if num_inputs < 2 or num_outputs != num_inputs:
        raise RuntimeError(
            f"expected x + states and log_probs + states, got "
            f"inputs={num_inputs} outputs={num_outputs}"
        )

    input_sizes = [int(acl.mdl.get_input_size_by_index(desc, i)) for i in range(num_inputs)]
    output_sizes = [
        int(acl.mdl.get_output_size_by_index(desc, i)) for i in range(num_outputs)
    ]
    for i in range(1, num_inputs):
        if input_sizes[i] != output_sizes[i]:
            raise RuntimeError(
                f"state size mismatch index={i} input={input_sizes[i]} "
                f"output={output_sizes[i]}"
            )

    def _malloc_zero(size: int, label: str) -> int:
        ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
        _check(f"acl.rt.malloc {label}", ret)
        _check(f"acl.rt.memset {label}", acl.rt.memset(ptr, size, 0, size))
        return ptr

    def _add_buffer(dataset: int, ptr: int, size: int, label: str) -> None:
        data_buf = acl.create_data_buffer(ptr, size)
        _, ret = acl.mdl.add_dataset_buffer(dataset, data_buf)
        _check(f"acl.mdl.add_dataset_buffer {label}", ret)

    x_ptr = _malloc_zero(input_sizes[0], "x")
    log_probs_ptr = _malloc_zero(output_sizes[0], "log_probs")
    state_a = [_malloc_zero(size, f"state_a_{i}") for i, size in enumerate(input_sizes[1:], 1)]
    state_b = [_malloc_zero(size, f"state_b_{i}") for i, size in enumerate(input_sizes[1:], 1)]

    input_dataset_a = acl.mdl.create_dataset()
    input_dataset_b = acl.mdl.create_dataset()
    output_dataset_a_to_b = acl.mdl.create_dataset()
    output_dataset_b_to_a = acl.mdl.create_dataset()

    _add_buffer(input_dataset_a, x_ptr, input_sizes[0], "input_a_x")
    _add_buffer(input_dataset_b, x_ptr, input_sizes[0], "input_b_x")
    _add_buffer(output_dataset_a_to_b, log_probs_ptr, output_sizes[0], "output_ab_log_probs")
    _add_buffer(output_dataset_b_to_a, log_probs_ptr, output_sizes[0], "output_ba_log_probs")
    for i, (ptr_a, ptr_b, size) in enumerate(zip(state_a, state_b, input_sizes[1:]), 1):
        _add_buffer(input_dataset_a, ptr_a, size, f"input_a_state_{i}")
        _add_buffer(input_dataset_b, ptr_b, size, f"input_b_state_{i}")
        _add_buffer(output_dataset_a_to_b, ptr_b, size, f"output_ab_state_{i}")
        _add_buffer(output_dataset_b_to_a, ptr_a, size, f"output_ba_state_{i}")

    # The copy mode uses A as input and B as output, then copies B back to A.
    state_pairs = list(zip(state_a, state_b, input_sizes[1:]))

    input_mb = sum(input_sizes) / 1024 / 1024
    output_mb = sum(output_sizes) / 1024 / 1024
    print(
        f"CTC_ONLINE_OM_INFO model={args.model} inputs={num_inputs} "
        f"outputs={num_outputs} input_mb={input_mb:.2f} output_mb={output_mb:.2f} "
        f"state_tensors={len(state_pairs)} state_mode={args.state_mode}"
    )
    sys.stdout.flush()

    for _ in range(max(0, args.warmup)):
        if args.state_mode == "pingpong":
            _check(
                "acl.mdl.execute warmup",
                acl.mdl.execute(model_id, input_dataset_a, output_dataset_a_to_b),
            )
            _check(
                "acl.mdl.execute warmup",
                acl.mdl.execute(model_id, input_dataset_b, output_dataset_b_to_a),
            )
        else:
            _check(
                "acl.mdl.execute warmup",
                acl.mdl.execute(model_id, input_dataset_a, output_dataset_a_to_b),
            )
        if args.state_mode == "copy":
            for in_ptr, out_ptr, size in state_pairs:
                _check(
                    "acl.rt.memcpy warmup state",
                    acl.rt.memcpy(in_ptr, size, out_ptr, size, ACL_MEMCPY_DEVICE_TO_DEVICE),
                )

    execute_ms: list[float] = []
    state_copy_ms: list[float] = []
    total_ms: list[float] = []
    use_a_as_input = True
    for _ in range(max(1, args.iters)):
        if args.state_mode == "pingpong":
            input_dataset = input_dataset_a if use_a_as_input else input_dataset_b
            output_dataset = output_dataset_a_to_b if use_a_as_input else output_dataset_b_to_a
            use_a_as_input = not use_a_as_input
        else:
            input_dataset = input_dataset_a
            output_dataset = output_dataset_a_to_b
        total_start = time.perf_counter()
        execute_start = time.perf_counter()
        _check("acl.mdl.execute", acl.mdl.execute(model_id, input_dataset, output_dataset))
        execute_done = time.perf_counter()
        if args.state_mode == "copy":
            copy_start = time.perf_counter()
            for in_ptr, out_ptr, size in state_pairs:
                _check(
                    "acl.rt.memcpy state",
                    acl.rt.memcpy(in_ptr, size, out_ptr, size, ACL_MEMCPY_DEVICE_TO_DEVICE),
                )
            copy_done = time.perf_counter()
            state_copy_ms.append((copy_done - copy_start) * 1000.0)
        total_done = time.perf_counter()
        execute_ms.append((execute_done - execute_start) * 1000.0)
        total_ms.append((total_done - total_start) * 1000.0)

    print("CTC_ONLINE_OM_BENCH " + _summarize("execute_ms", execute_ms))
    if state_copy_ms:
        print("CTC_ONLINE_OM_BENCH " + _summarize("state_copy_ms", state_copy_ms))
    print("CTC_ONLINE_OM_BENCH " + _summarize("total_tick_ms", total_ms))
    sys.stdout.flush()

    if args.wav:
        if not args.tokens or not args.onnx_model_for_features:
            raise RuntimeError("--wav requires --tokens and --onnx-model-for-features")
        text = _decode_wav_smoke(
            wav=args.wav,
            tokens=args.tokens,
            bbpe_model=args.bbpe_model,
            onnx_model=args.onnx_model_for_features,
            input_sizes=input_sizes,
            output_sizes=output_sizes,
            input_dataset_a=input_dataset_a,
            input_dataset_b=input_dataset_b,
            output_dataset_a_to_b=output_dataset_a_to_b,
            output_dataset_b_to_a=output_dataset_b_to_a,
            model_id=model_id,
            x_ptr=x_ptr,
            log_probs_ptr=log_probs_ptr,
            state_a=state_a,
            state_b=state_b,
        )
        print(f"CTC_ONLINE_OM_DECODE text={text!r}")
        sys.stdout.flush()

    # pyACL may abort during interpreter teardown after loading models. The
    # benchmark has already printed all results, so exit directly for repeatable
    # automation while the short-lived process returns success.
    os._exit(0)


def _read_wave(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path) as f:
        if f.getnchannels() != 1 or f.getsampwidth() != 2:
            raise RuntimeError("expected mono int16 WAV")
        samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        return samples.astype(np.float32) / 32768.0, int(f.getframerate())


def _decode_wav_smoke(
    *,
    wav: str,
    tokens: str,
    bbpe_model: str,
    onnx_model: str,
    input_sizes: list[int],
    output_sizes: list[int],
    input_dataset_a: int,
    input_dataset_b: int,
    output_dataset_a_to_b: int,
    output_dataset_b_to_a: int,
    model_id: int,
    x_ptr: int,
    log_probs_ptr: int,
    state_a: list[int],
    state_b: list[int],
) -> str:
    # Import lazily so pure performance benchmarks do not load ONNX Runtime.
    import sherpa_onnx

    chunk_length = 77
    feat_dim = 80
    chunk_shift = 64
    vocab_size = 1000
    batch = input_sizes[0] // (chunk_length * feat_dim * 4)
    num_output_frames = output_sizes[0] // (batch * vocab_size * 4)
    if input_sizes[0] != batch * chunk_length * feat_dim * 4:
        raise RuntimeError(f"unexpected x input size: {input_sizes[0]}")
    if output_sizes[0] != batch * num_output_frames * vocab_size * 4:
        raise RuntimeError(f"unexpected log_probs output size: {output_sizes[0]}")
    for ptr_a, ptr_b, size in zip(state_a, state_b, input_sizes[1:]):
        _check("acl.rt.memset decode state_a", acl.rt.memset(ptr_a, size, 0, size))
        _check("acl.rt.memset decode state_b", acl.rt.memset(ptr_b, size, 0, size))

    rec = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
        tokens=tokens,
        model=onnx_model,
        num_threads=1,
        provider="cpu",
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
    )
    stream = rec.create_stream()
    samples, sample_rate = _read_wave(wav)
    stream.accept_waveform(sample_rate, samples)
    stream.input_finished()

    # With snip_edges=false, sherpa's frame count is about duration*100 - 1.
    # Stay one frame conservative to avoid GetFrames aborting on boundary.
    num_frames = max(0, int(len(samples) * 100 / sample_rate) - 1)
    x = np.zeros((batch, chunk_length, feat_dim), dtype=np.float32)
    log_probs = np.empty((batch, num_output_frames, vocab_size), dtype=np.float32)
    emitted: list[int] = []
    prev = 0
    use_a_as_input = True
    ticks = 0
    debug_argmax: list[list[int]] = []
    debug_global: list[dict[str, int]] = []
    for start in range(0, max(0, num_frames - chunk_length + 1), chunk_shift):
        frames = np.asarray(stream.get_frames(start, chunk_length), dtype=np.float32)
        x[0, :, :] = frames.reshape(chunk_length, feat_dim)
        x[1:, :, :] = x[0:1, :, :]
        _check(
            "acl.rt.memcpy x",
            acl.rt.memcpy(
                x_ptr,
                input_sizes[0],
                acl.util.numpy_to_ptr(x),
                x.nbytes,
                ACL_MEMCPY_HOST_TO_DEVICE,
            ),
        )
        input_dataset = input_dataset_a if use_a_as_input else input_dataset_b
        output_dataset = output_dataset_a_to_b if use_a_as_input else output_dataset_b_to_a
        use_a_as_input = not use_a_as_input
        _check("acl.mdl.execute decode", acl.mdl.execute(model_id, input_dataset, output_dataset))
        _check(
            "acl.rt.memcpy log_probs",
            acl.rt.memcpy(
                acl.util.numpy_to_ptr(log_probs),
                log_probs.nbytes,
                log_probs_ptr,
                output_sizes[0],
                ACL_MEMCPY_DEVICE_TO_HOST,
            ),
        )
        token_ids = [int(i) for i in np.argmax(log_probs[0], axis=1).tolist()]
        if len(debug_argmax) < 5:
            debug_argmax.append(token_ids)
            all_ids = np.argmax(log_probs, axis=2)
            debug_global.append(
                {
                    "nonblank": int(np.count_nonzero(all_ids)),
                    "max_id": int(all_ids.max()),
                    "min_id": int(all_ids.min()),
                    "nan_count": int(np.count_nonzero(np.isnan(log_probs))),
                    "min_logp": float(np.nanmin(log_probs)),
                    "max_logp": float(np.nanmax(log_probs)),
                    "blank0": float(log_probs[0, 0, 0]),
                    "tok1": float(log_probs[0, 0, 1]),
                }
            )
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id != 0 and token_id != prev and token_id > 2:
                emitted.append(token_id)
            prev = token_id
        ticks += 1

    text = _decode_byte_tokens(tokens, emitted)
    print(
        f"CTC_ONLINE_OM_DECODE_INFO wav={wav} duration={len(samples) / sample_rate:.3f} "
        f"feature_frames={num_frames} ticks={ticks} emitted_tokens={len(emitted)} "
        f"debug_argmax={debug_argmax} debug_global={debug_global}"
    )
    return text


def _decode_byte_tokens(tokens_path: str, token_ids: list[int]) -> str:
    bpe_unk = chr(8263)
    printable_base_chars = (
        list(range(256, 287 + 1))
        + list(range(32, 126 + 1))
        + list(range(288, 305 + 1))
        + list(range(308, 318 + 1))
        + list(range(321, 328 + 1))
        + list(range(330, 382 + 1))
        + list(range(384, 422 + 1))
    )
    bchar_to_byte = {chr(ch): b for b, ch in enumerate(printable_base_chars)}
    bchar_to_byte[bpe_unk] = 32
    tokens: dict[int, str] = {}
    with open(tokens_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                tokens[int(parts[-1])] = parts[0]

    data = b""
    for token_id in token_ids:
        token = tokens.get(int(token_id), "")
        for ch in token:
            if ch == "▁":
                continue
            if ch in bchar_to_byte:
                data += bytes([bchar_to_byte[ch]])
    return data.decode("utf-8", errors="ignore")


if __name__ == "__main__":
    raise SystemExit(main())
