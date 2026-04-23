#!/usr/bin/env python3
"""Stress-test the emotion-recognition vLLM service.

Directly benchmarks the OpenAI-compatible chat completions endpoint exposed
by the AmphionSE model (default ``http://localhost:8222``), using synthetic
16 kHz mono PCM audio wrapped in WAV. Ramps concurrency upward and measures
end-to-end latency, RTF (real-time factor), throughput and error rate at
each level. Stops once RTF p50 exceeds a threshold or the error rate spikes.

Request payload mirrors ``backend.emotion.client._build_messages`` so the
benchmark exercises the exact same path the production service uses.

Usage examples
--------------

Default ramp 1..128 with 5 s synthetic audio:

    python scripts/bench_emotion_vllm.py

Ramp explicit list, multiple audio durations, save JSON:

    python scripts/bench_emotion_vllm.py \
        --base-url http://localhost:8222 \
        --model AmphionSE \
        --mode ser \
        --audio-secs 3,5,10,20 \
        --concurrency-list 1,2,4,8,16,32,64 \
        --output bench_emotion.json

"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import statistics
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np

# Allow running as a standalone script without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.emotion.prompt import get_prompt, normalize_mode  # noqa: E402


SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Audio synthesis
# ---------------------------------------------------------------------------


def make_wav_bytes(duration_s: float, kind: str = "sine", seed: int = 0) -> bytes:
    """Return a complete WAV (RIFF) byte string of *duration_s* seconds.

    ``seed`` controls per-request variation: with the same ``seed`` the bytes
    are identical (prefix-cache friendly); with a unique seed every call the
    audio differs and vLLM's prefix cache cannot reuse the prefill.
    """
    n = max(1, int(round(duration_s * SAMPLE_RATE)))
    if kind == "silence":
        pcm = np.zeros(n, dtype=np.int16)
        if seed:
            rng = np.random.default_rng(seed)
            pcm = (rng.standard_normal(n).astype(np.float32) * 1e-4 * 32767.0).astype(np.int16)
    elif kind == "sine":
        t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
        freq = 440.0 + (seed % 200)  # vary 440..639 Hz so cache cannot hit
        phase = (seed * 0.013) % (2 * math.pi)
        pcm = (0.05 * np.sin(2 * math.pi * freq * t + phase) * 32767.0).astype(np.int16)
    elif kind == "noise":
        rng = np.random.default_rng(seed or 1)
        pcm = (rng.standard_normal(n).astype(np.float32) * 0.02 * 32767.0).astype(np.int16)
    else:
        raise ValueError(f"Unknown audio kind: {kind}")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def make_wav_b64(duration_s: float, kind: str = "sine", seed: int = 0) -> str:
    return base64.b64encode(make_wav_bytes(duration_s, kind, seed)).decode("ascii")


# ---------------------------------------------------------------------------
# Request construction (kept in lock-step with backend/emotion/client.py)
# ---------------------------------------------------------------------------


_REQ_COUNTER = 0


def _next_seed() -> int:
    global _REQ_COUNTER
    _REQ_COUNTER += 1
    return _REQ_COUNTER


def build_payload(
    audio_b64: str,
    *,
    model: str,
    mode: str,
    max_tokens: int | None,
) -> dict[str, Any]:
    mode_norm = normalize_mode(mode)
    if max_tokens is None:
        max_tokens = 32 if mode_norm == "ser" else 256
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": get_prompt(mode_norm)},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    },
                ],
            }
        ],
        "max_tokens": int(max_tokens),
    }


# ---------------------------------------------------------------------------
# Per-request execution
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    ok: bool
    latency_s: float
    status: int = 0
    error: str = ""
    completion_tokens: int = 0
    prompt_tokens: int = 0
    output_text: str = ""


async def do_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    timeout: float,
) -> RequestResult:
    t0 = time.perf_counter()
    try:
        r = await client.post(url, json=payload, timeout=timeout)
        elapsed = time.perf_counter() - t0
        if r.status_code != 200:
            return RequestResult(
                ok=False,
                latency_s=elapsed,
                status=r.status_code,
                error=r.text[:200],
            )
        data = r.json()
        usage = data.get("usage") or {}
        choice = (data.get("choices") or [{}])[0]
        msg_content = (choice.get("message") or {}).get("content")
        if isinstance(msg_content, list):
            text = "".join(
                (c.get("text") or "") for c in msg_content if isinstance(c, dict)
            )
        else:
            text = str(msg_content or "")
        return RequestResult(
            ok=True,
            latency_s=elapsed,
            status=200,
            completion_tokens=int(usage.get("completion_tokens") or 0),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            output_text=text.strip()[:80],
        )
    except Exception as exc:  # noqa: BLE001
        return RequestResult(
            ok=False,
            latency_s=time.perf_counter() - t0,
            status=0,
            error=f"{type(exc).__name__}: {exc}"[:200],
        )


# ---------------------------------------------------------------------------
# Concurrency level runner
# ---------------------------------------------------------------------------


@dataclass
class LevelStats:
    audio_sec: float
    concurrency: int
    n_total: int
    n_ok: int
    n_err: int
    wall_s: float
    lat_p50: float
    lat_p90: float
    lat_p99: float
    lat_avg: float
    rtf_p50: float
    rtf_p90: float
    req_per_s: float
    audio_s_per_s: float
    out_tok_per_s: float
    sample_output: str = ""
    raw_latencies: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * (p / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return xs_sorted[lo]
    return xs_sorted[lo] + (xs_sorted[hi] - xs_sorted[lo]) * (k - lo)


async def run_level(
    *,
    client: httpx.AsyncClient,
    url: str,
    payload_factory,
    audio_sec: float,
    concurrency: int,
    n_requests: int,
    request_timeout: float,
) -> LevelStats:
    sem = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []

    async def _one() -> None:
        async with sem:
            payload = payload_factory()
            res = await do_request(client, url, payload, request_timeout)
            results.append(res)

    t0 = time.perf_counter()
    await asyncio.gather(*(_one() for _ in range(n_requests)))
    wall = time.perf_counter() - t0

    ok = [r for r in results if r.ok]
    err = [r for r in results if not r.ok]
    lats = [r.latency_s for r in ok]
    out_tokens = sum(r.completion_tokens for r in ok)

    sample = ok[0].output_text if ok else ""
    return LevelStats(
        audio_sec=audio_sec,
        concurrency=concurrency,
        n_total=len(results),
        n_ok=len(ok),
        n_err=len(err),
        wall_s=wall,
        lat_p50=_percentile(lats, 50),
        lat_p90=_percentile(lats, 90),
        lat_p99=_percentile(lats, 99),
        lat_avg=(statistics.fmean(lats) if lats else float("nan")),
        rtf_p50=_percentile(lats, 50) / audio_sec if lats else float("nan"),
        rtf_p90=_percentile(lats, 90) / audio_sec if lats else float("nan"),
        req_per_s=(len(ok) / wall) if wall > 0 else 0.0,
        audio_s_per_s=(len(ok) * audio_sec / wall) if wall > 0 else 0.0,
        out_tok_per_s=(out_tokens / wall) if wall > 0 else 0.0,
        sample_output=sample,
        raw_latencies=lats,
        errors=[r.error for r in err][:5],
    )


async def warmup(
    client: httpx.AsyncClient,
    url: str,
    payload_factory,
    n: int,
    request_timeout: float,
) -> tuple[int, int]:
    if n <= 0:
        return 0, 0
    results = await asyncio.gather(
        *(do_request(client, url, payload_factory(), request_timeout) for _ in range(n))
    )
    ok = sum(1 for r in results if r.ok)
    return ok, len(results) - ok


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _auto_n_requests(concurrency: int, override: int) -> int:
    if override > 0:
        return override
    return min(200, max(20, 5 * concurrency))


def _format_table(stats: list[LevelStats]) -> str:
    header = (
        f"{'audio_s':>7}  {'conc':>4}  {'n':>4}  {'ok':>4}  {'err':>4}  "
        f"{'lat_p50':>8}  {'lat_p90':>8}  {'lat_p99':>8}  "
        f"{'rtf_p50':>8}  {'rtf_p90':>8}  "
        f"{'req/s':>7}  {'audio_s/s':>9}  {'tok/s':>7}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for s in stats:
        lines.append(
            f"{s.audio_sec:>7.1f}  {s.concurrency:>4d}  "
            f"{s.n_total:>4d}  {s.n_ok:>4d}  {s.n_err:>4d}  "
            f"{s.lat_p50:>8.3f}  {s.lat_p90:>8.3f}  {s.lat_p99:>8.3f}  "
            f"{s.rtf_p50:>8.3f}  {s.rtf_p90:>8.3f}  "
            f"{s.req_per_s:>7.2f}  {s.audio_s_per_s:>9.2f}  {s.out_tok_per_s:>7.1f}"
        )
    lines.append(sep)
    return "\n".join(lines)


async def main_async(args: argparse.Namespace) -> int:
    url = args.base_url.rstrip("/") + "/v1/chat/completions"

    audio_secs = [float(x) for x in args.audio_secs.split(",") if x.strip()]
    if not audio_secs:
        print("--audio-secs is empty", file=sys.stderr)
        return 2

    if args.concurrency_list:
        ladder = [int(x) for x in args.concurrency_list.split(",") if x.strip()]
    else:
        ladder = [1, 2, 4, 8, 16, 32, 64, 128]
    ladder = [c for c in ladder if c <= args.max_concurrency]

    print(f"Target:    {url}")
    print(f"Model:     {args.model}")
    print(f"Mode:      {args.mode}")
    print(f"Audio:     {audio_secs} s ({args.audio_kind})")
    print(f"Ladder:    {ladder}")
    print(f"Stop when: rtf_p50 > {args.stop_rtf} OR error_rate > {args.stop_error_rate}")
    print()

    audio_b64_cache: dict[float, str] = {
        d: make_wav_b64(d, args.audio_kind, seed=0) for d in audio_secs
    }

    def _make_payload_factory(audio_sec: float):
        if args.unique_per_request:
            def factory() -> dict[str, Any]:
                seed = _next_seed()
                b64 = make_wav_b64(audio_sec, args.audio_kind, seed=seed)
                return build_payload(
                    b64,
                    model=args.model,
                    mode=args.mode,
                    max_tokens=args.max_tokens,
                )
            return factory

        cached_payload = build_payload(
            audio_b64_cache[audio_sec],
            model=args.model,
            mode=args.mode,
            max_tokens=args.max_tokens,
        )
        return lambda: cached_payload

    limits = httpx.Limits(
        max_connections=max(ladder) * 2 + 8,
        max_keepalive_connections=max(ladder) * 2 + 8,
    )
    timeout = httpx.Timeout(args.request_timeout, connect=10.0)

    all_stats: list[LevelStats] = []

    async with httpx.AsyncClient(limits=limits, timeout=timeout, http2=False) as client:
        for audio_sec in audio_secs:
            payload_factory = _make_payload_factory(audio_sec)

            print(f"=== Audio duration: {audio_sec:.2f} s ===")
            ok, err = await warmup(
                client, url, payload_factory, args.warmup, args.request_timeout
            )
            print(f"  warmup: ok={ok} err={err}")

            level_stats: list[LevelStats] = []
            for c in ladder:
                n = _auto_n_requests(c, args.requests_per_level)
                print(
                    f"  -> concurrency={c:<4d}  n={n:<4d}  running ...",
                    end="",
                    flush=True,
                )
                stats = await run_level(
                    client=client,
                    url=url,
                    payload_factory=payload_factory,
                    audio_sec=audio_sec,
                    concurrency=c,
                    n_requests=n,
                    request_timeout=args.request_timeout,
                )
                level_stats.append(stats)
                err_rate = stats.n_err / max(1, stats.n_total)
                print(
                    f"  lat_p50={stats.lat_p50:.3f}s  rtf_p50={stats.rtf_p50:.3f}  "
                    f"req/s={stats.req_per_s:.2f}  err={stats.n_err}"
                )
                if stats.sample_output:
                    print(f"     sample output: {stats.sample_output!r}")
                if stats.errors:
                    print(f"     first errors: {stats.errors[0]}")

                if err_rate > args.stop_error_rate:
                    print(
                        f"  [stop] error_rate={err_rate:.1%} > "
                        f"{args.stop_error_rate:.1%}"
                    )
                    break
                if not math.isnan(stats.rtf_p50) and stats.rtf_p50 > args.stop_rtf:
                    print(
                        f"  [stop] rtf_p50={stats.rtf_p50:.2f} > {args.stop_rtf:.2f}"
                    )
                    break

            print()
            print(_format_table(level_stats))
            print()
            all_stats.extend(level_stats)

    if args.output:
        out_path = Path(args.output)
        payload = {
            "config": {
                "base_url": args.base_url,
                "model": args.model,
                "mode": args.mode,
                "audio_secs": audio_secs,
                "audio_kind": args.audio_kind,
                "ladder": ladder,
                "warmup": args.warmup,
                "unique_per_request": args.unique_per_request,
                "requests_per_level": args.requests_per_level,
                "max_tokens": args.max_tokens,
                "request_timeout": args.request_timeout,
                "stop_rtf": args.stop_rtf,
                "stop_error_rate": args.stop_error_rate,
            },
            "results": [
                {
                    "audio_sec": s.audio_sec,
                    "concurrency": s.concurrency,
                    "n_total": s.n_total,
                    "n_ok": s.n_ok,
                    "n_err": s.n_err,
                    "wall_s": s.wall_s,
                    "lat_avg": s.lat_avg,
                    "lat_p50": s.lat_p50,
                    "lat_p90": s.lat_p90,
                    "lat_p99": s.lat_p99,
                    "rtf_p50": s.rtf_p50,
                    "rtf_p90": s.rtf_p90,
                    "req_per_s": s.req_per_s,
                    "audio_s_per_s": s.audio_s_per_s,
                    "out_tok_per_s": s.out_tok_per_s,
                    "sample_output": s.sample_output,
                    "raw_latencies": s.raw_latencies,
                    "errors": s.errors,
                }
                for s in all_stats
            ],
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"Saved JSON results -> {out_path}")

    print()
    print("=== Aggregate (all audio durations) ===")
    print(_format_table(all_stats))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stress-test the emotion-recognition vLLM service "
        "(OpenAI-compatible /v1/chat/completions)."
    )
    p.add_argument("--base-url", default="http://localhost:8222")
    p.add_argument("--model", default="AmphionSE")
    p.add_argument("--mode", default="ser", choices=("ser", "sec"))
    p.add_argument(
        "--audio-secs",
        default="5",
        help="Comma-separated audio durations in seconds (e.g. '3,5,10,20').",
    )
    p.add_argument(
        "--audio-kind",
        default="sine",
        choices=("sine", "silence", "noise"),
        help="Synthetic audio waveform.",
    )
    p.add_argument(
        "--concurrency-list",
        default="",
        help="Comma-separated concurrency ladder. Default: 1,2,4,8,16,32,64,128.",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=256,
        help="Filter ladder entries above this value.",
    )
    p.add_argument(
        "--requests-per-level",
        type=int,
        default=0,
        help="Requests per ladder step (0 = auto: max(20, 5*c) capped at 200).",
    )
    p.add_argument("--warmup", type=int, default=2, help="Warmup requests per duration.")
    p.add_argument(
        "--unique-per-request",
        action="store_true",
        help="Generate a unique audio payload per request to bypass vLLM's "
             "prefix cache. Recommended for realistic numbers.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Override max_tokens (0 = mode default: 32 for ser, 256 for sec).",
    )
    p.add_argument("--request-timeout", type=float, default=120.0)
    p.add_argument(
        "--stop-rtf",
        type=float,
        default=1.0,
        help="Stop ramping when rtf_p50 exceeds this value.",
    )
    p.add_argument(
        "--stop-error-rate",
        type=float,
        default=0.05,
        help="Stop ramping when per-level error rate exceeds this fraction.",
    )
    p.add_argument(
        "--output",
        default="",
        help="Optional path to dump JSON results (raw latencies included).",
    )
    args = p.parse_args(argv)
    if args.max_tokens == 0:
        args.max_tokens = None  # type: ignore[assignment]
    return args


def main() -> None:
    args = parse_args()
    try:
        rc = asyncio.run(main_async(args))
    except KeyboardInterrupt:
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()
