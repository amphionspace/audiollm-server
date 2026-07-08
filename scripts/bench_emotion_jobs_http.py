#!/usr/bin/env python3
"""Benchmark POST /api/emotion/jobs + poll (async HTTP emotion API)."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np

SAMPLE_RATE = 16000
DEFAULT_BASE = "http://127.0.0.1:8080"
DEFAULT_CONCURRENCY = [1, 5, 10, 20]
DEFAULT_DURATIONS = [5, 10, 20, 300]  # seconds; 300 = 5min
SERVER_TRIM_SEC = 20.0


def make_wav_bytes(duration_s: float, seed: int = 0) -> bytes:
    n = max(1, int(round(duration_s * SAMPLE_RATE)))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    rng = np.random.default_rng(seed)
    tone = 0.25 * np.sin(2 * math.pi * 220 * t)
    noise = 0.02 * rng.standard_normal(n)
    samples = np.clip(tone + noise, -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


@dataclass
class JobResult:
    ok: bool
    status: str
    submit_ms: float
    e2e_ms: float
    polls: int
    audio_sec: float
    effective_sec: float
    error: str = ""


@dataclass
class ScenarioResult:
    concurrency: int
    audio_sec: float
    effective_sec: float
    wall_ms: float
    submitted: int
    succeeded: int
    failed: int
    rejected_503: int
    submit_ms_p50: float
    e2e_ms_p50: float
    e2e_ms_p95: float
    rtf_p50: float
    throughput_rps: float
    errors: list[str]


async def run_one_job(
    client: httpx.AsyncClient,
    base: str,
    wav: bytes,
    *,
    poll_interval: float,
    max_wait: float,
    job_index: int,
) -> JobResult:
    audio_sec = len(wav) / (SAMPLE_RATE * 2)  # approximate from bytes
    effective = min(audio_sec, SERVER_TRIM_SEC) if SERVER_TRIM_SEC > 0 else audio_sec
    t0 = time.perf_counter()
    try:
        resp = await client.post(
            f"{base}/api/emotion/jobs",
            files={"audio": (f"bench_{job_index}.wav", wav, "audio/wav")},
            data={"mode": "ser"},
            timeout=120.0,
        )
        submit_ms = (time.perf_counter() - t0) * 1000.0
        if resp.status_code == 503:
            return JobResult(
                ok=False,
                status="503",
                submit_ms=submit_ms,
                e2e_ms=submit_ms,
                polls=0,
                audio_sec=audio_sec,
                effective_sec=effective,
                error=resp.text[:200],
            )
        if resp.status_code != 202:
            return JobResult(
                ok=False,
                status=str(resp.status_code),
                submit_ms=submit_ms,
                e2e_ms=submit_ms,
                polls=0,
                audio_sec=audio_sec,
                effective_sec=effective,
                error=resp.text[:200],
            )
        meta = resp.json()
        job_id = meta["job_id"]
        poll_url = meta.get("poll_url") or f"/api/emotion/jobs/{job_id}"
        if not poll_url.startswith("http"):
            poll_url = base.rstrip("/") + poll_url
        deadline = time.perf_counter() + max_wait
        polls = 0
        while time.perf_counter() < deadline:
            polls += 1
            pr = await client.get(poll_url, timeout=30.0)
            if pr.status_code == 404:
                return JobResult(
                    ok=False,
                    status="404",
                    submit_ms=submit_ms,
                    e2e_ms=(time.perf_counter() - t0) * 1000.0,
                    polls=polls,
                    audio_sec=audio_sec,
                    effective_sec=effective,
                    error="job not found",
                )
            body = pr.json()
            st = body.get("status")
            if st == "succeeded":
                e2e_ms = (time.perf_counter() - t0) * 1000.0
                return JobResult(
                    ok=True,
                    status="succeeded",
                    submit_ms=submit_ms,
                    e2e_ms=e2e_ms,
                    polls=polls,
                    audio_sec=audio_sec,
                    effective_sec=effective,
                )
            if st == "failed":
                err = body.get("error") or {}
                return JobResult(
                    ok=False,
                    status="failed",
                    submit_ms=submit_ms,
                    e2e_ms=(time.perf_counter() - t0) * 1000.0,
                    polls=polls,
                    audio_sec=audio_sec,
                    effective_sec=effective,
                    error=str(err.get("message", err)),
                )
            await asyncio.sleep(poll_interval)
        return JobResult(
            ok=False,
            status="timeout",
            submit_ms=submit_ms,
            e2e_ms=(time.perf_counter() - t0) * 1000.0,
            polls=polls,
            audio_sec=audio_sec,
            effective_sec=effective,
            error="poll timeout",
        )
    except Exception as exc:
        return JobResult(
            ok=False,
            status="exception",
            submit_ms=(time.perf_counter() - t0) * 1000.0,
            e2e_ms=(time.perf_counter() - t0) * 1000.0,
            polls=0,
            audio_sec=audio_sec,
            effective_sec=effective,
            error=str(exc),
        )


async def run_scenario(
    base: str,
    concurrency: int,
    audio_sec: float,
    *,
    poll_interval: float,
    max_wait: float,
    seed: int,
) -> tuple[ScenarioResult, list[JobResult]]:
    effective = min(audio_sec, SERVER_TRIM_SEC) if SERVER_TRIM_SEC > 0 else audio_sec
    wav = make_wav_bytes(audio_sec, seed=seed)
    limits = httpx.Limits(max_connections=max(concurrency + 4, 32))
    async with httpx.AsyncClient(base_url=base, limits=limits, timeout=120.0) as client:
        wall0 = time.perf_counter()
        results = await asyncio.gather(
            *[
                run_one_job(
                    client,
                    base,
                    wav,
                    poll_interval=poll_interval,
                    max_wait=max_wait,
                    job_index=i,
                )
                for i in range(concurrency)
            ]
        )
        wall_ms = (time.perf_counter() - wall0) * 1000.0

    succeeded = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok and r.status != "503"]
    rejected = [r for r in results if r.status == "503"]
    e2e_ok = [r.e2e_ms for r in succeeded]
    submit_all = [r.submit_ms for r in results]

    def pct(vals: list[float], p: float) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        k = (len(s) - 1) * p / 100.0
        f = int(math.floor(k))
        c = int(math.ceil(k))
        if f == c:
            return s[f]
        return s[f] + (s[c] - s[f]) * (k - f)

    rtf_p50 = 0.0
    if succeeded and effective > 0:
        rtf_p50 = pct([r.e2e_ms / 1000.0 / effective for r in succeeded], 50)

    errors = list({r.error for r in results if r.error})[:5]
    scen = ScenarioResult(
        concurrency=concurrency,
        audio_sec=audio_sec,
        effective_sec=effective,
        wall_ms=wall_ms,
        submitted=len(results),
        succeeded=len(succeeded),
        failed=len(failed),
        rejected_503=len(rejected),
        submit_ms_p50=pct(submit_all, 50),
        e2e_ms_p50=pct(e2e_ok, 50),
        e2e_ms_p95=pct(e2e_ok, 95),
        rtf_p50=rtf_p50,
        throughput_rps=len(succeeded) / (wall_ms / 1000.0) if wall_ms > 0 else 0.0,
        errors=errors,
    )
    return scen, results


def format_duration(sec: float) -> str:
    if sec >= 60:
        return f"{int(sec // 60)}min"
    return f"{int(sec)}s"


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    global SERVER_TRIM_SEC
    SERVER_TRIM_SEC = args.trim_sec
    durations = args.durations
    concurrencies = args.concurrency
    all_scenarios: list[dict[str, Any]] = []

    print(
        f"Base URL: {args.base_url}\n"
        f"Server trim cap: {SERVER_TRIM_SEC}s effective audio\n"
        f"Durations: {durations}\n"
        f"Concurrency (bs): {concurrencies}\n"
    )
    print(f"{'audio':>6} {'bs':>4} {'ok':>4} {'503':>4} {'fail':>4} "
          f"{'wall_s':>7} {'e2e_p50':>8} {'e2e_p95':>8} {'rtf_p50':>7} {'qps':>6}")
    print("-" * 72)

    seed = args.seed
    for audio_sec in durations:
        for bs in concurrencies:
            scen, _ = await run_scenario(
                args.base_url,
                bs,
                float(audio_sec),
                poll_interval=args.poll_interval,
                max_wait=args.max_wait,
                seed=seed,
            )
            seed += 1
            all_scenarios.append(asdict(scen))
            print(
                f"{format_duration(audio_sec):>6} {bs:4d} "
                f"{scen.succeeded:4d} {scen.rejected_503:4d} {scen.failed:4d} "
                f"{scen.wall_ms/1000:7.2f} {scen.e2e_ms_p50:8.0f} {scen.e2e_ms_p95:8.0f} "
                f"{scen.rtf_p50:7.3f} {scen.throughput_rps:6.2f}"
            )
            if scen.errors:
                print(f"       errors: {scen.errors[:2]}")
            if args.pause > 0:
                await asyncio.sleep(args.pause)

    payload = {
        "base_url": args.base_url,
        "server_trim_sec": SERVER_TRIM_SEC,
        "mode": "ser",
        "durations_sec": durations,
        "concurrency": concurrencies,
        "scenarios": all_scenarios,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark /api/emotion/jobs HTTP API")
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    parser.add_argument(
        "--durations",
        type=int,
        nargs="+",
        default=DEFAULT_DURATIONS,
        help="Audio lengths in seconds (300 = 5min)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=DEFAULT_CONCURRENCY,
        help="Concurrent jobs per scenario (bs1, bs5, ...)",
    )
    parser.add_argument("--poll-interval", type=float, default=0.3)
    parser.add_argument("--max-wait", type=float, default=180.0)
    parser.add_argument("--pause", type=float, default=2.0, help="Pause between scenarios")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path",
    )
    parser.add_argument(
        "--trim-sec",
        type=float,
        default=SERVER_TRIM_SEC,
        help="Match server emotion_max_audio_seconds",
    )
    args = parser.parse_args()
    payload = asyncio.run(main_async(args))
    out = args.output
    if out is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = (
            Path(__file__).resolve().parent.parent
            / "bench_results"
            / f"bench_emotion_jobs_{ts}.json"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
