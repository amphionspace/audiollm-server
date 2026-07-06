#!/usr/bin/env python3
"""Submit whole-utterance emotion inference via POST /api/emotion/jobs + poll."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

from audio_common import join_url


def poll_job(
    base_url: str,
    job_id: str,
    *,
    poll_url: str | None,
    interval_sec: float,
    max_wait_sec: float,
) -> dict:
    url = join_url(base_url, poll_url or f"/api/emotion/jobs/{job_id}")
    deadline = time.monotonic() + max_wait_sec
    while time.monotonic() < deadline:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        status = body.get("status")
        if status == "succeeded":
            return body.get("result") or {}
        if status == "failed":
            err = body.get("error") or {}
            raise RuntimeError(err.get("message") or "emotion job failed")
        time.sleep(interval_sec)
    raise TimeoutError(f"job {job_id} did not finish within {max_wait_sec}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Async emotion job client")
    parser.add_argument("audio", type=Path, help="WAV audio file")
    parser.add_argument(
        "--base-url",
        default="http://172.16.0.3:8082",
        help="HTTP base URL",
    )
    parser.add_argument("--mode", default="ser", choices=("ser", "sec"))
    parser.add_argument("--language", default="")
    parser.add_argument("--poll-interval", type=float, default=0.4)
    parser.add_argument("--max-wait", type=float, default=45.0)
    args = parser.parse_args()

    wav_bytes = args.audio.read_bytes()
    form = {"mode": args.mode}
    if args.language:
        form["language"] = args.language

    create = requests.post(
        join_url(args.base_url, "/api/emotion/jobs"),
        files={"audio": (args.audio.name, wav_bytes, "audio/wav")},
        data=form,
        timeout=60,
    )
    if create.status_code == 503:
        print("Server queue full; retry later.", file=sys.stderr)
        return 2
    create.raise_for_status()
    if create.status_code != 202:
        print(f"Unexpected status {create.status_code}", file=sys.stderr)
        return 1

    meta = create.json()
    job_id = meta["job_id"]
    print(f"job_id={job_id} status={meta.get('status')}", file=sys.stderr)

    result = poll_job(
        args.base_url,
        job_id,
        poll_url=meta.get("poll_url"),
        interval_sec=args.poll_interval,
        max_wait_sec=args.max_wait,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
