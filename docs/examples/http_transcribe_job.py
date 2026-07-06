#!/usr/bin/env python3
"""Offline long-audio transcription via POST /api/asr/transcriptions + poll.

Designed for meeting-minutes style recordings: upload a whole WAV (up to the
server's transcribe_max_audio_sec cap), then poll until the segmented
transcript is ready. Progress (segments done / total) is printed while the
job runs.
"""
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
    url = join_url(base_url, poll_url or f"/api/asr/transcriptions/{job_id}")
    deadline = time.monotonic() + max_wait_sec
    last_progress = ""
    while time.monotonic() < deadline:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        status = body.get("status")
        progress = body.get("progress") or {}
        line = f"status={status} segments={progress.get('segments_done')}/{progress.get('segments_total')}"
        if line != last_progress:
            print(line, file=sys.stderr)
            last_progress = line
        if status == "succeeded":
            return body.get("result") or {}
        if status == "failed":
            err = body.get("error") or {}
            raise RuntimeError(err.get("message") or "transcription job failed")
        time.sleep(interval_sec)
    raise TimeoutError(f"job {job_id} did not finish within {max_wait_sec}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Async long-audio transcription client")
    parser.add_argument("audio", type=Path, help="WAV audio file (compressed formats: convert to WAV first)")
    parser.add_argument(
        "--base-url",
        default="http://172.16.0.3:8082",
        help="HTTP base URL",
    )
    parser.add_argument("--language", default="", help="Language hint, empty = auto detect")
    parser.add_argument("--hotwords", default="", help="Comma-separated hotwords passed to every segment")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--max-wait", type=float, default=1800.0, help="Max seconds to wait for the transcript")
    parser.add_argument("--full-text-only", action="store_true", help="Print only full_text instead of the JSON result")
    args = parser.parse_args()

    wav_bytes = args.audio.read_bytes()
    form: dict[str, str] = {}
    if args.language:
        form["language"] = args.language
    if args.hotwords:
        form["hotwords"] = args.hotwords

    create = requests.post(
        join_url(args.base_url, "/api/asr/transcriptions"),
        files={"audio": (args.audio.name, wav_bytes, "audio/wav")},
        data=form,
        timeout=300,
    )
    if create.status_code == 503:
        print("Server queue full; retry later.", file=sys.stderr)
        return 2
    if create.status_code == 400:
        print(f"Rejected: {create.json().get('detail')}", file=sys.stderr)
        return 1
    create.raise_for_status()
    if create.status_code != 202:
        print(f"Unexpected status {create.status_code}", file=sys.stderr)
        return 1

    meta = create.json()
    job_id = meta["job_id"]
    print(
        f"job_id={job_id} status={meta.get('status')} duration={meta.get('duration_sec')}s",
        file=sys.stderr,
    )

    result = poll_job(
        args.base_url,
        job_id,
        poll_url=meta.get("poll_url"),
        interval_sec=args.poll_interval,
        max_wait_sec=args.max_wait,
    )
    if args.full_text_only:
        print(result.get("full_text", ""))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
