#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
from pathlib import Path

import requests


def post_multipart(url: str, audio_path: str, data: dict[str, str], verify: bool, timeout: float) -> dict:
    with open(audio_path, "rb") as audio_file:
        content_type = mimetypes.guess_type(audio_path)[0] or "application/octet-stream"
        files = {"audio": (Path(audio_path).name, audio_file, content_type)}
        response = requests.post(url, files=files, data=data, verify=verify, timeout=timeout)
    try:
        payload = response.json()
    except ValueError:
        payload = {"raw_text": response.text}
    if response.status_code >= 400:
        raise SystemExit(
            f"HTTP {response.status_code} {response.reason}\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
    return payload


def command_asr(args: argparse.Namespace) -> dict:
    url = join_url(args.base_url, "/api/asr/upload")
    return post_multipart(
        url,
        args.audio_file,
        data={"language": args.language, "hotwords": args.hotwords},
        verify=not args.insecure,
        timeout=args.timeout,
    )


def command_emotion(args: argparse.Namespace) -> dict:
    import time

    create_url = join_url(args.base_url, "/api/emotion/jobs")
    with open(args.audio_file, "rb") as audio_file:
        files = {"audio": (Path(args.audio_file).name, audio_file, "audio/wav")}
        create = requests.post(
            create_url,
            files=files,
            data={"mode": args.mode, "language": args.language},
            verify=not args.insecure,
            timeout=args.timeout,
        )
    if create.status_code != 202:
        try:
            payload = create.json()
        except ValueError:
            payload = {"raw_text": create.text}
        raise SystemExit(
            f"HTTP {create.status_code} {create.reason}\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
    meta = create.json()
    job_id = meta["job_id"]
    poll_url = join_url(args.base_url, meta.get("poll_url") or f"/api/emotion/jobs/{job_id}")
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        poll = requests.get(poll_url, verify=not args.insecure, timeout=30)
        try:
            body = poll.json()
        except ValueError:
            body = {"raw_text": poll.text}
        if poll.status_code >= 400:
            raise SystemExit(
                f"HTTP {poll.status_code} {poll.reason}\n"
                f"{json.dumps(body, ensure_ascii=False, indent=2)}"
            )
        if body.get("status") == "succeeded":
            return body.get("result") or {}
        if body.get("status") == "failed":
            err = body.get("error") or {}
            raise SystemExit(err.get("message") or "emotion job failed")
        time.sleep(0.4)
    raise SystemExit(f"emotion job {job_id} timed out after {args.timeout}s")


def command_analyze(args: argparse.Namespace) -> dict:
    url = join_url(args.base_url, "/api/audio/analyze")
    return post_multipart(
        url,
        args.audio_file,
        data={
            "language": args.language,
            "hotwords": args.hotwords,
            "emotion_mode": args.emotion_mode,
        },
        verify=not args.insecure,
        timeout=args.timeout,
    )


def join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("audio_file", help="Audio file to upload (ASR accepts WAV/MP3; other commands expect WAV)")
    parser.add_argument("--base-url", required=True, help="HTTP base URL, for example https://host:8443")
    parser.add_argument("--language", default="", help="Language code, for example zh/en/id/th")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP request timeout in seconds")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification for self-signed certificates")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call AudioLLM REST upload APIs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    asr = subparsers.add_parser("asr", help="POST /api/asr/upload")
    add_common(asr)
    asr.add_argument("--hotwords", default="", help="Comma-separated hotwords")
    asr.set_defaults(func=command_asr)

    emotion = subparsers.add_parser("emotion", help="POST /api/emotion/jobs (async)")
    add_common(emotion)
    emotion.add_argument("--mode", choices=("ser", "sec"), default="ser", help="Emotion mode")
    emotion.set_defaults(func=command_emotion)

    analyze = subparsers.add_parser("analyze", help="POST /api/audio/analyze")
    add_common(analyze)
    analyze.add_argument("--hotwords", default="", help="Comma-separated hotwords")
    analyze.add_argument(
        "--emotion-mode",
        choices=("ser", "sec"),
        default="ser",
        help="Emotion mode used by the analysis endpoint",
    )
    analyze.set_defaults(func=command_analyze)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = args.func(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
