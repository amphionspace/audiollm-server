#!/usr/bin/env python3
"""Capture the exact per-final WAV payload sent to AudioLLM."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import re
from pathlib import Path
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

from audio_common import chunk_bytes, make_ssl_context, read_audio_as_pcm

_AUDIO_SUFFIXES = {".pcm", ".raw", ".wav"}
_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9_.()-]+")


def discover_audio_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in _AUDIO_SUFFIXES:
            raise ValueError(f"unsupported audio file: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"input path does not exist: {input_path}")

    files = sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in _AUDIO_SUFFIXES
    )
    if not files:
        raise ValueError(f"no WAV/PCM/RAW files found in: {input_path}")
    return files


def safe_filename_part(value: str, fallback: str) -> str:
    cleaned = _UNSAFE_FILENAME.sub("_", value.strip()).strip("._")
    return cleaned or fallback


def save_model_audio(
    msg: dict[str, Any],
    *,
    source: Path,
    output_dir: Path,
    final_number: int,
) -> dict[str, Any]:
    encoded = msg.get("audio_b64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("final message has no audio_b64")

    wav = base64.b64decode(encoded, validate=True)
    if len(wav) < 12 or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        raise ValueError("final audio_b64 is not a WAV payload")

    source_stem = safe_filename_part(source.stem, "audio")
    segment_id = safe_filename_part(str(msg.get("id") or ""), f"final-{final_number:03d}")
    stem = f"{source_stem}__{segment_id}"
    audio_path = output_dir / f"{stem}.wav"
    result_path = output_dir / f"{stem}.json"
    audio_path.write_bytes(wav)

    record = {
        "segment_id": msg.get("id"),
        "text": msg.get("text", ""),
        "language": msg.get("language", ""),
        "duration_sec": msg.get("duration_sec"),
        "audio_path": str(audio_path.resolve()),
        "result_path": str(result_path.resolve()),
    }
    result = {
        "input": str(source.resolve()),
        "audio_path": record["audio_path"],
        # The WAV above is the decoded value. Keeping base64 out of this file
        # makes the inference metadata easy to inspect and diff.
        "final": {key: value for key, value in msg.items() if key != "audio_b64"},
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return record


async def receive_and_capture(
    ws: Any,
    *,
    source: Path,
    output_dir: Path,
    captured: list[dict[str, Any]],
) -> None:
    try:
        async for raw in ws:
            if not isinstance(raw, str):
                print(f"  <- ignored binary message ({len(raw)} bytes)")
                continue
            msg = json.loads(raw)
            msg_type = msg.get("type", "")
            if msg_type in {"partial", "partial_asr"}:
                print(f"  <- partial: {msg.get('text', '')}")
            elif msg_type in {"final", "final_asr"}:
                record = save_model_audio(
                    msg,
                    source=source,
                    output_dir=output_dir,
                    final_number=len(captured) + 1,
                )
                captured.append(record)
                print(
                    f"  <- final: {record['text']} "
                    f"(duration={record['duration_sec']}, saved={record['audio_path']})"
                )
            elif msg_type == "error":
                raise RuntimeError(f"server error: {json.dumps(msg, ensure_ascii=False)}")
            else:
                print(f"  <- {json.dumps(msg, ensure_ascii=False)}")
    except ConnectionClosed as exc:
        # Some deployed servers currently end after stop without a close frame.
        # Already received final payloads remain complete and usable.
        print(f"  <- connection closed (code={exc.code})")


async def capture_one(args: argparse.Namespace, source: Path) -> list[dict[str, Any]]:
    pcm = read_audio_as_pcm(str(source))
    chunk_size = chunk_bytes(args.chunk_ms)
    ssl_ctx = make_ssl_context(args.url, args.insecure)
    captured: list[dict[str, Any]] = []

    async with websockets.connect(
        args.url,
        ssl=ssl_ctx,
        open_timeout=args.timeout,
    ) as ws:
        ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=args.timeout))
        if ready.get("type") != "ready":
            raise RuntimeError(f"unexpected first message: {ready}")

        start_msg: dict[str, Any] = {
            "type": "start",
            "format": "pcm_s16le",
            "sample_rate_hz": 16000,
            "channels": 1,
        }
        if args.language:
            start_msg["language"] = args.language
        if args.hotwords:
            start_msg["hotwords"] = args.hotwords
        client_config: dict[str, Any] = {}
        if args.silence_threshold is not None:
            client_config["asr_silence_removal_threshold_sec"] = args.silence_threshold
        if args.silence_duration_ms is not None:
            client_config["silence_duration_ms"] = args.silence_duration_ms
        if args.vad_start_frames is not None:
            client_config["vad_start_frames"] = args.vad_start_frames
        if client_config:
            start_msg["config"] = client_config
        await ws.send(json.dumps(start_msg, ensure_ascii=False))

        recv_task = asyncio.create_task(
            receive_and_capture(
                ws,
                source=source,
                output_dir=args.output_dir,
                captured=captured,
            )
        )
        for offset in range(0, len(pcm), chunk_size):
            await ws.send(pcm[offset : offset + chunk_size])
            await asyncio.sleep(args.chunk_ms / 1000)

        await ws.send(json.dumps({"type": "stop"}))
        try:
            await asyncio.wait_for(recv_task, timeout=args.final_timeout)
        except asyncio.TimeoutError:
            recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await recv_task
            print(f"  <- timed out after stop ({args.final_timeout}s)")

    if not captured:
        raise RuntimeError("connection ended without a final audio payload")
    return captured


async def main_async(args: argparse.Namespace) -> int:
    sources = discover_audio_files(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "url": args.url,
        # Client overrides are best-effort by protocol: an older server may
        # ignore an unknown field, so do not label this as the effective value.
        "requested_silence_threshold_sec": args.silence_threshold,
        "requested_silence_duration_ms": args.silence_duration_ms,
        "requested_vad_start_frames": args.vad_start_frames,
        "requested_hotwords": args.hotwords,
        "files": [],
    }
    failures = 0

    for source in sources:
        print(f"[{source}]")
        record: dict[str, Any] = {"input": str(source.resolve())}
        try:
            record["segments"] = await capture_one(args, source)
            record["status"] = "ok"
        except Exception as exc:
            failures += 1
            record["status"] = "error"
            record["error"] = str(exc)
            print(f"  !! {exc}")
        manifest["files"].append(record)

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"manifest: {manifest_path.resolve()}")
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream one audio file or a directory to /transcribe-streaming and save "
            "each final.audio_b64 WAV exactly as it was sent to AudioLLM."
        )
    )
    parser.add_argument("input", type=Path, help="WAV/PCM/RAW file or a directory")
    parser.add_argument("--url", required=True, help="WebSocket /transcribe-streaming URL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("asr_model_audio"),
        help="Directory for captured per-final WAV files and manifest.json",
    )
    parser.add_argument("--language", default="zh", help="ASR language (default: zh)")
    parser.add_argument(
        "--hotwords",
        default="",
        help="Comma-separated temporary request hotwords",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=None,
        help=(
            "Client override for asr_silence_removal_threshold_sec; "
            "omit to use the server default"
        ),
    )
    parser.add_argument(
        "--silence-duration-ms",
        type=int,
        default=None,
        help="Client override for silence_duration_ms; omit to use the server default",
    )
    parser.add_argument(
        "--vad-start-frames",
        type=int,
        default=None,
        help="Client override for vad_start_frames; omit to use the server default",
    )
    parser.add_argument("--chunk-ms", type=int, default=80, help="PCM chunk size in ms")
    parser.add_argument("--timeout", type=float, default=10.0, help="Connect/read timeout")
    parser.add_argument(
        "--final-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for final messages after stop",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification for self-signed wss URLs",
    )
    args = parser.parse_args()
    args.hotwords = [item.strip() for item in args.hotwords.split(",") if item.strip()]
    if args.silence_threshold is not None and args.silence_threshold < 0:
        parser.error("--silence-threshold must be >= 0")
    if args.silence_duration_ms is not None and args.silence_duration_ms < 0:
        parser.error("--silence-duration-ms must be >= 0")
    if args.vad_start_frames is not None and args.vad_start_frames <= 0:
        parser.error("--vad-start-frames must be > 0")
    if args.chunk_ms <= 0:
        parser.error("--chunk-ms must be > 0")
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(parse_args())))
