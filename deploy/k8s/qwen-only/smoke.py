#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
import wave
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import httpx
import websockets

TERMINAL_STATUSES = {"succeeded", "failed"}


def websocket_url(base_url: str, path: str) -> str:
    parsed = urlparse(base_url.rstrip("/"))
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse((scheme, parsed.netloc, path, "", "", ""))


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2 or audio.getframerate() != 16_000:
            raise ValueError("smoke audio must be 16 kHz mono s16le WAV")
        return audio.readframes(audio.getnframes())


async def poll_job(client: httpx.AsyncClient, poll_url: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = await client.get(poll_url)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") in TERMINAL_STATUSES:
            if payload["status"] != "succeeded":
                raise RuntimeError(json.dumps(payload, ensure_ascii=False))
            return payload["result"]
        await asyncio.sleep(1)
    raise TimeoutError(f"job did not finish in {timeout:.0f}s: {poll_url}")


async def run_job(
    client: httpx.AsyncClient,
    path: str,
    audio_path: Path,
    data: dict[str, str],
    timeout: float,
) -> dict:
    with audio_path.open("rb") as audio:
        response = await client.post(
            path,
            data=data,
            files={"audio": (audio_path.name, audio, "audio/wav")},
        )
    response.raise_for_status()
    return await poll_job(client, response.json()["poll_url"], timeout)


async def run_ws(
    base_url: str,
    path: str,
    pcm: bytes,
    start: dict,
    expected_type: str,
    timeout: float,
) -> list[dict]:
    results: list[dict] = []
    async with websockets.connect(
        websocket_url(base_url, path),
        open_timeout=timeout,
        max_size=None,
    ) as socket:
        ready = json.loads(await asyncio.wait_for(socket.recv(), timeout))
        if ready.get("type") != "ready":
            raise RuntimeError(f"unexpected ready frame: {ready}")
        await socket.send(json.dumps(start, ensure_ascii=False))
        chunk_size = 16_000 * 2 // 10
        for offset in range(0, len(pcm), chunk_size):
            await socket.send(pcm[offset : offset + chunk_size])
        await socket.send(json.dumps({"type": "stop"}))
        while True:
            try:
                frame = json.loads(await asyncio.wait_for(socket.recv(), timeout))
            except websockets.ConnectionClosed:
                break
            if frame.get("type") == expected_type:
                results.append(frame)
                if expected_type == "final":
                    break
            if frame.get("type") == "error":
                raise RuntimeError(json.dumps(frame, ensure_ascii=False))
    if not results:
        raise RuntimeError(f"{path} returned no {expected_type} frames")
    return results


async def run_clean_stream(base_url: str, pcm: bytes, timeout: float) -> dict:
    async with websockets.connect(
        websocket_url(base_url, "/asr/v1/clean-stream"),
        open_timeout=timeout,
        max_size=None,
    ) as socket:
        created = json.loads(await asyncio.wait_for(socket.recv(), timeout))
        if created.get("type") != "session.created":
            raise RuntimeError(f"unexpected clean-stream greeting: {created}")
        await socket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "language": "zh",
                    "cleanup": {"level": "light", "text_emotion": True},
                    "hotwords": {"builtin": ["internet"], "custom": ["Amphion"]},
                },
                ensure_ascii=False,
            )
        )
        updated = json.loads(await asyncio.wait_for(socket.recv(), timeout))
        if updated.get("type") != "session.updated":
            raise RuntimeError(f"unexpected clean-stream update: {updated}")
        chunk_size = 16_000 * 2 // 10
        for offset in range(0, len(pcm), chunk_size):
            await socket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm[offset : offset + chunk_size]).decode(
                            "ascii"
                        ),
                    }
                )
            )
        await socket.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        while True:
            frame = json.loads(await asyncio.wait_for(socket.recv(), timeout))
            if frame.get("type") == "error":
                raise RuntimeError(json.dumps(frame, ensure_ascii=False))
            if frame.get("type") == "transcription.done":
                if not frame.get("text") or "cleaned_text" not in frame:
                    raise RuntimeError(f"incomplete clean-stream result: {frame}")
                return frame


async def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the qwen-only K8s profile")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()

    pcm = read_pcm(args.audio)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=args.timeout) as client:
        ready = await client.get("/readyz")
        ready.raise_for_status()

        transcription = await run_job(
            client,
            "/api/asr/transcriptions",
            args.audio,
            {
                "language": "zh",
                "config": json.dumps(
                    {
                        "language": "zh",
                        "cleanup": {"level": "light", "text_emotion": True},
                        "hotwords": {"custom": ["Amphion"]},
                    },
                    ensure_ascii=False,
                ),
            },
            args.timeout,
        )
        if "full_text" not in transcription or "cleanup_status" not in transcription:
            raise RuntimeError(f"incomplete transcription result: {transcription}")

        emotion_ser = await run_job(
            client,
            "/api/emotion/jobs",
            args.audio,
            {"mode": "ser", "language": "zh"},
            args.timeout,
        )
        if not emotion_ser.get("top_emotions") or "best_score" not in emotion_ser:
            raise RuntimeError(f"SER result lacks Top-K scores: {emotion_ser}")

        emotion_sec = await run_job(
            client,
            "/api/emotion/jobs",
            args.audio,
            {"mode": "sec", "language": "zh"},
            args.timeout,
        )
        if not emotion_sec.get("text"):
            raise RuntimeError(f"SEC result has no description: {emotion_sec}")

    asr_frames = await run_ws(
        args.base_url,
        "/transcribe-streaming",
        pcm,
        {"type": "start", "language": "zh", "hotwords": ["Amphion"]},
        "final",
        args.timeout,
    )
    clean_stream = await run_clean_stream(args.base_url, pcm, args.timeout)
    emotion_frames = await run_ws(
        args.base_url,
        "/emotion-segmented-streaming",
        pcm,
        {"type": "start", "mode": "sec", "language": "zh"},
        "final_emotion",
        args.timeout,
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "transcription": transcription,
                "emotion_ser": emotion_ser,
                "emotion_sec": emotion_sec,
                "asr_ws": asr_frames,
                "clean_stream_ws": clean_stream,
                "emotion_ws": emotion_frames,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
