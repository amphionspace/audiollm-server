#!/usr/bin/env python3
"""WebSocket test client for /transcribe-streaming endpoint.

Usage:
    python test_ws_client.py <audio_file> [--url URL] [--language LANG] [--chunk-ms MS] [--hotwords HW]

Examples:
    python test_ws_client.py test.wav
    python test_ws_client.py test.wav --language en
    python test_ws_client.py test.wav --hotwords "武新华,挚音科技,张硕"
    python test_ws_client.py raw.pcm --chunk-ms 80
"""

import argparse
import asyncio
import json
import ssl
import sys
import wave
from pathlib import Path

import websockets


def read_wav_as_s16le_16k(filepath: str) -> bytes:
    """Read a WAV file and return 16 kHz mono s16le PCM bytes."""
    with wave.open(filepath, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    print(f"  WAV info: {framerate} Hz, {n_channels} ch, {sample_width * 8}-bit, "
          f"{n_frames} frames ({n_frames / framerate:.2f}s)")

    import numpy as np

    if sample_width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)[:, 0]

    if framerate != 16000:
        from fractions import Fraction
        ratio = Fraction(16000, framerate)
        target_len = int(len(samples) * ratio)
        indices = np.linspace(0, len(samples) - 1, target_len)
        samples = np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)
        print(f"  Resampled {framerate} -> 16000 Hz ({len(samples)} samples, "
              f"{len(samples) / 16000:.2f}s)")

    pcm_int16 = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    return pcm_int16.tobytes()


def read_raw_pcm(filepath: str) -> bytes:
    """Read a raw PCM file (assumed 16 kHz mono s16le)."""
    data = Path(filepath).read_bytes()
    n_samples = len(data) // 2
    print(f"  Raw PCM: {n_samples} samples ({n_samples / 16000:.2f}s)")
    return data


async def run_client(url: str, audio_file: str, language: str, chunk_ms: int,
                     hotwords: list[str] | None = None):
    suffix = Path(audio_file).suffix.lower()
    print(f"Loading audio: {audio_file}")

    if suffix in (".wav", ".wave"):
        pcm_bytes = read_wav_as_s16le_16k(audio_file)
    elif suffix in (".pcm", ".raw"):
        pcm_bytes = read_raw_pcm(audio_file)
    else:
        print(f"Unknown format '{suffix}', treating as raw s16le 16kHz PCM")
        pcm_bytes = read_raw_pcm(audio_file)

    total_samples = len(pcm_bytes) // 2
    duration = total_samples / 16000
    chunk_bytes = 32 * chunk_ms  # 16kHz * 1ch * 2bytes = 32 bytes/ms
    print(f"  Total: {duration:.2f}s, chunk: {chunk_ms}ms ({chunk_bytes} bytes)")
    print()

    ssl_ctx = None
    if url.startswith("wss://"):
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

    print(f"Connecting to {url}?language={language} ...")
    async with websockets.connect(
        f"{url}?language={language}",
        ssl=ssl_ctx,
    ) as ws:
        recv_task = asyncio.create_task(_receive_messages(ws))

        # 1) Wait for ready
        print("Waiting for ready ...")
        await asyncio.sleep(0.1)

        # 2) Send start
        start_msg = {
            "type": "start",
            "mode": "asr_only",
            "format": "pcm_s16le",
            "sample_rate_hz": 16000,
            "channels": 1,
        }
        await ws.send(json.dumps(start_msg))
        print("-> Sent: start")

        # 2.5) Send hotwords if provided
        if hotwords:
            hw_msg = {"type": "update_hotwords", "hotwords": hotwords}
            await ws.send(json.dumps(hw_msg))
            print(f"-> Sent: update_hotwords {hotwords}")

        # 3) Stream PCM chunks, simulating real-time pace
        offset = 0
        chunk_count = 0
        while offset < len(pcm_bytes):
            end = min(offset + chunk_bytes, len(pcm_bytes))
            await ws.send(pcm_bytes[offset:end])
            chunk_count += 1
            offset = end
            await asyncio.sleep(chunk_ms / 1000.0)

        print(f"-> Sent: {chunk_count} PCM chunks ({duration:.2f}s audio)")

        # 4) Send stop
        await ws.send(json.dumps({"type": "stop"}))
        print("-> Sent: stop")
        print()

        # 5) Wait for remaining responses
        try:
            await asyncio.wait_for(recv_task, timeout=30.0)
        except asyncio.TimeoutError:
            print("[timeout] No more messages after 30s, closing.")
            recv_task.cancel()


async def _receive_messages(ws):
    """Print all messages from server until connection closes."""
    try:
        async for raw_msg in ws:
            try:
                msg = json.loads(raw_msg)
            except (json.JSONDecodeError, TypeError):
                print(f"<- [binary] {len(raw_msg)} bytes")
                continue

            msg_type = msg.get("type", "?")
            if msg_type == "ready":
                print("<- ready")
            elif msg_type == "partial_asr":
                print(f"<- partial_asr: {msg.get('text', '')}")
            elif msg_type == "final_asr":
                print(f"<- FINAL_ASR:   {msg.get('text', '')}  "
                      f"(language={msg.get('language', '')})")
            elif msg_type == "error":
                print(f"<- ERROR: {msg.get('message', '')}")
            else:
                print(f"<- {msg_type}: {json.dumps(msg, ensure_ascii=False)}")
    except websockets.exceptions.ConnectionClosed:
        pass
    print("[connection closed]")


def main():
    parser = argparse.ArgumentParser(description="Test WS client for /transcribe-streaming")
    parser.add_argument("audio_file", help="Path to audio file (WAV or raw PCM s16le 16kHz)")
    parser.add_argument("--url", default="wss://localhost:8443/transcribe-streaming",
                        help="WebSocket URL (default: wss://localhost:8443/transcribe-streaming)")
    parser.add_argument("--language", default="zh", help="Language code (default: zh)")
    parser.add_argument("--chunk-ms", type=int, default=80,
                        help="Chunk size in ms (default: 80)")
    parser.add_argument("--hotwords", default="",
                        help="Comma-separated hotwords (e.g. \"武新华,挚音科技\")")
    args = parser.parse_args()

    if not Path(args.audio_file).is_file():
        print(f"Error: file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    hw_list = [w.strip() for w in args.hotwords.split(",") if w.strip()] if args.hotwords else None
    asyncio.run(run_client(args.url, args.audio_file, args.language, args.chunk_ms, hw_list))


if __name__ == "__main__":
    main()
