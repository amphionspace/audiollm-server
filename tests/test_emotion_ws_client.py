#!/usr/bin/env python3
"""WebSocket test client for /emotion-streaming and /emotion-segmented-streaming.

Sends an entire audio file through the WS, then ``stop``. With the default
URL (``/emotion-streaming``) the server replies with one ``final_emotion``
and the client returns. When pointed at ``/emotion-segmented-streaming``
(via ``--url`` or ``--segmented``) the client keeps receiving until the
server closes the connection, so it can print every per-segment
``final_emotion``.

Usage:
    python test_emotion_ws_client.py <audio_file> [--url URL] [--chunk-ms MS] [--config K=V ...]

Examples:
    python test_emotion_ws_client.py audio.wav
    python test_emotion_ws_client.py raw.pcm --chunk-ms 80
    python test_emotion_ws_client.py audio.wav --config emotion_request_timeout=20
    python test_emotion_ws_client.py audio.wav --segmented
    python test_emotion_ws_client.py audio.wav \
        --url wss://localhost:8443/emotion-segmented-streaming
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
    data = Path(filepath).read_bytes()
    n_samples = len(data) // 2
    print(f"  Raw PCM: {n_samples} samples ({n_samples / 16000:.2f}s)")
    return data


def _parse_config_value(v: str) -> object:
    low = v.lower()
    if low in ("true", "1", "yes"):
        return True
    if low in ("false", "0", "no"):
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


async def run_client(url: str, audio_file: str, chunk_ms: int, mode: str | None,
                     config_overrides: dict | None = None,
                     segmented: bool = False):
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
    chunk_bytes = 32 * chunk_ms
    print(f"  Total: {duration:.2f}s, chunk: {chunk_ms}ms ({chunk_bytes} bytes)")
    print()

    ssl_ctx = None
    if url.startswith("wss://"):
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

    print(f"Connecting to {url} ...")
    async with websockets.connect(url, ssl=ssl_ctx) as ws:
        recv_task = asyncio.create_task(_receive_messages(ws, segmented=segmented))

        await asyncio.sleep(0.1)

        start_msg: dict = {
            "type": "start",
            "format": "pcm_s16le",
            "sample_rate_hz": 16000,
            "channels": 1,
        }
        if mode:
            start_msg["mode"] = mode
        if config_overrides:
            start_msg["config"] = config_overrides
        await ws.send(json.dumps(start_msg))
        parts: list[str] = []
        if mode:
            parts.append(f"mode={mode}")
        if config_overrides:
            parts.append(f"config={list(config_overrides.keys())}")
        print(f"-> Sent: start ({', '.join(parts) or 'no overrides'})")

        offset = 0
        chunk_count = 0
        while offset < len(pcm_bytes):
            end = min(offset + chunk_bytes, len(pcm_bytes))
            await ws.send(pcm_bytes[offset:end])
            chunk_count += 1
            offset = end
            await asyncio.sleep(chunk_ms / 1000.0)

        print(f"-> Sent: {chunk_count} PCM chunks ({duration:.2f}s audio)")

        await ws.send(json.dumps({"type": "stop"}))
        print("-> Sent: stop")
        print()

        try:
            await asyncio.wait_for(recv_task, timeout=60.0)
        except asyncio.TimeoutError:
            print("[timeout] No final_emotion within 60s, closing.")
            recv_task.cancel()


async def _receive_messages(ws, *, segmented: bool = False):
    final_count = 0
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
            elif msg_type == "final_emotion":
                mode = msg.get("mode", "?")
                final_count += 1
                tag = f"#{final_count} " if segmented else ""
                if mode == "sec":
                    print(f"<- {tag}FINAL_EMOTION[sec]: label={msg.get('label')!r} "
                          f"text={msg.get('text', '')!r} "
                          f"duration={msg.get('duration_sec')}s")
                else:
                    print(f"<- {tag}FINAL_EMOTION[ser]: label={msg.get('label')!r} "
                          f"duration={msg.get('duration_sec')}s")
                # Whole-utterance endpoint guarantees exactly one final per
                # cycle, so we can return early. Segmented streaming may emit
                # several finals (one per VAD segment), so keep draining until
                # the server closes the WebSocket.
                if not segmented:
                    return
            elif msg_type == "error":
                print(f"<- ERROR: {msg.get('message', '')}")
            else:
                print(f"<- {msg_type}: {json.dumps(msg, ensure_ascii=False)}")
    except websockets.exceptions.ConnectionClosed:
        pass
    print(f"[connection closed] received {final_count} final_emotion message(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Test WS client for /emotion-streaming and /emotion-segmented-streaming"
    )
    parser.add_argument("audio_file", help="Path to audio file (WAV or raw PCM s16le 16kHz)")
    parser.add_argument("--url", default=None,
                        help="WebSocket URL. Defaults to "
                             "wss://localhost:8443/emotion-streaming, or "
                             "wss://localhost:8443/emotion-segmented-streaming "
                             "when --segmented is set.")
    parser.add_argument("--segmented", action="store_true",
                        help="Target the VAD-segmented endpoint and keep "
                             "receiving until the server closes the WebSocket. "
                             "Implies the default URL switch shown above.")
    parser.add_argument("--chunk-ms", type=int, default=80, help="Chunk size in ms (default: 80)")
    parser.add_argument("--mode", choices=("ser", "sec"), default=None,
                        help="Emotion task variant. ser=8-class label; "
                             "sec=free-form caption. Default: server-side config "
                             "(emotion_task_mode, usually 'ser').")
    parser.add_argument("--config", nargs="*", default=[],
                        help="Config overrides as key=value pairs "
                             "(e.g. emotion_request_timeout=20)")
    args = parser.parse_args()

    if not Path(args.audio_file).is_file():
        print(f"Error: file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    if args.url is None:
        args.url = (
            "wss://localhost:8443/emotion-segmented-streaming"
            if args.segmented
            else "wss://localhost:8443/emotion-streaming"
        )
    # Auto-detect segmented mode from an explicit URL so users don't have to
    # remember to pass both --url and --segmented.
    segmented = args.segmented or "segmented" in args.url

    cfg_overrides: dict | None = None
    if args.config:
        cfg_overrides = {}
        for item in args.config:
            if "=" not in item:
                print(f"Error: config item must be key=value, got: {item}", file=sys.stderr)
                sys.exit(1)
            k, v = item.split("=", 1)
            cfg_overrides[k.strip()] = _parse_config_value(v.strip())

    asyncio.run(run_client(
        args.url, args.audio_file, args.chunk_ms, args.mode,
        cfg_overrides, segmented=segmented,
    ))


if __name__ == "__main__":
    main()
