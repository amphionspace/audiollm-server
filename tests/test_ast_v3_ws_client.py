#!/usr/bin/env python3
"""WebSocket test client for the /tuling/ast/v3 endpoint (iFlytek Tuling AST v3).

Speaks the AST v3 envelope protocol: audio is base64-encoded inside JSON frames
and ``header.status`` drives the state machine (0 first, 1 middle, 2 last).
Mirrors the reference Java SDK's framing (FIRST/CONTINUE/LAST + 4096-byte audio
chunks) so it doubles as a contract check.

Usage:
    python test_ast_v3_ws_client.py <audio_file> [--url URL] [--hotwords HW]
                                     [--biz-id ID] [--app-id ID] [--chunk-ms MS]

Examples:
    python test_ast_v3_ws_client.py test.wav
    python test_ast_v3_ws_client.py test.wav --url ws://127.0.0.1:8080/tuling/ast/v3
    python test_ast_v3_ws_client.py test.wav --hotwords "挚音科技,张硕"
"""

import argparse
import asyncio
import base64
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

    print(
        f"  WAV info: {framerate} Hz, {n_channels} ch, {sample_width * 8}-bit, "
        f"{n_frames} frames ({n_frames / framerate:.2f}s)"
    )

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
        target_len = int(len(samples) * 16000 / framerate)
        indices = np.linspace(0, len(samples) - 1, target_len)
        samples = np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)
        print(
            f"  Resampled {framerate} -> 16000 Hz ({len(samples)} samples, "
            f"{len(samples) / 16000:.2f}s)"
        )

    pcm_int16 = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    return pcm_int16.tobytes()


def read_raw_pcm(filepath: str) -> bytes:
    """Read a raw PCM file (assumed 16 kHz mono s16le)."""
    data = Path(filepath).read_bytes()
    n_samples = len(data) // 2
    print(f"  Raw PCM: {n_samples} samples ({n_samples / 16000:.2f}s)")
    return data


def _frame(status: int, *, trace_id: str, app_id: str, biz_id: str,
           audio_b64: str | None = None, hotwords: str | None = None,
           enrollment_id: str = "", asr_config: dict | None = None) -> str:
    """Build one AST v3 request envelope as a JSON string."""
    header: dict = {"traceId": trace_id, "status": status}
    if app_id:
        header["appId"] = app_id
    if biz_id:
        header["bizId"] = biz_id
    # engine stays log-only; asr_config carries per-connection config overrides.
    parameter: dict = {"engine": {}}
    if enrollment_id:
        asr_config = dict(asr_config or {})
        asr_config["enable_role_separation"] = False
        asr_config["enrollment_enable"] = True
        asr_config["enrollment_id"] = enrollment_id
    if asr_config:
        parameter["asr_config"] = asr_config
    payload: dict = {}
    if audio_b64 is not None:
        payload["audio"] = {"audio": audio_b64}
    if hotwords:
        payload["text"] = {"text": hotwords}
    return json.dumps({"header": header, "parameter": parameter, "payload": payload})


async def run_client(url: str, audio_file: str, *, hotwords: str, biz_id: str,
                     app_id: str, chunk_ms: int, enrollment_id: str = "",
                     asr_config: dict | None = None):
    suffix = Path(audio_file).suffix.lower()
    print(f"Loading audio: {audio_file}")
    if suffix in (".wav", ".wave"):
        pcm_bytes = read_wav_as_s16le_16k(audio_file)
    else:
        pcm_bytes = read_raw_pcm(audio_file)

    total_samples = len(pcm_bytes) // 2
    duration = total_samples / 16000
    chunk_bytes = 32 * chunk_ms  # 16kHz * 1ch * 2 bytes = 32 bytes/ms
    print(f"  Total: {duration:.2f}s, chunk: {chunk_ms}ms ({chunk_bytes} bytes)")
    print()

    ssl_ctx = None
    if url.startswith("wss://"):
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

    trace_id = "ast-v3-demo"
    print(f"Connecting to {url} ...")
    async with websockets.connect(url, ssl=ssl_ctx) as ws:
        recv_task = asyncio.create_task(_receive_messages(ws))

        # Build the chunk list, then label them FIRST(0) / CONTINUE(1) / LAST(2).
        chunks = [
            pcm_bytes[off : off + chunk_bytes]
            for off in range(0, max(len(pcm_bytes), 1), chunk_bytes)
        ]
        if not chunks:
            chunks = [b""]

        for i, chunk in enumerate(chunks):
            if i == 0:
                status = 0
            elif i == len(chunks) - 1:
                status = 2
            else:
                status = 1
            await ws.send(
                _frame(
                    status,
                    trace_id=trace_id,
                    app_id=app_id,
                    biz_id=biz_id,
                    audio_b64=base64.b64encode(chunk).decode(),
                    # Hotwords + enrollment + asr_config ride the first frame only.
                    hotwords=hotwords if i == 0 else None,
                    enrollment_id=enrollment_id if i == 0 else "",
                    asr_config=asr_config if i == 0 else None,
                )
            )
            await asyncio.sleep(chunk_ms / 1000.0)

        print(f"-> Sent: {len(chunks)} AST v3 frames ({duration:.2f}s audio)")
        print()

        try:
            await asyncio.wait_for(recv_task, timeout=30.0)
        except asyncio.TimeoutError:
            print("[timeout] No more messages after 30s, closing.")
            recv_task.cancel()


def _format_result(result: dict) -> str:
    msgtype = result.get("msgtype", "?")
    candidates = [
        cw
        for item in (result.get("ws") or [])
        for cw in (item.get("cw") or [])
    ]
    words = "".join(
        cw.get("w", "") for cw in candidates
    )
    if msgtype == "sentence":
        role = candidates[0].get("rl") if candidates else None
        return (f"FINAL   segId={result.get('segId')} sn={result.get('sn')} "
                f"bg={result.get('bg')}ms ed={result.get('ed')}ms "
                f"rl={role}: {words}")
    return f"{msgtype}: {words}"


async def _receive_messages(ws):
    """Print AST v3 server frames until the connection closes."""
    try:
        async for raw_msg in ws:
            try:
                msg = json.loads(raw_msg)
            except (json.JSONDecodeError, TypeError):
                print(f"<- [binary] {len(raw_msg)} bytes")
                continue

            header = msg.get("header", {})
            code = header.get("code", 0)
            status = header.get("status")
            if code != 0:
                print(f"<- ERROR code={code}: {header.get('message', '')}")
                continue
            if status == 2 and not msg.get("payload", {}).get("result", {}).get("ws"):
                print(f"<- END (sid={header.get('sid')})")
                continue
            result = msg.get("payload", {}).get("result")
            if result:
                print(f"<- {_format_result(result)}")
            else:
                print(f"<- {json.dumps(msg, ensure_ascii=False)}")
    except websockets.exceptions.ConnectionClosed:
        pass
    print("[connection closed]")


def _build_asr_config(args: argparse.Namespace) -> dict:
    """Assemble parameter.asr_config from --config KEY=VALUE plus shortcuts.

    Each value is parsed as JSON when possible (0.45 -> float, false -> bool,
    300 -> int), otherwise kept as a string. The server whitelists and coerces
    fields, so unknown or restricted keys are simply ignored downstream.
    """
    cfg: dict = {}
    for item in args.config:
        key, sep, val = item.partition("=")
        if not sep:
            continue
        key = key.strip()
        try:
            cfg[key] = json.loads(val)
        except json.JSONDecodeError:
            cfg[key] = val
    if args.language:
        cfg["language"] = args.language
    if args.vad_threshold is not None:
        cfg["vad_threshold"] = args.vad_threshold
    if args.no_pseudo_stream:
        cfg["enable_pseudo_stream"] = False
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Test WS client for /tuling/ast/v3 (AST v3)"
    )
    parser.add_argument("audio_file", help="Path to audio (WAV or raw PCM s16le 16kHz)")
    parser.add_argument(
        "--url",
        default="wss://localhost:8443/tuling/ast/v3",
        help="WebSocket URL (default: wss://localhost:8443/tuling/ast/v3)",
    )
    parser.add_argument("--hotwords", default="",
                        help='Comma-separated hotwords (e.g. "挚音科技,张硕")')
    parser.add_argument("--enrollment-id", default="",
                        help="Target-speaker id from POST /api/asr/enrollment (-> parameter.asr_config.enrollment_id)")
    parser.add_argument("--language", default="",
                        help="会话语言代码，写入 parameter.asr_config.language")
    parser.add_argument("--vad-threshold", type=float, default=None,
                        help="覆写 VAD 阈值 parameter.asr_config.vad_threshold")
    parser.add_argument("--no-pseudo-stream", action="store_true",
                        help="关闭伪流式中间结果 enable_pseudo_stream=false")
    parser.add_argument("--config", action="append", default=[], metavar="KEY=VALUE",
                        help="通用 parameter.asr_config 覆写，可多次，如 --config silence_duration_ms=300")
    parser.add_argument("--biz-id", default="12345", help="header.bizId (required field)")
    parser.add_argument("--app-id", default="ast", help="header.appId")
    parser.add_argument("--chunk-ms", type=int, default=128,
                        help="Chunk size in ms (default: 128 ~= SDK's 4096 bytes)")
    args = parser.parse_args()

    if not Path(args.audio_file).is_file():
        print(f"Error: file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(
        run_client(
            args.url,
            args.audio_file,
            hotwords=args.hotwords,
            biz_id=args.biz_id,
            app_id=args.app_id,
            chunk_ms=args.chunk_ms,
            enrollment_id=args.enrollment_id,
            asr_config=_build_asr_config(args),
        )
    )


if __name__ == "__main__":
    main()
