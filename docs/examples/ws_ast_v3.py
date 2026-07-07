#!/usr/bin/env python3
"""Call /tuling/ast/v3 (iFlytek Tuling AST v3) over WebSocket.

Audio is base64-encoded inside JSON envelopes; header.status drives the state
machine (0 first frame, 1 middle frames, 2 last frame). Mirrors the reference
Java SDK framing (FIRST / CONTINUE / LAST).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json

import websockets
from websockets.exceptions import ConnectionClosed

from audio_common import chunk_bytes, make_ssl_context, read_audio_as_pcm


def _frame(status: int, *, trace_id: str, app_id: str, biz_id: str,
           audio: bytes, hotwords: str = "", enrollment_id: str = "",
           asr_config: dict | None = None) -> str:
    header: dict[str, object] = {"traceId": trace_id, "status": status}
    if app_id:
        header["appId"] = app_id
    if biz_id:
        header["bizId"] = biz_id
    # engine stays log-only; asr_config carries per-connection config overrides.
    parameter: dict[str, object] = {"engine": {}}
    if enrollment_id:
        asr_config = dict(asr_config or {})
        asr_config["enable_role_separation"] = False
        asr_config["enrollment_enable"] = True
        asr_config["enrollment_id"] = enrollment_id
    if asr_config:
        parameter["asr_config"] = asr_config
    payload: dict[str, object] = {"audio": {"audio": base64.b64encode(audio).decode()}}
    if hotwords:
        payload["text"] = {"text": hotwords}
    return json.dumps(
        {"header": header, "parameter": parameter, "payload": payload},
        ensure_ascii=False,
    )


def _build_asr_config(args: argparse.Namespace) -> dict:
    """Assemble parameter.asr_config from --config KEY=VALUE plus shortcuts.

    Each value is parsed as JSON when possible (so 0.45 -> float, false -> bool,
    300 -> int), otherwise kept as a string. The server whitelists and coerces
    fields, so unknown or restricted keys are simply ignored downstream.
    """
    cfg: dict[str, object] = {}
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


async def main_async(args: argparse.Namespace) -> None:
    pcm = read_audio_as_pcm(args.audio_file)
    chunk_size = chunk_bytes(args.chunk_ms)
    ssl_ctx = make_ssl_context(args.url, args.insecure)

    chunks = [pcm[off : off + chunk_size] for off in range(0, max(len(pcm), 1), chunk_size)]
    if not chunks:
        chunks = [b""]

    asr_config = _build_asr_config(args)

    async with websockets.connect(args.url, ssl=ssl_ctx, open_timeout=args.timeout) as ws:
        recv_task = asyncio.create_task(receive_messages(ws))
        for i, chunk in enumerate(chunks):
            status = 0 if i == 0 else (2 if i == len(chunks) - 1 else 1)
            await ws.send(
                _frame(
                    status,
                    trace_id=args.trace_id,
                    app_id=args.app_id,
                    biz_id=args.biz_id,
                    audio=chunk,
                    # Hotwords + enrollment + asr_config ride the first frame only.
                    hotwords=args.hotwords if i == 0 else "",
                    enrollment_id=args.enrollment_id if i == 0 else "",
                    asr_config=asr_config if i == 0 else None,
                )
            )
            await asyncio.sleep(args.chunk_ms / 1000)
        print(f"-> sent {len(chunks)} frames (status 0..2)")

        try:
            await asyncio.wait_for(recv_task, timeout=args.final_timeout)
        except asyncio.TimeoutError:
            recv_task.cancel()
            print(f"[timeout] no more messages within {args.final_timeout}s")


async def receive_messages(ws) -> None:
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                print(f"<- binary {len(raw)} bytes")
                continue

            header = msg.get("header", {})
            if header.get("code", 0) != 0:
                print(f"<- error code={header.get('code')}: {header.get('message', '')}")
                return
            result = msg.get("payload", {}).get("result", {})
            words = "".join(
                cw.get("w", "")
                for item in (result.get("ws") or [])
                for cw in (item.get("cw") or [])
            )
            if header.get("status") == 2 and not result.get("ws"):
                print(f"<- end (sid={header.get('sid')})")
            elif result.get("msgtype") == "sentence":
                print(f"<- final: {words} (bg={result.get('bg')}ms ed={result.get('ed')}ms)")
            elif result.get("msgtype") == "Progressive":
                print(f"<- partial: {words}")
            else:
                print(f"<- {json.dumps(msg, ensure_ascii=False)}")
    except ConnectionClosed:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call /tuling/ast/v3 over WebSocket.")
    parser.add_argument("audio_file", help="WAV file or raw 16 kHz mono s16le PCM file")
    parser.add_argument("--url", required=True,
                        help="WebSocket URL, e.g. ws://host:8080/tuling/ast/v3")
    parser.add_argument("--hotwords", default="",
                        help="Comma-separated hotwords (first frame only)")
    parser.add_argument("--enrollment-id", default="",
                        help="Target-speaker id from POST /api/asr/enrollment")
    parser.add_argument("--language", default="",
                        help="会话语言代码，写入 parameter.asr_config.language")
    parser.add_argument("--vad-threshold", type=float, default=None,
                        help="覆写 VAD 阈值 parameter.asr_config.vad_threshold")
    parser.add_argument("--no-pseudo-stream", action="store_true",
                        help="关闭伪流式中间结果 enable_pseudo_stream=false")
    parser.add_argument("--config", action="append", default=[], metavar="KEY=VALUE",
                        help="通用 parameter.asr_config 覆写，可多次，如 --config silence_duration_ms=300")
    parser.add_argument("--trace-id", default="ast-v3-demo",
                        help="header.traceId (echoed back)")
    parser.add_argument("--biz-id", default="12345", help="header.bizId (required field)")
    parser.add_argument("--app-id", default="ast", help="header.appId")
    parser.add_argument("--chunk-ms", type=int, default=128,
                        help="PCM chunk size in ms (~4096 bytes)")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="Connect/read timeout in seconds")
    parser.add_argument("--final-timeout", type=float, default=30.0,
                        help="Wait time after last frame")
    parser.add_argument("--insecure", action="store_true",
                        help="Disable TLS cert verification (self-signed)")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
