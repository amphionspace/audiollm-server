"""Pluggable wire protocols for :class:`StreamingSession`.

A :class:`WireProtocol` decouples the *on-the-wire* message framing from the
session's internal semantics. The session only knows two inbound primitives
(a control dict, or a chunk of PCM bytes) and a small set of outbound message
dicts (``ready`` / ``partial`` / ``final`` / ``error`` / ...). Each protocol
translates between those internal primitives and whatever the client speaks.

Two protocols ship here:

- :class:`NativeProtocol` is the historical 1:1 framing used by
  ``/transcribe-streaming`` and ``/emotion-segmented-streaming``: text frames
  are JSON control messages, binary frames are raw PCM, and outbound messages
  go out verbatim. It is the default so those endpoints are untouched.
- :class:`AstV3Protocol` speaks the iFlytek Tuling AST v3 envelope
  (``header`` / ``parameter`` / ``payload``): audio arrives base64-encoded
  inside JSON frames, ``header.status`` (0/1/2) drives the start/stop state
  machine, and results are repackaged into the ``payload.result`` lattice
  structure. The current ASR stack only produces whole-sentence text, so the
  word-level ``ws[].cw[]`` fields are filled with one cw per sentence and the
  per-word timing/score fields carry segment-level approximations / defaults
  (see ``docs/tuling-ast-v3-protocol.md``).
"""

from __future__ import annotations

import base64
import json
import logging
import re
import secrets
import string
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inbound primitives the session understands
# ---------------------------------------------------------------------------


@dataclass
class ControlAction:
    """A normalized control message (``start`` / ``stop`` / ``update_hotwords`` ...)."""

    ctrl: dict


@dataclass
class PcmAction:
    """A chunk of raw signed-16-bit little-endian PCM bytes to feed the stream."""

    data: bytes


InboundAction = ControlAction | PcmAction


@runtime_checkable
class WireProtocol(Protocol):
    """Translate between the on-the-wire framing and session primitives."""

    def decode_inbound(self, msg: dict) -> list[InboundAction]:
        """Decode one raw WebSocket ``receive()`` dict into ordered actions."""

    def encode_outbound(self, payload: dict) -> dict | None:
        """Encode one internal message dict for the wire (``None`` = suppress)."""

    def encode_terminal(self) -> dict | None:
        """Optional final frame sent once when the session ends (``None`` = none)."""


# ---------------------------------------------------------------------------
# Native protocol (current behavior, default)
# ---------------------------------------------------------------------------


class NativeProtocol:
    """Historical framing: text = JSON control, bytes = PCM, output verbatim."""

    def decode_inbound(self, msg: dict) -> list[InboundAction]:
        text = msg.get("text")
        if text:
            try:
                ctrl = json.loads(text)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON from client: %.200s", text)
                return []
            if isinstance(ctrl, dict):
                return [ControlAction(ctrl)]
            return []
        data = msg.get("bytes")
        if data:
            return [PcmAction(data)]
        return []

    def encode_outbound(self, payload: dict) -> dict | None:
        return payload

    def encode_terminal(self) -> dict | None:
        return None


# ---------------------------------------------------------------------------
# AST v3 protocol (iFlytek Tuling)
# ---------------------------------------------------------------------------

_SID_ALPHABET = string.ascii_uppercase + string.digits
_HOTWORD_SPLIT = re.compile(r"[,，、;；\n]+")

# Map our model-side / client language strings to the short codes the AST v3
# ``cw.lg`` field expects (the spec example uses "zh").
_LANG_TO_CODE: dict[str, str] = {
    "chinese": "zh",
    "english": "en",
    "indonesian": "id",
    "thai": "th",
    "cn": "zh",
    "zh": "zh",
    "en": "en",
    "id": "id",
    "th": "th",
}

# Generic, non-zero error code. The AST v3 spec only pins code 0 = success and
# leaves the failure code space to the implementation, so we document this in
# docs/tuling-ast-v3-protocol.md rather than inventing a per-error taxonomy.
_ERROR_CODE = -1

# Safety cap while locating the WAV ``data`` chunk in a header-prefixed stream;
# beyond this we stop buffering and treat the bytes as raw PCM so a malformed
# header can never stall the session indefinitely.
_WAV_HEADER_SCAN_LIMIT = 8192


def _gen_sid() -> str:
    return "AST_" + "".join(secrets.choice(_SID_ALPHABET) for _ in range(13))


def _short_lang(value: object) -> str:
    code = str(value or "").strip().lower()
    if not code:
        return ""
    return _LANG_TO_CODE.get(code, code)


def _parse_hotword_text(raw: object) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    return [tok.strip() for tok in _HOTWORD_SPLIT.split(text) if tok.strip()]


def _coerce_status(raw: object) -> int:
    """Read ``header.status`` leniently (int per spec, but tolerate "2"/2.0).

    Be liberal inbound so a client that JSON-encodes status as a string still
    drives the state machine; outbound framing stays strict per the spec.
    Unparseable values fall back to 1 (middle frame), which only feeds audio
    and never strands the session — end-of-session is still guaranteed by the
    terminal frame on socket close.
    """
    if isinstance(raw, bool):
        return 1
    if isinstance(raw, int):
        return raw
    try:
        return int(float(raw))  # handles "0" / "2" / 2.0
    except (TypeError, ValueError):
        return 1


class AstV3Protocol:
    """iFlytek Tuling AST v3 envelope protocol (stateful, one per connection).

    Inbound: each text frame is a ``{header, parameter, payload}`` envelope.
    The first frame is mapped to a synthesized ``start`` control (capturing
    ``traceId`` and any ``payload.text.text`` hotwords); ``payload.audio.audio``
    is base64-decoded to PCM; ``header.status == 2`` appends a ``stop`` control.

    Outbound: ``final`` -> ``msgtype: sentence`` lattice frame (status 1),
    ``partial`` -> ``msgtype: Progressive`` (status 1), ``error`` -> non-zero
    ``header.code``. A single terminal frame with ``header.status == 2`` marks
    end-of-session. ``ready`` and other native-only messages are suppressed.
    """

    def __init__(self) -> None:
        self.sid = _gen_sid()
        self.trace_id = ""
        self._inbound_started = False
        self._terminated = False
        # Result counters. segId identifies a speech segment; sn is the result
        # sequence number. We emit one sentence per segment, so both advance in
        # lockstep on every final (segId from 0, sn from 1 per the spec sample).
        self._seg_id = 0
        self._sn = 1
        # Stateful audio decode: strip a leading WAV header (the reference Java
        # SDK chunks an entire .wav file) and keep PCM frames 16-bit aligned.
        self._pcm_resolved = False
        self._lead_buf = b""
        self._byte_carry = b""

    # -- inbound ------------------------------------------------------------

    def decode_inbound(self, msg: dict) -> list[InboundAction]:
        if msg.get("bytes"):
            logger.debug("AST v3: ignoring unexpected binary frame")
            return []
        text = msg.get("text")
        if not text:
            return []
        try:
            frame = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("AST v3: invalid JSON frame: %.200s", text)
            return []
        if not isinstance(frame, dict):
            return []

        header = frame.get("header") or {}
        parameter = frame.get("parameter") or {}
        payload = frame.get("payload") or {}
        status = _coerce_status(header.get("status", 1))

        actions: list[InboundAction] = []

        if not self._inbound_started:
            self._inbound_started = True
            self.trace_id = str(header.get("traceId") or "")
            logger.info(
                "AST v3 session start: sid=%s traceId=%s appId=%s bizId=%s "
                "resIdList=%s (ignored)",
                self.sid,
                self.trace_id,
                header.get("appId"),
                header.get("bizId"),
                header.get("resIdList"),
            )
            # parameter.engine (or the SDK's parameter.service) are engine
            # passthrough knobs with no equivalent in this stack; log them so
            # operators can see what was requested, but do not map behavior.
            engine_params = parameter.get("engine") or parameter.get("service")
            if engine_params:
                logger.info(
                    "AST v3 engine passthrough params (not applied): %s",
                    engine_params,
                )
            start_ctrl: dict = {"type": "start"}
            if self.trace_id:
                start_ctrl["gateway_trace_id"] = self.trace_id
            hotwords = _parse_hotword_text(self._payload_text(payload))
            if hotwords:
                start_ctrl["hotwords"] = hotwords
            # parameter.asr_config is this service's per-connection tuning slot
            # (distinct from the log-only iFlytek engine block). language is not
            # a Config field so it rides start.language; the rest becomes
            # start.config and is whitelist-filtered downstream in the session.
            language, hotword_pool_id, enrollment_id, cfg_overrides = (
                self._extract_asr_config(parameter)
            )
            if language:
                start_ctrl["language"] = language
            if hotword_pool_id:
                start_ctrl["hotword_pool_id"] = hotword_pool_id
            if enrollment_id:
                start_ctrl["enrollment_id"] = enrollment_id
            if cfg_overrides:
                start_ctrl["config"] = cfg_overrides
            actions.append(ControlAction(start_ctrl))

        pcm = self._decode_audio(payload)
        if pcm:
            actions.append(PcmAction(pcm))

        if status == 2:
            actions.append(ControlAction({"type": "stop"}))

        return actions

    @staticmethod
    def _payload_text(payload: dict) -> object:
        text_obj = payload.get("text")
        if isinstance(text_obj, dict):
            return text_obj.get("text")
        return None

    @staticmethod
    def _extract_asr_config(parameter: dict) -> tuple[str, str, str, dict]:
        """Split ``parameter.asr_config`` into routed fields and overrides.

        ``asr_config`` is this service's extension slot for per-connection
        tuning; the iFlytek ``parameter.engine`` block stays log-only. The
        ``language`` key is pulled out because it is not a ``Config`` field (the
        session maps it separately via ``start.language``), and
        ``hotword_pool_id`` is pulled out as the Triton hotword-pool isolation
        key. ``enrollment_id`` is the only AST v3 target-speaker field. Every
        other key is forwarded verbatim as ``start.config`` and is
        whitelist-filtered by ``Config.override_client`` downstream, so no
        validation happens here.
        Returns empty values when the slot is absent or not a dict.
        """
        cfg = parameter.get("asr_config")
        if not isinstance(cfg, dict) or not cfg:
            return "", "", "", {}
        overrides = dict(cfg)
        language = str(overrides.pop("language", "") or "").strip()
        hotword_pool_id = str(overrides.pop("hotword_pool_id", "") or "").strip()
        enrollment_id = str(overrides.pop("enrollment_id", "") or "").strip()
        overrides.pop("user_id", None)
        return language, hotword_pool_id, enrollment_id, overrides

    def _decode_audio(self, payload: dict) -> bytes:
        audio_obj = payload.get("audio")
        b64 = audio_obj.get("audio") if isinstance(audio_obj, dict) else None
        if not b64:
            return b""
        try:
            raw = base64.b64decode(b64, validate=False)
        except (ValueError, TypeError):
            logger.warning("AST v3: invalid base64 audio chunk")
            return b""
        return self._extract_pcm(raw)

    def _extract_pcm(self, raw: bytes) -> bytes:
        """Return 16-bit-aligned PCM, stripping a one-time leading WAV header."""
        if not self._pcm_resolved:
            self._lead_buf += raw
            if len(self._lead_buf) < 12:
                return b""
            if self._lead_buf[:4] == b"RIFF" and self._lead_buf[8:12] == b"WAVE":
                idx = self._lead_buf.find(b"data")
                if idx == -1:
                    if len(self._lead_buf) <= _WAV_HEADER_SCAN_LIMIT:
                        return b""
                    # Pathological header; stop stalling and treat as PCM.
                    data = self._lead_buf
                else:
                    data = self._lead_buf[idx + 8:]  # 'data' (4) + size (4)
                    logger.info("AST v3: stripped WAV header (%d bytes)", idx + 8)
            else:
                data = self._lead_buf
            self._lead_buf = b""
            self._pcm_resolved = True
        else:
            data = raw

        if self._byte_carry:
            data = self._byte_carry + data
            self._byte_carry = b""
        if len(data) % 2:
            self._byte_carry = data[-1:]
            data = data[:-1]
        return data

    # -- outbound -----------------------------------------------------------

    def encode_outbound(self, payload: dict) -> dict | None:
        mtype = payload.get("type")
        if mtype == "final":
            text = str(payload.get("text") or "")
            if not text:
                # Empty final = "nothing heard"; the terminal status=2 frame is
                # the canonical end-of-session signal, so do not emit a frame.
                return None
            return self._sentence_frame(text, payload)
        if mtype == "partial":
            text = str(payload.get("text") or "")
            if not text:
                return None
            return self._progressive_frame(text, payload)
        if mtype == "error":
            return self._error_frame(str(payload.get("message") or "error"))
        # ready / extract_hotwords_* / unknown have no AST v3 representation.
        return None

    def encode_terminal(self) -> dict | None:
        if self._terminated:
            return None
        self._terminated = True
        result = {
            "segId": self._seg_id,
            "bg": 0,
            "ed": 0,
            "ei": 0,
            "ls": True,
            "metadata": "",
            "msgtype": "sentence",
            "sn": self._sn,
            "pa": 0,
        }
        return self._envelope(result, status=2)

    def _envelope(self, result: dict, *, status: int) -> dict:
        return {
            "header": {
                "code": 0,
                "message": "success",
                "sid": self.sid,
                "traceId": self.trace_id,
                "status": status,
            },
            "payload": {"result": result},
        }

    def _sentence_frame(self, text: str, payload: dict) -> dict:
        bg_ms = max(0, int(round(float(payload.get("bg_ms") or 0.0))))
        ed_ms = max(bg_ms, int(round(float(payload.get("ed_ms") or 0.0))))
        bg_f = bg_ms // 10
        ed_f = ed_ms // 10
        lg = _short_lang(payload.get("language"))

        seg_id = self._seg_id
        sn = self._sn
        self._seg_id += 1
        self._sn += 1

        result = {
            "segId": seg_id,
            "bg": bg_ms,
            "ed": ed_ms,
            "ei": 0,
            "ls": False,
            "metadata": "",
            "msgtype": "sentence",
            "sn": sn,
            "pa": 0,
            "vad": {"ws": [{"bg": bg_f, "ed": ed_f}]},
            "ws": [
                {
                    "bg": bg_f,
                    "cw": [
                        {
                            "lg": lg,
                            "ng": "0.00",
                            "ph": "phone",
                            "rl": 0,
                            "sc": "0.00",
                            "w": text,
                            "wb": bg_f,
                            "wc": "0.00",
                            "we": ed_f,
                            "wp": "n",
                        }
                    ],
                }
            ],
        }
        return self._envelope(result, status=1)

    def _progressive_frame(self, text: str, payload: dict) -> dict:
        lg = _short_lang(payload.get("language"))
        result = {
            "segId": self._seg_id,
            "ls": False,
            "msgtype": "Progressive",
            "ws": [
                {
                    "bg": 0,
                    "cw": [
                        {
                            "lg": lg,
                            "ng": "0.00",
                            "ph": "phone",
                            "sc": "0.00",
                            "w": text,
                            "wb": 0,
                            "wc": "0.00",
                            "we": 0,
                            "wp": "n",
                        }
                    ],
                }
            ],
        }
        return self._envelope(result, status=1)

    def _error_frame(self, message: str) -> dict:
        return {
            "header": {
                "code": _ERROR_CODE,
                "message": message,
                "sid": self.sid,
                "traceId": self.trace_id,
                "status": 1,
            }
        }
