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


def _coerce_bool(value: object) -> bool:
    """Read a JSON-ish boolean leniently (bool / 0-1 / "true"/"false"/...)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


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
        # Role-separation / enrollment result state (contract V0.4-review).
        # Role separation defaults on and takes priority over enrollment, so
        # sentence frames carry cw[].rl and enrollment_used=False until the
        # first frame's parameter.asr_config resolves the effective mode.
        # ``enrollment_used`` is corrected by the session after it actually
        # resolves the enrollment id (see set_enrollment_used).
        self._role_separation_active = True
        self._enrollment_used = False

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
            (
                language,
                role_sep,
                enrollment_enable,
                enrollment_id,
                hotword_pool_id,
                cfg_overrides,
            ) = self._extract_asr_config(parameter)
            logger.info(
                "AST v3 session start: sid=%s traceId=%s appId=%s bizId=%s "
                "enable_role_separation=%s enrollment_enable=%s enrollment_id=%s",
                self.sid,
                self.trace_id,
                header.get("appId"),
                header.get("bizId"),
                role_sep,
                enrollment_enable,
                bool(enrollment_id),
            )
            # header.resIdList is deprecated as a voiceprint entry (contract
            # V0.4-review) and is no longer honored; log it if a stale client
            # still sends it so operators can see the obsolete field.
            if header.get("resIdList"):
                logger.info(
                    "AST v3: ignoring deprecated header.resIdList=%s",
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
            # Resolve the effective mode from the role-separation/enrollment
            # matrix. Role separation defaults on (None/True) and takes
            # priority over enrollment; only a closed role separation with
            # enrollment_enable and an empty id is a parameter error.
            role_separation_on = role_sep is not False
            mode = self._resolve_mode(
                role_separation_on, enrollment_enable, enrollment_id
            )
            if mode == "param_error":
                # Surface an explicit parameter error instead of silently
                # falling back. The session sends the error frame and ends the
                # session (see StreamingSession._handle_control).
                self._role_separation_active = False
                self._enrollment_used = False
                logger.warning(
                    "AST v3 parameter error: enrollment_enable=true requires a "
                    "non-empty enrollment_id (sid=%s traceId=%s)",
                    self.sid,
                    self.trace_id,
                )
                return [
                    ControlAction(
                        {
                            "type": "protocol_error",
                            "message": (
                                "enrollment_enable=true requires a non-empty "
                                "enrollment_id"
                            ),
                        }
                    )
                ]
            # cw[].rl is only emitted when role separation is active; the value
            # is a protocol-compatible placeholder because neither the Amphion
            # ASR model nor the split decode path produces real speaker labels
            # (see change ast-v3-role-separation-and-enrollment design).
            self._role_separation_active = mode == "role_separation"
            self._enrollment_used = False

            start_ctrl: dict = {"type": "start"}
            if self.trace_id:
                start_ctrl["trace_id"] = self.trace_id
            hotwords = _parse_hotword_text(self._payload_text(payload))
            if hotwords:
                start_ctrl["hotwords"] = hotwords
            # hotword_pool_id selects which RAG-ASR pool feeds recall for this
            # session (empty = default pool). Routed explicitly so it is not
            # dropped by the start.config whitelist.
            if hotword_pool_id:
                start_ctrl["hotword_pool_id"] = hotword_pool_id
            # Only route the enrollment id downstream when the mode is
            # enrollment. The session resolves it through the shared enrollment
            # store and degrades to plain ASR on an unknown/expired id, then
            # reports the real outcome back via set_enrollment_used.
            if mode == "enrollment":
                start_ctrl["enrollment_id"] = enrollment_id
            # parameter.asr_config language rides start.language; the remaining
            # keys become start.config and are whitelist-filtered downstream.
            if language:
                start_ctrl["language"] = language
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

    def set_enrollment_used(self, used: bool) -> None:
        """Record whether enrollment was actually applied for this session.

        Called by the session after it resolves the enrollment id, so sentence
        frames report a truthful ``enrollment_used`` (an unknown/expired id
        that fell back to plain ASR reports ``False``).
        """
        self._enrollment_used = bool(used)

    @staticmethod
    def _resolve_mode(
        role_separation_on: bool, enrollment_enable: bool, enrollment_id: str
    ) -> str:
        """Resolve the effective mode from the role-separation matrix.

        Returns one of ``role_separation`` / ``enrollment`` / ``plain_asr`` /
        ``param_error``. Role separation is highest priority; enrollment only
        applies when role separation is closed. A closed role separation with
        ``enrollment_enable`` but an empty id is a parameter error rather than a
        silent fallback. A non-empty but unknown/expired id resolves to
        ``enrollment`` here and degrades to plain ASR in the session.
        """
        if role_separation_on:
            return "role_separation"
        if enrollment_enable:
            return "enrollment" if enrollment_id else "param_error"
        return "plain_asr"

    @staticmethod
    def _extract_asr_config(
        parameter: dict,
    ) -> tuple[str, bool | None, bool, str, str, dict]:
        """Split ``parameter.asr_config`` into routed fields plus overrides.

        Returns ``(language, enable_role_separation, enrollment_enable,
        enrollment_id, hotword_pool_id, config-overrides)``.
        ``enable_role_separation`` is ``None`` when the client omits it
        (equivalent to on) so the caller can distinguish "omitted" from an
        explicit ``false``. The role-separation, enrollment, and
        hotword_pool_id keys are consumed here (routed explicitly, not through
        ``start.config``); language is pulled out because it is not a
        ``Config`` field; every remaining key is forwarded as ``start.config``
        and whitelist-filtered by ``Config.override_client`` downstream.
        Returns defaults when the slot is absent or not a dict.
        """
        cfg = parameter.get("asr_config")
        if not isinstance(cfg, dict) or not cfg:
            return "", None, False, "", "", {}
        overrides = dict(cfg)
        language = str(overrides.pop("language", "") or "").strip()
        raw_role_sep = overrides.pop("enable_role_separation", None)
        role_sep = None if raw_role_sep is None else _coerce_bool(raw_role_sep)
        enrollment_enable = _coerce_bool(overrides.pop("enrollment_enable", False))
        enrollment_id = str(overrides.pop("enrollment_id", "") or "").strip()
        hotword_pool_id = str(overrides.pop("hotword_pool_id", "") or "").strip()
        return (
            language,
            role_sep,
            enrollment_enable,
            enrollment_id,
            hotword_pool_id,
            overrides,
        )

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
        if mtype == "speech_started":
            # VAD-activated notification.  Clients use the arrival time of this
            # frame as the base for TTFT measurement so leading silence is
            # excluded.  Sent as a status=1 result frame with msgtype=speech_started.
            return {
                "header": {"sid": self.sid, "status": 1},
                "payload": {
                    "result": {
                        "segId": self._seg_id,
                        "msgtype": "speech_started",
                    }
                },
            }
        # ready / extract_hotwords_* / unknown have no AST v3 representation.
        return None

    def encode_terminal(self) -> dict | None:
        if self._terminated:
            return None
        self._terminated = True
        # CONTRACT — DO NOT REMOVE the empty ``ws`` below.
        # The terminal (status=2) frame MUST carry an empty ``ws`` placeholder.
        # This is a customer/SDK agreement: the client always receives ``ws`` and
        # parses it even though the word is empty. (An older note in
        # docs/tuling-ast-v3-protocol.md line 379 claiming ``ws`` is forbidden is
        # OUTDATED — ignore it. See the ascend-asr-delivery skill for the rule.)
        # Only add fields to this shape in the future; never delete ``ws``.
        cw = {
            "lg": "",
            "ng": "0.00",
            "ph": "phone",
            "sc": "0.00",
            "w": "",
            "wb": 0,
            "wc": "0.00",
            "we": 0,
            "wp": "n",
        }
        if self._role_separation_active:
            cw["rl"] = 0
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
            "enrollment_used": self._enrollment_used,
            "ws": [{"bg": 0, "cw": [cw]}],
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

        cw = {
            "lg": lg,
            "ng": "0.00",
            "ph": "phone",
            "sc": "0.00",
            "w": text,
            "wb": bg_f,
            "wc": "0.00",
            "we": ed_f,
            "wp": "n",
        }
        # cw[].rl (role/speaker number) is only present on sentence results
        # while role separation is active; it is a protocol placeholder (see
        # set_enrollment_used / design). Progressive results never carry it.
        if self._role_separation_active:
            cw["rl"] = 0
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
            "enrollment_used": self._enrollment_used,
            "vad": {"ws": [{"bg": bg_f, "ed": ed_f}]},
            "ws": [{"bg": bg_f, "cw": [cw]}],
        }
        return self._envelope(result, status=1)

    def _progressive_frame(self, text: str, payload: dict) -> dict:
        lg = _short_lang(payload.get("language"))
        # Progressive (intermediate) results never carry cw[].rl regardless of
        # role separation; enrollment_used is still reported for observability.
        result = {
            "segId": self._seg_id,
            "ls": False,
            "msgtype": "Progressive",
            "enrollment_used": self._enrollment_used,
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
