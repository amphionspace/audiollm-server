import base64
import io
import struct
import wave

import numpy as np


class Resampler48to16:
    """Streaming 48 kHz -> 16 kHz resampler.

    Uses a Kaiser-windowed sinc FIR low-pass filter (65 taps, beta=6)
    applied via overlap-save convolution, followed by factor-3 decimation.
    Maintains internal state so successive `process()` calls are seamless.

    Approximate filter characteristics at fs=48 kHz:
      passband  < 6.5 kHz  (~0.1 dB ripple)
      stopband  > 9.5 kHz  (~60 dB attenuation)
    """

    RATIO = 3  # 48000 / 16000

    def __init__(self, n_taps: int = 65, beta: float = 6.0) -> None:
        cutoff = 1.0 / self.RATIO
        half = n_taps // 2
        t = np.arange(-half, half + 1, dtype=np.float64)
        h = np.sinc(2.0 * cutoff * t) * (2.0 * cutoff)
        h *= np.kaiser(n_taps, beta)
        h /= h.sum()
        self._kernel = h.astype(np.float32)
        self._overlap = np.zeros(n_taps - 1, dtype=np.float32)
        self._tail = np.empty(0, dtype=np.float32)

    def process(self, pcm_48k: np.ndarray) -> np.ndarray:
        """Feed a chunk of 48 kHz float32 PCM; returns 16 kHz float32 PCM."""
        buf = (
            np.concatenate([self._tail, pcm_48k])
            if self._tail.size
            else pcm_48k
        )
        usable = (buf.size // self.RATIO) * self.RATIO
        if usable == 0:
            self._tail = buf.copy()
            return np.empty(0, dtype=np.float32)
        self._tail = buf[usable:].copy()
        seg = buf[:usable]
        ext = np.concatenate([self._overlap, seg])
        flt = np.convolve(ext, self._kernel, mode="valid")
        self._overlap = ext[-(self._kernel.size - 1) :].copy()
        return flt[:: self.RATIO].astype(np.float32)


def pcm_to_wav_bytes(pcm: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Convert float32 PCM array to WAV file bytes (16-bit)."""
    pcm_int16 = np.clip(pcm * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    num_samples = len(pcm_int16)
    data_size = num_samples * 2  # 16-bit = 2 bytes per sample

    # WAV header (44 bytes, mono, 16-bit)
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))           # chunk size
    buf.write(struct.pack("<H", 1))            # PCM format
    buf.write(struct.pack("<H", 1))            # mono
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * 2))  # byte rate
    buf.write(struct.pack("<H", 2))            # block align
    buf.write(struct.pack("<H", 16))           # bits per sample
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm_int16.tobytes())

    return buf.getvalue()


def pcm_to_wav_base64(pcm: np.ndarray, sample_rate: int = 16000) -> str:
    """Convert float32 PCM array to base64-encoded WAV string."""
    wav_bytes = pcm_to_wav_bytes(pcm, sample_rate)
    return base64.b64encode(wav_bytes).decode("ascii")


_RESAMPLER_48_TO_16: Resampler48to16 | None = None


def _resample_linear(pcm: np.ndarray, src_sr: int, dst_sr: int = 16000) -> np.ndarray:
    """Fallback linear-interpolation resampler for arbitrary ratios.

    Good enough for enrollment/demo audio. Hot paths (live streaming) use the
    Kaiser-windowed FIR in :class:`Resampler48to16`.
    """
    if src_sr == dst_sr or pcm.size == 0:
        return pcm.astype(np.float32, copy=False)
    ratio = dst_sr / float(src_sr)
    target_len = max(1, int(round(pcm.size * ratio)))
    xp = np.arange(pcm.size, dtype=np.float64)
    x = np.linspace(0.0, pcm.size - 1, target_len, dtype=np.float64)
    return np.interp(x, xp, pcm).astype(np.float32)


def wav_base64_to_pcm_16k_mono(b64: str) -> np.ndarray:
    """Decode a base64-encoded WAV string to float32 mono PCM at 16 kHz.

    Accepts PCM WAVs with 8/16/24/32-bit integer or 32-bit float samples, any
    channel count, and any sample rate. Multi-channel input is averaged to
    mono; non-16 kHz rates are resampled (48 kHz uses the FIR resampler, all
    other rates use a linear fallback).

    Returns a 1-D ``np.ndarray[np.float32]`` in the range ``[-1, 1]``.

    Raises:
        ValueError: when the input is not a parseable PCM WAV (e.g. empty
            payload, non-PCM compressed WAV).
    """
    try:
        wav_bytes = base64.b64decode(b64, validate=False)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid base64 payload: {exc}") from exc
    if not wav_bytes:
        raise ValueError("Empty WAV payload")

    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except wave.Error as exc:
        raise ValueError(f"Invalid WAV container: {exc}") from exc

    if n_frames <= 0 or not raw:
        return np.empty(0, dtype=np.float32)

    if sample_width == 1:
        samples = (
            np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
        ) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 3:
        # 24-bit little-endian packed; unpack into int32 then normalize.
        buf = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        ints = (
            buf[:, 0].astype(np.int32)
            | (buf[:, 1].astype(np.int32) << 8)
            | (buf[:, 2].astype(np.int32) << 16)
        )
        sign_bit = 1 << 23
        ints = np.where(ints & sign_bit, ints - (1 << 24), ints)
        samples = ints.astype(np.float32) / float(sign_bit)
    elif sample_width == 4:
        samples = (
            np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        )
    else:
        raise ValueError(f"Unsupported sample width: {sample_width} bytes")

    if n_channels > 1:
        usable = (samples.size // n_channels) * n_channels
        samples = samples[:usable].reshape(-1, n_channels).mean(axis=1)

    if framerate == 16000:
        return samples.astype(np.float32, copy=False)

    if framerate == 48000:
        global _RESAMPLER_48_TO_16
        if _RESAMPLER_48_TO_16 is None:
            _RESAMPLER_48_TO_16 = Resampler48to16()
        # Use a fresh stateless instance for one-shot decode to avoid leaking
        # filter state across enrollment uploads.
        return Resampler48to16().process(samples.astype(np.float32))

    return _resample_linear(samples.astype(np.float32), framerate, 16000)
