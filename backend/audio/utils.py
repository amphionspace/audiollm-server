import base64
import io
import struct

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
