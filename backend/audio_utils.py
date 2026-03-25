import base64
import io
import struct

import numpy as np


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
