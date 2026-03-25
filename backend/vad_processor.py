import numpy as np
from ten_vad import TenVad

from .config import HOP_SIZE, SILENCE_DURATION_MS, VAD_THRESHOLD


class VADProcessor:
    def __init__(
        self,
        hop_size: int = HOP_SIZE,
        threshold: float = VAD_THRESHOLD,
        silence_duration_ms: int = SILENCE_DURATION_MS,
    ):
        self.vad = TenVad()
        self.hop_size = hop_size
        self.threshold = threshold
        self.silence_frames = silence_duration_ms // 10
        self.audio_buffer: list[np.ndarray] = []
        self.silent_count = 0
        self.is_speaking = False

    def process(self, pcm_frame: np.ndarray) -> np.ndarray | None:
        """Feed one frame (hop_size samples, float32).
        Returns the full speech segment when speech-to-silence transition
        is detected, otherwise None.
        """
        prob = self.vad.process(pcm_frame)

        if prob > self.threshold:
            self.is_speaking = True
            self.silent_count = 0
            self.audio_buffer.append(pcm_frame.copy())
        elif self.is_speaking:
            self.silent_count += 1
            self.audio_buffer.append(pcm_frame.copy())
            if self.silent_count >= self.silence_frames:
                # Trim trailing silence (keep a small tail for natural sound)
                keep_tail = min(10, self.silence_frames)
                trim = self.silence_frames - keep_tail
                if trim > 0:
                    del self.audio_buffer[-trim:]
                segment = np.concatenate(self.audio_buffer)
                self._reset()
                return segment

        return None

    def flush(self) -> np.ndarray | None:
        """Flush any remaining buffered speech (e.g. on disconnect)."""
        if self.audio_buffer and self.is_speaking:
            segment = np.concatenate(self.audio_buffer)
            self._reset()
            return segment
        self._reset()
        return None

    def _reset(self):
        self.audio_buffer.clear()
        self.silent_count = 0
        self.is_speaking = False
