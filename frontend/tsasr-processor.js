/**
 * AudioWorklet processor for the TS-ASR demo.
 *
 * Runs inside an ``AudioContext({sampleRate: 16000})`` so the backend can
 * consume the raw frames directly — the generic /transcribe-target-streaming
 * endpoint expects 16 kHz mono PCM16 (matching the ``VadSegmentedStream``
 * contract), without the 48 kHz resampling the legacy /ws/audio path does.
 */
class TsAsrCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._size = 480; // 30 ms at 16 kHz
    this._buf = new Float32Array(this._size + 128);
    this._pos = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const samples = input[0];

    this._buf.set(samples, this._pos);
    this._pos += samples.length;

    while (this._pos >= this._size) {
      this.port.postMessage({
        type: 'audio',
        samples: new Float32Array(this._buf.subarray(0, this._size)),
      });
      this._buf.copyWithin(0, this._size, this._pos);
      this._pos -= this._size;
    }
    return true;
  }
}

registerProcessor('tsasr-capture-processor', TsAsrCaptureProcessor);
