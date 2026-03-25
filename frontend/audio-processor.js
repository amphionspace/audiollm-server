/**
 * AudioWorklet processor: captures raw PCM from the browser's audio input,
 * downsamples from the native sample rate (usually 48kHz) to 16kHz,
 * and posts Int16 PCM buffers to the main thread.
 */
class AudioCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = [];
    this._targetRate = 16000;
    this._ratio = sampleRate / this._targetRate;
    this._accumulator = 0;
    // Send every ~30ms worth of 16kHz samples (480 samples)
    this._chunkSize = 480;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const samples = input[0]; // Float32Array, native sample rate

    // Simple linear-interpolation downsampling
    for (let i = 0; i < samples.length; i++) {
      this._accumulator++;
      if (this._accumulator >= this._ratio) {
        this._accumulator -= this._ratio;
        this._buffer.push(samples[i]);

        if (this._buffer.length >= this._chunkSize) {
          const chunk = new Float32Array(this._buffer);
          this.port.postMessage({ type: 'audio', samples: chunk });
          this._buffer = [];
        }
      }
    }

    return true;
  }
}

registerProcessor('audio-capture-processor', AudioCaptureProcessor);
