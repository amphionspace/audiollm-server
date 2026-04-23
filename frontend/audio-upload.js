/**
 * Shared helpers for the "upload local audio file" feature on each demo page.
 *
 * The flow is now uniform across demos:
 *   1. Decode the user-picked file with the browser's audio decoder.
 *   2. Down-mix to mono and resample to 16 kHz (what every backend model
 *      actually consumes).
 *   3. Encode the float32 PCM as a 16-bit WAV blob.
 *   4. POST the WAV as multipart/form-data to a one-shot REST endpoint
 *      (no WebSocket, no chunking). The server returns the final result
 *      in a single JSON response.
 *
 * Exposed as a small global namespace (`window.AmphionAudioUpload`) so the
 * three demo scripts can pull it in without needing a module bundler.
 */
(() => {
  'use strict';

  /**
   * Decode an audio file (wav/mp3/m4a/ogg/flac, browser-dependent) into a
   * single-channel Float32Array sampled at `targetSr` Hz.
   *
   * Internally we first decode at the file's native rate using a temporary
   * AudioContext (Safari needs a real AudioContext for `decodeAudioData`),
   * then run the buffer through an OfflineAudioContext to do the resample
   * and channel down-mix in one pass.
   */
  async function decodeFileToMono(file, targetSr) {
    if (!file) throw new Error('No file provided');
    const arrayBuffer = await file.arrayBuffer();

    const TempCtx = window.AudioContext || window.webkitAudioContext;
    if (!TempCtx) throw new Error('AudioContext is not available');

    const tempCtx = new TempCtx();
    let decoded;
    try {
      decoded = await new Promise((resolve, reject) => {
        // Both promise- and callback-style decodeAudioData exist; Safari only
        // supports the callback form so we cover both.
        try {
          const maybe = tempCtx.decodeAudioData(arrayBuffer.slice(0), resolve, reject);
          if (maybe && typeof maybe.then === 'function') {
            maybe.then(resolve, reject);
          }
        } catch (err) {
          reject(err);
        }
      });
    } finally {
      try { tempCtx.close(); } catch (_) { /* ignore */ }
    }

    const srcSr = decoded.sampleRate;
    const srcLen = decoded.length;
    if (srcLen === 0) {
      return new Float32Array(0);
    }

    const outLen = Math.max(1, Math.round(srcLen * (targetSr / srcSr)));
    const Offline = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    if (!Offline) throw new Error('OfflineAudioContext is not available');

    const offline = new Offline(1, outLen, targetSr);
    const src = offline.createBufferSource();
    src.buffer = decoded;

    if (decoded.numberOfChannels > 1) {
      // Average the channels into the single offline output channel.
      const merger = offline.createGain();
      merger.gain.value = 1 / decoded.numberOfChannels;
      src.connect(merger);
      merger.connect(offline.destination);
    } else {
      src.connect(offline.destination);
    }
    src.start(0);

    const rendered = await offline.startRendering();
    return rendered.getChannelData(0).slice();
  }

  /**
   * Encode a Float32 mono PCM array as a little-endian 16-bit PCM WAV.
   * Returns a Uint8Array of the full WAV byte-stream (header + data).
   *
   * Bit-for-bit compatible with the WAV the streaming pipeline produces
   * server-side via `pcm_to_wav_base64`, so the model sees identical input
   * regardless of whether the audio came in over WS or REST.
   */
  function encodeWavBytes(float32, sampleRate) {
    const numSamples = float32.length;
    const bytesPerSample = 2;
    const byteRate = sampleRate * bytesPerSample;
    const dataSize = numSamples * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    // RIFF header
    writeAscii(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeAscii(view, 8, 'WAVE');
    // fmt chunk
    writeAscii(view, 12, 'fmt ');
    view.setUint32(16, 16, true);          // fmt chunk size
    view.setUint16(20, 1, true);           // PCM
    view.setUint16(22, 1, true);           // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, bytesPerSample, true); // block align
    view.setUint16(34, 16, true);          // bits per sample
    // data chunk
    writeAscii(view, 36, 'data');
    view.setUint32(40, dataSize, true);

    let offset = 44;
    for (let i = 0; i < numSamples; i++) {
      const s = Math.max(-1, Math.min(1, float32[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }
    return new Uint8Array(buffer);
  }

  function writeAscii(view, offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  /**
   * Convert a Uint8Array WAV (or any byte array) to a base64 string. Used for
   * passing the enrollment WAV to the TS-ASR endpoint as a form field.
   */
  function bytesToBase64(bytes) {
    let binary = '';
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) {
      binary += String.fromCharCode.apply(
        null,
        bytes.subarray(i, Math.min(i + chunk, bytes.length))
      );
    }
    return btoa(binary);
  }

  /**
   * POST a WAV byte buffer to a REST endpoint as multipart/form-data and
   * return the parsed JSON response.
   *
   *   url        — destination, e.g. '/api/asr/upload'
   *   wavBytes   — Uint8Array of the WAV blob to upload (`audio` field)
   *   formFields — extra string/string entries to add to the form
   *   options    — optional { signal, fileName, mimeType, onError }
   *
   * Throws an Error whose `.status` mirrors the HTTP status on non-2xx
   * responses. The thrown error includes the server-supplied `detail` (or
   * the raw response text) as its message.
   */
  async function postWavToEndpoint(url, wavBytes, formFields, options) {
    const opts = options || {};
    const blob = new Blob([wavBytes], { type: opts.mimeType || 'audio/wav' });
    const form = new FormData();
    form.append('audio', blob, opts.fileName || 'upload.wav');
    if (formFields) {
      for (const [k, v] of Object.entries(formFields)) {
        if (v === undefined || v === null) continue;
        form.append(k, String(v));
      }
    }

    let resp;
    try {
      resp = await fetch(url, {
        method: 'POST',
        body: form,
        signal: opts.signal,
      });
    } catch (err) {
      // Network / abort. Re-throw with a tagged message so callers can pick
      // an appropriate i18n key based on `.name === 'AbortError'`.
      throw err;
    }

    let payload = null;
    let raw = null;
    const ctype = resp.headers.get('content-type') || '';
    if (ctype.includes('application/json')) {
      try { payload = await resp.json(); } catch (_) { payload = null; }
    } else {
      try { raw = await resp.text(); } catch (_) { raw = null; }
    }

    if (!resp.ok) {
      const detail =
        (payload && (payload.detail || payload.message))
        || raw
        || `HTTP ${resp.status}`;
      const err = new Error(
        typeof detail === 'string' ? detail : JSON.stringify(detail)
      );
      err.status = resp.status;
      err.payload = payload;
      throw err;
    }
    return payload;
  }

  /** Convenience: run decodeFileToMono → encodeWavBytes in one step. */
  async function decodeFileToWavBytes(file, targetSr) {
    const pcm = await decodeFileToMono(file, targetSr);
    if (!pcm || pcm.length === 0) return { pcm: new Float32Array(0), wav: null };
    const wav = encodeWavBytes(pcm, targetSr);
    return { pcm, wav };
  }

  /**
   * Format a sample count as a "Xs" / "X.Ys" duration string for status UIs.
   */
  function formatSamples(samples, sampleRate) {
    const sec = samples / sampleRate;
    if (sec >= 10) return sec.toFixed(0) + 's';
    return sec.toFixed(1) + 's';
  }

  window.AmphionAudioUpload = {
    decodeFileToMono,
    decodeFileToWavBytes,
    encodeWavBytes,
    bytesToBase64,
    postWavToEndpoint,
    formatSamples,
  };
})();
