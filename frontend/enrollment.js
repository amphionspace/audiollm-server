/**
 * Target-speaker enrollment controller for the realtime ASR page.
 *
 * Responsibilities
 * ----------------
 * - Capture an enrollment clip (5–10 s) either by file upload or by
 *   microphone recording, decode/resample to 16 kHz mono WAV in the
 *   browser, and ``POST /api/asr/enrollment`` to obtain an opaque
 *   ``enrollment_id``. The caller (app.js) then forwards the id on
 *   every WS session-control message and REST upload so the primary
 *   model prepends the clip into the TS-ASR dual-audio prompt.
 * - Gate the buttons so the user can only set/clear enrollment when
 *   the microphone is idle (``must_before_start`` decision) and reset
 *   the UI back to "no enrollment" once cleared.
 * - Provide a tiny pub/sub interface so app.js doesn't have to know
 *   anything about MediaRecorder / OfflineAudioContext.
 *
 * Why a separate module
 * ---------------------
 * The realtime ASR page already has ~1.3k LOC of state; the
 * enrollment flow is an independent capability (REST + MediaRecorder
 * + UI) that has zero overlap with the WS streaming pipeline. Keeping
 * it in its own file means app.js only needs to know two things:
 *   1. ``Amphion.Enrollment.attach(...)`` to wire up the DOM, and
 *   2. ``ctrl.getEnrollmentId()`` to read the current id when
 *      composing WS session-control messages.
 */
(() => {
  'use strict';

  const TARGET_SR = 16000;
  // Mirror the backend's 5–10 s window so users see
  // a friendly error before the request even leaves the browser.
  const MIN_DURATION_SEC = 5.0;
  const MAX_DURATION_SEC = 10.0;
  // Let users stop manually once they have recorded enough speech, and
  // hard-stop at the upper bound so every automatic recording is valid.
  const DEFAULT_RECORD_SEC = MAX_DURATION_SEC;

  /**
   * Pick a MediaRecorder mime type the browser can actually encode. The
   * media is fed through decodeAudioData afterwards, so any container
   * the browser can both record and decode works; we just need it to
   * succeed on the current device.
   */
  function pickRecorderMime() {
    if (typeof MediaRecorder === 'undefined') return '';
    const candidates = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/mp4',
    ];
    for (const m of candidates) {
      try {
        if (MediaRecorder.isTypeSupported(m)) return m;
      } catch (_) { /* ignore */ }
    }
    return '';
  }

  /**
   * Decode a Blob (recorded clip) into a 16 kHz mono Float32Array.
   * Uses the shared decode helper so the WAV bytes are byte-compatible
   * with what the upload-file path produces.
   */
  async function blobToWavBytes(blob) {
    const file = new File([blob], 'enrollment.webm', { type: blob.type });
    const upload = window.AmphionAudioUpload;
    if (!upload) throw new Error('AmphionAudioUpload helper not available');
    const decoded = await upload.decodeFileToWavBytes(file, TARGET_SR);
    if (!decoded || !decoded.wav || decoded.pcm.length === 0) {
      throw new Error('decoded clip is empty');
    }
    return decoded;
  }

  /**
   * Attach the enrollment UI controller to a set of DOM elements.
   *
   * The host page provides an ``opts`` bag with:
   *   - elements: { card, uploadBtn, fileInput, recordBtn, clearBtn,
   *                 statusPill, hint }
   *   - isMicRecording: () => boolean — used to gate inputs
   *   - t: (key, vars?) => string — i18n helper (or a passthrough)
   *   - onChange: (state) => void — invoked when the id changes
   *       state = { id: string|null, durationSec: number|null }
   *
   * Returns a controller object the host can use to peek the current
   * enrollment id and to dispose all listeners on navigation.
   */
  function attach(opts) {
    const els = opts.elements;
    const isMicRecording = opts.isMicRecording || (() => false);
    const t = opts.t || ((k) => k);
    const onChange = opts.onChange || (() => {});

    let enrollmentId = null;
    let durationSec = null;
    let inFlight = false;
    let mediaRecorder = null;
    let recordStream = null;
    let recordTimer = null;
    // Keep the most recent WAV's blob URL around so the user can replay
    // their enrollment clip and verify it actually captured speech (not
    // silence due to a wrong mic device / muted headset / etc.). This
    // is the missing feedback channel that made earlier "I recorded
    // but nothing came through" failures invisible.
    let lastWavBlobUrl = null;
    let lastPlaybackAudio = null;

    function setStatus(stateKey, vars) {
      if (!els.statusPill) return;
      const state =
        stateKey === 'asr.enroll.status.idle' ? 'waiting' :
        stateKey === 'asr.enroll.status.ready' ? 'ready' :
        stateKey === 'asr.enroll.status.error' ? 'offline' :
        stateKey === 'asr.enroll.status.recording' ? 'ready' :
        stateKey === 'asr.enroll.status.uploading' ? 'waiting' : 'waiting';
      els.statusPill.dataset.state = state;
      els.statusPill.textContent = t(stateKey, vars || undefined);
      els.statusPill.setAttribute('data-dyn-key', stateKey);
      if (vars) {
        els.statusPill.setAttribute('data-dyn-vars', JSON.stringify(vars));
      } else {
        els.statusPill.removeAttribute('data-dyn-vars');
      }
    }

    function setHint(stateKey, vars) {
      if (!els.hint) return;
      if (!stateKey) {
        els.hint.hidden = true;
        els.hint.textContent = '';
        els.hint.removeAttribute('data-dyn-key');
        els.hint.removeAttribute('data-dyn-vars');
        els.hint.className = 'text-[11px] text-faint mt-2';
        return;
      }
      els.hint.hidden = false;
      els.hint.textContent = t(stateKey, vars || undefined);
      els.hint.setAttribute('data-dyn-key', stateKey);
      if (vars) {
        els.hint.setAttribute('data-dyn-vars', JSON.stringify(vars));
      } else {
        els.hint.removeAttribute('data-dyn-vars');
      }
      els.hint.className = 'text-[11px] mt-2';
      els.hint.style.color = (
        stateKey === 'asr.enroll.error.tooShort' ||
        stateKey === 'asr.enroll.error.tooLong' ||
        stateKey === 'asr.enroll.error.decode' ||
        stateKey === 'asr.enroll.error.upload' ||
        stateKey === 'asr.enroll.error.micDenied' ||
        stateKey === 'asr.enroll.error.unsupported' ||
        stateKey === 'asr.enroll.error.busyRecording'
      ) ? 'var(--danger)' : 'var(--ink-mute)';
    }

    function refreshButtons() {
      const micBusy = isMicRecording();
      const lockAll = inFlight || micBusy;
      if (els.uploadBtn) els.uploadBtn.disabled = lockAll;
      if (els.recordBtn) els.recordBtn.disabled = lockAll;
      if (els.clearBtn) {
        // Clearing is allowed even when uploading nothing — it's a
        // local-only mutation. Hide it when there is no enrollment to
        // keep the row tidy.
        els.clearBtn.hidden = !enrollmentId;
        els.clearBtn.disabled = lockAll && !enrollmentId;
      }
      if (els.playBtn) {
        // Replay only makes sense when we still have the source bytes
        // in browser memory; refreshes / navigations drop the blob URL.
        els.playBtn.hidden = !lastWavBlobUrl;
        els.playBtn.disabled = lockAll;
      }
      // The mic-record button changes label depending on state, but
      // its enabled bit is the gating one above.
    }

    function revokePlaybackUrl() {
      if (lastPlaybackAudio) {
        try { lastPlaybackAudio.pause(); } catch (_) { /* ignore */ }
        lastPlaybackAudio = null;
      }
      if (lastWavBlobUrl) {
        try { URL.revokeObjectURL(lastWavBlobUrl); } catch (_) { /* ignore */ }
        lastWavBlobUrl = null;
      }
    }

    function stashPlaybackWav(wavBytes) {
      revokePlaybackUrl();
      const blob = new Blob([wavBytes], { type: 'audio/wav' });
      lastWavBlobUrl = URL.createObjectURL(blob);
    }

    function setEnrollment(id, dur) {
      enrollmentId = id || null;
      durationSec = typeof dur === 'number' ? dur : null;
      if (enrollmentId) {
        setStatus('asr.enroll.status.ready', { sec: durationSec.toFixed(1) });
        setHint(null);
      } else {
        setStatus('asr.enroll.status.idle');
        revokePlaybackUrl();
      }
      refreshButtons();
      try { onChange({ id: enrollmentId, durationSec }); } catch (_) { /* ignore */ }
    }

    async function uploadWavBytes(wavBytes, pcm) {
      const upload = window.AmphionAudioUpload;
      if (!upload) throw new Error('upload helper unavailable');
      // Audible-energy guard: if the PCM the browser handed back is all
      // zeros (or essentially zero), don't even bother round-tripping
      // to the server — the model would just return silence and the
      // user would be left wondering why TS-ASR is hallucinating an
      // empty transcript. We surface a targeted hint instead.
      if (pcm && pcm.length) {
        let peak = 0.0;
        // Sample every 32-th element — peak detection doesn't need
        // every value and a tight loop over 50k+ Float32s is wasteful.
        for (let i = 0; i < pcm.length; i += 32) {
          const a = Math.abs(pcm[i]);
          if (a > peak) peak = a;
        }
        if (peak < 0.005) {
          setStatus('asr.enroll.status.error');
          setHint('asr.enroll.error.silent');
          return;
        }
      }
      setStatus('asr.enroll.status.uploading');
      inFlight = true;
      refreshButtons();
      try {
        const result = await upload.postWavToEndpoint(
          '/api/asr/enrollment',
          wavBytes,
          {},
          { fileName: 'enrollment.wav' },
        );
        const id = result && result.enrollment_id;
        const dur = (result && typeof result.duration_sec === 'number')
          ? result.duration_sec
          : null;
        if (!id) throw new Error('server returned no enrollment_id');
        // Keep the WAV around for replay BEFORE flipping the visible
        // state so the play button shows up in the same paint as the
        // "Enrolled (Xs)" pill.
        stashPlaybackWav(wavBytes);
        setEnrollment(id, dur);
      } catch (err) {
        // The backend serialises validation errors as
        // {detail: {code, message}}. The audio-upload helper has
        // already stringified ``detail`` — match on known codes so we
        // can surface targeted i18n messages rather than the raw text.
        const raw = String(err && err.message || '');
        let key = 'asr.enroll.error.upload';
        if (/too_short/i.test(raw)) key = 'asr.enroll.error.tooShort';
        else if (/too_long/i.test(raw)) key = 'asr.enroll.error.tooLong';
        else if (/decode/i.test(raw)) key = 'asr.enroll.error.decode';
        setStatus('asr.enroll.status.error');
        setHint(key, { raw });
        enrollmentId = null;
        durationSec = null;
        try { onChange({ id: null, durationSec: null }); } catch (_) { /* ignore */ }
      } finally {
        inFlight = false;
        refreshButtons();
      }
    }

    async function handleFile(file) {
      if (isMicRecording()) {
        setHint('asr.enroll.error.busyRecording');
        return;
      }
      let decoded;
      try {
        const upload = window.AmphionAudioUpload;
        if (!upload) throw new Error('upload helper unavailable');
        decoded = await upload.decodeFileToWavBytes(file, TARGET_SR);
      } catch (err) {
        setStatus('asr.enroll.status.error');
        setHint('asr.enroll.error.decode', { raw: String(err && err.message || '') });
        return;
      }
      if (!decoded || !decoded.wav || decoded.pcm.length === 0) {
        setStatus('asr.enroll.status.error');
        setHint('asr.enroll.error.decode');
        return;
      }
      const sec = decoded.pcm.length / TARGET_SR;
      if (sec < MIN_DURATION_SEC) {
        setStatus('asr.enroll.status.error');
        setHint('asr.enroll.error.tooShort', { sec: sec.toFixed(2), min: MIN_DURATION_SEC });
        return;
      }
      // No early reject on overflow — the backend tail-trims to 10 s
      // and reports the canonical duration in its response. Surface
      // the trim via a hint after upload succeeds.
      await uploadWavBytes(decoded.wav, decoded.pcm);
    }

    async function startRecording() {
      if (mediaRecorder) return;
      if (isMicRecording()) {
        setHint('asr.enroll.error.busyRecording');
        return;
      }
      const mime = pickRecorderMime();
      if (!mime || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setStatus('asr.enroll.status.error');
        setHint('asr.enroll.error.unsupported');
        return;
      }

      try {
        recordStream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
        });
      } catch (err) {
        setStatus('asr.enroll.status.error');
        setHint('asr.enroll.error.micDenied');
        return;
      }

      const chunks = [];
      try {
        mediaRecorder = new MediaRecorder(recordStream, { mimeType: mime });
      } catch (err) {
        try { recordStream.getTracks().forEach((tr) => tr.stop()); } catch (_) { /* ignore */ }
        recordStream = null;
        setStatus('asr.enroll.status.error');
        setHint('asr.enroll.error.unsupported');
        return;
      }

      mediaRecorder.addEventListener('dataavailable', (ev) => {
        if (ev.data && ev.data.size > 0) chunks.push(ev.data);
      });

      mediaRecorder.addEventListener('stop', async () => {
        try { recordStream.getTracks().forEach((tr) => tr.stop()); } catch (_) { /* ignore */ }
        recordStream = null;
        const mr = mediaRecorder;
        mediaRecorder = null;
        if (recordTimer) {
          clearTimeout(recordTimer);
          recordTimer = null;
        }
        if (els.recordBtn) {
          els.recordBtn.setAttribute('data-dyn-key', 'asr.enroll.record');
          els.recordBtn.textContent = t('asr.enroll.record');
        }
        if (!chunks.length) {
          setStatus('asr.enroll.status.error');
          setHint('asr.enroll.error.decode');
          refreshButtons();
          return;
        }
        const blob = new Blob(chunks, { type: mr.mimeType || mime });
        try {
          const decoded = await blobToWavBytes(blob);
          const sec = decoded.pcm.length / TARGET_SR;
          if (sec < MIN_DURATION_SEC) {
            setStatus('asr.enroll.status.error');
            setHint('asr.enroll.error.tooShort', { sec: sec.toFixed(2), min: MIN_DURATION_SEC });
            refreshButtons();
            return;
          }
          await uploadWavBytes(decoded.wav, decoded.pcm);
        } catch (err) {
          setStatus('asr.enroll.status.error');
          setHint('asr.enroll.error.decode', { raw: String(err && err.message || '') });
          refreshButtons();
        }
      });

      setStatus('asr.enroll.status.recording');
      setHint(null);
      if (els.recordBtn) {
        els.recordBtn.setAttribute('data-dyn-key', 'asr.enroll.recordStop');
        els.recordBtn.textContent = t('asr.enroll.recordStop');
      }
      // ``start(timeslice)`` makes the browser flush ``dataavailable``
      // events periodically. On Chrome, short single-shot webm clips
      // produced via ``start()`` (no arg) occasionally lack a Cluster
      // timecode duration header, which then makes the downstream
      // ``decodeAudioData`` return a 0-length AudioBuffer — i.e. the
      // user sees their recording silently vanish. Forcing periodic
      // flushes side-steps that container bug.
      mediaRecorder.start(250);

      // Hard stop after DEFAULT_RECORD_SEC so the user doesn't accidentally
      // overflow the 10 s cap. The user can also stop early via the
      // toggle button (handled in the click handler below).
      recordTimer = setTimeout(() => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
          mediaRecorder.stop();
        }
      }, Math.round(DEFAULT_RECORD_SEC * 1000));
    }

    function stopRecording() {
      if (recordTimer) {
        clearTimeout(recordTimer);
        recordTimer = null;
      }
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
      }
    }

    async function clearEnrollment() {
      const prev = enrollmentId;
      setEnrollment(null, null);
      setHint(null);
      if (!prev) return;
      try {
        await fetch(`/api/asr/enrollment/${encodeURIComponent(prev)}`, {
          method: 'DELETE',
        });
      } catch (_) {
        // Best-effort: the server TTLs the entry anyway, and the
        // user-visible state is already "no enrollment".
      }
    }

    function playEnrollment() {
      if (!lastWavBlobUrl) return;
      if (lastPlaybackAudio && !lastPlaybackAudio.paused) {
        try { lastPlaybackAudio.pause(); } catch (_) { /* ignore */ }
        lastPlaybackAudio.currentTime = 0;
        if (els.playBtn) els.playBtn.classList.remove('is-playing');
        return;
      }
      try {
        const audio = new Audio(lastWavBlobUrl);
        lastPlaybackAudio = audio;
        if (els.playBtn) els.playBtn.classList.add('is-playing');
        audio.addEventListener('ended', () => {
          if (els.playBtn) els.playBtn.classList.remove('is-playing');
          if (lastPlaybackAudio === audio) lastPlaybackAudio = null;
        });
        audio.addEventListener('error', () => {
          if (els.playBtn) els.playBtn.classList.remove('is-playing');
        });
        audio.play().catch(() => {
          if (els.playBtn) els.playBtn.classList.remove('is-playing');
        });
      } catch (_) {
        if (els.playBtn) els.playBtn.classList.remove('is-playing');
      }
    }

    // ---- DOM wiring -----------------------------------------------------

    function onUploadClick() {
      if (els.uploadBtn.disabled) return;
      if (!els.fileInput) return;
      els.fileInput.value = '';
      els.fileInput.click();
    }

    function onFileChange() {
      const file = els.fileInput.files && els.fileInput.files[0];
      if (file) handleFile(file);
    }

    function onRecordClick() {
      if (els.recordBtn.disabled) return;
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        stopRecording();
      } else {
        startRecording();
      }
    }

    function onClearClick() {
      if (els.clearBtn && els.clearBtn.disabled) return;
      clearEnrollment();
    }

    function onPlayClick() {
      if (els.playBtn && els.playBtn.disabled) return;
      playEnrollment();
    }

    if (els.uploadBtn) els.uploadBtn.addEventListener('click', onUploadClick);
    if (els.fileInput) els.fileInput.addEventListener('change', onFileChange);
    if (els.recordBtn) els.recordBtn.addEventListener('click', onRecordClick);
    if (els.clearBtn) els.clearBtn.addEventListener('click', onClearClick);
    if (els.playBtn) els.playBtn.addEventListener('click', onPlayClick);

    setEnrollment(null, null);

    return {
      getEnrollmentId() { return enrollmentId; },
      getDurationSec() { return durationSec; },
      isBusy() { return inFlight || !!mediaRecorder; },
      refresh: refreshButtons,
      refreshLabels() {
        // Called from the host's i18n change subscriber so the dynamic
        // strings (status pill, hint, record button) re-render.
        if (els.statusPill && els.statusPill.getAttribute('data-dyn-key')) {
          const key = els.statusPill.getAttribute('data-dyn-key');
          let vars;
          try { vars = JSON.parse(els.statusPill.getAttribute('data-dyn-vars') || 'null'); } catch (_) { vars = null; }
          els.statusPill.textContent = t(key, vars || undefined);
        }
        if (els.hint && els.hint.getAttribute('data-dyn-key')) {
          const key = els.hint.getAttribute('data-dyn-key');
          let vars;
          try { vars = JSON.parse(els.hint.getAttribute('data-dyn-vars') || 'null'); } catch (_) { vars = null; }
          els.hint.textContent = t(key, vars || undefined);
        }
        if (els.recordBtn) {
          const key = els.recordBtn.getAttribute('data-dyn-key') || 'asr.enroll.record';
          els.recordBtn.textContent = t(key);
        }
      },
      dispose() {
        stopRecording();
        try { if (recordStream) recordStream.getTracks().forEach((tr) => tr.stop()); } catch (_) { /* ignore */ }
        if (recordTimer) {
          clearTimeout(recordTimer);
          recordTimer = null;
        }
        revokePlaybackUrl();
        if (els.uploadBtn) els.uploadBtn.removeEventListener('click', onUploadClick);
        if (els.fileInput) els.fileInput.removeEventListener('change', onFileChange);
        if (els.recordBtn) els.recordBtn.removeEventListener('click', onRecordClick);
        if (els.clearBtn) els.clearBtn.removeEventListener('click', onClearClick);
        if (els.playBtn) els.playBtn.removeEventListener('click', onPlayClick);
      },
    };
  }

  window.Amphion = window.Amphion || {};
  window.Amphion.Enrollment = {
    attach,
    constants: {
      MIN_DURATION_SEC,
      MAX_DURATION_SEC,
      DEFAULT_RECORD_SEC,
      TARGET_SR,
    },
  };
})();
