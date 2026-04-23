/**
 * TS-ASR demo page.
 *
 * Flow:
 *   1. User records a short enrollment clip in-browser (WebAudio, 16 kHz mono).
 *   2. Clip is wrapped in a WAV container, base64-encoded, and stashed locally.
 *   3. On mic click the page opens a fresh WS to /transcribe-target-streaming
 *      and sends:
 *          { type: "start", enrollment_audio: "<b64>",
 *            enrollment_format: "wav", voice_traits?, language?, config? }
 *      If the backend replies with ``enrollment_ok`` the mic starts streaming;
 *      if it replies with an ``error`` the session is torn down and the user
 *      is prompted to re-record.
 *   4. Live PCM is sent as binary frames (Int16 mono @16 kHz) on the same WS.
 *      Final transcripts arrive as ``{type:"final", text, task:"tsasr"}``.
 */

(() => {
  'use strict';

  const i18n = window.Amphion && window.Amphion.i18n;
  const t = (key, vars) => (i18n ? i18n.t(key, vars) : (vars && vars.defaultValue) || key);
  const onLangChange = (fn) => (i18n ? i18n.onChange(fn) : () => {});

  const MIN_ENROLL_SEC = 1.0;
  // Backend VAD-trims longer uploads to 5s (see tsasr_enrollment_max_sec).
  // Keeping the browser auto-stop in sync avoids uploading material we know
  // will be discarded and lets the progress bar fill cleanly at 5s.
  const MAX_ENROLL_SEC = 5.0;
  const TARGET_SAMPLE_RATE = 16000;

  // -------------------- State --------------------
  let ws = null;
  let liveCtx = null;
  let liveNode = null;
  let liveStream = null;
  let isRecordingLive = false;

  let enrollCtx = null;
  let enrollNode = null;
  let enrollStream = null;
  let enrollChunks = []; // Float32Array pieces at 16 kHz
  let enrollStartAt = 0;
  let enrollTimerId = null;
  let enrollPcm = null; // concatenated Float32Array
  let enrollWavB64 = null;
  let enrollDurationSec = 0;
  let isEnrollRecording = false;
  let enrollPreviewUrl = null;

  // Track current displayed enrollment status so we can re-render on lang switch.
  let enrollStatusDyn = { state: 'idle', key: 'tsasr.enroll.notRecorded', vars: null };

  // -------------------- Segment replay cache --------------------
  // Keyed by the backend-assigned `final.id`. Each value is a blob URL that
  // can be handed to an <audio> element. We eagerly create the blob on
  // message arrival so the replay button responds instantly, then revoke on
  // reset or page unload to avoid leaking audio blobs.
  const segmentAudio = new Map();
  let activeReplayAudio = null;

  function b64ToWavBlobUrl(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }));
  }

  function clearSegmentAudio() {
    if (activeReplayAudio) {
      try { activeReplayAudio.pause(); } catch { /* noop */ }
      activeReplayAudio = null;
    }
    segmentAudio.forEach((url) => {
      try { URL.revokeObjectURL(url); } catch { /* noop */ }
    });
    segmentAudio.clear();
    document.querySelectorAll('.replay-btn.is-playing').forEach((b) => {
      b.classList.remove('is-playing');
    });
  }

  window.addEventListener('beforeunload', clearSegmentAudio);

  // -------------------- DOM refs --------------------
  const micBtn = document.getElementById('mic-btn');
  const micIcon = document.getElementById('mic-icon');
  const micStatus = document.getElementById('mic-status');
  const pulseRings = document.querySelectorAll('.pulse-ring');
  const chatArea = document.getElementById('chat-area');

  const enrollStatusPill = document.getElementById('enroll-status');
  const enrollRecBtn = document.getElementById('enroll-rec-btn');
  const enrollRecLabel = document.getElementById('enroll-rec-label');
  const enrollResetBtn = document.getElementById('enroll-reset-btn');
  const enrollTimer = document.getElementById('enroll-timer');
  const enrollProgressBar = document.getElementById('enroll-progress-bar');
  const enrollProgress = document.getElementById('enroll-progress');
  const enrollPreviewEl = document.getElementById('enroll-preview');

  const uploadBtn = document.getElementById('upload-btn');
  const uploadBtnLabel = uploadBtn ? uploadBtn.querySelector('.btn-upload-label') : null;
  const uploadInput = document.getElementById('upload-input');
  const uploadStatus = document.getElementById('upload-status');

  const enrollUploadBtn = document.getElementById('enroll-upload-btn');
  const enrollUploadBtnLabel = enrollUploadBtn
    ? enrollUploadBtn.querySelector('.btn-upload-label')
    : null;
  const enrollUploadInput = document.getElementById('enroll-upload-input');
  const enrollUploadStatus = document.getElementById('enroll-upload-status');
  let isEnrollUploading = false;
  let currentEnrollUploadDyn = null; // { key, vars }

  // Upload state for the transcription stage. The transcription upload now
  // hits POST /api/tsasr/upload as a one-shot REST call (mixed audio in
  // multipart, enrollment WAV inlined as base64) and never opens a WS.
  let isUploading = false;
  let uploadController = null;     // AbortController for in-flight fetch
  let currentUploadDyn = null;     // { key, vars }
  const TSASR_UPLOAD_MAX_SECONDS = 60;

  function langDisplayName(value) {
    if (!value) return '';
    const v = String(value).trim();
    if (!v) return '';
    return t(`lang.name.${v}`, { defaultValue: v });
  }

  // -------------------- Connection status --------------------
  function setConnStatus(state) {
    if (!window.AmphionSidebar || !window.AmphionSidebar.setConnectionState) return;
    if (state === 'connected') {
      window.AmphionSidebar.setConnectionState('connected');
    } else if (state === 'pending') {
      window.AmphionSidebar.setConnectionState('pending');
    } else {
      window.AmphionSidebar.setConnectionState('idle');
    }
  }
  setConnStatus('disconnected');

  // -------------------- Enrollment status pill --------------------
  function setEnrollStatus(state, key, vars) {
    enrollStatusPill.className = 'status-pill';
    if (state === 'recording') {
      enrollStatusPill.dataset.state = 'recording';
    } else if (state === 'ready') {
      enrollStatusPill.dataset.state = 'ready';
    } else if (state === 'error') {
      enrollStatusPill.dataset.state = 'error';
    } else if (state === 'pending') {
      enrollStatusPill.dataset.state = 'pending';
    } else {
      enrollStatusPill.dataset.state = 'idle';
    }
    enrollStatusDyn = { state, key, vars: vars || null };
    enrollStatusPill.textContent = t(key, vars || undefined);
  }

  function updateMicGate(messageKey) {
    const enabled =
      enrollWavB64 !== null
      && !isEnrollRecording
      && !isUploading
      && !isEnrollUploading;
    micBtn.disabled = !enabled;
    if (uploadBtn) {
      uploadBtn.disabled =
        enrollWavB64 === null
        || isEnrollRecording
        || isRecordingLive
        || isUploading
        || isEnrollUploading;
    }
    if (enrollUploadBtn) {
      enrollUploadBtn.disabled =
        isEnrollRecording || isRecordingLive || isEnrollUploading;
    }
    if (isRecordingLive) {
      micStatus.textContent = t('tsasr.mic.listening');
      micStatus.setAttribute('data-dyn-key', 'tsasr.mic.listening');
    } else if (!enabled) {
      const k = messageKey || 'tsasr.mic.gateDisabled';
      micStatus.textContent = t(k);
      micStatus.setAttribute('data-dyn-key', k);
    } else {
      micStatus.textContent = t('tsasr.mic.start');
      micStatus.setAttribute('data-dyn-key', 'tsasr.mic.start');
    }
  }

  // -------------------- WAV encoder (Float32 mono @16k -> WAV bytes) --------
  function floatToPcm16(samples) {
    const out = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return out;
  }

  function encodeWav(floatSamples, sampleRate = TARGET_SAMPLE_RATE) {
    const pcm16 = floatToPcm16(floatSamples);
    const byteLength = pcm16.length * 2;
    const buffer = new ArrayBuffer(44 + byteLength);
    const view = new DataView(buffer);
    const writeStr = (off, str) => {
      for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
    };
    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + byteLength, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // byte rate
    view.setUint16(32, 2, true); // block align
    view.setUint16(34, 16, true); // bits per sample
    writeStr(36, 'data');
    view.setUint32(40, byteLength, true);
    new Int16Array(buffer, 44).set(pcm16);
    return new Uint8Array(buffer);
  }

  function bytesToBase64(bytes) {
    // Chunked to avoid call-stack limits on large buffers.
    const CHUNK = 0x8000;
    let binary = '';
    for (let i = 0; i < bytes.length; i += CHUNK) {
      binary += String.fromCharCode.apply(
        null, bytes.subarray(i, i + CHUNK)
      );
    }
    return btoa(binary);
  }

  // -------------------- Audio context setup --------------------
  async function openSixteenKContext() {
    // Some browsers can't honor 16 kHz (e.g. Safari), but Chrome/Firefox do.
    // If the request isn't honored we still proceed, and the session will
    // send at whatever the browser returns — the backend already expects
    // 16 kHz so we'll warn loudly in that case.
    const mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: { ideal: TARGET_SAMPLE_RATE },
        echoCancellation: true,
        noiseSuppression: true,
      },
    });
    const ctx = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
    if (ctx.sampleRate !== TARGET_SAMPLE_RATE) {
      console.warn(
        `AudioContext honored ${ctx.sampleRate} Hz instead of ${TARGET_SAMPLE_RATE} Hz; ` +
        'audio will be uploaded at that rate.'
      );
    }
    await ctx.audioWorklet.addModule('tsasr-processor.js?v=' + Date.now());
    const source = ctx.createMediaStreamSource(mediaStream);
    const node = new AudioWorkletNode(ctx, 'tsasr-capture-processor');
    source.connect(node);
    node.connect(ctx.destination);
    return { ctx, node, mediaStream };
  }

  // -------------------- Enrollment recording --------------------
  async function startEnrollRecording() {
    try {
      const { ctx, node, mediaStream } = await openSixteenKContext();
      enrollCtx = ctx;
      enrollNode = node;
      enrollStream = mediaStream;
    } catch (err) {
      console.error(err);
      setEnrollStatus('error', 'tsasr.enroll.micDenied');
      alert(t('tsasr.enroll.micAlert'));
      return;
    }

    enrollChunks = [];
    enrollNode.port.onmessage = (evt) => {
      if (evt.data.type === 'audio') {
        enrollChunks.push(evt.data.samples);
      }
    };
    isEnrollRecording = true;
    enrollStartAt = performance.now();
    enrollRecLabel.textContent = t('tsasr.enroll.stop');
    enrollRecLabel.setAttribute('data-i18n', 'tsasr.enroll.stop');
    enrollRecBtn.classList.add('enroll-recording');
    setEnrollStatus('recording', 'tsasr.enroll.recording');
    enrollResetBtn.disabled = true;
    enrollPreviewEl.classList.add('hidden');

    enrollTimerId = setInterval(tickEnrollTimer, 80);
  }

  function tickEnrollTimer() {
    const dt = (performance.now() - enrollStartAt) / 1000;
    enrollTimer.textContent = `${dt.toFixed(1)}s`;
    const pct = Math.min(100, (dt / MAX_ENROLL_SEC) * 100);
    enrollProgressBar.style.width = `${pct}%`;
    if (dt >= MAX_ENROLL_SEC) {
      stopEnrollRecording();
    }
  }

  async function stopEnrollRecording() {
    if (!isEnrollRecording) return;
    isEnrollRecording = false;
    clearInterval(enrollTimerId);
    enrollTimerId = null;
    enrollRecBtn.classList.remove('enroll-recording');
    enrollRecLabel.textContent = t('tsasr.enroll.start');
    enrollRecLabel.setAttribute('data-i18n', 'tsasr.enroll.start');

    if (enrollNode) {
      enrollNode.port.onmessage = null;
      enrollNode.disconnect();
      enrollNode = null;
    }
    if (enrollCtx) {
      await enrollCtx.close();
      enrollCtx = null;
    }
    if (enrollStream) {
      enrollStream.getTracks().forEach((tr) => tr.stop());
      enrollStream = null;
    }

    const total = enrollChunks.reduce((n, b) => n + b.length, 0);
    const sr = TARGET_SAMPLE_RATE;
    const duration = total / sr;
    enrollDurationSec = duration;
    enrollTimer.textContent = `${duration.toFixed(1)}s`;

    if (duration < MIN_ENROLL_SEC) {
      setEnrollStatus('error', 'tsasr.enroll.tooShort', { dur: duration.toFixed(1) });
      enrollChunks = [];
      enrollPcm = null;
      enrollWavB64 = null;
      enrollResetBtn.disabled = true;
      updateMicGate();
      return;
    }

    const merged = new Float32Array(total);
    let offset = 0;
    for (const chunk of enrollChunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    enrollPcm = merged;
    const wavBytes = encodeWav(merged, sr);
    enrollWavB64 = bytesToBase64(wavBytes);

    // Preview
    if (enrollPreviewUrl) URL.revokeObjectURL(enrollPreviewUrl);
    enrollPreviewUrl = URL.createObjectURL(
      new Blob([wavBytes], { type: 'audio/wav' })
    );
    enrollPreviewEl.src = enrollPreviewUrl;
    enrollPreviewEl.classList.remove('hidden');

    setEnrollStatus('ready', 'tsasr.enroll.ready', { dur: duration.toFixed(1) });
    enrollResetBtn.disabled = false;
    updateMicGate();
  }

  function resetEnrollment() {
    if (isEnrollRecording) stopEnrollRecording();
    enrollChunks = [];
    enrollPcm = null;
    enrollWavB64 = null;
    enrollDurationSec = 0;
    enrollProgressBar.style.width = '0%';
    enrollTimer.textContent = '0.0s';
    enrollPreviewEl.classList.add('hidden');
    if (enrollPreviewUrl) {
      URL.revokeObjectURL(enrollPreviewUrl);
      enrollPreviewUrl = null;
    }
    enrollResetBtn.disabled = true;
    setEnrollStatus('idle', 'tsasr.enroll.notRecorded');
    if (uploadController) {
      try { uploadController.abort(); } catch (_) { /* noop */ }
      uploadController = null;
    }
    setUploadStatus(null, null);
    setEnrollUploadStatus(null, null);
    updateMicGate();
  }

  enrollRecBtn.addEventListener('click', () => {
    if (isRecordingLive) return; // ignore while streaming
    if (isEnrollRecording) {
      stopEnrollRecording();
    } else {
      startEnrollRecording();
    }
  });
  enrollResetBtn.addEventListener('click', () => {
    if (isRecordingLive) return;
    resetEnrollment();
  });

  // -------------------- Enrollment via uploaded audio --------------------
  // Mirrors stopEnrollRecording's tail-end work (truncate, encode WAV,
  // populate preview + enrollWavB64) but the source PCM comes from a
  // user-picked file decoded by AmphionAudioUpload instead of the mic.

  function setEnrollUploadStatus(state, key, vars) {
    if (!enrollUploadStatus) return;
    if (!key) {
      enrollUploadStatus.hidden = true;
      enrollUploadStatus.textContent = '';
      enrollUploadStatus.removeAttribute('data-state');
      currentEnrollUploadDyn = null;
      return;
    }
    enrollUploadStatus.hidden = false;
    enrollUploadStatus.dataset.state = state || 'info';
    currentEnrollUploadDyn = { key, vars: vars || null };
    enrollUploadStatus.textContent = t(key, vars || undefined);
  }

  function setEnrollUploadBusy(busy) {
    isEnrollUploading = busy;
    if (enrollUploadBtn) {
      enrollUploadBtn.disabled = busy || isEnrollRecording || isRecordingLive;
    }
    if (enrollUploadBtnLabel) {
      enrollUploadBtnLabel.textContent = t(
        busy ? 'tsasr.enrollUpload.uploading' : 'tsasr.enrollUpload.label'
      );
    }
    // While uploading enrollment, also gate the mic / file-upload for the
    // transcription stage so the user can't fire two flows at once.
    enrollRecBtn.disabled = busy;
    enrollResetBtn.disabled = busy || enrollWavB64 === null;
    updateMicGate();
  }

  async function handleEnrollUploadFile(file) {
    if (!file) return;
    if (isEnrollRecording || isRecordingLive || isEnrollUploading || isUploading) {
      setEnrollUploadStatus('error', 'tsasr.enrollUpload.error.busy');
      return;
    }
    const upload = window.AmphionAudioUpload;
    if (!upload) {
      setEnrollUploadStatus('error', 'tsasr.enrollUpload.error.unsupported');
      return;
    }

    setEnrollUploadBusy(true);
    setEnrollUploadStatus('info', 'tsasr.enrollUpload.decoding');

    let pcm;
    try {
      pcm = await upload.decodeFileToMono(file, TARGET_SAMPLE_RATE);
    } catch (err) {
      console.error('Enrollment upload decode failed:', err);
      setEnrollUploadBusy(false);
      setEnrollUploadStatus('error', 'tsasr.enrollUpload.error.decode');
      return;
    }
    if (!pcm || pcm.length === 0) {
      setEnrollUploadBusy(false);
      setEnrollUploadStatus('error', 'tsasr.enrollUpload.error.empty');
      return;
    }

    const sr = TARGET_SAMPLE_RATE;
    const totalSec = pcm.length / sr;
    if (totalSec < MIN_ENROLL_SEC) {
      setEnrollUploadBusy(false);
      setEnrollUploadStatus('error', 'tsasr.enrollUpload.error.tooShort', {
        dur: totalSec.toFixed(1),
        min: MIN_ENROLL_SEC.toFixed(1),
      });
      return;
    }

    let trimmedNote = null;
    if (totalSec > MAX_ENROLL_SEC) {
      // Match the live recorder's auto-stop behavior: keep the leading
      // MAX_ENROLL_SEC seconds, drop the rest. The backend VAD-trims
      // anyway, but trimming up-front gives a tidy preview waveform.
      pcm = new Float32Array(pcm.subarray(0, Math.floor(sr * MAX_ENROLL_SEC)));
      trimmedNote = totalSec.toFixed(1);
    }
    const finalDuration = pcm.length / sr;

    // Update the recorder UI state so the existing "ready" affordances
    // (preview, reset button, status pill, gating) light up as if the
    // enrollment had been recorded live.
    enrollPcm = pcm;
    enrollDurationSec = finalDuration;
    enrollTimer.textContent = `${finalDuration.toFixed(1)}s`;
    enrollProgressBar.style.width = `${Math.min(100, (finalDuration / MAX_ENROLL_SEC) * 100)}%`;

    const wavBytes = encodeWav(pcm, sr);
    enrollWavB64 = bytesToBase64(wavBytes);

    if (enrollPreviewUrl) URL.revokeObjectURL(enrollPreviewUrl);
    enrollPreviewUrl = URL.createObjectURL(
      new Blob([wavBytes], { type: 'audio/wav' })
    );
    enrollPreviewEl.src = enrollPreviewUrl;
    enrollPreviewEl.classList.remove('hidden');

    setEnrollStatus('ready', 'tsasr.enroll.ready', { dur: finalDuration.toFixed(1) });
    enrollResetBtn.disabled = false;
    setEnrollUploadBusy(false);
    if (trimmedNote !== null) {
      setEnrollUploadStatus('warn', 'tsasr.enrollUpload.trimmed', {
        max: MAX_ENROLL_SEC.toFixed(1),
        actual: trimmedNote,
      });
    } else {
      setEnrollUploadStatus('success', 'tsasr.enrollUpload.done', {
        dur: finalDuration.toFixed(1),
      });
    }
    updateMicGate();
  }

  if (enrollUploadBtn && enrollUploadInput) {
    enrollUploadBtn.addEventListener('click', () => {
      if (enrollUploadBtn.disabled) return;
      enrollUploadInput.value = '';
      enrollUploadInput.click();
    });
    enrollUploadInput.addEventListener('change', () => {
      const file = enrollUploadInput.files && enrollUploadInput.files[0];
      if (file) handleEnrollUploadFile(file);
    });
  }

  // -------------------- Transcript UI --------------------
  function replaySegment(segId, btn) {
    // Toggle behavior: clicking the playing button (or any button while
    // something is playing) stops the current audio. Clicking a different
    // segment's button starts a fresh playback.
    if (activeReplayAudio) {
      activeReplayAudio.pause();
      const prevBtn = document.querySelector('.replay-btn.is-playing');
      if (prevBtn) prevBtn.classList.remove('is-playing');
      const wasSame = activeReplayAudio._segId === segId;
      activeReplayAudio = null;
      if (wasSame) return;
    }
    const url = segmentAudio.get(segId);
    if (!url) return;
    const audio = new Audio(url);
    audio._segId = segId;
    if (btn) btn.classList.add('is-playing');
    audio.addEventListener('ended', () => {
      if (btn) btn.classList.remove('is-playing');
      if (activeReplayAudio === audio) activeReplayAudio = null;
    });
    audio.play().catch(() => {
      if (btn) btn.classList.remove('is-playing');
    });
    activeReplayAudio = audio;
  }

  function addFinalBubble(text, langValue, durationSec, segId) {
    const wrapper = document.createElement('div');
    wrapper.className = 'chat-row chat-row-ai chat-bubble-float';
    if (segId) wrapper.id = `ai-${segId}`;

    const metaParts = [];
    if (langValue) {
      metaParts.push(
        `<span data-dyn-key="tsasr.meta.lang"
               data-dyn-vars='${escapeHtml(JSON.stringify({ lang: langValue }))}'>${escapeHtml(t('tsasr.meta.lang', { lang: langDisplayName(langValue) }))}</span>`
      );
    }
    if (typeof durationSec === 'number') {
      metaParts.push(`<span>${durationSec.toFixed(1)}s</span>`);
    }
    const metaLine = metaParts.length
      ? `<div class="text-[11px] text-faint mt-1">${metaParts.join(' \u00b7 ')}</div>`
      : '';
    const hasAudio = Boolean(segId) && segmentAudio.has(segId);
    // Keep the replay button inline with the paragraph so users scan text
    // first and see the play affordance as a subtle trailing glyph, not a
    // separate control.
    const replayTitle = escapeHtml(t('tsasr.replayTitle'));
    const replayBtn = hasAudio
      ? `<button class="replay-btn" type="button" title="${replayTitle}">
           <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
             <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z"/>
           </svg>
         </button>`
      : '';

    wrapper.innerHTML = `
      <div class="flex gap-3 max-w-2xl items-start">
        <div class="chat-avatar flex-shrink-0">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/>
          </svg>
        </div>
        <div class="chat-bubble chat-bubble-ai ai-content">
          <div class="flex items-start gap-2">
            <p class="text-sm leading-relaxed flex-1">${escapeHtml(text)}</p>
            ${replayBtn}
          </div>
          ${metaLine}
        </div>
      </div>
    `;

    if (hasAudio) {
      const btn = wrapper.querySelector('.replay-btn');
      if (btn) {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          replaySegment(segId, e.currentTarget);
        });
      }
    }

    chatArea.appendChild(wrapper);
    chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' });
  }

  function addErrorBubble(code, message) {
    const wrapper = document.createElement('div');
    wrapper.className = 'chat-row chat-row-ai chat-bubble-float';
    const vars = { code: code || 'error', msg: message || '' };
    wrapper.innerHTML = `
      <div class="flex gap-3 max-w-2xl items-start">
        <div class="chat-avatar flex-shrink-0"
             style="background:var(--danger-soft); border-color:rgba(180,80,74,0.4); color:var(--danger)">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"
               style="color:var(--danger)">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
          </svg>
        </div>
        <div class="chat-bubble chat-bubble-ai text-sm" style="color:var(--danger)"
             data-dyn-key="tsasr.error.serverPrefix"
             data-dyn-vars='${escapeHtml(JSON.stringify(vars))}'>
          ${escapeHtml(t('tsasr.error.serverPrefix', vars))}
        </div>
      </div>
    `;
    chatArea.appendChild(wrapper);
    chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' });
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = String(text == null ? '' : text);
    return div.innerHTML;
  }

  // -------------------- Upload (transcription stage only) --------------------

  function setUploadStatus(state, key, vars) {
    if (!uploadStatus) return;
    if (!key) {
      uploadStatus.hidden = true;
      uploadStatus.textContent = '';
      uploadStatus.removeAttribute('data-state');
      currentUploadDyn = null;
      return;
    }
    uploadStatus.hidden = false;
    uploadStatus.dataset.state = state || 'info';
    currentUploadDyn = { key, vars: vars || null };
    uploadStatus.textContent = t(key, vars || undefined);
  }

  function setUploadBusy(busy) {
    isUploading = busy;
    if (uploadBtnLabel) {
      uploadBtnLabel.textContent = t(busy ? 'tsasr.upload.uploading' : 'tsasr.upload.label');
    }
    updateMicGate();
  }

  async function handleUploadFile(file) {
    if (!file) return;
    if (!enrollWavB64) {
      setUploadStatus('error', 'tsasr.upload.error.noEnroll');
      return;
    }
    if (isUploading || isRecordingLive || isEnrollRecording || isEnrollUploading) {
      setUploadStatus('error', 'tsasr.upload.error.busy');
      return;
    }
    const upload = window.AmphionAudioUpload;
    if (!upload) {
      setUploadStatus('error', 'tsasr.upload.error.unsupported');
      return;
    }

    setUploadBusy(true);
    setUploadStatus('info', 'tsasr.upload.decoding');

    let decoded;
    try {
      decoded = await upload.decodeFileToWavBytes(file, TARGET_SAMPLE_RATE);
    } catch (err) {
      console.error('Upload decode failed:', err);
      setUploadBusy(false);
      setUploadStatus('error', 'tsasr.upload.error.decode');
      return;
    }
    if (!decoded || !decoded.wav || !decoded.pcm.length) {
      setUploadBusy(false);
      setUploadStatus('error', 'tsasr.upload.error.empty');
      return;
    }

    let pcm = decoded.pcm;
    let wavBytes = decoded.wav;
    const totalSec = pcm.length / TARGET_SAMPLE_RATE;
    let trimmedNote = null;
    if (totalSec > TSASR_UPLOAD_MAX_SECONDS) {
      pcm = new Float32Array(
        pcm.subarray(0, Math.floor(TSASR_UPLOAD_MAX_SECONDS * TARGET_SAMPLE_RATE))
      );
      wavBytes = upload.encodeWavBytes(pcm, TARGET_SAMPLE_RATE);
      trimmedNote = totalSec.toFixed(1);
    }

    setUploadStatus('info', 'tsasr.upload.analyzing');
    uploadController = new AbortController();
    let result;
    try {
      result = await upload.postWavToEndpoint(
        '/api/tsasr/upload',
        wavBytes,
        {
          enrollment_wav_base64: enrollWavB64,
          // Pass through the same hot-word/voice-trait knobs the live mic
          // session would, even though the server currently ignores them
          // unless ``tsasr_enable_hotwords`` is on. Voice traits are not
          // exposed in the demo UI yet, so they go in empty.
          hotwords: '',
          voice_traits: '',
        },
        { signal: uploadController.signal, fileName: file.name || 'upload.wav' }
      );
    } catch (err) {
      console.error('Upload request failed:', err);
      uploadController = null;
      setUploadBusy(false);
      const aborted = err && err.name === 'AbortError';
      const detail = err && err.message ? err.message : 'Upload failed';
      // Surface enrollment-validation errors with the same chat bubble the
      // WS path uses so users get a consistent failure mode regardless of
      // whether the bad enrollment came from the mic or a file.
      if (err && err.payload && typeof err.payload.detail === 'object') {
        const d = err.payload.detail;
        addErrorBubble(d.code || 'error', d.message || detail);
      } else if (!aborted) {
        addErrorBubble('upload_error', detail);
      }
      setUploadStatus(
        aborted ? 'info' : 'error',
        aborted ? 'tsasr.upload.aborted' : 'tsasr.upload.error.serverPrefix',
        aborted ? null : { msg: detail }
      );
      return;
    }
    uploadController = null;

    const text = (result && result.text) || '';
    if (text.trim()) {
      // Mirror the WS ``final`` payload's replay-button wiring.
      if (result.audio_b64) {
        const synthId = `upload-${Date.now()}`;
        try {
          segmentAudio.set(synthId, b64ToWavBlobUrl(result.audio_b64));
        } catch (err) {
          console.warn('Failed to decode uploaded segment audio:', err);
        }
        const langValue =
          result.language && result.language !== 'N/A' ? result.language : null;
        const durationSec =
          typeof result.duration_sec === 'number' ? result.duration_sec : null;
        addFinalBubble(text.trim(), langValue, durationSec, synthId);
      } else {
        addFinalBubble(text.trim(), result.language || null, null, null);
      }
    }

    setUploadBusy(false);
    if (trimmedNote !== null) {
      setUploadStatus('warn', 'tsasr.upload.trimmed', {
        max: TSASR_UPLOAD_MAX_SECONDS,
        actual: trimmedNote,
      });
    } else {
      setUploadStatus('success', 'tsasr.upload.done');
    }
  }

  if (uploadBtn && uploadInput) {
    uploadBtn.addEventListener('click', () => {
      if (uploadBtn.disabled) return;
      uploadInput.value = '';
      uploadInput.click();
    });
    uploadInput.addEventListener('change', () => {
      const file = uploadInput.files && uploadInput.files[0];
      if (file) handleUploadFile(file);
    });
  }

  // -------------------- WebSocket --------------------
  function openWsAndStart() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${proto}://${location.host}/transcribe-target-streaming`;
    setConnStatus('pending');
    setEnrollStatus('pending', 'tsasr.enroll.sending');
    ws = new WebSocket(url);

    ws.onopen = () => {
      setConnStatus('connected');
      const payload = {
        type: 'start',
        format: 'pcm_s16le',
        sample_rate_hz: TARGET_SAMPLE_RATE,
        channels: 1,
        enrollment_audio: enrollWavB64,
        enrollment_format: 'wav',
      };
      ws.send(JSON.stringify(payload));
    };

    ws.onmessage = (evt) => {
      let data;
      try {
        data = JSON.parse(evt.data);
      } catch {
        return;
      }
      handleServerMessage(data);
    };

    ws.onerror = () => {
      setConnStatus('disconnected');
    };

    ws.onclose = () => {
      setConnStatus('disconnected');
      ws = null;
      if (isRecordingLive) {
        stopLiveStreaming({ sendStop: false, reason: 'ws_closed' });
      }
    };
  }

  function handleServerMessage(data) {
    switch (data.type) {
      case 'ready':
        // Server accepted the WS; waiting for our start ack.
        break;
      case 'enrollment_ok':
        setEnrollStatus(
          'ready',
          'tsasr.enroll.ready',
          { dur: (data.duration_sec || enrollDurationSec).toFixed(1) }
        );
        startLiveStreaming();
        break;
      case 'final':
        if (data.text && data.text.trim()) {
          // Cache the mixed-audio blob *before* rendering so addFinalBubble
          // can mount the replay button in the initial DOM pass. Always
          // overwrite on collision (backend should mint unique ids, but if
          // anything ever reuses a key we want the freshest audio, not the
          // stale one — a stale cache hit would make the replay button play
          // audio from a completely different transcript).
          if (data.id && data.audio_b64) {
            const prev = segmentAudio.get(data.id);
            if (prev) {
              try { URL.revokeObjectURL(prev); } catch { /* noop */ }
            }
            try {
              segmentAudio.set(data.id, b64ToWavBlobUrl(data.audio_b64));
            } catch (err) {
              console.warn('Failed to decode segment audio:', err);
            }
          }
          const langValue =
            data.language && data.language !== 'N/A' ? data.language : null;
          const durationSec =
            typeof data.duration_sec === 'number' ? data.duration_sec : null;
          addFinalBubble(
            data.text.trim(),
            langValue,
            durationSec,
            data.id || null,
          );
        }
        break;
      case 'partial':
        // Not rendered by default; TS-ASR partial is usually disabled server-side.
        break;
      case 'error': {
        const code = data.code || 'error';
        // Pill stays short -- show the short code only; the full message
        // lives in the error chat bubble below.
        const shortCode = code.replace(/^enrollment_/, '');
        setEnrollStatus(
          'error',
          'tsasr.enroll.errorCode',
          { code: shortCode, defaultValue: shortCode }
        );
        addErrorBubble(code, data.message || '');
        if (code.startsWith('enrollment_')) {
          // Enrollment was rejected: tear down WS, let user re-record.
          stopLiveStreaming({ sendStop: false, reason: 'enrollment_rejected' });
          if (ws) {
            try { ws.close(); } catch { /* noop */ }
          }
        }
        break;
      }
    }
  }

  // -------------------- Live streaming --------------------
  async function startLiveStreaming() {
    if (isRecordingLive) return;
    try {
      const { ctx, node, mediaStream } = await openSixteenKContext();
      liveCtx = ctx;
      liveNode = node;
      liveStream = mediaStream;
    } catch (err) {
      console.error(err);
      addErrorBubble('mic_denied', t('tsasr.enroll.micAlert'));
      setConnStatus('disconnected');
      return;
    }

    liveNode.port.onmessage = (evt) => {
      if (evt.data.type !== 'audio') return;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const pcm16 = floatToPcm16(evt.data.samples);
      ws.send(pcm16.buffer);
    };

    isRecordingLive = true;
    micBtn.classList.add('recording');
    micIcon.setAttribute('fill', 'currentColor');
    pulseRings.forEach((r) => r.classList.add('active'));
    updateMicGate();
  }

  async function stopLiveStreaming({ sendStop, reason } = { sendStop: true }) {
    if (!isRecordingLive && !liveCtx) return;

    if (liveNode) {
      liveNode.port.onmessage = null;
      liveNode.disconnect();
      liveNode = null;
    }
    if (liveCtx) {
      try { await liveCtx.close(); } catch { /* noop */ }
      liveCtx = null;
    }
    if (liveStream) {
      liveStream.getTracks().forEach((tr) => tr.stop());
      liveStream = null;
    }

    isRecordingLive = false;
    micBtn.classList.remove('recording');
    micIcon.setAttribute('fill', 'none');
    pulseRings.forEach((r) => r.classList.remove('active'));
    updateMicGate();

    if (sendStop && ws && ws.readyState === WebSocket.OPEN) {
      try { ws.send(JSON.stringify({ type: 'stop' })); } catch { /* noop */ }
    }
    if (ws && ws.readyState === WebSocket.OPEN && reason !== 'keep_ws') {
      try { ws.close(1000); } catch { /* noop */ }
    }
  }

  micBtn.addEventListener('click', async () => {
    if (micBtn.disabled) return;
    if (isRecordingLive || (ws && ws.readyState === WebSocket.OPEN)) {
      await stopLiveStreaming({ sendStop: true });
    } else {
      if (!enrollWavB64) return;
      openWsAndStart();
    }
  });

  // -------------------- Refresh on language change --------------------
  function refreshDynamic() {
    if (enrollStatusDyn.key) {
      enrollStatusPill.textContent = t(
        enrollStatusDyn.key,
        enrollStatusDyn.vars || undefined
      );
    }
    enrollRecLabel.textContent = isEnrollRecording
      ? t('tsasr.enroll.stop')
      : t('tsasr.enroll.start');
    if (uploadBtnLabel) {
      uploadBtnLabel.textContent = t(isUploading ? 'tsasr.upload.uploading' : 'tsasr.upload.label');
    }
    if (currentUploadDyn && uploadStatus) {
      uploadStatus.textContent = t(currentUploadDyn.key, currentUploadDyn.vars || undefined);
    }
    if (enrollUploadBtnLabel) {
      enrollUploadBtnLabel.textContent = t(
        isEnrollUploading ? 'tsasr.enrollUpload.uploading' : 'tsasr.enrollUpload.label'
      );
    }
    if (currentEnrollUploadDyn && enrollUploadStatus) {
      enrollUploadStatus.textContent = t(
        currentEnrollUploadDyn.key,
        currentEnrollUploadDyn.vars || undefined,
      );
    }
    updateMicGate();
    // Walk dyn nodes inside the chat area to refresh transcript meta + errors.
    chatArea.querySelectorAll('[data-dyn-key]').forEach((el) => {
      const key = el.getAttribute('data-dyn-key');
      let vars = null;
      const raw = el.getAttribute('data-dyn-vars');
      if (raw) {
        try { vars = JSON.parse(raw); } catch { vars = null; }
      }
      if (key === 'tsasr.meta.lang' && vars && vars.lang) {
        el.textContent = t(key, { lang: langDisplayName(vars.lang) });
      } else {
        el.textContent = t(key, vars || undefined);
      }
    });
  }

  onLangChange(refreshDynamic);

  // -------------------- Init --------------------
  setEnrollStatus('idle', 'tsasr.enroll.notRecorded');
  updateMicGate();
})();
