/**
 * Emotion recognition panel (independent of the main ASR pipeline).
 *
 * Flow: click Start -> record 16 kHz PCM in the browser -> click Stop ->
 * POST /api/emotion/jobs (async) -> poll until final_emotion.
 *
 * Backend selection: this page targets the AmphionSPEC checkpoint
 * internally while the public protocol exposes only ``ser`` (8-way
 * classification) and ``sec`` (free-form paralinguistic description).
 *
 * Capture warm-up (mic + AudioContext + worklet) is kept alive across
 * sessions so successive Start clicks skip getUserMedia / worklet load. An
 * idle timer (IDLE_RELEASE_MS) releases the warm capture if the panel is
 * left untouched, so the browser's mic indicator doesn't stay on forever.
 */
(() => {
  'use strict';

  const EMOTION_ENDPOINT = '/api/emotion/jobs';

  // Emotion demo page module.
  //
  // Wrapped in an ``init`` factory so the SPA router can mount and tear
  // down this page repeatedly within a single document. ``init``
  // returns a ``dispose`` callback the router calls before swapping the
  // page out — that aborts in-flight jobs/uploads, releases the
  // microphone + AudioContext, cancels the idle-release timer, and
  // unsubscribes from i18n change events.
  function initEmotion() {
    const i18n = window.Amphion && window.Amphion.i18n;
    const t = (key, vars) => (i18n ? i18n.t(key, vars) : (vars && vars.defaultValue) || key);
    const onLangChange = (fn) => (i18n ? i18n.onChange(fn) : () => {});
    let i18nUnsub = null;

  const MODE_LABEL_KEYS = { ser: 'emotion.mode.tag.ser', sec: 'emotion.mode.tag.sec' };
  const modeTag = (mode) => t(MODE_LABEL_KEYS[mode] || 'emotion.mode.tag.ser', {
    defaultValue: (mode || 'ser').toUpperCase(),
  });

  // Status badges now derive their palette from CSS via `data-state`.
  // Kept here only as a whitelist so we don't accidentally write unknown states.
  const KNOWN_STATES = new Set([
    'idle', 'ready', 'listening', 'analyzing', 'done', 'error',
  ]);

  // Sidebar dot state maps onto the shared connection dot.
  const CONN_DOT_STATE = {
    idle: 'idle',
    ready: 'ready',
    listening: 'listening',
    analyzing: 'analyzing',
    done: 'ready',
    error: 'error',
  };

  const IDLE_RELEASE_MS = 30000;
  const MAX_HISTORY = 8;

  let mediaStream = null;
  let audioCtx = null;
  let workletNode = null;
  let sourceNode = null;
  let isCaptureWarm = false;
  let isGraphAttached = false;

  let isRecording = false;
  let isJobRunning = false;
  let jobController = null;
  let recordChunks = [];
  let recordSampleCount = 0;
  let idleReleaseTimer = null;
  const EMOTION_JOB_POLL_MS = 400;
  const EMOTION_JOB_MAX_WAIT_MS = 45000;

  // Last-known UI state so we can re-derive labels on language change.
  let currentStatus = { state: 'idle', labelKey: 'emotion.status.idle', labelVars: null };
  let currentButton = 'idle';
  let currentResult = { kind: 'placeholder', key: 'emotion.result.placeholder' };
  let lastFinalData = null; // for re-rendering result after lang switch

  const btn = document.getElementById('emotion-btn');
  const btnText = document.getElementById('emotion-btn-text');
  const pulseRings = document.querySelectorAll('.pulse-ring');
  const statusBadge = document.getElementById('emotion-status');
  const liveDot = document.getElementById('emotion-live-dot');
  const modeSelect = document.getElementById('emotion-mode');
  const resultBox = document.getElementById('emotion-result');
  const historyList = document.getElementById('emotion-history');
  const historyClear = document.getElementById('emotion-history-clear');
  const uploadBtn = document.getElementById('upload-btn');
  const uploadBtnLabel = uploadBtn ? uploadBtn.querySelector('.btn-upload-label') : null;
  const uploadInput = document.getElementById('upload-input');
  const uploadStatus = document.getElementById('upload-status');

  // Upload / async job state (shared POST /api/emotion/jobs + poll).
  let isUploading = false;
  let uploadController = null;     // AbortController for in-flight fetch
  let currentUploadDyn = null;     // { key, vars } | null when hidden
  const EMOTION_UPLOAD_SAMPLE_RATE = 16000;
  // Server caps emotion uploads at emotion_max_audio_seconds (default 20s,
  // tail-trimmed). We give the client a slightly more generous ceiling and
  // let the server do the final trim — the user-visible behaviour matches
  // what the live mic flow has always done.
  const EMOTION_UPLOAD_MAX_SECONDS = 60;

  if (!btn || !btnText || !statusBadge || !modeSelect || !resultBox) {
    return;
  }

  let historyEntries = [];

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = String(text == null ? '' : text);
    return div.innerHTML;
  }

  function setStatus(state, labelKey, labelVars) {
    const resolved = KNOWN_STATES.has(state) ? state : 'idle';
    currentStatus = { state: resolved, labelKey, labelVars: labelVars || null };
    const label = labelKey ? t(labelKey, labelVars || undefined) : '';
    statusBadge.textContent = label;
    statusBadge.className = 'status-pill';
    statusBadge.dataset.state = resolved;

    if (liveDot) {
      const isLive = resolved === 'listening' || resolved === 'analyzing';
      liveDot.classList.toggle('is-active', isLive);
    }

    if (window.AmphionSidebar && window.AmphionSidebar.setConnectionState) {
      const dotState = CONN_DOT_STATE[resolved] || 'idle';
      window.AmphionSidebar.setConnectionState(dotState, label);
    }
  }

  function setButton(state) {
    currentButton = state;
    btn.disabled = false;
    btn.classList.remove('recording');
    pulseRings.forEach((r) => r.classList.remove('active'));
    if (state === 'idle') {
      btnText.textContent = t('emotion.btn.start');
    } else if (state === 'recording') {
      btnText.textContent = t('emotion.btn.recording');
      btn.classList.add('recording');
      pulseRings.forEach((r) => r.classList.add('active'));
    } else if (state === 'waiting') {
      btnText.textContent = t('emotion.btn.analyzing');
      btn.disabled = true;
    } else if (state === 'connecting') {
      btnText.textContent = t('emotion.btn.connecting');
      btn.disabled = true;
    } else if (state === 'opening') {
      btnText.textContent = t('emotion.btn.opening');
      btn.disabled = true;
    }
  }

  function setIdleStatus() {
    if (isCaptureWarm) {
      setStatus('ready', 'emotion.status.ready');
    } else {
      setStatus('idle', 'emotion.status.idle');
    }
  }

  function resetResult(messageKey) {
    const key = messageKey || 'emotion.result.placeholder';
    currentResult = { kind: 'placeholder', key };
    lastFinalData = null;
    resultBox.innerHTML =
      '<span class="text-faint">' + escapeHtml(t(key)) + '</span>';
  }

  function renderResult(data) {
    lastFinalData = data;
    currentResult = { kind: 'final' };
    const mode = data.mode || modeSelect.value || 'ser';
    const label = String(data.label || '').trim();
    const text = String(data.text || '').trim();
    const topEmotions = Array.isArray(data.top_emotions) ? data.top_emotions : [];
    const duration = typeof data.duration_sec === 'number' ? data.duration_sec : 0;

    const tag = modeTag(mode);
    const durTag = duration > 0 ? duration.toFixed(2) + 's' : '—';

    let body = '';
    if (mode === 'sec') {
      const caption = text || t('emotion.result.empty');
      const labelHint = label
        ? '<div class="mt-2 text-[11px] text-muted">'
            + escapeHtml(t('emotion.result.taxonomyHint', { label }))
            + '</div>'
        : '';
      body =
        '<div class="text-sm leading-relaxed">' + escapeHtml(caption) + '</div>'
        + labelHint;
    } else {
      const displayLabel = label || t('emotion.result.unparsed');
      body =
        '<div class="flex items-center gap-2">'
        + '<span class="text-base font-semibold">' + escapeHtml(displayLabel) + '</span>'
        + '</div>';
      if (topEmotions.length) {
        body += '<div class="mt-2 text-[11px] text-muted">'
          + topEmotions.map((item) => {
            const score = Number(item && item.score);
            const suffix = Number.isFinite(score) ? ' ' + (score * 100).toFixed(1) + '%' : '';
            return escapeHtml(String((item && item.label) || '')) + suffix;
          }).join(' · ')
          + '</div>';
      }
      if (!label && text && text !== label) {
        body += '<div class="mt-1 text-[11px] text-muted">'
          + escapeHtml(t('emotion.result.raw', { text }))
          + '</div>';
      }
    }

    resultBox.innerHTML =
      '<div class="flex items-center justify-between text-[11px] text-faint mb-1">'
      + '<span>' + escapeHtml(tag) + '</span>'
      + '<span>' + escapeHtml(durTag) + '</span>'
      + '</div>'
      + body;
  }

  function pushHistory(data) {
    if (!historyList) return;
    historyEntries.unshift({
      mode: data.mode || modeSelect.value || 'ser',
      label: String(data.label || '').trim(),
      text: String(data.text || '').trim(),
      duration: typeof data.duration_sec === 'number' ? data.duration_sec : 0,
      ts: new Date(),
    });
    if (historyEntries.length > MAX_HISTORY) {
      historyEntries.length = MAX_HISTORY;
    }
    renderHistory();
  }

  function renderHistory() {
    if (!historyList) return;
    if (historyEntries.length === 0) {
      historyList.innerHTML =
        '<li class="text-[11px] text-faint italic">'
        + escapeHtml(t('emotion.history.empty'))
        + '</li>';
      return;
    }
    historyList.innerHTML = historyEntries.map((entry) => {
      const hh = String(entry.ts.getHours()).padStart(2, '0');
      const mm = String(entry.ts.getMinutes()).padStart(2, '0');
      const ss = String(entry.ts.getSeconds()).padStart(2, '0');
      const tag = modeTag(entry.mode);
      const durTag = entry.duration > 0 ? entry.duration.toFixed(2) + 's' : '—';
      const primary = entry.mode === 'sec'
        ? (entry.text || t('emotion.result.empty'))
        : (entry.label || entry.text || t('emotion.result.unparsed'));
      return (
        '<li class="rounded-lg border px-3 py-2 text-xs"'
        + ' style="border-color:var(--line); background:var(--paper-sunk); color:var(--ink)">'
        + '<div class="flex items-center justify-between text-[10px] text-faint mb-0.5">'
        + '<span>' + escapeHtml(tag) + ' &middot; ' + escapeHtml(durTag) + '</span>'
        + '<span>' + hh + ':' + mm + ':' + ss + '</span>'
        + '</div>'
        + '<div class="leading-snug">' + escapeHtml(primary) + '</div>'
        + '</li>'
      );
    }).join('');
  }

  // --- Capture lifecycle (warm-once, release on idle) ---------------------

  async function warmCapture() {
    if (isCaptureWarm) return;

    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: { ideal: 16000 },
        echoCancellation: true,
        noiseSuppression: true,
      },
    });

    audioCtx = new AudioContext({ sampleRate: 16000 });
    if (audioCtx.sampleRate !== 16000) {
      console.warn(
        '[emotion] AudioContext sample rate is %d Hz (expected 16000); '
        + 'browser may not honor the requested rate.',
        audioCtx.sampleRate,
      );
    }
    await audioCtx.audioWorklet.addModule('audio-processor.js');

    sourceNode = audioCtx.createMediaStreamSource(mediaStream);
    workletNode = new AudioWorkletNode(audioCtx, 'audio-capture-processor');

    workletNode.port.onmessage = (evt) => {
      if (!isRecording) return;
      if (evt.data.type !== 'audio') return;
      const float32 = evt.data.samples;
      if (!float32 || !float32.length) return;
      recordChunks.push(new Float32Array(float32));
      recordSampleCount += float32.length;
    };

    isCaptureWarm = true;
    isGraphAttached = false;
  }

  function attachGraph() {
    if (!isCaptureWarm || isGraphAttached) return;
    sourceNode.connect(workletNode);
    workletNode.connect(audioCtx.destination);
    isGraphAttached = true;
  }

  function detachGraph() {
    if (!isCaptureWarm || !isGraphAttached) return;
    try { sourceNode.disconnect(); } catch (_) { /* ignore */ }
    try { workletNode.disconnect(); } catch (_) { /* ignore */ }
    isGraphAttached = false;
  }

  function releaseCapture() {
    cancelIdleRelease();
    detachGraph();
    if (workletNode) {
      try { workletNode.port.onmessage = null; } catch (_) { /* ignore */ }
      workletNode = null;
    }
    sourceNode = null;
    if (audioCtx) {
      try { audioCtx.close(); } catch (_) { /* ignore */ }
      audioCtx = null;
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach((tr) => {
        try { tr.stop(); } catch (_) { /* ignore */ }
      });
      mediaStream = null;
    }
    isCaptureWarm = false;
  }

  function scheduleIdleRelease() {
    cancelIdleRelease();
    if (!isCaptureWarm) return;
    idleReleaseTimer = setTimeout(() => {
      idleReleaseTimer = null;
      if (!isRecording && !isJobRunning) {
        releaseCapture();
        setIdleStatus();
      }
    }, IDLE_RELEASE_MS);
  }

  function cancelIdleRelease() {
    if (idleReleaseTimer != null) {
      clearTimeout(idleReleaseTimer);
      idleReleaseTimer = null;
    }
  }

  function clearRecordingBuffer() {
    recordChunks = [];
    recordSampleCount = 0;
  }

  function mergeRecordedPcm() {
    if (recordSampleCount <= 0) {
      return new Float32Array(0);
    }
    const merged = new Float32Array(recordSampleCount);
    let offset = 0;
    for (let i = 0; i < recordChunks.length; i++) {
      const chunk = recordChunks[i];
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    return merged;
  }

  function finishSession({
    reasonKey = null,
    reasonVars = null,
    state = null,
    labelKey = null,
    releaseNow = false,
  } = {}) {
    isRecording = false;
    isJobRunning = false;
    detachGraph();
    clearRecordingBuffer();
    if (jobController) {
      try { jobController.abort(); } catch (_) { /* ignore */ }
      jobController = null;
    }
    setButton('idle');
    if (reasonKey) {
      currentResult = { kind: 'placeholder', key: reasonKey, vars: reasonVars };
      lastFinalData = null;
      resultBox.innerHTML =
        '<span class="text-faint">'
        + escapeHtml(t(reasonKey, reasonVars || undefined))
        + '</span>';
    }
    if (releaseNow) {
      releaseCapture();
    } else {
      scheduleIdleRelease();
    }
    if (state && labelKey) {
      setStatus(state, labelKey);
    } else {
      setIdleStatus();
    }
  }

  async function runEmotionJob(wavBytes, { trimmedNote = null } = {}) {
    const upload = window.AmphionAudioUpload;
    if (!upload || !upload.submitEmotionJobAndPoll) {
      throw new Error('emotion job API unavailable');
    }
    isJobRunning = true;
    jobController = new AbortController();
    setButton('waiting');
    setStatus('analyzing', 'emotion.status.analyzing');
    currentResult = { kind: 'placeholder', key: 'emotion.result.analyzing' };
    resultBox.innerHTML =
      '<span class="text-faint">' + escapeHtml(t('emotion.result.analyzing')) + '</span>';
    btn.disabled = true;

    let result;
    try {
      result = await upload.submitEmotionJobAndPoll(
        wavBytes,
        { mode: modeSelect.value || 'ser' },
        {
          endpoint: EMOTION_ENDPOINT,
          signal: jobController.signal,
          pollIntervalMs: EMOTION_JOB_POLL_MS,
          maxWaitMs: EMOTION_JOB_MAX_WAIT_MS,
        },
      );
    } finally {
      jobController = null;
      isJobRunning = false;
    }

    renderResult(result);
    pushHistory(result);
    setStatus('done', 'emotion.status.done');
    setIdleStatus();
    setButton('idle');
    btn.disabled = isUploading;
    if (trimmedNote !== null && uploadStatus) {
      setUploadStatus('warn', 'emotion.upload.trimmed', {
        max: EMOTION_UPLOAD_MAX_SECONDS,
        actual: trimmedNote,
      });
    }
  }

  async function start() {
    if (isRecording || isJobRunning || isUploading) return;
    cancelIdleRelease();
    setButton(isCaptureWarm ? 'connecting' : 'opening');
    btn.disabled = true;
    const placeholderKey = isCaptureWarm
      ? 'emotion.result.connecting'
      : 'emotion.result.opening';
    setStatus('ready', isCaptureWarm ? 'emotion.status.ready' : 'emotion.status.idle');
    currentResult = { kind: 'placeholder', key: placeholderKey };
    resultBox.innerHTML =
      '<span class="text-faint">' + escapeHtml(t(placeholderKey)) + '</span>';

    try {
      await warmCapture();
    } catch (err) {
      const msg = err && err.message ? err.message : String(err);
      finishSession({
        reasonKey: 'emotion.error.mic',
        reasonVars: { msg },
        state: 'error',
        labelKey: 'emotion.status.micErr',
        releaseNow: true,
      });
      return;
    }

    clearRecordingBuffer();
    attachGraph();
    isRecording = true;
    setStatus('listening', 'emotion.status.listening');
    setButton('recording');
    currentResult = { kind: 'placeholder', key: 'emotion.result.speakNow' };
    resultBox.innerHTML =
      '<span class="text-faint">' + escapeHtml(t('emotion.result.speakNow')) + '</span>';
    btn.disabled = false;
  }

  async function stop() {
    if (!isRecording) return;
    isRecording = false;
    detachGraph();

    const upload = window.AmphionAudioUpload;
    const pcm = mergeRecordedPcm();
    clearRecordingBuffer();
    if (!pcm.length || !upload) {
      finishSession({
        reasonKey: 'emotion.error.emptyRecording',
        state: 'error',
        labelKey: 'emotion.status.error',
      });
      return;
    }

    const wavBytes = upload.encodeWavBytes(pcm, EMOTION_UPLOAD_SAMPLE_RATE);
    try {
      await runEmotionJob(wavBytes);
      scheduleIdleRelease();
    } catch (err) {
      console.error('Emotion job failed:', err);
      const aborted = err && err.name === 'AbortError';
      const msg = err && err.message ? err.message : String(err);
      finishSession({
        reasonKey: aborted ? 'emotion.upload.aborted' : 'emotion.error.serverPrefix',
        reasonVars: aborted ? null : { msg },
        state: 'error',
        labelKey: 'emotion.status.error',
      });
    }
  }

  // --- Upload flow ----------------------------------------------------------

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
    if (uploadBtn) {
      uploadBtn.disabled = busy;
    }
    if (uploadBtnLabel) {
      uploadBtnLabel.textContent = t(busy ? 'emotion.upload.uploading' : 'emotion.upload.label');
    }
    btn.disabled = busy || isRecording || isJobRunning;
  }

  async function handleUploadFile(file) {
    if (!file) return;
    // The live mic flow shares the result panel and idle-release timer; we
    // refuse uploads while either is active so the two flows can't fight
    // for the same UI slot.
    if (isRecording || isJobRunning || isUploading) {
      setUploadStatus('error', 'emotion.upload.error.busy');
      return;
    }
    const upload = window.AmphionAudioUpload;
    if (!upload) {
      setUploadStatus('error', 'emotion.upload.error.unsupported');
      return;
    }

    setUploadBusy(true);
    setUploadStatus('info', 'emotion.upload.decoding');

    let decoded;
    try {
      decoded = await upload.decodeFileToWavBytes(file, EMOTION_UPLOAD_SAMPLE_RATE);
    } catch (err) {
      console.error('Upload decode failed:', err);
      setUploadBusy(false);
      setUploadStatus('error', 'emotion.upload.error.decode');
      return;
    }
    if (!decoded || !decoded.wav || !decoded.pcm.length) {
      setUploadBusy(false);
      setUploadStatus('error', 'emotion.upload.error.empty');
      return;
    }

    let pcm = decoded.pcm;
    let wavBytes = decoded.wav;
    const totalSec = pcm.length / EMOTION_UPLOAD_SAMPLE_RATE;
    let trimmedNote = null;
    if (totalSec > EMOTION_UPLOAD_MAX_SECONDS) {
      pcm = new Float32Array(
        pcm.subarray(0, Math.floor(EMOTION_UPLOAD_MAX_SECONDS * EMOTION_UPLOAD_SAMPLE_RATE))
      );
      wavBytes = upload.encodeWavBytes(pcm, EMOTION_UPLOAD_SAMPLE_RATE);
      trimmedNote = totalSec.toFixed(1);
    }

    // Use the existing "analyzing" placeholders so the result panel is
    // visually consistent with the live-mic flow.
    setStatus('analyzing', 'emotion.status.analyzing');
    setUploadStatus('info', 'emotion.upload.analyzing');
    currentResult = { kind: 'placeholder', key: 'emotion.result.analyzing' };
    resultBox.innerHTML =
      '<span class="text-faint">' + escapeHtml(t('emotion.result.analyzing')) + '</span>';

    uploadController = new AbortController();
    jobController = uploadController;
    isJobRunning = true;
    try {
      const result = await upload.submitEmotionJobAndPoll(
        wavBytes,
        { mode: modeSelect.value || 'ser' },
        {
          endpoint: EMOTION_ENDPOINT,
          signal: uploadController.signal,
          fileName: file.name || 'upload.wav',
          pollIntervalMs: EMOTION_JOB_POLL_MS,
          maxWaitMs: EMOTION_JOB_MAX_WAIT_MS,
        },
      );
      uploadController = null;
      jobController = null;
      isJobRunning = false;
      setUploadBusy(false);
      renderResult(result);
      pushHistory(result);
      setStatus('done', 'emotion.status.done');
      setIdleStatus();
      if (trimmedNote !== null) {
        setUploadStatus('warn', 'emotion.upload.trimmed', {
          max: EMOTION_UPLOAD_MAX_SECONDS,
          actual: trimmedNote,
        });
      } else {
        setUploadStatus('success', 'emotion.upload.done');
      }
    } catch (err) {
      console.error('Upload job failed:', err);
      uploadController = null;
      jobController = null;
      isJobRunning = false;
      setUploadBusy(false);
      const aborted = err && err.name === 'AbortError';
      const msg = err && err.message ? err.message : 'Upload failed';
      finishSession({
        reasonKey: aborted ? 'emotion.upload.aborted' : 'emotion.error.serverPrefix',
        reasonVars: aborted ? null : { msg },
        state: 'error',
        labelKey: 'emotion.status.error',
      });
      setUploadStatus(aborted ? 'info' : 'error',
        aborted ? 'emotion.upload.aborted' : 'emotion.upload.error.serverPrefix',
        aborted ? null : { msg });
    }
  }

  if (uploadBtn && uploadInput) {
    uploadBtn.addEventListener('click', () => {
      if (isUploading || isRecording || isJobRunning) return;
      uploadInput.value = '';
      uploadInput.click();
    });
    uploadInput.addEventListener('change', () => {
      const file = uploadInput.files && uploadInput.files[0];
      if (file) handleUploadFile(file);
    });
  }

  btn.addEventListener('click', () => {
    if (isUploading) return;
    if (isRecording) {
      stop();
    } else if (!isJobRunning) {
      start();
    }
  });

  if (historyClear) {
    historyClear.addEventListener('click', () => {
      historyEntries = [];
      renderHistory();
    });
  }

  // ``beforeunload`` still fires on a real tab close. Wrap so dispose
  // can ``removeEventListener`` it during SPA navigation — otherwise
  // every emotion mount would stack a duplicate listener.
  function onBeforeUnload() {
    try { releaseCapture(); } catch (_) { /* ignore */ }
  }
  window.addEventListener('beforeunload', onBeforeUnload);

  // --- Refresh on language change ---
  i18nUnsub = onLangChange(() => {
    if (currentStatus.labelKey) {
      setStatus(currentStatus.state, currentStatus.labelKey, currentStatus.labelVars);
    }
    setButton(currentButton);
    if (currentResult.kind === 'placeholder') {
      resultBox.innerHTML =
        '<span class="text-faint">'
        + escapeHtml(t(currentResult.key, currentResult.vars || undefined))
        + '</span>';
    } else if (currentResult.kind === 'final' && lastFinalData) {
      renderResult(lastFinalData);
    }
    renderHistory();
    if (uploadBtnLabel) {
      uploadBtnLabel.textContent = t(isUploading ? 'emotion.upload.uploading' : 'emotion.upload.label');
    }
    if (currentUploadDyn && uploadStatus) {
      uploadStatus.textContent = t(currentUploadDyn.key, currentUploadDyn.vars || undefined);
    }
  });

  setButton('idle');
  setIdleStatus();
  renderHistory();

    // --- Dispose ---
    // Called by the SPA router before the emotion <main> is swapped
    // out. Release every external resource:
    //   * In-flight upload AbortController
    //   * In-flight emotion job
    //   * Microphone + AudioContext + worklet (warm-capture state)
    //   * Idle-release timer that would otherwise fire after we're
    //     gone and try to call setIdleStatus on detached DOM nodes
    //   * i18n change subscription
    return function disposeEmotion() {
      try {
        window.removeEventListener('beforeunload', onBeforeUnload);
      } catch (_) { /* ignore */ }
      if (uploadController) {
        try { uploadController.abort(); } catch (_) { /* ignore */ }
        uploadController = null;
      }
      if (jobController) {
        try { jobController.abort(); } catch (_) { /* ignore */ }
        jobController = null;
      }
      try { releaseCapture(); } catch (_) { /* ignore */ }
      try { cancelIdleRelease(); } catch (_) { /* ignore */ }
      isRecording = false;
      isJobRunning = false;
      if (typeof i18nUnsub === 'function') {
        try { i18nUnsub(); } catch (_) { /* ignore */ }
        i18nUnsub = null;
      }
    };
  }

  window.AmphionPages = window.AmphionPages || {};
  window.AmphionPages.emotion = { init: initEmotion };
})();
