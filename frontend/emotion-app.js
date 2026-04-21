/**
 * Emotion recognition panel (independent of the main ASR pipeline).
 *
 * Flow: click Start -> open WS to /emotion-streaming at 16 kHz -> stream PCM
 * -> click Stop -> send {type:"stop"} -> render the single final_emotion reply.
 *
 * Capture warm-up (mic + AudioContext + worklet) is kept alive across
 * sessions so successive Start clicks skip getUserMedia / worklet load. An
 * idle timer (IDLE_RELEASE_MS) releases the warm capture if the panel is
 * left untouched, so the browser's mic indicator doesn't stay on forever.
 *
 * The server does not resample /emotion-streaming input, so the capture
 * AudioContext is forced to 16 kHz. The shared audio-capture-processor
 * worklet is sample-rate agnostic (it groups samples by count, not duration),
 * so we reuse it here.
 */
(() => {
  'use strict';

  const i18n = window.Amphion && window.Amphion.i18n;
  const t = (key, vars) => (i18n ? i18n.t(key, vars) : (vars && vars.defaultValue) || key);
  const onLangChange = (fn) => (i18n ? i18n.onChange(fn) : () => {});

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

  let ws = null;
  let isRecording = false;
  let awaitingFinal = false;
  let idleReleaseTimer = null;

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
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      if (evt.data.type !== 'audio') return;
      const float32 = evt.data.samples;
      const int16 = new Int16Array(float32.length);
      for (let i = 0; i < float32.length; i++) {
        const s = Math.max(-1, Math.min(1, float32[i]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      ws.send(int16.buffer);
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
      if (!isRecording && !awaitingFinal && !ws) {
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

  // --- WebSocket lifecycle ------------------------------------------------

  function closeWsSilently() {
    if (!ws) return;
    try {
      ws.onopen = null;
      ws.onclose = null;
      ws.onerror = null;
      ws.onmessage = null;
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
      }
    } catch (_) { /* ignore */ }
    ws = null;
  }

  function finishSession({
    reasonKey = null,
    reasonVars = null,
    state = null,
    labelKey = null,
    releaseNow = false,
  } = {}) {
    isRecording = false;
    awaitingFinal = false;
    detachGraph();
    closeWsSilently();
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

  function handleServerMessage(data) {
    if (!data || typeof data !== 'object') return;
    if (data.type === 'ready') {
      const startMsg = {
        type: 'start',
        format: 'pcm_s16le',
        sample_rate_hz: 16000,
        channels: 1,
        mode: modeSelect.value || 'ser',
      };
      ws.send(JSON.stringify(startMsg));
      isRecording = true;
      setStatus('listening', 'emotion.status.listening');
      setButton('recording');
      currentResult = { kind: 'placeholder', key: 'emotion.result.speakNow' };
      resultBox.innerHTML =
        '<span class="text-faint">' + escapeHtml(t('emotion.result.speakNow')) + '</span>';
    } else if (data.type === 'final_emotion') {
      renderResult(data);
      pushHistory(data);
      finishSession({ state: 'done', labelKey: 'emotion.status.done' });
    } else if (data.type === 'error') {
      const msg = data.message || t('emotion.error.unknown');
      finishSession({
        reasonKey: 'emotion.error.serverPrefix',
        reasonVars: { msg },
        state: 'error',
        labelKey: 'emotion.status.error',
      });
    }
  }

  async function start() {
    if (isRecording || awaitingFinal || ws) return;
    cancelIdleRelease();
    setButton(isCaptureWarm ? 'connecting' : 'opening');
    btn.disabled = true;
    setStatus('analyzing', 'emotion.status.connecting');
    const placeholderKey = isCaptureWarm
      ? 'emotion.result.connecting'
      : 'emotion.result.opening';
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
    attachGraph();

    try {
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      ws = new WebSocket(proto + '//' + location.host + '/emotion-streaming');
      ws.binaryType = 'arraybuffer';
    } catch (err) {
      const msg = err && err.message ? err.message : String(err);
      finishSession({
        reasonKey: 'emotion.error.ws',
        reasonVars: { msg },
        state: 'error',
        labelKey: 'emotion.status.wsErr',
      });
      return;
    }

    ws.onmessage = (evt) => {
      try {
        handleServerMessage(JSON.parse(evt.data));
      } catch (_) { /* non-JSON frames are ignored */ }
    };
    ws.onerror = () => {
      if (awaitingFinal || isRecording) {
        finishSession({
          reasonKey: 'emotion.error.wsGeneric',
          state: 'error',
          labelKey: 'emotion.status.wsErr',
        });
      }
    };
    ws.onclose = () => {
      if (awaitingFinal || isRecording) {
        finishSession({
          reasonKey: 'emotion.error.closedBeforeFinal',
          state: 'error',
          labelKey: 'emotion.status.closed',
        });
      }
    };
  }

  function stop() {
    if (!isRecording) return;
    isRecording = false;
    detachGraph();

    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify({ type: 'stop' }));
      } catch (_) { /* ignore */ }
      awaitingFinal = true;
      setButton('waiting');
      setStatus('analyzing', 'emotion.status.analyzing');
      currentResult = { kind: 'placeholder', key: 'emotion.result.analyzing' };
      resultBox.innerHTML =
        '<span class="text-faint">' + escapeHtml(t('emotion.result.analyzing')) + '</span>';
    } else {
      finishSession({
        reasonKey: 'emotion.error.connLost',
        state: 'error',
        labelKey: 'emotion.status.closed',
      });
    }
  }

  btn.addEventListener('click', () => {
    if (isRecording) {
      stop();
    } else if (!awaitingFinal && !ws) {
      start();
    }
  });

  if (historyClear) {
    historyClear.addEventListener('click', () => {
      historyEntries = [];
      renderHistory();
    });
  }

  window.addEventListener('beforeunload', () => {
    try { releaseCapture(); } catch (_) { /* ignore */ }
    try { closeWsSilently(); } catch (_) { /* ignore */ }
  });

  // --- Refresh on language change ---
  onLangChange(() => {
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
  });

  setButton('idle');
  setIdleStatus();
  renderHistory();
})();
