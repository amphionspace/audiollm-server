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

  const MODE_LABELS = { ser: 'SER', sec: 'SEC' };

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

  function setStatus(state, label) {
    const resolved = KNOWN_STATES.has(state) ? state : 'idle';
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
    btn.disabled = false;
    btn.classList.remove('recording');
    pulseRings.forEach((r) => r.classList.remove('active'));
    if (state === 'idle') {
      btnText.textContent = 'Click to start';
    } else if (state === 'recording') {
      btnText.textContent = 'Listening… click to stop';
      btn.classList.add('recording');
      pulseRings.forEach((r) => r.classList.add('active'));
    } else if (state === 'waiting') {
      btnText.textContent = 'Analyzing…';
      btn.disabled = true;
    }
  }

  function setIdleStatus() {
    if (isCaptureWarm) {
      setStatus('ready', 'Ready');
    } else {
      setStatus('idle', 'Idle');
    }
  }

  function resetResult(message) {
    resultBox.innerHTML =
      '<span class="text-faint">' + escapeHtml(message || 'Result will appear here.') + '</span>';
  }

  function renderResult(data) {
    const mode = data.mode || modeSelect.value || 'ser';
    const label = String(data.label || '').trim();
    const text = String(data.text || '').trim();
    const duration = typeof data.duration_sec === 'number' ? data.duration_sec : 0;

    const modeTag = MODE_LABELS[mode] || mode.toUpperCase();
    const durTag = duration > 0 ? duration.toFixed(2) + 's' : '—';

    let body = '';
    if (mode === 'sec') {
      const caption = text || '(empty)';
      const labelHint = label
        ? '<div class="mt-2 text-[11px] text-muted">Taxonomy hint: '
            + escapeHtml(label) + '</div>'
        : '';
      body =
        '<div class="text-sm leading-relaxed">' + escapeHtml(caption) + '</div>'
        + labelHint;
    } else {
      const displayLabel = label || '(unparsed)';
      body =
        '<div class="flex items-center gap-2">'
        + '<span class="text-base font-semibold">' + escapeHtml(displayLabel) + '</span>'
        + '</div>';
      if (!label && text && text !== label) {
        body += '<div class="mt-1 text-[11px] text-muted">Raw: '
          + escapeHtml(text) + '</div>';
      }
    }

    resultBox.innerHTML =
      '<div class="flex items-center justify-between text-[11px] text-faint mb-1">'
      + '<span>' + escapeHtml(modeTag) + '</span>'
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
        '<li class="text-[11px] text-faint italic">No sessions yet.</li>';
      return;
    }
    historyList.innerHTML = historyEntries.map((entry) => {
      const hh = String(entry.ts.getHours()).padStart(2, '0');
      const mm = String(entry.ts.getMinutes()).padStart(2, '0');
      const ss = String(entry.ts.getSeconds()).padStart(2, '0');
      const modeTag = MODE_LABELS[entry.mode] || entry.mode.toUpperCase();
      const durTag = entry.duration > 0 ? entry.duration.toFixed(2) + 's' : '—';
      const primary = entry.mode === 'sec'
        ? (entry.text || '(empty)')
        : (entry.label || entry.text || '(unparsed)');
      return (
        '<li class="rounded-lg border px-3 py-2 text-xs"'
        + ' style="border-color:var(--line); background:var(--paper-sunk); color:var(--ink)">'
        + '<div class="flex items-center justify-between text-[10px] text-faint mb-0.5">'
        + '<span>' + escapeHtml(modeTag) + ' &middot; ' + escapeHtml(durTag) + '</span>'
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
      mediaStream.getTracks().forEach((t) => {
        try { t.stop(); } catch (_) { /* ignore */ }
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
    reason = null,
    state = null,
    label = null,
    releaseNow = false,
  } = {}) {
    isRecording = false;
    awaitingFinal = false;
    detachGraph();
    closeWsSilently();
    setButton('idle');
    if (reason) {
      resetResult(reason);
    }
    if (releaseNow) {
      releaseCapture();
    } else {
      scheduleIdleRelease();
    }
    if (state && label) {
      setStatus(state, label);
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
      setStatus('listening', 'Listening');
      setButton('recording');
      resetResult('Speak now…');
    } else if (data.type === 'final_emotion') {
      renderResult(data);
      pushHistory(data);
      finishSession({ state: 'done', label: 'Done' });
    } else if (data.type === 'error') {
      const msg = data.message || 'unknown error';
      finishSession({
        reason: 'Error: ' + msg,
        state: 'error',
        label: 'Error',
      });
    }
  }

  async function start() {
    if (isRecording || awaitingFinal || ws) return;
    cancelIdleRelease();
    setButton('waiting');
    btnText.textContent = isCaptureWarm ? 'Connecting…' : 'Opening…';
    btn.disabled = true;
    setStatus('analyzing', 'Connecting');
    resetResult(isCaptureWarm ? 'Connecting…' : 'Opening mic…');

    try {
      await warmCapture();
    } catch (err) {
      finishSession({
        reason: 'Microphone error: ' + (err && err.message ? err.message : String(err)),
        state: 'error',
        label: 'Mic error',
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
      finishSession({
        reason: 'WebSocket error: ' + (err && err.message ? err.message : String(err)),
        state: 'error',
        label: 'WS error',
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
          reason: 'WebSocket error.',
          state: 'error',
          label: 'WS error',
        });
      }
    };
    ws.onclose = () => {
      if (awaitingFinal || isRecording) {
        finishSession({
          reason: 'Connection closed before final result.',
          state: 'error',
          label: 'Closed',
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
      setStatus('analyzing', 'Analyzing');
      resetResult('Analyzing…');
    } else {
      finishSession({
        reason: 'Connection lost.',
        state: 'error',
        label: 'Closed',
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

  setButton('idle');
  setIdleStatus();
  renderHistory();
})();
