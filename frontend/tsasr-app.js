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

  const voiceTraitsInput = document.getElementById('voice-traits');
  const languageSelect = document.getElementById('session-language');

  // -------------------- Connection status --------------------
  function setConnStatus(state) {
    if (!window.AmphionSidebar || !window.AmphionSidebar.setConnectionState) return;
    if (state === 'connected') {
      window.AmphionSidebar.setConnectionState('connected', 'Connected');
    } else if (state === 'pending') {
      window.AmphionSidebar.setConnectionState('pending', 'Connecting...');
    } else {
      window.AmphionSidebar.setConnectionState('idle', 'Idle');
    }
  }
  setConnStatus('disconnected');

  // -------------------- Enrollment status pill --------------------
  function setEnrollStatus(state, text) {
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
    enrollStatusPill.textContent = text;
  }

  function updateMicGate(message) {
    const enabled = enrollWavB64 !== null && !isEnrollRecording;
    micBtn.disabled = !enabled;
    if (isRecordingLive) {
      micStatus.textContent = 'Listening...';
    } else if (!enabled) {
      micStatus.textContent = message || 'Complete enrollment to enable';
    } else {
      micStatus.textContent = 'Click to start';
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
      setEnrollStatus('error', 'Mic denied');
      alert('Microphone access denied. Please allow microphone access.');
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
    enrollRecLabel.textContent = 'Stop';
    enrollRecBtn.classList.add('enroll-recording');
    setEnrollStatus('recording', 'Recording...');
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
    enrollRecLabel.textContent = 'Start recording';

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
      enrollStream.getTracks().forEach((t) => t.stop());
      enrollStream = null;
    }

    const total = enrollChunks.reduce((n, b) => n + b.length, 0);
    const sr = TARGET_SAMPLE_RATE;
    const duration = total / sr;
    enrollDurationSec = duration;
    enrollTimer.textContent = `${duration.toFixed(1)}s`;

    if (duration < MIN_ENROLL_SEC) {
      setEnrollStatus('error', `Too short (${duration.toFixed(1)}s)`);
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

    setEnrollStatus('ready', `Ready (${duration.toFixed(1)}s)`);
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
    setEnrollStatus('idle', 'Not recorded');
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

  function addFinalBubble(text, meta, segId) {
    const wrapper = document.createElement('div');
    wrapper.className = 'chat-row chat-row-ai chat-bubble-float';
    if (segId) wrapper.id = `ai-${segId}`;

    const metaLine = meta
      ? `<div class="text-[11px] text-faint mt-1">${escapeHtml(meta)}</div>`
      : '';
    const hasAudio = Boolean(segId) && segmentAudio.has(segId);
    // Keep the replay button inline with the paragraph so users scan text
    // first and see the play affordance as a subtle trailing glyph, not a
    // separate control.
    const replayBtn = hasAudio
      ? `<button class="replay-btn" type="button" title="Replay audio">
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

  function addErrorBubble(message) {
    const wrapper = document.createElement('div');
    wrapper.className = 'chat-row chat-row-ai chat-bubble-float';
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
        <div class="chat-bubble chat-bubble-ai text-sm" style="color:var(--danger)">
          ${escapeHtml(message)}
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

  // -------------------- WebSocket --------------------
  function openWsAndStart() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const lang = languageSelect.value || '';
    const url = `${proto}://${location.host}/transcribe-target-streaming${
      lang ? `?language=${encodeURIComponent(lang)}` : ''
    }`;
    setConnStatus('pending');
    setEnrollStatus('pending', 'Sending enrollment...');
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
      const traits = (voiceTraitsInput.value || '').trim();
      if (traits) payload.voice_traits = traits;
      if (lang) payload.language = lang;
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
          `Ready (${(data.duration_sec || enrollDurationSec).toFixed(1)}s)`
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
          const metaParts = [];
          if (data.language && data.language !== 'N/A') {
            metaParts.push(`language: ${data.language}`);
          }
          if (typeof data.duration_sec === 'number') {
            metaParts.push(`${data.duration_sec.toFixed(1)}s`);
          }
          addFinalBubble(
            data.text.trim(),
            metaParts.join(' \u00b7 ') || null,
            data.id || null,
          );
        }
        break;
      case 'partial':
        // Not rendered by default; TS-ASR partial is usually disabled server-side.
        break;
      case 'error': {
        const code = data.code || 'error';
        setEnrollStatus('error', code.replace(/^enrollment_/, ''));
        addErrorBubble(`Server error [${code}]: ${data.message || ''}`);
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
      addErrorBubble('Microphone access denied.');
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
      liveStream.getTracks().forEach((t) => t.stop());
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

  // -------------------- Init --------------------
  setEnrollStatus('idle', 'Not recorded');
  updateMicGate();
})();
