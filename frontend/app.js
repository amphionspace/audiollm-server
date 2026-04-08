(() => {
  'use strict';

  // --- State ---
  let ws = null;
  let audioCtx = null;
  let workletNode = null;
  let mediaStream = null;
  let isRecording = false;
  let hotwords = [];
  let hotwordEnabled = localStorage.getItem('hotword_enabled') !== '0';
  let sessionHitCount = 0;
  let extractRequestId = null;
  let activeReplayAudio = null;
  const segmentAudio = new Map();
  const MAX_EXTRACTED_HOTWORD_LENGTH = 10;

  const HOTWORD_BUCKETS = ['auto', 'chinese', 'english', 'indonesian', 'thai'];
  const HOTWORDS_PER_LANG_MIGRATED = 'hotwords_per_lang_migrated';
  const UI_TO_API_LANG = {
    auto: 'N/A',
    chinese: 'Chinese',
    english: 'English',
    indonesian: 'Indonesian',
    thai: 'Thai',
  };

  function migrateLegacyHotwords() {
    if (localStorage.getItem(HOTWORDS_PER_LANG_MIGRATED) === '1') return;
    const legacy = localStorage.getItem('hotwords');
    if (legacy) {
      try {
        const arr = JSON.parse(legacy);
        if (Array.isArray(arr)) {
          HOTWORD_BUCKETS.forEach((b) => {
            if (localStorage.getItem(`hotwords_${b}`) === null) {
              localStorage.setItem(`hotwords_${b}`, JSON.stringify(arr));
            }
          });
        }
      } catch {
        /* ignore */
      }
    }
    localStorage.setItem(HOTWORDS_PER_LANG_MIGRATED, '1');
  }

  function readHotwordBucket(langForUi) {
    const raw = localStorage.getItem(`hotwords_${langForUi}`);
    if (raw === null) return [];
    try {
      const arr = JSON.parse(raw);
      return Array.isArray(arr) ? arr : [];
    } catch {
      return [];
    }
  }

  function writeHotwordBucket(langForUi, words) {
    localStorage.setItem(`hotwords_${langForUi}`, JSON.stringify(words));
  }

  function apiLangFromUi(langForUi) {
    return UI_TO_API_LANG[langForUi] || 'N/A';
  }

  migrateLegacyHotwords();
  let srcLangUi = localStorage.getItem('asr_src_lang') || 'auto';
  if (!HOTWORD_BUCKETS.includes(srcLangUi)) srcLangUi = 'auto';

  function b64ToWavBlobUrl(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }));
  }

  // --- DOM refs ---
  const micBtn = document.getElementById('mic-btn');
  const micIcon = document.getElementById('mic-icon');
  const micStatus = document.getElementById('mic-status');
  const pulseRings = document.querySelectorAll('.pulse-ring');
  const chatArea = document.getElementById('chat-area');
  const connDot = document.getElementById('conn-dot');
  const connLabel = document.getElementById('conn-label');
  const hotwordInput = document.getElementById('hotword-input');
  const hotwordAddBtn = document.getElementById('hotword-add-btn');
  const hotwordList = document.getElementById('hotword-list');
  const hotwordClearBtn = document.getElementById('hotword-clear-btn');
  const hotwordEnabledInput = document.getElementById('hotword-enabled');
  const hotwordSyncStatus = document.getElementById('hotword-sync-status');
  const hotwordCount = document.getElementById('hotword-count');
  const hotwordHitCount = document.getElementById('hotword-hit-count');
  const hotwordTextarea = document.getElementById('hotword-textarea');
  const hotwordExtractBtn = document.getElementById('hotword-extract-btn');
  const hotwordExtractStatus = document.getElementById('hotword-extract-status');
  const asrLangSelect = document.getElementById('asr-lang-select');

  // --- Hotword management ---
  function sanitizeHotwords(sourceWords) {
    const result = [];
    (Array.isArray(sourceWords) ? sourceWords : []).forEach((item) => {
      const value = String(item || '').trim();
      if (!value || result.includes(value)) return;
      result.push(value);
    });
    return result;
  }

  function enforceHotwordLimit() {
    hotwords = sanitizeHotwords(hotwords);
  }

  function renderHotwords() {
    hotwordList.innerHTML = '';
    hotwords.forEach((word, idx) => {
      const tag = document.createElement('span');
      tag.className =
        'inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm ' +
        'bg-white/6 text-white/90 border border-white/14 backdrop-blur-sm';
      tag.innerHTML =
        `<span>${escapeHtml(word)}</span>` +
        `<button class="hover:text-red-400 transition-colors text-white/50 ml-0.5" data-idx="${idx}">&times;</button>`;
      tag.querySelector('button').addEventListener('click', () => removeHotword(idx));
      hotwordList.appendChild(tag);
    });
    hotwordCount.textContent = `${hotwords.length} hotwords`;
  }

  function getEffectiveHotwords() {
    return hotwordEnabled ? hotwords : [];
  }

  function setHotwordSyncStatus(state) {
    if (!hotwordSyncStatus) return;
    if (state === 'synced') {
      hotwordSyncStatus.textContent = hotwordEnabled ? 'Active' : 'Paused';
      hotwordSyncStatus.className =
        'text-[11px] px-2 py-0.5 rounded-full border border-emerald-300/35 text-emerald-200/90 bg-emerald-300/10';
      return;
    }
    if (state === 'offline') {
      hotwordSyncStatus.textContent = 'Offline';
      hotwordSyncStatus.className =
        'text-[11px] px-2 py-0.5 rounded-full border border-amber-300/35 text-amber-200/90 bg-amber-300/10';
      return;
    }
    hotwordSyncStatus.textContent = 'Waiting';
    hotwordSyncStatus.className =
      'text-[11px] px-2 py-0.5 rounded-full border border-white/15 text-white/65 bg-white/6';
  }

  function syncHotwords() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: 'update_hotwords',
          hotwords: getEffectiveHotwords(),
          src_lang: apiLangFromUi(srcLangUi),
        })
      );
      setHotwordSyncStatus('synced');
    } else {
      setHotwordSyncStatus('offline');
    }
  }

  function saveAndSyncHotwords() {
    enforceHotwordLimit();
    writeHotwordBucket(srcLangUi, hotwords);
    localStorage.setItem('hotwords', JSON.stringify(hotwords));
    renderHotwords();
    syncHotwords();
  }

  function updateHitCounter() {
    hotwordHitCount.textContent = String(sessionHitCount);
  }

  function setExtractStatus(state, text) {
    if (!hotwordExtractStatus) return;
    hotwordExtractStatus.textContent = text;
    hotwordExtractStatus.className = 'hotword-extract-status';
    if (state === 'loading') {
      hotwordExtractStatus.classList.add('is-loading');
    } else if (state === 'success') {
      hotwordExtractStatus.classList.add('is-success');
    } else if (state === 'error') {
      hotwordExtractStatus.classList.add('is-error');
    }
  }

  function setExtractBusy(busy) {
    if (!hotwordExtractBtn || !hotwordTextarea) return;
    hotwordExtractBtn.disabled = busy;
    hotwordExtractBtn.textContent = busy ? 'Extracting...' : 'Extract and Add';
    hotwordExtractBtn.classList.toggle('opacity-60', busy);
    hotwordExtractBtn.classList.toggle('cursor-not-allowed', busy);
    hotwordTextarea.disabled = busy;
    updateExtractButtonAttention();
  }

  function updateExtractButtonAttention() {
    if (!hotwordExtractBtn || !hotwordTextarea) return;
    const hasText = hotwordTextarea.value.trim().length > 0;
    hotwordExtractBtn.classList.toggle(
      'is-attention',
      hasText && !hotwordExtractBtn.disabled
    );
  }

  function mergeExtractedHotwords(words) {
    const normalized = Array.isArray(words)
      ? words
          .map((w) => String(w || '').trim())
          .filter((w) => w && w.length < MAX_EXTRACTED_HOTWORD_LENGTH)
      : [];
    if (normalized.length === 0) return { added: 0, total: 0 };
    let added = 0;
    normalized.forEach((word) => {
      if (!hotwords.includes(word)) {
        hotwords.push(word);
        added += 1;
      }
    });
    if (added > 0) {
      saveAndSyncHotwords();
    } else {
      renderHotwords();
    }
    return { added, total: normalized.length };
  }

  function requestHotwordExtraction(text) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setExtractStatus('error', 'WebSocket offline');
      return;
    }
    const payloadText = String(text || '').trim();
    if (!payloadText) {
      setExtractStatus('error', 'Please paste text first');
      return;
    }
    if (extractRequestId) {
      setExtractStatus('error', 'Extraction already running');
      return;
    }

    extractRequestId = `extract-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
    setExtractBusy(true);
    setExtractStatus('loading', 'Extracting...');
    ws.send(
      JSON.stringify({
        type: 'extract_hotwords',
        request_id: extractRequestId,
        text: payloadText,
      })
    );
  }

  function addHotword(text) {
    const words = text
      .split(/[,，\n]/)
      .map((w) => w.trim())
      .filter((w) => w && !hotwords.includes(w));
    if (words.length === 0) return;
    hotwords.push(...words);
    saveAndSyncHotwords();
  }

  function removeHotword(idx) {
    hotwords.splice(idx, 1);
    saveAndSyncHotwords();
  }

  function clearHotwords() {
    hotwords = [];
    saveAndSyncHotwords();
  }

  hotwordAddBtn.addEventListener('click', () => {
    addHotword(hotwordInput.value);
    hotwordInput.value = '';
  });

  hotwordInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      addHotword(hotwordInput.value);
      hotwordInput.value = '';
    }
  });

  hotwordClearBtn.addEventListener('click', clearHotwords);
  hotwordExtractBtn.addEventListener('click', () => {
    requestHotwordExtraction(hotwordTextarea.value);
  });
  hotwordTextarea.addEventListener('input', updateExtractButtonAttention);

  hotwordEnabledInput.checked = hotwordEnabled;
  hotwordEnabledInput.addEventListener('change', () => {
    hotwordEnabled = hotwordEnabledInput.checked;
    localStorage.setItem('hotword_enabled', hotwordEnabled ? '1' : '0');
    syncHotwords();
  });

  if (asrLangSelect) {
    asrLangSelect.value = srcLangUi;
    asrLangSelect.addEventListener('change', () => {
      const next = asrLangSelect.value;
      if (!HOTWORD_BUCKETS.includes(next)) return;
      writeHotwordBucket(srcLangUi, sanitizeHotwords(hotwords));
      srcLangUi = next;
      localStorage.setItem('asr_src_lang', srcLangUi);
      hotwords = sanitizeHotwords(readHotwordBucket(srcLangUi));
      localStorage.setItem('hotwords', JSON.stringify(hotwords));
      renderHotwords();
      syncHotwords();
    });
  }

  hotwords = sanitizeHotwords(readHotwordBucket(srcLangUi));
  localStorage.setItem('hotwords', JSON.stringify(hotwords));
  renderHotwords();
  updateHitCounter();
  setHotwordSyncStatus('waiting');
  setExtractStatus('idle', 'Idle');
  updateExtractButtonAttention();

  // --- Connection status ---
  function setConnected(connected) {
    if (connected) {
      connDot.className = 'w-2.5 h-2.5 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.35)]';
      connLabel.textContent = 'Connected';
    } else {
      connDot.className = 'w-2.5 h-2.5 rounded-full bg-red-400 shadow-[0_0_8px_rgba(248,113,113,0.35)]';
      connLabel.textContent = 'Disconnected';
    }
  }

  // --- WebSocket ---
  function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${location.host}/ws/audio`);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      setConnected(true);
      syncHotwords();
    };

    ws.onclose = () => {
      setConnected(false);
      setHotwordSyncStatus('offline');
      if (extractRequestId) {
        extractRequestId = null;
        setExtractBusy(false);
        setExtractStatus('error', 'Connection closed');
      }
      stopRecording();
      setTimeout(connectWS, 2000);
    };

    ws.onerror = () => {
      setConnected(false);
      setHotwordSyncStatus('offline');
      if (extractRequestId) {
        extractRequestId = null;
        setExtractBusy(false);
        setExtractStatus('error', 'Connection error');
      }
    };

    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        handleServerMessage(data);
      } catch {
        // ignore non-JSON
      }
    };
  }

  function handleServerMessage(data) {
    switch (data.type) {
      case 'vad_event':
        if (data.event === 'segment_detected') {
          if (data.audio_b64) {
            segmentAudio.set(data.id, b64ToWavBlobUrl(data.audio_b64));
          }
          addUserBubble(data.id, data.duration || '');
          addAIBubble(data.id);
        }
        break;
      case 'status':
        updateAIBubble(data.id, null, 'processing');
        break;
      case 'response':
        updateAIBubble(data.id, data.text, 'done', data.model_hotwords, {
          textPrimary: data.text_primary,
          textSecondary: data.text_secondary,
          fusionMeta: data.fusion_meta,
          srcLangDetected: data.src_lang_detected,
        });
        break;
      case 'discard':
        removeSegmentBubbles(data.id);
        break;
      case 'error':
        updateAIBubble(data.id, `Error: ${data.message}`, 'error');
        break;
      case 'extract_hotwords_result':
        if (!extractRequestId || data.request_id !== extractRequestId) {
          break;
        }
        extractRequestId = null;
        setExtractBusy(false);
        {
          const merged = mergeExtractedHotwords(data.hotwords || []);
          setExtractStatus('success', `Added ${merged.added}/${merged.total}`);
        }
        break;
      case 'extract_hotwords_error':
        if (!extractRequestId || data.request_id !== extractRequestId) {
          break;
        }
        extractRequestId = null;
        setExtractBusy(false);
        setExtractStatus('error', data.message || 'Extract failed');
        break;
    }
  }

  // --- Chat bubbles ---
  function replaySegment(segId, btn) {
    if (activeReplayAudio) {
      activeReplayAudio.pause();
      const prevBtn = document.querySelector('.replay-btn.is-playing');
      if (prevBtn) prevBtn.classList.remove('is-playing');
      if (activeReplayAudio._segId === segId) {
        activeReplayAudio = null;
        return;
      }
      activeReplayAudio = null;
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

  function addUserBubble(segId, duration) {
    const wrapper = document.createElement('div');
    wrapper.className = 'chat-row chat-row-user chat-bubble-float';
    wrapper.id = `user-${segId}`;

    const hasAudio = segmentAudio.has(segId);
    wrapper.innerHTML = `
      <div class="chat-bubble chat-bubble-user text-white">
        <div class="flex items-center gap-2">
          <svg class="w-4 h-4 text-white/70" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/>
          </svg>
          <span class="text-sm font-medium tracking-wide">Voice ${duration}</span>
          ${hasAudio ? `<button class="replay-btn" data-seg="${segId}" title="Replay audio">
            <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
              <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z"/>
            </svg>
          </button>` : ''}
        </div>
        <div class="mt-2 flex gap-0.5 items-end h-4">
          ${generateWaveformBars()}
        </div>
      </div>
    `;

    if (hasAudio) {
      wrapper.querySelector('.replay-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        replaySegment(segId, e.currentTarget);
      });
    }

    chatArea.appendChild(wrapper);
    scrollChatToBottom();
  }

  function generateWaveformBars() {
    let bars = '';
    for (let i = 0; i < 20; i++) {
      const h = 4 + Math.random() * 12;
      bars += `<div class="w-1 rounded-full bg-white/50" style="height:${h}px"></div>`;
    }
    return bars;
  }

  function addAIBubble(segId) {
    const wrapper = document.createElement('div');
    wrapper.className = 'chat-row chat-row-ai chat-bubble-float';
    wrapper.id = `ai-${segId}`;

    wrapper.innerHTML = `
      <div class="flex gap-3 max-w-2xl items-start">
        <div class="chat-avatar flex-shrink-0">
          <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/>
          </svg>
        </div>
        <div class="chat-bubble chat-bubble-ai ai-processing text-white/90 ai-content">
          <div class="shimmer-lines">
            <div class="shimmer-line w-48 h-3 mb-2"></div>
            <div class="shimmer-line w-36 h-3 mb-2"></div>
            <div class="shimmer-line w-24 h-3"></div>
          </div>
        </div>
      </div>
    `;

    chatArea.appendChild(wrapper);
    scrollChatToBottom();
  }

  function removeSegmentBubbles(segId) {
    const user = document.getElementById(`user-${segId}`);
    const ai = document.getElementById(`ai-${segId}`);
    const targets = [user, ai].filter((el) => el && el.parentNode);
    if (targets.length === 0) {
      const url = segmentAudio.get(segId);
      if (url) URL.revokeObjectURL(url);
      segmentAudio.delete(segId);
      return;
    }
    let removed = 0;
    targets.forEach((el) => {
      el.classList.add('chat-bubble-discard');
      el.addEventListener('animationend', () => {
        if (el.parentNode) el.parentNode.removeChild(el);
        removed++;
        if (removed >= targets.length) {
          const url = segmentAudio.get(segId);
          if (url) URL.revokeObjectURL(url);
          segmentAudio.delete(segId);
        }
      }, { once: true });
    });
  }

  function renderDualAsrDebug(debugInfo) {
    if (!debugInfo) return '';
    const primary = String(debugInfo.textPrimary || '').trim();
    const secondary = String(debugInfo.textSecondary || '').trim();
    const meta = debugInfo.fusionMeta || null;
    if (!primary && !secondary) return '';

    const selected = meta && meta.selected ? escapeHtml(String(meta.selected)) : '-';
    const reason = meta && meta.reason ? escapeHtml(String(meta.reason)) : '-';
    const similarity =
      meta && typeof meta.similarity === 'number' ? String(meta.similarity) : '-';

    return `
      <div class="mt-3 rounded-lg border border-white/12 bg-black/20 p-2 text-xs text-white/70 space-y-1">
        <div class="text-[11px] text-white/50">DEBUG Dual ASR</div>
        <div><span class="text-white/50">Primary:</span> ${escapeHtml(primary)}</div>
        <div><span class="text-white/50">Secondary:</span> ${escapeHtml(secondary)}</div>
        <div><span class="text-white/50">Selected:</span> ${selected} | <span class="text-white/50">Reason:</span> ${reason} | <span class="text-white/50">Sim:</span> ${similarity}</div>
      </div>
    `;
  }

  function streamRevealContent(container, htmlString, charDelayMs = 12) {
    const temp = document.createElement('div');
    temp.innerHTML = htmlString;
    let idx = 0;

    function wrapTextNodes(node) {
      if (node.nodeType === Node.TEXT_NODE) {
        const text = node.textContent;
        if (!text) return;
        const frag = document.createDocumentFragment();
        for (const ch of text) {
          const span = document.createElement('span');
          span.className = 'stream-char';
          span.style.animationDelay = `${idx * charDelayMs}ms`;
          span.textContent = ch;
          frag.appendChild(span);
          if (ch.trim()) idx++;
        }
        node.parentNode.replaceChild(frag, node);
      } else if (node.nodeType === Node.ELEMENT_NODE) {
        [...node.childNodes].forEach(wrapTextNodes);
      }
    }

    wrapTextNodes(temp);
    container.innerHTML = temp.innerHTML;
  }

  function updateAIBubble(segId, text, status, modelHotwords = null, debugInfo = null) {
    const bubble = document.getElementById(`ai-${segId}`);
    if (!bubble) return;
    const content = bubble.querySelector('.ai-content');
    if (!content) return;

    if (status === 'processing') {
      content.classList.add('ai-processing');
      content.innerHTML = `
        <div class="shimmer-lines">
          <div class="shimmer-line w-48 h-3 mb-2"></div>
          <div class="shimmer-line w-36 h-3 mb-2"></div>
          <div class="shimmer-line w-24 h-3"></div>
        </div>
        <div class="text-xs text-white/40 mt-2">Processing...</div>
      `;
    } else if (status === 'done') {
      content.classList.remove('ai-processing');
      const wordsForHighlight = Array.from(
        new Set([
          ...((Array.isArray(modelHotwords) ? modelHotwords : []).map((w) => String(w || '').trim()).filter(Boolean)),
          ...getEffectiveHotwords(),
        ])
      );
      const highlighted = highlightHotwords(text || '', wordsForHighlight);
      if (highlighted.count > 0) {
        sessionHitCount += highlighted.count;
        updateHitCounter();
      }
      const hitMeta =
        highlighted.count > 0
          ? `<div class="text-[11px] text-sky-200/85 mt-2 stream-meta">Hotword hits: ${highlighted.count}</div>`
          : '';
      const langDetectedMeta =
        debugInfo &&
        debugInfo.srcLangDetected &&
        srcLangUi === 'auto' &&
        String(debugInfo.srcLangDetected).trim()
          ? `<div class="text-[11px] text-violet-200/80 mt-2 stream-meta">Detected language: ${escapeHtml(String(debugInfo.srcLangDetected).trim())}</div>`
          : '';
      const debugBlock = renderDualAsrDebug(debugInfo);

      const textP = document.createElement('p');
      textP.className = 'text-sm leading-relaxed';
      streamRevealContent(textP, highlighted.html);
      content.innerHTML = '';
      content.appendChild(textP);
      if (hitMeta || langDetectedMeta || debugBlock) {
        const extra = document.createElement('div');
        extra.innerHTML = langDetectedMeta + hitMeta + debugBlock;
        content.appendChild(extra);
      }
    } else if (status === 'error') {
      content.classList.remove('ai-processing');
      content.innerHTML = `<p class="text-sm text-red-400">${escapeHtml(text)}</p>`;
    }

    scrollChatToBottom();
  }

  function scrollChatToBottom() {
    requestAnimationFrame(() => {
      chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' });
    });
  }

  // --- Audio capture ---
  async function startRecording() {
    if (isRecording) return;

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: { ideal: 48000 },
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
    } catch (err) {
      alert('Microphone access denied. Please allow microphone access and try again.');
      return;
    }

    audioCtx = new AudioContext({ sampleRate: 48000 });
    await audioCtx.audioWorklet.addModule('audio-processor.js?v=' + Date.now());

    const source = audioCtx.createMediaStreamSource(mediaStream);
    workletNode = new AudioWorkletNode(audioCtx, 'audio-capture-processor');

    workletNode.port.onmessage = (evt) => {
      if (evt.data.type === 'audio' && ws && ws.readyState === WebSocket.OPEN) {
        const float32 = evt.data.samples;
        const int16 = new Int16Array(float32.length);
        for (let i = 0; i < float32.length; i++) {
          const s = Math.max(-1, Math.min(1, float32[i]));
          int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        ws.send(int16.buffer);
      }
    };

    source.connect(workletNode);
    workletNode.connect(audioCtx.destination);

    isRecording = true;
    micBtn.classList.add('recording');
    micIcon.setAttribute('fill', 'currentColor');
    micStatus.textContent = 'Listening...';
    pulseRings.forEach((r) => r.classList.add('active'));
  }

  function stopRecording() {
    if (!isRecording) return;

    if (workletNode) {
      workletNode.disconnect();
      workletNode = null;
    }
    if (audioCtx) {
      audioCtx.close();
      audioCtx = null;
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }

    isRecording = false;
    micBtn.classList.remove('recording');
    micIcon.setAttribute('fill', 'none');
    micStatus.textContent = 'Click to start';
    pulseRings.forEach((r) => r.classList.remove('active'));
  }

  micBtn.addEventListener('click', () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  });

  // --- Utilities ---
  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function escapeRegExp(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  function highlightHotwords(text, candidateHotwords = null) {
    const source = String(text || '');
    const activeSource = Array.isArray(candidateHotwords)
      ? candidateHotwords
      : getEffectiveHotwords();
    const active = activeSource
      .map((w) => w.trim())
      .filter(Boolean);

    if (!source || active.length === 0) {
      return { html: escapeHtml(source), count: 0 };
    }

    const ranges = [];
    active.forEach((word) => {
      const re = new RegExp(escapeRegExp(word), 'gi');
      let match = re.exec(source);
      while (match) {
        ranges.push({
          start: match.index,
          end: match.index + match[0].length,
        });
        match = re.exec(source);
      }
    });

    if (ranges.length === 0) {
      return { html: escapeHtml(source), count: 0 };
    }

    ranges.sort((a, b) => {
      if (a.start !== b.start) return a.start - b.start;
      return b.end - a.end;
    });

    const merged = [];
    ranges.forEach((r) => {
      const last = merged[merged.length - 1];
      if (!last || r.start >= last.end) {
        merged.push(r);
      }
    });

    let html = '';
    let cursor = 0;
    merged.forEach((r) => {
      html += escapeHtml(source.slice(cursor, r.start));
      html += `<mark class="hotword-hit">${escapeHtml(source.slice(r.start, r.end))}</mark>`;
      cursor = r.end;
    });
    html += escapeHtml(source.slice(cursor));

    return { html, count: merged.length };
  }

  // --- Init ---
  connectWS();
})();
