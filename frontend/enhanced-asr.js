(() => {
  'use strict';

  const SAMPLE_RATE = 16000;
  const CLEAN_STREAM_URL = 'wss://amphion.top/asr/v1/clean-stream';

  function initEnhancedAsr() {
    const i18n = window.Amphion && window.Amphion.i18n;
    const t = (key, vars) => (i18n ? i18n.t(key, vars) : (vars && vars.defaultValue) || key);
    const chatArea = document.getElementById('enhanced-chat-area');
    const micBtn = document.getElementById('enhanced-mic-btn');
    const micIcon = document.getElementById('enhanced-mic-icon');
    const micStatus = document.getElementById('enhanced-mic-status');
    const statusPill = document.getElementById('enhanced-status-pill');
    const pulseRings = document.querySelectorAll('.pulse-ring');
    const apiKeyInput = document.getElementById('enhanced-api-key');
    const languageSelect = document.getElementById('enhanced-language');
    const cleanupSelect = document.getElementById('enhanced-cleanup-level');
    const emotionInput = document.getElementById('enhanced-emotion');
    const emotionLabel = document.getElementById('enhanced-emotion-label');
    const translateInput = document.getElementById('enhanced-translate');
    const translateLabel = document.getElementById('enhanced-translate-label');
    const targetRow = document.getElementById('enhanced-target-row');
    const targetSelect = document.getElementById('enhanced-target-language');
    const builtinSelect = document.getElementById('enhanced-builtin-hotwords');
    const customInput = document.getElementById('enhanced-custom-hotwords');
    const controls = [apiKeyInput, languageSelect, cleanupSelect, emotionInput, translateInput, targetSelect, builtinSelect, customInput];

    if (!chatArea || !micBtn) return () => {};

    let disposed = false;
    let state = 'idle';
    let ws = null;
    let audioCtx = null;
    let mediaStream = null;
    let sourceNode = null;
    let workletNode = null;
    let currentResult = null;
    let startedAt = 0;
    let terminalReceived = false;
    let sessionTimer = null;
    const postprocessSegments = new Map();

    function setSidebar(stateName, label) {
      if (window.AmphionSidebar && window.AmphionSidebar.setConnectionState) {
        window.AmphionSidebar.setConnectionState(stateName, label);
      }
    }

    function setUiState(next, messageKey) {
      state = next;
      const stateMap = {
        idle: ['waiting', 'enhanced.status.idle', 'enhanced.mic.start', 'idle'],
        connecting: ['waiting', 'enhanced.status.connecting', 'enhanced.mic.connecting', 'pending'],
        listening: ['ready', 'enhanced.status.listening', 'enhanced.mic.listening', 'listening'],
        finalizing: ['waiting', 'enhanced.status.refining', 'enhanced.mic.refining', 'analyzing'],
        done: ['ready', 'enhanced.status.done', 'enhanced.mic.again', 'ready'],
        error: ['offline', 'enhanced.status.error', 'enhanced.mic.again', 'error'],
      };
      const cfg = stateMap[next] || stateMap.idle;
      statusPill.dataset.state = cfg[0];
      statusPill.textContent = t(cfg[1]);
      micStatus.textContent = messageKey ? t(messageKey) : t(cfg[2]);
      setSidebar(cfg[3]);
      const active = next === 'connecting' || next === 'listening' || next === 'finalizing';
      controls.forEach((el) => { if (el) el.disabled = active; });
      micBtn.disabled = next === 'connecting' || next === 'finalizing';
      micBtn.classList.toggle('recording', next === 'listening');
      micIcon.setAttribute('fill', next === 'listening' ? 'currentColor' : 'none');
      pulseRings.forEach((ring) => ring.classList.toggle('active', next === 'listening'));
    }

    function scrollToBottom() {
      requestAnimationFrame(() => chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' }));
    }

    function createResultCard() {
      const userRow = document.createElement('div');
      userRow.className = 'chat-row chat-row-user chat-bubble-float';
      const userBubble = document.createElement('div');
      userBubble.className = 'chat-bubble chat-bubble-user';
      userBubble.textContent = t('enhanced.recording');
      userRow.appendChild(userBubble);

      const row = document.createElement('div');
      row.className = 'chat-row chat-row-ai chat-bubble-float';
      const wrap = document.createElement('div');
      wrap.className = 'flex gap-3 max-w-2xl items-start enhanced-result-wrap';
      const avatar = document.createElement('div');
      avatar.className = 'chat-avatar flex-shrink-0';
      avatar.innerHTML = '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3a3 3 0 00-3 3v6a3 3 0 006 0V6a3 3 0 00-3-3zM19 11a7 7 0 01-14 0M12 18v3M8 21h8"/></svg>';
      const card = document.createElement('div');
      card.className = 'chat-bubble chat-bubble-ai enhanced-result-card ai-processing';

      const rawSection = document.createElement('section');
      rawSection.className = 'enhanced-result-section';
      const rawLabel = document.createElement('span');
      rawLabel.className = 'enhanced-result-label';
      rawLabel.textContent = t('enhanced.result.raw');
      const rawText = document.createElement('p');
      rawText.className = 'enhanced-result-text is-placeholder';
      rawText.textContent = t('enhanced.result.waiting');
      rawSection.append(rawLabel, rawText);

      const refinedSection = document.createElement('section');
      refinedSection.className = 'enhanced-result-section enhanced-refined-section';
      const refinedLabel = document.createElement('span');
      refinedLabel.className = 'enhanced-result-label';
      refinedLabel.textContent = translateInput.checked ? t('enhanced.result.translation') : t('enhanced.result.refined');
      const refinedText = document.createElement('p');
      refinedText.className = 'enhanced-result-text is-placeholder';
      refinedText.textContent = cleanupSelect.value === 'off' && !translateInput.checked
        ? t('enhanced.result.disabled') : t('enhanced.result.waiting');
      refinedSection.append(refinedLabel, refinedText);

      const emotionRow = document.createElement('div');
      emotionRow.className = 'enhanced-emotion-results';
      emotionRow.hidden = true;
      const meta = document.createElement('div');
      meta.className = 'enhanced-result-meta';
      card.append(rawSection, refinedSection, emotionRow, meta);
      wrap.append(avatar, card);
      row.appendChild(wrap);
      chatArea.append(userRow, row);
      scrollToBottom();
      return { userBubble, card, rawText, refinedText, refinedLabel, refinedSection, emotionRow, meta };
    }

    function showError(message) {
      const row = document.createElement('div');
      row.className = 'chat-row chat-row-ai chat-bubble-float';
      const bubble = document.createElement('div');
      bubble.className = 'chat-bubble enhanced-error-bubble';
      bubble.textContent = message || t('enhanced.error.connection');
      row.appendChild(bubble);
      chatArea.appendChild(row);
      scrollToBottom();
    }

    function parseHotwords() {
      return String(customInput.value || '')
        .split(/[\n,，]+/)
        .map((word) => word.trim())
        .filter(Boolean);
    }

    function toBase64(arrayBuffer) {
      const bytes = new Uint8Array(arrayBuffer);
      let binary = '';
      for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
      return btoa(binary);
    }

    function floatToPcm16(samples) {
      const output = new ArrayBuffer(samples.length * 2);
      const view = new DataView(output);
      for (let i = 0; i < samples.length; i += 1) {
        const value = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(i * 2, value < 0 ? value * 0x8000 : value * 0x7fff, true);
      }
      return output;
    }

    function sendSessionUpdate() {
      const builtin = builtinSelect.value ? [builtinSelect.value] : [];
      ws.send(JSON.stringify({
        type: 'session.update',
        language: languageSelect.value,
        translate_mode: translateInput.checked,
        target_language: translateInput.checked ? targetSelect.value : undefined,
        cleanup: {
          level: cleanupSelect.value,
          text_emotion: emotionInput.checked,
        },
        hotwords: {
          builtin,
          custom: parseHotwords(),
        },
      }));
    }

    function attachAudioSender() {
      if (!workletNode || !ws) return;
      workletNode.port.onmessage = (event) => {
        if (state !== 'listening' || !ws || ws.readyState !== WebSocket.OPEN) return;
        if (!event.data || event.data.type !== 'audio') return;
        ws.send(JSON.stringify({
          type: 'input_audio_buffer.append',
          audio: toBase64(floatToPcm16(event.data.samples)),
        }));
      };
    }

    function stopAudio() {
      if (workletNode) {
        workletNode.port.onmessage = null;
        try { workletNode.disconnect(); } catch (_) { /* noop */ }
        workletNode = null;
      }
      if (sourceNode) {
        try { sourceNode.disconnect(); } catch (_) { /* noop */ }
        sourceNode = null;
      }
      if (mediaStream) {
        mediaStream.getTracks().forEach((track) => track.stop());
        mediaStream = null;
      }
      if (audioCtx) {
        try { audioCtx.close(); } catch (_) { /* noop */ }
        audioCtx = null;
      }
    }

    function closeSocket() {
      if (sessionTimer) clearTimeout(sessionTimer);
      sessionTimer = null;
      if (ws) {
        ws.onopen = null;
        ws.onmessage = null;
        ws.onerror = null;
        ws.onclose = null;
        try { ws.close(); } catch (_) { /* noop */ }
        ws = null;
      }
    }

    function fail(message) {
      stopAudio();
      closeSocket();
      if (currentResult) currentResult.card.classList.remove('ai-processing');
      showError(message);
      setUiState('error');
      syncOptions();
    }

    function finishResult(data) {
      terminalReceived = true;
      if (!currentResult) return;
      const raw = String(data.text || '').trim();
      const finalText = String(data.translated_text || data.cleaned_text || raw).trim();
      currentResult.rawText.textContent = raw || t('enhanced.result.empty');
      currentResult.rawText.classList.remove('is-placeholder');
      if (translateInput.checked || cleanupSelect.value !== 'off') {
        currentResult.refinedText.textContent = finalText || t('enhanced.result.empty');
        currentResult.refinedText.classList.remove('is-placeholder');
      } else {
        currentResult.refinedSection.hidden = true;
      }
      const seconds = data.usage && (data.usage.seconds || data.usage.duration_seconds);
      const pieces = [];
      if (data.language) pieces.push(String(data.language));
      if (seconds != null) pieces.push(`${Number(seconds).toFixed(1)}s`);
      if (data.cleanup_status || data.translation_status) pieces.push(String(data.cleanup_status || data.translation_status));
      currentResult.meta.textContent = pieces.join(' · ');
      currentResult.card.classList.remove('ai-processing');
      stopAudio();
      closeSocket();
      setUiState('done');
      syncOptions();
      scrollToBottom();
    }

    function handleMessage(data) {
      if (!data || !data.type) return;
      if (data.type === 'session.created') {
        sendSessionUpdate();
      } else if (data.type === 'session.waiting') {
        setUiState('connecting', 'enhanced.mic.waiting');
      } else if (data.type === 'session.updated') {
        attachAudioSender();
        setUiState('listening');
        startedAt = Date.now();
      } else if (data.type === 'transcription.delta' && currentResult) {
        currentResult.rawText.textContent = data.text || data.delta || '';
        currentResult.rawText.classList.remove('is-placeholder');
        scrollToBottom();
      } else if (data.type === 'postprocess.delta' && currentResult) {
        if (data.segment_index != null) {
          postprocessSegments.set(Number(data.segment_index), String(data.text || ''));
          currentResult.refinedText.textContent = Array.from(postprocessSegments.keys())
            .sort((a, b) => a - b).map((key) => postprocessSegments.get(key)).join('');
        } else {
          currentResult.refinedText.textContent = data.text || data.delta || '';
        }
        currentResult.refinedText.classList.remove('is-placeholder');
        scrollToBottom();
      } else if (data.type === 'emotion.bucket' && currentResult) {
        const chip = document.createElement('span');
        chip.className = 'enhanced-emotion-chip';
        chip.textContent = (data.emotion && data.emotion.label) || t('enhanced.emotion.unknown');
        currentResult.emotionRow.hidden = false;
        currentResult.emotionRow.appendChild(chip);
        scrollToBottom();
      } else if (data.type === 'transcription.done') {
        finishResult(data);
      } else if (data.type === 'error') {
        terminalReceived = true;
        fail(data.message || data.code || t('enhanced.error.connection'));
      }
    }

    async function startSession() {
      const apiKey = String(apiKeyInput.value || '').trim();
      if (!apiKey) {
        apiKeyInput.focus();
        setUiState('error', 'enhanced.error.keyRequired');
        return;
      }
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        fail(t('asrtest.mic.insecure'));
        return;
      }

      terminalReceived = false;
      postprocessSegments.clear();
      currentResult = createResultCard();
      setUiState('connecting');

      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, sampleRate: { ideal: SAMPLE_RATE }, echoCancellation: true, noiseSuppression: true },
        });
        if (disposed) return;
        audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
        await audioCtx.audioWorklet.addModule('audio-processor.js?v=' + Date.now());
        sourceNode = audioCtx.createMediaStreamSource(mediaStream);
        workletNode = new AudioWorkletNode(audioCtx, 'audio-capture-processor');
        sourceNode.connect(workletNode);
        workletNode.connect(audioCtx.destination);

        const url = new URL(CLEAN_STREAM_URL);
        url.searchParams.set('api_key', apiKey);
        ws = new WebSocket(url.toString());
        ws.onmessage = (event) => {
          try { handleMessage(JSON.parse(event.data)); } catch (_) { /* ignore non-JSON */ }
        };
        ws.onerror = () => {
          if (!terminalReceived && !disposed) fail(t('enhanced.error.connection'));
        };
        ws.onclose = (event) => {
          if (!terminalReceived && !disposed && state !== 'idle') {
            fail(event.code === 4003 ? t('enhanced.error.auth') : t('enhanced.error.closed'));
          }
        };
        sessionTimer = setTimeout(() => {
          if (state === 'connecting') fail(t('enhanced.error.timeout'));
        }, 12000);
      } catch (error) {
        fail(error && error.name === 'NotAllowedError' ? t('asr.mic.alert.denied') : t('enhanced.error.mic'));
      }
    }

    function commitSession() {
      if (state !== 'listening') return;
      const duration = Math.max(0, (Date.now() - startedAt) / 1000);
      currentResult.userBubble.textContent = t('enhanced.recorded', { seconds: duration.toFixed(1) });
      stopAudio();
      setUiState('finalizing');
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'input_audio_buffer.commit', final: true }));
      } else {
        fail(t('enhanced.error.closed'));
      }
    }

    function syncOptions() {
      emotionLabel.textContent = t(emotionInput.checked ? 'enhanced.on' : 'enhanced.off');
      translateLabel.textContent = t(translateInput.checked ? 'enhanced.on' : 'enhanced.off');
      targetRow.hidden = !translateInput.checked;
      cleanupSelect.disabled = translateInput.checked || state === 'connecting' || state === 'listening' || state === 'finalizing';
    }

    micBtn.addEventListener('click', () => {
      if (state === 'listening') commitSession();
      else if (!['connecting', 'finalizing'].includes(state)) startSession();
    });
    emotionInput.addEventListener('change', syncOptions);
    translateInput.addEventListener('change', syncOptions);
    const unsubscribe = i18n ? i18n.onChange(() => {
      setUiState(state);
      syncOptions();
    }) : () => {};
    setUiState('idle');
    syncOptions();

    return () => {
      disposed = true;
      unsubscribe();
      stopAudio();
      closeSocket();
    };
  }

  window.AmphionPages = window.AmphionPages || {};
  window.AmphionPages['enhanced-asr'] = { init: initEnhancedAsr };
})();
