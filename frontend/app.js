(() => {
  'use strict';

  // --- State ---
  let ws = null;
  let audioCtx = null;
  let workletNode = null;
  let mediaStream = null;
  let isRecording = false;
  let hotwords = JSON.parse(localStorage.getItem('hotwords') || '[]');

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
  const hotwordToggle = document.getElementById('hotword-toggle');
  const hotwordPanel = document.getElementById('hotword-panel');

  // --- Hotword management ---
  function renderHotwords() {
    hotwordList.innerHTML = '';
    hotwords.forEach((word, idx) => {
      const tag = document.createElement('span');
      tag.className =
        'inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm ' +
        'bg-white/10 text-white/90 border border-white/20 backdrop-blur-sm';
      tag.innerHTML =
        `<span>${escapeHtml(word)}</span>` +
        `<button class="hover:text-red-400 transition-colors text-white/50 ml-0.5" data-idx="${idx}">&times;</button>`;
      tag.querySelector('button').addEventListener('click', () => removeHotword(idx));
      hotwordList.appendChild(tag);
    });
  }

  function saveAndSyncHotwords() {
    localStorage.setItem('hotwords', JSON.stringify(hotwords));
    renderHotwords();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'update_hotwords', hotwords }));
    }
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

  hotwordToggle.addEventListener('click', () => {
    hotwordPanel.classList.toggle('hidden');
    hotwordToggle.querySelector('.toggle-arrow').classList.toggle('rotate-180');
  });

  renderHotwords();

  // --- Connection status ---
  function setConnected(connected) {
    if (connected) {
      connDot.className = 'w-2.5 h-2.5 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)]';
      connLabel.textContent = 'Connected';
    } else {
      connDot.className = 'w-2.5 h-2.5 rounded-full bg-red-400 shadow-[0_0_8px_rgba(248,113,113,0.6)]';
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
      if (hotwords.length > 0) {
        ws.send(JSON.stringify({ type: 'update_hotwords', hotwords }));
      }
    };

    ws.onclose = () => {
      setConnected(false);
      stopRecording();
      setTimeout(connectWS, 2000);
    };

    ws.onerror = () => {
      setConnected(false);
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
          addUserBubble(data.id, data.duration || '');
          addAIBubble(data.id);
        }
        break;
      case 'status':
        updateAIBubble(data.id, null, 'processing');
        break;
      case 'response':
        updateAIBubble(data.id, data.text, 'done');
        break;
      case 'error':
        updateAIBubble(data.id, `Error: ${data.message}`, 'error');
        break;
    }
  }

  // --- Chat bubbles ---
  function addUserBubble(segId, duration) {
    const wrapper = document.createElement('div');
    wrapper.className = 'flex justify-end mb-4 animate-fade-in';
    wrapper.id = `user-${segId}`;

    wrapper.innerHTML = `
      <div class="max-w-xs px-4 py-3 rounded-2xl rounded-tr-md
                  bg-gradient-to-br from-indigo-500/80 to-purple-600/80
                  backdrop-blur-md border border-white/10 text-white shadow-lg">
        <div class="flex items-center gap-2">
          <svg class="w-4 h-4 text-white/70" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/>
          </svg>
          <span class="text-sm font-medium">Voice ${duration}</span>
        </div>
        <div class="mt-1.5 flex gap-0.5 items-end h-4">
          ${generateWaveformBars()}
        </div>
      </div>
    `;

    chatArea.appendChild(wrapper);
    chatArea.scrollTop = chatArea.scrollHeight;
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
    wrapper.className = 'flex justify-start mb-4 animate-fade-in';
    wrapper.id = `ai-${segId}`;

    wrapper.innerHTML = `
      <div class="flex gap-3 max-w-md">
        <div class="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-cyan-400 to-blue-500
                    flex items-center justify-center shadow-lg shadow-cyan-500/20">
          <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/>
          </svg>
        </div>
        <div class="px-4 py-3 rounded-2xl rounded-tl-md
                    bg-white/10 backdrop-blur-md border border-white/10
                    text-white/90 shadow-lg ai-content">
          <div class="shimmer-lines">
            <div class="shimmer-line w-48 h-3 mb-2"></div>
            <div class="shimmer-line w-36 h-3 mb-2"></div>
            <div class="shimmer-line w-24 h-3"></div>
          </div>
        </div>
      </div>
    `;

    chatArea.appendChild(wrapper);
    chatArea.scrollTop = chatArea.scrollHeight;
  }

  function updateAIBubble(segId, text, status) {
    const bubble = document.getElementById(`ai-${segId}`);
    if (!bubble) return;
    const content = bubble.querySelector('.ai-content');
    if (!content) return;

    if (status === 'processing') {
      content.innerHTML = `
        <div class="shimmer-lines">
          <div class="shimmer-line w-48 h-3 mb-2"></div>
          <div class="shimmer-line w-36 h-3 mb-2"></div>
          <div class="shimmer-line w-24 h-3"></div>
        </div>
        <div class="text-xs text-white/40 mt-2">Processing...</div>
      `;
    } else if (status === 'done') {
      content.innerHTML = `<p class="text-sm leading-relaxed typewriter">${escapeHtml(text)}</p>`;
    } else if (status === 'error') {
      content.innerHTML = `<p class="text-sm text-red-400">${escapeHtml(text)}</p>`;
    }

    chatArea.scrollTop = chatArea.scrollHeight;
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
    await audioCtx.audioWorklet.addModule('audio-processor.js');

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

  // --- Init ---
  connectWS();
})();
