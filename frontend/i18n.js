/**
 * Tiny client-side i18n for the Amphion demos.
 *
 * Exposes window.Amphion.i18n with:
 *   t(key, vars?)     - look up a string in the active locale, with {var}
 *                        interpolation and a graceful fallback to the key
 *                        (or `vars.defaultValue` if provided).
 *   getLang()         - currently active locale ('en' | 'zh').
 *   setLang(lang)     - switch locale, persist, refresh DOM, notify listeners.
 *   onChange(handler) - subscribe to language changes; returns an unsubscribe.
 *   applyTranslations(root?) - rescan a DOM subtree for data-i18n attrs.
 *
 * HTML markup conventions:
 *   data-i18n="some.key"             -> textContent
 *   data-i18n-html="some.key"        -> innerHTML  (use sparingly)
 *   data-i18n-attr-<attr>="some.key" -> set element attribute (e.g. placeholder,
 *                                       aria-label, title)
 *   data-i18n-doc-title="some.key"   -> sets document.title (only honored if the
 *                                       node is the <title> element or any node
 *                                       carrying this attribute)
 *
 * The dictionaries below cover all visible UI labels across the three demo
 * pages and the shared sidebar. Backend-emitted free-form text (model
 * transcripts, error.message) is intentionally NOT translated; only the
 * front-end labels around it.
 */
(() => {
  'use strict';

  const STORAGE_KEY = 'amphion_lang';
  const SUPPORTED = ['en', 'zh'];

  const EN = {
    // ---- Generic / sidebar ----
    'common.idle': 'Idle',
    'common.connected': 'Connected',
    'common.disconnected': 'Disconnected',
    'common.connecting': 'Connecting...',
    'common.listening': 'Listening',
    'common.analyzing': 'Analyzing',
    'common.busy': 'Working',
    'common.error': 'Error',
    'common.offline': 'Offline',

    'sidebar.brand.title': 'Amphion',
    'sidebar.brand.sub': 'Speech Demo',
    'sidebar.lang.aria': 'Language',
    'nav.asr': 'Realtime ASR',
    'nav.emotion': 'Emotion',
    'nav.tsasr': 'Target Speaker',

    // ---- ASR page ----
    'asr.titleTag': 'Amphion Demo',
    'asr.title': 'Realtime ASR',
    'asr.subtitle': "Stream your voice and watch the transcript arrive as you speak.",
    'asr.greeting.html':
      "Ready when you are. Click the microphone to begin speaking and I'll transcribe your audio in real time."
      + '<br/><br/>'
      + '<span class="text-muted">Tip: add hotwords in the right panel to improve recognition on domain-specific terms.</span>',
    'asr.mic.start': 'Click to start',
    'asr.mic.listening': 'Listening...',
    'asr.mic.aria': 'Toggle microphone',
    'asr.mic.alert.denied': 'Microphone access denied. Please allow microphone access and try again.',

    'asr.hotword.title': 'Hotwords',
    'asr.hotword.tip': 'Boost recognition for domain-specific terms.',
    'asr.hotword.lang': 'Language',
    'asr.hotword.langTip': 'Hotwords are saved per language.',
    'asr.hotword.langOption.auto': 'Auto (detect)',
    'asr.hotword.langOption.chinese': 'Chinese',
    'asr.hotword.langOption.english': 'English',
    'asr.hotword.langOption.indonesian': 'Indonesian',
    'asr.hotword.langOption.thai': 'Thai',
    'asr.hotword.langSelect.aria': 'ASR input language',
    'asr.hotword.toggle.title': 'Toggle hotword influence',
    'asr.hotword.toggle.on': 'ON',
    'asr.emotion.title': 'Emotion recognition',
    'asr.emotion.tip': 'Show the speaker\u2019s emotion and tone alongside each finished transcript.',
    'asr.emotion.toggle.title': 'Toggle emotion recognition',
    'asr.emotion.toggle.on': 'ON',
    'asr.emotion.toggle.off': 'OFF',
    'asr.emotion.result.ser': 'Emotion',
    'asr.emotion.result.sec': 'Tone',
    'asr.hotword.placeholder': 'Add hotword (comma-separated for batch)',
    'asr.hotword.add': 'Add',
    'asr.hotword.clear': 'Clear',
    'asr.hotword.textarea.placeholder': 'Paste long text here to extract hotwords with LLM',
    'asr.hotword.extract': 'Extract and Add',
    'asr.hotword.extracting': 'Extracting...',
    'asr.hotword.removeAria': 'Remove hotword',
    'asr.hotword.count': '{n} hotwords',
    'asr.hotword.sessionHits': 'Session hits:',

    'asr.sync.active': 'Active',
    'asr.sync.paused': 'Paused',
    'asr.sync.waiting': 'Waiting',
    'asr.sync.offline': 'Offline',

    'asr.extract.idle': 'Idle',
    'asr.extract.loading': 'Extracting...',
    'asr.extract.added': 'Added {added}/{total}',
    'asr.extract.wsOffline': 'WebSocket offline',
    'asr.extract.pasteFirst': 'Please paste text first',
    'asr.extract.alreadyRunning': 'Extraction already running',
    'asr.extract.connClosed': 'Connection closed',
    'asr.extract.connError': 'Connection error',
    'asr.extract.failed': 'Extract failed',
    'asr.extract.raw': '{msg}',

    'asr.user.speaking': 'Speaking\u2026',
    'asr.user.voice': 'Voice {dur}',
    'asr.user.replayTitle': 'Replay audio',
    'asr.processing': 'Processing...',
    'asr.streamingHint': 'Listening\u2026',
    'asr.errorPrefix': 'Error: {msg}',

    'asr.debug.title': 'DEBUG Dual ASR',
    'asr.debug.primary': 'Primary:',
    'asr.debug.secondary': 'Secondary:',
    'asr.debug.selected': 'Selected:',
    'asr.debug.reason': 'Reason:',
    'asr.debug.sim': 'Sim:',
    'asr.debug.langDetected': 'Detected language: {lang}',
    'asr.debug.hotwordHits': 'Hotword hits: {n}',

    // ---- ASR upload (one-shot REST) ----
    'asr.upload.label': 'Upload audio',
    'asr.upload.uploading': 'Uploading…',
    'asr.upload.aria': 'Upload local audio file',
    'asr.upload.decoding': 'Decoding…',
    'asr.upload.analyzing': 'Analyzing {sec}s clip…',
    'asr.upload.done': 'Done ({elapsed}s)',
    'asr.upload.aborted': 'Upload cancelled',
    'asr.upload.trimmed': 'Trimmed to {max}s (file was {actual}s)',
    'asr.upload.error.decode': 'Could not decode the audio file.',
    'asr.upload.error.empty': 'Audio file is empty.',
    'asr.upload.error.unsupported': 'Audio upload is not supported in this browser.',
    'asr.upload.error.request': 'Upload failed. Please try again.',
    'asr.upload.error.busyRecording': 'Stop recording before uploading a file.',

    // ---- Emotion page ----
    'emotion.titleTag': 'Amphion Emotion Demo',
    'emotion.title': 'Emotion Recognition',
    'emotion.subtitle': 'SER and SEC inference on a full spoken utterance.',
    'emotion.live.title': 'Live emotion inference',
    'emotion.live.tip':
      'Press the microphone to start, speak naturally, then press again to stop.'
      + ' The model uses the full utterance; clips longer than 20 seconds are trimmed'
      + ' to the trailing 20s.',
    'emotion.mode.label': 'Mode',
    'emotion.mode.aria': 'Emotion task mode',
    'emotion.mode.option.ser.html': 'SER &middot; label',
    'emotion.mode.option.sec.html': 'SEC &middot; caption',
    'emotion.mode.tag.ser': 'SER',
    'emotion.mode.tag.sec': 'SEC',
    'emotion.result.placeholder': 'Result will appear here.',
    'emotion.result.connecting': 'Connecting…',
    'emotion.result.opening': 'Opening mic…',
    'emotion.result.speakNow': 'Speak now…',
    'emotion.result.analyzing': 'Analyzing…',
    'emotion.result.empty': '(empty)',
    'emotion.result.unparsed': '(unparsed)',
    'emotion.result.taxonomyHint': 'Taxonomy hint: {label}',
    'emotion.result.raw': 'Raw: {text}',
    'emotion.history.title': 'Recent results',
    'emotion.history.clear': 'Clear',
    'emotion.history.empty': 'No sessions yet.',
    'emotion.labels.note':
      'SER labels: Neutral, Happy, Sad, Angry, Fear, Disgust, Surprise, Other/Complex.'
      + ' SEC returns a free-form caption of emotional tone, tempo, and prosody.',
    'emotion.btn.start': 'Click to start',
    'emotion.btn.recording': 'Listening… click to stop',
    'emotion.btn.analyzing': 'Analyzing…',
    'emotion.btn.connecting': 'Connecting…',
    'emotion.btn.opening': 'Opening…',
    'emotion.btn.aria': 'Toggle emotion recording',
    'emotion.status.idle': 'Idle',
    'emotion.status.ready': 'Ready',
    'emotion.status.connecting': 'Connecting',
    'emotion.status.listening': 'Listening',
    'emotion.status.analyzing': 'Analyzing',
    'emotion.status.done': 'Done',
    'emotion.status.error': 'Error',
    'emotion.status.micErr': 'Mic error',
    'emotion.status.wsErr': 'WS error',
    'emotion.status.closed': 'Closed',
    'emotion.error.mic': 'Microphone error: {msg}',
    'emotion.error.ws': 'WebSocket error: {msg}',
    'emotion.error.wsGeneric': 'WebSocket error.',
    'emotion.error.closedBeforeFinal': 'Connection closed before final result.',
    'emotion.error.connLost': 'Connection lost.',
    'emotion.error.serverPrefix': 'Error: {msg}',
    'emotion.error.unknown': 'unknown error',

    // ---- Emotion upload ----
    'emotion.upload.label': 'Upload audio',
    'emotion.upload.uploading': 'Uploading…',
    'emotion.upload.aria': 'Upload local audio file',
    'emotion.upload.decoding': 'Decoding…',
    'emotion.upload.analyzing': 'Analyzing uploaded audio…',
    'emotion.upload.done': 'Done',
    'emotion.upload.aborted': 'Upload cancelled',
    'emotion.upload.trimmed': 'Trimmed to {max}s (file was {actual}s)',
    'emotion.upload.error.decode': 'Could not decode the audio file.',
    'emotion.upload.error.empty': 'Audio file is empty.',
    'emotion.upload.error.unsupported': 'Audio upload is not supported in this browser.',
    'emotion.upload.error.busy': 'A session is already in progress.',
    'emotion.upload.error.serverPrefix': 'Server error: {msg}',

    // ---- TS-ASR page ----
    'tsasr.titleTag': 'Amphion TS-ASR Demo',
    'tsasr.title': 'Target-Speaker ASR',
    'tsasr.subtitle': 'Enroll a voice once, then transcribe only that speaker amid overlapping audio.',
    'tsasr.greeting':
      'Record a 1-5s clean sample of the target speaker first. Once enrollment is accepted, you can'
      + ' start speaking with any interfering audio in the background and the model will transcribe only'
      + ' the enrolled voice.',
    'tsasr.enroll.title': 'Speaker Enrollment',
    'tsasr.enroll.tip': 'Record 1-5 seconds of the target speaker in a quiet environment.',
    'tsasr.enroll.start': 'Start recording',
    'tsasr.enroll.stop': 'Stop',
    'tsasr.enroll.reset': 'Reset',
    'tsasr.enroll.minMax': 'min 1.0s \u00b7 max 5.0s',
    'tsasr.enroll.notRecorded': 'Not recorded',
    'tsasr.enroll.recording': 'Recording...',
    'tsasr.enroll.ready': 'Ready ({dur}s)',
    'tsasr.enroll.tooShort': 'Too short ({dur}s)',
    'tsasr.enroll.sending': 'Sending enrollment...',
    'tsasr.enroll.micDenied': 'Mic denied',
    'tsasr.enroll.micAlert': 'Microphone access denied. Please allow microphone access.',
    'tsasr.enroll.errorCode': '{code}',
    'tsasr.mic.gateDisabled': 'Complete enrollment to enable',
    'tsasr.mic.start': 'Click to start',
    'tsasr.mic.listening': 'Listening...',
    'tsasr.mic.aria': 'Toggle microphone',
    'tsasr.footer.note':
      'Short-term demo: model = primary ASR endpoint.'
      + ' Prompt and optional switches (hotwords, voice traits) are expected'
      + ' to evolve before a dedicated TS-ASR checkpoint ships.',
    'tsasr.error.serverPrefix': 'Server error [{code}]: {msg}',
    'tsasr.meta.lang': 'language: {lang}',
    'tsasr.replayTitle': 'Replay audio',

    // ---- TS-ASR upload (transcription stage only) ----
    'tsasr.upload.label': 'Upload audio',
    'tsasr.upload.uploading': 'Uploading…',
    'tsasr.upload.aria': 'Upload local audio file (transcription)',
    'tsasr.upload.decoding': 'Decoding…',
    'tsasr.upload.analyzing': 'Analyzing uploaded audio…',
    'tsasr.upload.done': 'Done',
    'tsasr.upload.aborted': 'Upload cancelled',
    'tsasr.upload.trimmed': 'Trimmed to {max}s (file was {actual}s)',
    'tsasr.upload.error.decode': 'Could not decode the audio file.',
    'tsasr.upload.error.empty': 'Audio file is empty.',
    'tsasr.upload.error.unsupported': 'Audio upload is not supported in this browser.',
    'tsasr.upload.error.busy': 'A session is already in progress.',
    'tsasr.upload.error.noEnroll': 'Complete enrollment first.',
    'tsasr.upload.error.serverPrefix': 'Server error: {msg}',

    // ---- TS-ASR enrollment upload (speaker registration via file) ----
    'tsasr.enrollUpload.label': 'Upload enrollment',
    'tsasr.enrollUpload.uploading': 'Uploading…',
    'tsasr.enrollUpload.aria': 'Upload enrollment audio file',
    'tsasr.enrollUpload.decoding': 'Decoding…',
    'tsasr.enrollUpload.done': 'Loaded ({dur}s)',
    'tsasr.enrollUpload.trimmed': 'Trimmed to {max}s (file was {actual}s)',
    'tsasr.enrollUpload.error.decode': 'Could not decode the audio file.',
    'tsasr.enrollUpload.error.empty': 'Audio file is empty.',
    'tsasr.enrollUpload.error.unsupported': 'Audio upload is not supported in this browser.',
    'tsasr.enrollUpload.error.busy': 'A session is already in progress.',
    'tsasr.enrollUpload.error.tooShort': 'File is {dur}s, need at least {min}s.',

    // ---- Fusion enums (frontend lookups; no backend coupling) ----
    'fusion.selected.primary_hotword_hit': 'primary_hotword_hit',
    'fusion.selected.primary_agreement': 'primary_agreement',
    'fusion.selected.primary_hotword_advantage': 'primary_hotword_advantage',
    'fusion.selected.secondary_qwen_fallback': 'secondary_qwen_fallback',
    'fusion.reason.primary_hits_hotword': 'primary_hits_hotword',
    'fusion.reason.primary_hallucination_risk': 'primary_hallucination_risk',
    'fusion.reason.high_similarity_and_primary_valid': 'high_similarity_and_primary_valid',
    'fusion.reason.primary_score_margin': 'primary_score_margin',
    'fusion.reason.primary_not_confident': 'primary_not_confident',

    // Language name lookup (from upstream model output / select values)
    'lang.name.Chinese': 'Chinese',
    'lang.name.English': 'English',
    'lang.name.Indonesian': 'Indonesian',
    'lang.name.Thai': 'Thai',
    'lang.name.zh': 'Chinese',
    'lang.name.en': 'English',
    'lang.name.id': 'Indonesian',
    'lang.name.th': 'Thai',
  };

  const ZH = {
    'common.idle': '空闲',
    'common.connected': '已连接',
    'common.disconnected': '已断开',
    'common.connecting': '连接中…',
    'common.listening': '聆听中',
    'common.analyzing': '分析中',
    'common.busy': '处理中',
    'common.error': '错误',
    'common.offline': '离线',

    'sidebar.brand.title': 'Amphion',
    'sidebar.brand.sub': '语音演示',
    'sidebar.lang.aria': '语言',
    'nav.asr': '实时识别',
    'nav.emotion': '情感识别',
    'nav.tsasr': '目标说话人',

    'asr.titleTag': 'Amphion 演示',
    'asr.title': '实时语音识别',
    'asr.subtitle': '边说边看，转写实时呈现。',
    'asr.greeting.html':
      '准备就绪。点击麦克风开始说话，我会实时转写你的语音。'
      + '<br/><br/>'
      + '<span class="text-muted">提示：在右侧添加热词，可提升专业术语的识别效果。</span>',
    'asr.mic.start': '点击开始',
    'asr.mic.listening': '聆听中…',
    'asr.mic.aria': '切换麦克风',
    'asr.mic.alert.denied': '麦克风权限被拒。请在浏览器中允许麦克风访问后重试。',

    'asr.hotword.title': '热词',
    'asr.hotword.tip': '提升专有名词与领域词的识别准确率。',
    'asr.hotword.lang': '语言',
    'asr.hotword.langTip': '热词按语言分别保存。',
    'asr.hotword.langOption.auto': '自动检测',
    'asr.hotword.langOption.chinese': '中文',
    'asr.hotword.langOption.english': '英文',
    'asr.hotword.langOption.indonesian': '印尼语',
    'asr.hotword.langOption.thai': '泰语',
    'asr.hotword.langSelect.aria': 'ASR 输入语种',
    'asr.hotword.toggle.title': '热词开关',
    'asr.hotword.toggle.on': '开',
    'asr.emotion.title': '情感识别',
    'asr.emotion.tip': '在每条转写结果旁，附上说话人的情绪和语气描述。',
    'asr.emotion.toggle.title': '切换情感识别',
    'asr.emotion.toggle.on': '开',
    'asr.emotion.toggle.off': '关',
    'asr.emotion.result.ser': '情绪',
    'asr.emotion.result.sec': '语气',
    'asr.hotword.placeholder': '添加热词（多个用逗号分隔）',
    'asr.hotword.add': '添加',
    'asr.hotword.clear': '清空',
    'asr.hotword.textarea.placeholder': '在此粘贴长文，使用大模型抽取热词',
    'asr.hotword.extract': '抽取并添加',
    'asr.hotword.extracting': '抽取中…',
    'asr.hotword.removeAria': '删除热词',
    'asr.hotword.count': '共 {n} 个热词',
    'asr.hotword.sessionHits': '本次命中：',

    'asr.sync.active': '生效中',
    'asr.sync.paused': '已暂停',
    'asr.sync.waiting': '等待中',
    'asr.sync.offline': '离线',

    'asr.extract.idle': '空闲',
    'asr.extract.loading': '抽取中…',
    'asr.extract.added': '已添加 {added}/{total}',
    'asr.extract.wsOffline': 'WebSocket 离线',
    'asr.extract.pasteFirst': '请先粘贴文本',
    'asr.extract.alreadyRunning': '抽取任务进行中',
    'asr.extract.connClosed': '连接已关闭',
    'asr.extract.connError': '连接错误',
    'asr.extract.failed': '抽取失败',
    'asr.extract.raw': '{msg}',

    'asr.user.speaking': '说话中…',
    'asr.user.voice': '语音 {dur}',
    'asr.user.replayTitle': '重新播放',
    'asr.processing': '处理中…',
    'asr.streamingHint': '聆听中…',
    'asr.errorPrefix': '错误：{msg}',

    'asr.debug.title': '调试：双路 ASR',
    'asr.debug.primary': '主路：',
    'asr.debug.secondary': '副路：',
    'asr.debug.selected': '采用：',
    'asr.debug.reason': '原因：',
    'asr.debug.sim': '相似度：',
    'asr.debug.langDetected': '检测语种：{lang}',
    'asr.debug.hotwordHits': '热词命中：{n}',

    'asr.upload.label': '上传音频',
    'asr.upload.uploading': '上传中…',
    'asr.upload.aria': '上传本地音频文件',
    'asr.upload.decoding': '解码中…',
    'asr.upload.analyzing': '正在识别 {sec} 秒音频…',
    'asr.upload.done': '完成（耗时 {elapsed}s）',
    'asr.upload.aborted': '上传已取消',
    'asr.upload.trimmed': '已截取至 {max} 秒（原始 {actual}s）',
    'asr.upload.error.decode': '无法解码该音频文件。',
    'asr.upload.error.empty': '音频文件为空。',
    'asr.upload.error.unsupported': '当前浏览器不支持音频上传。',
    'asr.upload.error.request': '上传请求失败，请重试。',
    'asr.upload.error.busyRecording': '请先停止录音再上传文件。',

    'emotion.titleTag': 'Amphion 情感识别演示',
    'emotion.title': '情感识别',
    'emotion.subtitle': '对完整语句进行 SER 与 SEC 推理。',
    'emotion.live.title': '实时情感推理',
    'emotion.live.tip':
      '点击麦克风开始，自然说话后再次点击停止。'
      + '模型基于完整语句推理；超过 20 秒的片段会截取末尾 20 秒。',
    'emotion.mode.label': '模式',
    'emotion.mode.aria': '情感任务模式',
    'emotion.mode.option.ser.html': 'SER &middot; 标签',
    'emotion.mode.option.sec.html': 'SEC &middot; 描述',
    'emotion.mode.tag.ser': 'SER',
    'emotion.mode.tag.sec': 'SEC',
    'emotion.result.placeholder': '结果将显示在这里。',
    'emotion.result.connecting': '连接中…',
    'emotion.result.opening': '正在打开麦克风…',
    'emotion.result.speakNow': '请开始说话…',
    'emotion.result.analyzing': '分析中…',
    'emotion.result.empty': '（空）',
    'emotion.result.unparsed': '（未解析）',
    'emotion.result.taxonomyHint': '类别提示：{label}',
    'emotion.result.raw': '原始：{text}',
    'emotion.history.title': '最近结果',
    'emotion.history.clear': '清空',
    'emotion.history.empty': '暂无记录。',
    'emotion.labels.note':
      'SER 标签：中性、开心、悲伤、愤怒、恐惧、厌恶、惊讶、其他/复合。'
      + 'SEC 返回关于情感语调、节奏与韵律的自由文本描述。',
    'emotion.btn.start': '点击开始',
    'emotion.btn.recording': '聆听中…再次点击停止',
    'emotion.btn.analyzing': '分析中…',
    'emotion.btn.connecting': '连接中…',
    'emotion.btn.opening': '打开中…',
    'emotion.btn.aria': '切换情感录音',
    'emotion.status.idle': '空闲',
    'emotion.status.ready': '就绪',
    'emotion.status.connecting': '连接中',
    'emotion.status.listening': '聆听中',
    'emotion.status.analyzing': '分析中',
    'emotion.status.done': '完成',
    'emotion.status.error': '错误',
    'emotion.status.micErr': '麦克风错误',
    'emotion.status.wsErr': 'WS 错误',
    'emotion.status.closed': '已关闭',
    'emotion.error.mic': '麦克风错误：{msg}',
    'emotion.error.ws': 'WebSocket 错误：{msg}',
    'emotion.error.wsGeneric': 'WebSocket 错误。',
    'emotion.error.closedBeforeFinal': '在收到结果前连接已关闭。',
    'emotion.error.connLost': '连接已断开。',
    'emotion.error.serverPrefix': '错误：{msg}',
    'emotion.error.unknown': '未知错误',

    'emotion.upload.label': '上传音频',
    'emotion.upload.uploading': '上传中…',
    'emotion.upload.aria': '上传本地音频文件',
    'emotion.upload.decoding': '解码中…',
    'emotion.upload.analyzing': '正在分析上传的音频…',
    'emotion.upload.done': '完成',
    'emotion.upload.aborted': '上传已取消',
    'emotion.upload.trimmed': '已截取至 {max} 秒（原始 {actual}s）',
    'emotion.upload.error.decode': '无法解码该音频文件。',
    'emotion.upload.error.empty': '音频文件为空。',
    'emotion.upload.error.unsupported': '当前浏览器不支持音频上传。',
    'emotion.upload.error.busy': '已有任务进行中。',
    'emotion.upload.error.serverPrefix': '服务端错误：{msg}',

    'tsasr.titleTag': 'Amphion 目标说话人识别',
    'tsasr.title': '目标说话人 ASR',
    'tsasr.subtitle': '注册一次目标说话人，即可在重叠音频中只转写该说话人。',
    'tsasr.greeting':
      '请先录制 1-5 秒目标说话人的纯净样本。注册被接受后，即使背景中有其他人声，'
      + '模型也只会转写该说话人。',
    'tsasr.enroll.title': '说话人注册',
    'tsasr.enroll.tip': '在安静环境下录制 1-5 秒目标说话人的语音。',
    'tsasr.enroll.start': '开始录音',
    'tsasr.enroll.stop': '停止',
    'tsasr.enroll.reset': '重置',
    'tsasr.enroll.minMax': '最少 1.0 秒 \u00b7 最多 5.0 秒',
    'tsasr.enroll.notRecorded': '未录制',
    'tsasr.enroll.recording': '录音中…',
    'tsasr.enroll.ready': '就绪（{dur}s）',
    'tsasr.enroll.tooShort': '过短（{dur}s）',
    'tsasr.enroll.sending': '正在发送注册音频…',
    'tsasr.enroll.micDenied': '麦克风被拒',
    'tsasr.enroll.micAlert': '麦克风权限被拒。请在浏览器中允许麦克风访问。',
    'tsasr.enroll.errorCode': '{code}',
    'tsasr.mic.gateDisabled': '完成注册后启用',
    'tsasr.mic.start': '点击开始',
    'tsasr.mic.listening': '聆听中…',
    'tsasr.mic.aria': '切换麦克风',
    'tsasr.footer.note':
      '短期演示：模型 = 主路 ASR 端点。'
      + '后续将引入专用 TS-ASR 模型；提示与可选参数（热词、声音特征）也将随之演进。',
    'tsasr.error.serverPrefix': '服务端错误 [{code}]：{msg}',
    'tsasr.meta.lang': '语种：{lang}',
    'tsasr.replayTitle': '重新播放',

    'tsasr.upload.label': '上传音频',
    'tsasr.upload.uploading': '上传中…',
    'tsasr.upload.aria': '上传本地音频文件（用于转写）',
    'tsasr.upload.decoding': '解码中…',
    'tsasr.upload.analyzing': '正在分析上传的音频…',
    'tsasr.upload.done': '完成',
    'tsasr.upload.aborted': '上传已取消',
    'tsasr.upload.trimmed': '已截取至 {max} 秒（原始 {actual}s）',
    'tsasr.upload.error.decode': '无法解码该音频文件。',
    'tsasr.upload.error.empty': '音频文件为空。',
    'tsasr.upload.error.unsupported': '当前浏览器不支持音频上传。',
    'tsasr.upload.error.busy': '已有任务进行中。',
    'tsasr.upload.error.noEnroll': '请先完成说话人注册。',
    'tsasr.upload.error.serverPrefix': '服务端错误：{msg}',

    'tsasr.enrollUpload.label': '上传注册音频',
    'tsasr.enrollUpload.uploading': '上传中…',
    'tsasr.enrollUpload.aria': '上传说话人注册音频文件',
    'tsasr.enrollUpload.decoding': '解码中…',
    'tsasr.enrollUpload.done': '已加载（{dur}s）',
    'tsasr.enrollUpload.trimmed': '已截取至 {max} 秒（原始 {actual}s）',
    'tsasr.enrollUpload.error.decode': '无法解码该音频文件。',
    'tsasr.enrollUpload.error.empty': '音频文件为空。',
    'tsasr.enrollUpload.error.unsupported': '当前浏览器不支持音频上传。',
    'tsasr.enrollUpload.error.busy': '已有任务进行中。',
    'tsasr.enrollUpload.error.tooShort': '文件时长 {dur} 秒，至少需要 {min} 秒。',

    'fusion.selected.primary_hotword_hit': '主路命中热词',
    'fusion.selected.primary_agreement': '主副路一致',
    'fusion.selected.primary_hotword_advantage': '主路热词优势',
    'fusion.selected.secondary_qwen_fallback': '回退副路（Qwen）',
    'fusion.reason.primary_hits_hotword': '主路命中热词',
    'fusion.reason.primary_hallucination_risk': '主路疑似幻觉',
    'fusion.reason.high_similarity_and_primary_valid': '主副路高相似且主路可用',
    'fusion.reason.primary_score_margin': '主路得分占优',
    'fusion.reason.primary_not_confident': '主路置信不足',

    'lang.name.Chinese': '中文',
    'lang.name.English': '英文',
    'lang.name.Indonesian': '印尼语',
    'lang.name.Thai': '泰语',
    'lang.name.zh': '中文',
    'lang.name.en': '英文',
    'lang.name.id': '印尼语',
    'lang.name.th': '泰语',
  };

  const DICTS = { en: EN, zh: ZH };

  function detectInitialLang() {
    let stored = null;
    try {
      stored = localStorage.getItem(STORAGE_KEY);
    } catch (_) {
      stored = null;
    }
    if (stored && SUPPORTED.includes(stored)) return stored;
    const nav = (typeof navigator !== 'undefined'
      && (navigator.language || navigator.userLanguage)) || '';
    return nav.toLowerCase().startsWith('zh') ? 'zh' : 'en';
  }

  let currentLang = detectInitialLang();
  const listeners = new Set();

  function interpolate(template, vars) {
    if (!vars) return template;
    return String(template).replace(/\{(\w+)\}/g, (m, key) =>
      Object.prototype.hasOwnProperty.call(vars, key) ? String(vars[key]) : m
    );
  }

  function t(key, vars) {
    const dict = DICTS[currentLang] || EN;
    let value = dict[key];
    if (value == null) {
      value = EN[key];
    }
    if (value == null) {
      if (vars && Object.prototype.hasOwnProperty.call(vars, 'defaultValue')) {
        value = vars.defaultValue;
      } else {
        value = key;
      }
    }
    return interpolate(value, vars);
  }

  function applyTranslations(root) {
    const scope = root || document;

    scope.querySelectorAll('[data-i18n]').forEach((el) => {
      const key = el.getAttribute('data-i18n');
      if (!key) return;
      el.textContent = t(key);
    });

    scope.querySelectorAll('[data-i18n-html]').forEach((el) => {
      const key = el.getAttribute('data-i18n-html');
      if (!key) return;
      el.innerHTML = t(key);
    });

    scope.querySelectorAll('*').forEach((el) => {
      if (!el.attributes) return;
      for (let i = 0; i < el.attributes.length; i++) {
        const attr = el.attributes[i];
        if (!attr.name.startsWith('data-i18n-attr-')) continue;
        const targetAttr = attr.name.slice('data-i18n-attr-'.length);
        const key = attr.value;
        if (!key) continue;
        el.setAttribute(targetAttr, t(key));
      }
    });

    const titleNode = scope.querySelector('[data-i18n-doc-title]');
    if (titleNode) {
      const key = titleNode.getAttribute('data-i18n-doc-title');
      if (key) document.title = t(key);
    }

    if (document.documentElement) {
      document.documentElement.lang = currentLang === 'zh' ? 'zh-CN' : 'en';
    }
  }

  function setLang(lang) {
    if (!SUPPORTED.includes(lang)) return;
    if (lang === currentLang) return;
    currentLang = lang;
    try {
      localStorage.setItem(STORAGE_KEY, lang);
    } catch (_) {
      /* ignore */
    }
    applyTranslations(document);
    listeners.forEach((fn) => {
      try { fn(currentLang); } catch (_) { /* ignore */ }
    });
  }

  function getLang() {
    return currentLang;
  }

  function onChange(fn) {
    if (typeof fn !== 'function') return () => {};
    listeners.add(fn);
    return () => listeners.delete(fn);
  }

  function ready() {
    applyTranslations(document);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', ready, { once: true });
  } else {
    ready();
  }

  window.Amphion = window.Amphion || {};
  window.Amphion.i18n = {
    t,
    getLang,
    setLang,
    onChange,
    applyTranslations,
    SUPPORTED,
  };
})();
