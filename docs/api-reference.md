# AudioLLM API Reference

本文档面向外部系统集成方，说明如何远程调用 AudioLLM 服务完成语音转写和情感识别测试。

## 基础信息

| 项目 | 说明 |
|---|---|
| Base URL | `http://172.16.0.3:8082`（systemd 生产部署） |
| WebSocket Base URL | `ws://172.16.0.3:8082` |
| 鉴权 | 当前服务不要求 API Key、Token 或自定义请求头 |
| 音频格式 | WebSocket 发送原始 PCM：16 kHz、mono、signed 16-bit little-endian |
| REST 上传 | 使用 multipart/form-data 上传 WAV 文件，服务端会解码为 16 kHz mono |
| 默认端口 | systemd 生产部署 `8082`（HTTP）；`start.sh` 开发启动为 `8443`（HTTPS） |

生产环境如需访问控制、限流或 IP 白名单，应在 API 网关、负载均衡或反向代理层配置。

## 接口总览

### WebSocket 任务接口

| 接口 | 任务 | 适用场景 | 结果消息 |
|---|---|---|---|
| `/transcribe-streaming` | 通用流式 ASR | 实时语音转写、Triton 热词召回转写 | `partial` / `partial_asr`、`final` / `final_asr` |
| `/emotion-segmented-streaming` | 分段情感识别 | 长连接中按 VAD 语音段持续返回情感 | 多条 `final_emotion` |
| `/tuling/ast/v3` | 通用流式 ASR（讯飞图灵 AST v3 协议） | 对接讯飞 tuling-ast-sdk 或按 AST v3 信封集成 | `payload.result` 词图（msgtype sentence / Progressive） |
| `/astv3-test-proxy` | AST v3 同源代理（测试用） | 仅供 HTTPS 前端规避 mixed content，透明转发到写死的远程 AST v3 后端 | 同 `/tuling/ast/v3`（透明转发） |

`/transcribe-streaming` 的 `final` / `final_asr` 消息除文本外会带当前语音分段的 `audio_b64`（WAV base64）、`duration_sec` 和 `effective_hotwords`（本段音频经 RAG-ASR/Triton 实际召回的热词列表，不含临时请求热词），主前端用音频字段做分段回放、可用 `effective_hotwords` 展示本段召回命中。k2 模式下该音频是同一段送入 LLM ASR 的 k2 段缓冲，不再经过本地 VAD 段首/段尾二次裁剪；完整字段见 [实时转写 WebSocket 协议](protocols/transcribe-streaming-protocol.md)。服务端开启 `debug_dump_enabled`（`defaults.debug`，运维级、不在客户端覆写白名单）后，`ready` 带 `session_id`/`dump_dir`、`final` 带 `dump_id`，并把每段音频+元信息落盘到 `<dump_dir>/<session_id>/<seg_id>.{wav,json}`，前端在气泡上显示可复制的 `dump_id`，用于回放/最终结果对账，详见协议文档“调试落盘”小节。

`/tuling/ast/v3` 与上面两个任务接口的线上协议不同：音频以 base64 放在 JSON 帧，`header.status`（0/1/2）驱动状态机，无 `ready`/`start`/`stop`，结果为词图结构。模型组合上也不同：本端点恒为 primary-only（强制关闭副模型/本地 Qwen/融合，客户端无法经 `parameter.asr_config` 重开），主模型由 `astv3_vllm_*` 指定（当前留空，回退全局 primary `vllm_base_url`），而 `/transcribe-streaming` 仍按 `config.yaml` 走双模型。临时热词放 `payload.text.text`；热词池隔离只用首帧 `parameter.asr_config.hotword_pool_id`；目标说话人先经 `POST /api/asr/enrollment` 注册，再在首帧设置 `enable_role_separation=false`、`enrollment_enable=true` 和 `enrollment_id`。`header.resIdList` 仅记录并忽略。它不遵循下文“WebSocket 调用流程”，详见 [实时转写 AST v3 WebSocket](protocols/tuling-ast-v3-protocol.md)。

`/astv3-test-proxy` 是为「实时语音识别（测试用）」前端页面临时搭的同源 WebSocket 代理。该页经 HTTPS 提供，浏览器 mixed-content 策略禁止它直接打开明文 `ws://` 的远程 AST v3 后端；由后端在同源 `wss://`（经反向代理）接入后，把每一帧原样双向转发到写死的上游 `ws://159.138.9.106:18082/tuling/ast/v3`。它不解析 AST v3 信封，线上协议与 `/tuling/ast/v3` 完全一致（见 [实时转写 AST v3 WebSocket](protocols/tuling-ast-v3-protocol.md)）；上游连接失败时服务端以 close code 1011 关闭连接。临时测试设施：上游地址写死、前端不暴露任何可选项，外部集成请直接使用 `/tuling/ast/v3`。

### REST 上传接口

| 方法 | 路径 | 任务 | 表单字段 |
|---|---|---|---|
| POST | `/api/asr/upload` | 上传整段音频做 ASR（短音频，尾截 60 秒） | `audio`、`language`、`hotwords`、`hotword_pool_id`、`enrollment_id` |
| POST | `/api/asr/transcriptions` | 异步长音频离线转写（202 + 轮询，会议纪要场景） | `audio`、`language`、`hotwords`、`hotword_pool_id` |
| GET | `/api/asr/transcriptions/{job_id}` | 查询转写任务状态、进度与分段结果 | — |
| POST | `/api/asr/enrollment` | 上传目标说话人音频（1-8 秒）注册 | `audio` |
| GET | `/api/asr/enrollment/{enrollment_id}` | 查询声纹 ID 是否可用于后续 ASR | — |
| DELETE | `/api/asr/enrollment/{enrollment_id}` | 删除注册音频 | — |
| GET | `/api/asr/hotword-pool` | 查询热词池 | `hotword_pool_id`、`query`、`limit`、`offset` |
| POST | `/api/asr/hotword-pool` | 向热词池添加热词 | JSON `hotword_pool_id`、`hotwords` |
| DELETE | `/api/asr/hotword-pool` | 从热词池删除热词 | JSON `hotword_pool_id`、`hotwords` |
| POST | `/api/asr/hotword-pool/delete` | 从热词池删除热词，兼容不稳定支持 DELETE body 的客户端 | JSON `hotword_pool_id`、`hotwords` |
| POST | `/api/asr/hotword-pool/clear` | 清空指定热词池 | JSON 或 query `hotword_pool_id` |
| POST | `/api/asr/hotword-pool/reload` | 让 RAG-ASR 从热词池文件 reload 热词 | JSON 或 query `hotword_pool_id` |
| POST | `/api/emotion/jobs` | 异步整段情感识别（202 + 轮询） | `audio`、`mode`、`language` |
| GET | `/api/emotion/jobs/{job_id}` | 查询情感任务状态与结果 | — |
| POST | `/api/audio/analyze` | 非实时聚合分析：ASR 原始结果、文本清洗、情感标签和情感描述 | `audio`、`language`、`hotwords`、`enrollment_id` |

## WebSocket 调用流程

所有任务型 WebSocket 接口共享同一条基本流程：

1. 连接 `ws://172.16.0.3:8082/<endpoint>`。
2. 等待服务端发送 `{"type":"ready"}`。
3. 发送一条 `start` JSON 消息，声明音频格式和任务参数。
4. 持续发送二进制 PCM 音频帧。
5. 接收中间结果或最终结果。
6. 发送 `{"type":"stop"}` 结束本次音频输入。
7. 等待服务端处理尾部音频并返回最终结果，然后关闭连接。

推荐每帧 30-80 ms PCM。16 kHz、mono、s16le 的字节数计算为：

```text
bytes_per_ms = 16000 * 1 * 2 / 1000 = 32
80 ms chunk = 2560 bytes
```

### 通用 start 消息

```json
{
  "type": "start",
  "format": "pcm_s16le",
  "sample_rate_hz": 16000,
  "channels": 1
}
```

各任务可以在此基础上增加字段，例如 ASR 的 `language` / `hotword_pool_id` / `hotwords` / `enrollment_id`、情感识别的 `mode`。`hotword_pool_id` 是热词池隔离 ID，默认 `default`；`hotwords` 是临时请求热词字段，当前 ASR 偏置来自该热词池召回。`/transcribe-streaming` 携带 `enrollment_id` 时会切换为目标说话人模式，详见 [通用流式 ASR WebSocket](protocols/transcribe-streaming-protocol.md)。

### 临时配置覆写

参数取值优先级（后者覆盖前者）：`backend/config.py` 内置默认 → `config.yaml` 服务端默认（实际生效默认值，重启生效）→ 客户端临时覆写（仅当前连接生效、不落盘）。`config.py` 内置默认与 `config.yaml` 不一致时以 `config.yaml` 为准，内置默认仅为文件缺字段时的兜底。

客户端临时覆写对任务型 WebSocket 端点统一生效，承载位置不同：`/transcribe-streaming` 与 `/emotion-segmented-streaming` 用 `start.config`，`/tuling/ast/v3` 用首帧 `parameter.asr_config`（见 [实时转写 AST v3 WebSocket](protocols/tuling-ast-v3-protocol.md)）。两者都只接受扁平字段名（与 `config.yaml` 是否分组无关）。

覆写字段受服务端白名单（`backend/config.py` 的 `CLIENT_OVERRIDABLE_FIELDS`）约束：只放调参类字段；模型地址（`*_vllm_base_url`，避免 SSRF）、模型 prompt 模板（`*_prompt_template`）、密钥（`text_cleanup_api_key*`）、连接池与任务队列等进程级基础设施字段不可覆写。白名单外字段、未知字段与非法值都会被忽略并保持服务端默认，不会中断连接。完整白名单按类别如下：

| 类别 | 字段 |
|---|---|
| VAD / 分段 | vad_threshold、silence_duration_ms、vad_smoothing_alpha、vad_start_frames、vad_pre_speech_ms、vad_keep_tail_ms、min_segment_duration_ms |
| 伪流式 | enable_pseudo_stream、pseudo_stream_interval_ms、pseudo_stream_first_partial_ms |
| ASR 模型组合 / 超时 | enable_primary_asr、enable_secondary_asr、enable_dual_asr_fusion、primary_asr_timeout、asr_request_timeout、debug_show_dual_asr |
| AST v3 协议字段 | enable_role_separation（AST v3 下不作为普通 Config 覆写，而是参与声纹矩阵与 `cw[].rl` 出参） |
| 融合阈值 | fusion_similarity_threshold、fusion_min_primary_score、fusion_max_repetition_ratio、fusion_disagreement_threshold、fusion_hotword_boost、fusion_primary_score_margin |
| 热词召回 | enable_hotword_recall、recall_top_k |
| TS-ASR | asr_enrollment_min_sec、asr_enrollment_max_sec、asr_enrollment_ttl_sec |
| 情感（仅情感端点有效） | emotion_task_mode、emotion_request_timeout、emotion_max_audio_seconds、emotion_spec_task_mode、emotion_spec_request_timeout、emotion_spec_max_audio_seconds |

`pseudo_stream_first_partial_ms` 是每段语音首个 partial（伪流式中间结果）的触发门槛，只对会输出 partial 的端点生效：`/transcribe-streaming` 与 `/tuling/ast/v3`。`/emotion-segmented-streaming` 不产 partial（服务端固定关闭），传入无效。它与 `vad_start_frames` 一起按 max 决定本地伪流式首字延迟；调低只让首字更早出，不改变 final 段的短噪声过滤（仍由 `min_segment_duration_ms` 控制，不变量 `pseudo_stream_first_partial_ms ≤ min_segment_duration_ms`）。

服务端可启用 `k2_enabled=true` 让 `/transcribe-streaming` 与 `/tuling/ast/v3` 的 partial 改由外部 k2 gRPC 流式 ASR 产生；final 仍走本服务 LLM ASR。k2 只做纯识别，不接热词、不接目标说话人、不返回 token timestamps。`k2_target`、`k2_max_segment_sec`、`k2_idle_keep_ms`、`k2_voice_gate_*` 等均为服务端配置，不在临时覆写白名单内。k2 模式下，切段权威是 k2 endpoint，本服务只用 `k2_idle_keep_ms` 限制起音前旧静音、用 `k2_max_segment_sec` 防止无 endpoint 时缓冲无限增长，并用 `k2_voice_gate_*` 在 partial/final 进入下游前确认有人声证据；voice gate 只决定放行或丢弃，不再用本地 VAD 裁剪段首/段尾。上表 VAD / 伪流式间隔字段仍会被接受，但不再决定这两个端点的切点或首字时机；`enable_pseudo_stream=false` 仍会抑制 partial 下发。

final 文本规范化开关（enable_asr_itn、asr_itn_enable_0_to_9、enable_asr_plate_normalize）与解码退化重复折叠开关（enable_asr_repetition_fix）为服务端配置，不在上表白名单内，客户端无法临时覆写。语义与示例见各协议文档的“文本规范化”小节与 [README 文本规范化](../README.md)。

`enable_role_separation` 在 `/tuling/ast/v3` 中是协议字段：默认 true，省略等价于开启，并且优先级高于 `enrollment_enable/enrollment_id`。当前版本不做真实角色分离；开启时 sentence 的 `cw[].rl` 固定返回 0 作为兼容占位，关闭时 sentence/Progressive 均不返回 `cw[].rl`。完整声纹矩阵见 [实时转写 AST v3 WebSocket](protocols/tuling-ast-v3-protocol.md)。

ASR 模型组合开关的语义矩阵（`enable_dual_asr_fusion=true` 但 `enable_secondary_asr=false` 会在 load 时自动降级为 false）：

| enable_secondary_asr | enable_dual_asr_fusion | Partial（WS） | Final（WS / REST） |
|---|---|---|---|
| true | true | 双调 + 副模型静音门 + 发主模型文本 | 双调 + 融合矫正 |
| true | false | 双调 + 副模型静音门 + 发主模型文本 | 仅主模型（REST 上传也跳过副模型） |
| false | (自动 false) | 仅主模型、无静音门 | 仅主模型 |

示例：

```json
{
  "type": "start",
  "format": "pcm_s16le",
  "sample_rate_hz": 16000,
  "channels": 1,
  "config": {
    "vad_threshold": 0.45,
    "min_segment_duration_ms": 500
  }
}
```

`/tuling/ast/v3` 没有 `start` 消息，等价覆写是把上面 `config` 里的字段放进首帧 `parameter.asr_config`（另可选 `language`），详见 [实时转写 AST v3 WebSocket](protocols/tuling-ast-v3-protocol.md)。

## REST 上传调用

REST 接口适合离线测试或一次性上传完整录音。请求使用 `multipart/form-data`，音频字段名固定为 `audio`。

### 非实时聚合分析

`POST /api/audio/analyze` 会对同一段音频执行 ASR、文本清洗和情感理解。情感理解默认同时返回分类标签和文本描述。

`hotwords` 只传给 ASR 模型，用于 ASR 原生热词识别；文本清洗阶段不会接收热词，也不会根据热词做事后替换。

```bash
python docs/examples/rest_upload.py analyze sample.wav \
  --base-url http://172.16.0.3:8082 \
  --language zh \
  --hotwords "挚音科技,张硕"
```

响应示例：

```json
{
  "type": "audio_analysis",
  "duration_sec": 8.24,
  "language": "zh",
  "hotwords": ["挚音科技", "张硕"],
  "asr": {
    "text": "原始融合转写文本",
    "language": "zh"
  },
  "cleaned_asr": {
    "text": "清洗后的转写文本。"
  },
  "emotion": {
    "type": "final_emotion_pair",
    "mode": "both",
    "ser": {
      "type": "final_emotion",
      "mode": "ser",
      "label": "Neutral",
      "text": "Neutral",
      "duration_sec": 8.24
    },
    "sec": {
      "type": "final_emotion",
      "mode": "sec",
      "label": "Neutral",
      "text": "The speaker sounds calm and neutral.",
      "duration_sec": 8.24
    }
  }
}
```

### ASR 上传

```bash
python docs/examples/rest_upload.py asr sample.wav \
  --base-url http://172.16.0.3:8082 \
  --language zh \
  --hotwords "挚音科技,张硕"
```

响应示例：

```json
{
  "type": "final",
  "text": "你好，欢迎使用语音识别服务。",
  "language": "zh",
  "duration_sec": 3.42,
  "effective_hotwords": ["挚音科技", "张硕"],
  "enrollment_used": false
}
```

`effective_hotwords` 是该音频经 RAG-ASR/Triton 实际召回的热词列表，不包含本次请求临时传入的 `hotwords` 追加部分；召回关闭、失败或无结果时为空数组。`text`（流式 `final` 与上传响应一致）默认已做逆文本规范化（ITN，仅中文）与车牌规范化：`六五四三八`→`65438`、`辽b二四五零七`→`辽B24507`；`partial`/中间结果保持口语形式。省份简称被声学误识别成字母（`冀`→`J`）属识别错误，后处理只修数字/字母、不还原省份字。开关 `enable_asr_itn`、`asr_itn_enable_0_to_9`、`enable_asr_plate_normalize` 为服务端 `config.yaml` 配置（`defaults.itn` 分组），不在客户端覆写白名单内；详见各协议文档的“文本规范化”小节。

如需让模型只转写指定说话人的话，先用 `POST /api/asr/enrollment` 上传 1-8 秒目标人语音、拿到 `enrollment_id`，再把它作为表单字段附加到 `/api/asr/upload`，响应里的 `enrollment_used` 会变为 `true`。注册字段、错误码与生命周期见下文“目标说话人注册”。

### 长音频离线转写（会议纪要）

`POST /api/asr/transcriptions` 面向整段会议录音等长音频（默认上限 3 小时 / 512 MB，超时长直接 400 拒绝而非截断）。服务端先按与流式端点相同的 VAD 状态机把录音切成语音段（切段停顿阈值可经 `transcribe_silence_duration_ms` 独立调参、不影响实时端点；连续无停顿语音超过 `transcribe_max_segment_sec` 会强制切分），再对每段并行执行与 `/api/asr/upload` 相同的双模型转写（含 ITN / 车牌规范化），最后按时间序拼出全文。表单字段为 `audio`（WAV）、`language`、`hotwords`；不支持 `enrollment_id`（目标说话人过滤与多人会议语义相反）。

```bash
curl -X POST http://172.16.0.3:8082/api/asr/transcriptions \
  -F "audio=@meeting.wav" \
  -F "language=zh" \
  -F "hotwords=挚音科技,张硕"
```

受理后轮询 `GET /api/asr/transcriptions/{job_id}` 取进度与结果：

```json
{
  "job_id": "tr_6f0c2a8e9b3d41a7c5e21f08",
  "status": "succeeded",
  "progress": { "segments_total": 636, "segments_done": 636 },
  "result": {
    "type": "transcription",
    "language": "zh",
    "duration_sec": 1949.076,
    "failed_segments": 0,
    "full_text": "师傅好啊，师傅好啊！\n009，我是。\n…",
    "segments": [
      { "id": 0, "start_ms": 21400, "end_ms": 22300, "text": "师傅好啊，师傅好啊！", "language": "zh" }
    ]
  }
}
```

`segments[*].start_ms` / `end_ms` 为段级近似时间戳（非词级对齐）。单段失败重试一次后以 `error` 占位、不拖垮整个任务；结果内存保留 `transcribe_job_ttl_sec`（默认 1 小时）。完整的请求/响应字段表、状态机、部分失败语义、错误码、`config.yaml` 调参（`defaults.transcribe` 分组）与切段停顿调参建议见 [长音频离线转写 API](api/transcription-jobs-api.md)，命令行客户端见 `docs/examples/http_transcribe_job.py`。

### 目标说话人注册

`POST /api/asr/enrollment` 上传一段目标说话人音频，返回不透明的 `enrollment_id` 供后续请求复用：`/transcribe-streaming` 放进 `start.enrollment_id`、`/tuling/ast/v3` 需在首帧同时设置 `parameter.asr_config.enable_role_separation=false`、`enrollment_enable=true` 和 `enrollment_id`、REST 的 `/api/asr/upload` 与 `/api/audio/analyze` 作为表单字段 `enrollment_id`。

```bash
curl -X POST http://172.16.0.3:8082/api/asr/enrollment \
  -F "audio=@speaker_enroll.wav"
# 也支持 speaker_enroll.mp3；raw PCM 请按 16 kHz mono s16le 上传并使用 .pcm 后缀。
```

响应：

```json
{
  "enrollment_id": "ule8QilVjZql30Q9oy9kiQ",
  "duration_sec": 3.0
}
```

#### 请求约束

音频支持 WAV、MP3 和 raw PCM。WAV/MP3 会解码并规范化为 16 kHz mono，其中 MP3 解码依赖服务运行环境可执行 `ffmpeg`；raw PCM 必须是 16 kHz mono s16le，服务端按文件后缀 `.pcm`/`.raw` 或 `audio/pcm` 等 content type 识别，不从裸字节猜测。短于 `asr_enrollment_min_sec`（默认 1.0 秒）返回 400 且 `detail.code=too_short`；长于 `asr_enrollment_max_sec`（默认 8.0 秒）不拒绝，服务端尾截到上限。上传体为空或解码后无音频返回 `detail.code=empty`，格式不在支持范围返回 `detail.code=unsupported_format`，容器损坏或解码失败返回 `detail.code=decode_failed`。

#### 生命周期

生命周期决定何时需要重新注册，集成方必读：

| 项目 | 行为 |
|---|---|
| 存储 | 默认 `enable_triton_enrollment_store=false` 时为 demo 进程内内存缓存，服务重启全部失效，不跨实例共享；打开后，新注册音频会转发到 RAG-ASR 管理服务，由 RAG-ASR 将 projector frames tensor 与 JSON 元数据落盘到 `enrollment_store_dir/<enrollment_scope_id>/<enrollment_id>.pt/.json`（默认 `var/enrollments`），不保存原始注册音频 |
| 有效期 | demo 进程内缓存由 `asr_enrollment_ttl_sec`（默认 3600 秒）控制并在使用时续期；RAG-ASR 落盘存储当前不做 TTL 自动过期，查询/使用会更新元数据 `last_used_at`（受 RAG-ASR `enrollment_metadata_touch_interval_sec` 节流） |
| 断连 / 重启 | WebSocket 断开不删除。demo 进程内缓存重启即丢；RAG-ASR 落盘存储在管理服务重启后仍可按同一 `enrollment_scope_id` + `enrollment_id` 读取 |
| 容量 | demo 进程内缓存上限 `asr_enrollment_max_entries`（默认 256），超出按 LRU 淘汰；RAG-ASR 的 `enrollment_cache_max_entries` 只是内存热缓存上限，不删除磁盘上的 enrollment 文件 |
| 删除 | `DELETE /api/asr/enrollment/{enrollment_id}` 立即清除；RAG-ASR 下沉链路会删除对应 `.pt` 与 `.json` 文件。未知 id 也返回 204，可安全重试 |

`enrollment_id` 不可用（demo 本地缓存过期 / 重启 / 被 LRU 淘汰 / 删除，或 RAG-ASR 下沉链路缺失对应落盘文件、embedding 与当前模型/adapter 不兼容）后再被使用时，默认兼容语义是回退为普通 ASR：AST v3 结果返回 `enrollment_applied=false` 并尽量给出 `enrollment_fallback_reason`，REST `/api/asr/upload` 响应 `enrollment_used=false`。集成方应对失效有预期，必要时重新注册并更新所携带的 id。

`GET /api/asr/enrollment/{enrollment_id}` 可查询该 ID 当前是否可直接用于后续 ASR。响应固定为 `{ "enrollment_id": "...", "available": true/false, "reason": "ok|not_found|incompatible|deleted|upstream_unavailable" }`，不返回原始注册音频、PCM、embedding 或其他声纹敏感材料。该接口只负责诊断，不保证查询后的实际识别一定应用声纹；最终仍以本次 AST v3 结果中的 `enrollment_applied` 或 REST 响应中的 `enrollment_used` 为准。

`asr_enrollment_min_sec` / `asr_enrollment_max_sec` / `asr_enrollment_ttl_sec` 虽在客户端覆写白名单内（见“临时配置覆写”），但注册是独立的 REST 调用、恒按服务端默认执行；流式端点首帧覆写这些值不会改变已注册 id 的行为。通用流式端点的 `start.enrollment_id` / `update_hotwords.enrollment_id` 用法与 TS-ASR 双音频 prompt 模板见 [通用流式 ASR WebSocket](protocols/transcribe-streaming-protocol.md)；AST v3 集成只需按本节注册，并按 [实时转写 AST v3 WebSocket](protocols/tuling-ast-v3-protocol.md) 发送 `enable_role_separation=false`、`enrollment_enable=true` 和 `enrollment_id`。

### 热词池管理

ASR 热词偏置按 `hotword_pool_id` 维护热词池。final 段先在当前热词池内召回 `recall_top_k` 个相关热词，再让少量请求临时 `hotwords`（默认 `recall_custom_hotword_limit=8`，去重、不写入热词池）优先进入主 ASR prompt，并过滤与请求热词精确重复或整词同音（忽略声调）的召回词；伪流式 partial 不执行召回、不注入热词、也不走 encoder bypass，只使用纯 vLLM raw-audio 推理。池管理接口优先代理 `services.recall_management` 指向的 RAG-ASR HTTP 管理服务；未配置管理服务时使用 Triton 管理路径，不在 demo 进程内复制热词状态。未传热词池 ID 时使用 `config.yaml` 的 `hotword_pool_id`，默认 `default`。

```bash
curl 'http://172.16.0.3:8082/api/asr/hotword-pool?hotword_pool_id=tenant-a&limit=20'
curl -X POST http://172.16.0.3:8082/api/asr/hotword-pool \
  -H 'content-type: application/json' \
  -d '{"hotword_pool_id":"tenant-a","hotwords":["挚音科技","张硕"]}'
curl -X DELETE http://172.16.0.3:8082/api/asr/hotword-pool \
  -H 'content-type: application/json' \
  -d '{"hotword_pool_id":"tenant-a","hotwords":["张硕"]}'
curl -X POST http://172.16.0.3:8082/api/asr/hotword-pool/clear \
  -H 'content-type: application/json' \
  -d '{"hotword_pool_id":"tenant-a"}'
curl -X POST 'http://172.16.0.3:8082/api/asr/hotword-pool/reload?hotword_pool_id=tenant-a'
curl -X POST http://172.16.0.3:8082/api/asr/hotword-pool/reload \
  -H 'content-type: application/json' \
  -d '{"hotword_pool_id":"tenant-a"}'
```

| 接口 | 请求 | 响应 |
|---|---|---|
| `GET /api/asr/hotword-pool` | query 参数 `hotword_pool_id`、`query`、`limit`、`offset` | RAG-ASR 返回的 `status`、`hotwords`、`total_count`、分页元信息 |
| `POST /api/asr/hotword-pool` | JSON `{ "hotword_pool_id": "tenant-a", "hotwords": ["词1", "词2"] }` | `added_count`、`duplicate_count`、`invalid_count`、`ignored_hotwords`、`total_count` |
| `DELETE /api/asr/hotword-pool` | JSON `{ "hotword_pool_id": "tenant-a", "hotwords": ["词1", "词2"] }` | `deleted_count`、`missing_count`、`missing_hotwords`、`total_count` |
| `POST /api/asr/hotword-pool/delete` | JSON `{ "hotword_pool_id": "tenant-a", "hotwords": ["词1", "词2"] }` | 与 `DELETE /api/asr/hotword-pool` 相同 |
| `POST /api/asr/hotword-pool/clear` | JSON 或 query 参数 `hotword_pool_id` | 清空指定热词池后的总量；不影响其他池 |
| `POST /api/asr/hotword-pool/reload` | JSON 或 query 参数 `hotword_pool_id`；两者同时存在且不一致时返回 400 | 从 RAG-ASR 对应热词池文件重载后的总量 |

热词添加 / 删除响应不透出上游内部兼容字段 `added`、`skipped_duplicates`、`duplicates`、`deleted`、`missing`、`invalid`；客户端应只依赖上表列出的对外字段。

### 情感上传

```bash
python docs/examples/rest_upload.py emotion sample.wav \
  --base-url http://172.16.0.3:8082 \
  --mode ser \
  --language zh
```

响应示例：

```json
{
  "type": "final_emotion",
  "mode": "ser",
  "label": "Happy",
  "text": "Happy",
  "duration_sec": 3.42,
  "language": "zh"
}
```

## Python 示例

先安装依赖：

```bash
pip install websockets requests numpy
```

运行 WebSocket ASR：

```bash
python docs/examples/ws_transcribe.py sample.wav \
  --url ws://172.16.0.3:8082/transcribe-streaming \
  --language zh
```

运行整段情感识别（异步 HTTP）：

```bash
python docs/examples/http_emotion_job.py sample.wav \
  --base-url http://172.16.0.3:8082 \
  --mode ser
```

或：

```bash
python docs/examples/rest_upload.py emotion sample.wav \
  --base-url http://172.16.0.3:8082 \
  --mode ser
```

运行分段情感识别（WebSocket）：

```bash
python tests/test_emotion_ws_client.py sample.wav \
  --url ws://172.16.0.3:8082/emotion-segmented-streaming \
  --segmented \
  --language zh
```

使用 `bash start.sh`（`https://172.16.0.3:8443`）时，示例脚本可加 `--insecure` 跳过自签证书校验。

## 错误处理

### WebSocket 错误消息

服务端遇到可恢复错误时会发送：

```json
{
  "type": "error",
  "message": "model inference failed"
}
```

部分错误事件还会带 `id`（语音段标识）或服务端自定义 `code`。客户端应至少记录完整错误 payload，并在收到错误后停止发送音频或主动关闭连接。

AST v3 中 `enrollment_enable=true` 但缺少 `enrollment_id` 属于参数错误，会返回 code 非 0 的 error 帧并结束本次会话。已启用但不可用的 `enrollment_id` 不触发 error；服务端回退普通 ASR，并在结果中返回 `enrollment_applied=false` 与可用时的 `enrollment_fallback_reason`。客户端可用 `GET /api/asr/enrollment/{id}` 判断是否需要重新注册。

### REST 错误响应

REST 接口使用标准 HTTP 状态码：

| 状态码 | 含义 |
|---|---|
| 400 | 请求字段缺失、音频为空、音频无法解码、注册音频校验失败、转写音频超过 `transcribe_max_audio_sec` 时长上限 |
| 413 | 上传文件超过服务端大小限制（转写接口为 `transcribe_max_upload_bytes`，默认 512 MB） |
| 422 | multipart 字段类型或必填字段不符合 FastAPI 校验 |
| 502 | 后端模型服务推理失败 |
| 502 | `/api/audio/analyze` 的 ASR、情感或文本清洗模型调用失败 |
| 204 | `DELETE /api/asr/enrollment/{id}` 删除成功（未知 id 也返回 204） |
| 202 | `POST /api/emotion/jobs` / `POST /api/asr/transcriptions` 已受理（需轮询 GET） |
| 503 | 情感 / 转写任务队列已满（`Retry-After`） |
| 404 | `GET /api/emotion/jobs/{id}` / `GET /api/asr/transcriptions/{id}` 任务不存在或已过期 |

普通错误体示例：

```json
{
  "detail": "audio file is empty"
}
```

`/api/asr/enrollment` 返回的是结构化错误体：

```json
{
  "detail": {
    "code": "too_short",
    "message": "enrollment audio is 0.30s, need at least 1.00s"
  }
}
```

## 相关文档

- [公网非实时音频分析 API](api/public-audio-analyze-api.md)
- [非实时音频分析 API](api/audio-analyze-api.md)
- [长音频离线转写 API](api/transcription-jobs-api.md)
- [通用流式 ASR WebSocket](protocols/transcribe-streaming-protocol.md)
- [实时转写 AST v3 WebSocket](protocols/tuling-ast-v3-protocol.md)
- [整段情感识别 HTTP（异步）](protocols/emotion-streaming-protocol.md)
- [分段情感识别 WebSocket](protocols/emotion-segmented-streaming-protocol.md)
