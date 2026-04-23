# /transcribe-target-streaming 协议与 TS-ASR Demo 说明

> 端点命名约定：本服务按任务一类一个 WebSocket 端点（`/<task>-streaming`）。本文档描述目标说话人语音识别（Target-Speaker ASR）任务。它与 `/transcribe-streaming`、`/emotion-streaming` 共享相同的控制消息基础结构（`ready` / `start` / `stop` / `update_hotwords` / `error` / `start.config` 覆写机制），仅 `start` 的必填字段与输出语义不同。

## 概述

TS-ASR 用一段「参考/注册音频」锁定目标说话人的音色，在包含干扰人声/背景噪声的混合音频中，只转写该说话人的内容。

当前是短期方案：复用主 ASR vLLM 端点（默认 `Amphion/Amphion-3B`），Prompt 模板、热词开关、说话人音色描述等仍在迭代。未来路线包括切换到专用 TS-ASR checkpoint、支持持久化说话人注册与多说话人并行。

## 架构

整体走与常规 ASR 相同的流式架构，但所有 TS-ASR 相关逻辑被限制在独立的子包与任务引擎中：

```
frontend/tsasr.html
frontend/tsasr-app.js
frontend/tsasr.css
frontend/tsasr-processor.js
backend/main.py                       -> /transcribe-target-streaming 端点
backend/streaming/                    -> 通用会话层（无需改动）
backend/tasks/ts_asr.py               -> TsAsrTaskEngine（thin orchestrator）
backend/tsasr/
  prompt.py                           -> build_tsasr_content（Prompt 构建器）
  client.py                           -> query_tsasr_model（vLLM 请求）
  enrollment.py                       -> decode_enrollment（注册音频校验）
backend/audio/utils.py                -> wav_base64_to_pcm_16k_mono
```

将 Prompt / Client / Enrollment 放到独立子包的目的：TS-ASR 的业务语义仍在演进，后续新增 hotwords / voice_traits / 多说话人等开关都只在 `backend/tsasr/` 内完成，不扩散到 `streaming/` 与 `tasks/` 的其余部分。

## 连接

### URL

```
ws://<host>:<port>/transcribe-target-streaming?language=<lang>
wss://<host>:<port>/transcribe-target-streaming?language=<lang>
```

### Query 参数

| 参数     | 必填 | 说明                                                |
|----------|------|-----------------------------------------------------|
| language | 否   | 语言代码，如 zh/en/id/th。默认空（自动检测）        |

## 消息时序

```
Client                                 Server
  |                                      |
  |  ---- WebSocket 连接 -------------->  |
  |  <--------  ready  ---------------   |
  |  ----  start (含注册音频) -------->   |
  |  <--------  enrollment_ok  -------   |
  |  ----  binary PCM chunk  -------->   |
  |  ----  binary PCM chunk  -------->   |
  |  <--------  final  ---------------   |  每段语音一条
  |  ...                                 |
  |  ----  stop  -------------------->   |
  |  <--------  final  ---------------   |  stop 后兜底返回一条
  |  ---- 连接关闭 ---                    |
```

若 `start.enrollment_audio` 缺失或校验失败，服务端返回一条 `error` 消息并停止后续推理，但不会主动关闭 WebSocket；客户端可自行 close 并重录。

## 客户端 → 服务端

### start

发送 PCM 前必须先发 `start`，一次会话只能发一条。

| 字段                      | 必填 | 类型     | 说明                                                             |
|---------------------------|------|----------|------------------------------------------------------------------|
| type                      | 是   | string   | 固定 `"start"`                                                   |
| format                    | 否   | string   | 固定 `"pcm_s16le"`                                               |
| sample_rate_hz            | 否   | int      | 16000                                                            |
| channels                  | 否   | int      | 1                                                                |
| language                  | 否   | string   | zh/en/id/th，与 query 参数二选一                                 |
| enrollment_audio          | 是   | string   | 注册音频的 base64 WAV。支持任意采样率/声道，服务端下采样到 16kHz |
| enrollment_format         | 否   | string   | 固定 `"wav"`                                                     |
| voice_traits              | 否   | string   | 说话人音色自由描述，插入在 Prompt 的注册音频之后                 |
| hotwords                  | 否   | string[] | 热词列表。仅当服务端 `tsasr_enable_hotwords=true` 时生效         |
| config                    | 否   | object   | 服务端参数覆写，仅接受白名单字段                                 |

### 其他

| 消息                           | 说明                                                              |
|--------------------------------|-------------------------------------------------------------------|
| 二进制 PCM 帧                  | 16kHz 单声道 s16le，建议每帧 30-80ms                              |
| `{"type":"update_hotwords"}`   | 中途更新热词（仅在 `tsasr_enable_hotwords=true` 下影响推理）      |
| `{"type":"stop"}`              | 结束音频流。服务端处理尾部残余音频并保证返回一条 `final`          |

## 服务端 → 客户端

| 消息                                              | 说明                                                         |
|---------------------------------------------------|--------------------------------------------------------------|
| `{"type":"ready"}`                                | WebSocket 就绪，可发送 start                                 |
| `{"type":"enrollment_ok","duration_sec":..}`      | 注册音频校验通过，可开始发送 PCM                             |
| `{"type":"partial","text":"...","task":"tsasr"}`  | 中间结果（默认关闭，取决于 `tsasr_enable_partial`）          |
| `{"type":"final","text":"...","task":"tsasr"}`    | 最终结果（每段语音一条；stop 后即便无文本也兜底一条空 final）|
| `{"type":"error","code":"...","message":"..."}`   | 错误通知。注册相关错误的 code 以 `enrollment_` 为前缀        |

注册相关错误的 code 枚举：

| code                       | 触发                                 |
|----------------------------|--------------------------------------|
| enrollment_missing         | start 未带 enrollment_audio          |
| enrollment_empty           | 解码结果为空 PCM                     |
| enrollment_too_short       | 时长 < `tsasr_enrollment_min_sec`    |
| enrollment_too_long        | 时长 > `tsasr_enrollment_max_sec`    |
| enrollment_decode_failed   | base64 / WAV 头解析失败              |
| enrollment_unsupported_format | enrollment_format 非 `wav`        |

## Prompt 模板

TS-ASR 的多模态 `content` 布局（对齐 AmphionASR ms-swift SFT 训练时的结构）：

```
[ {text: "Given the speaker's voice:"},
  {input_audio: enrollment},
  (可选) {text: "\nSpeaker traits: ..."},
  (可选) {text: "\nHotwords: a,b,c."},
  {text: "\nTranscribe what this speaker says in the following audio:"},
  {input_audio: mixed} ]
```

由 `backend.tsasr.prompt.build_tsasr_content` 统一构建，想调整时只改这一个函数。

## 配置

所有 TS-ASR 相关参数都以 `tsasr_` 前缀统一命名，默认值留空/继承 `vllm_*`，便于后续切到独立服务。

| 字段                         | 默认值                                    | 说明                                                             |
|------------------------------|-------------------------------------------|------------------------------------------------------------------|
| tsasr_base_url               | ""                                        | 留空时回落到 `vllm_base_url`                                     |
| tsasr_model_name             | ""                                        | 留空时回落到 `vllm_model_name`                                   |
| tsasr_request_timeout        | 30.0                                      | 单次 HTTP 请求超时（秒）                                         |
| tsasr_enrollment_min_sec     | 1.0                                       | 注册音频最短时长（秒）                                           |
| tsasr_enrollment_max_sec     | 30.0                                      | 注册音频最长时长（秒）                                           |
| tsasr_max_audio_seconds      | 30.0                                      | 单段混合音频的时长上限；超出则保留尾部                           |
| tsasr_enable_partial         | false                                     | 是否开启伪流式 partial。双音频推理 RTF 较高，默认关闭            |
| tsasr_enable_hotwords        | false                                     | 是否把 `ctx.hotwords` 注入到 Prompt。默认关闭，短期方案未验证   |

以上字段都可通过 `start.config` 覆写。

## 短期方案的限制

- 双音频推理时长近似为普通 ASR 的 2 倍，因此：
  - 默认关闭 pseudo-streaming partial；
  - 未启用双模型 fusion（Qwen3-ASR 不接受双音频输入）。
- 注册音频每会话一次，随 WS 断开丢失，不做持久化。未来计划加入注册表或 speaker embedding 缓存。
- Prompt 中的 hotwords / voice_traits 分支尚未对齐训练数据，启用前请做离线验证。

## 前端 Demo

静态页入口：`frontend/tsasr.html`（由 FastAPI StaticFiles 自动暴露在 `/tsasr.html`）。

使用流程：

1. 在右侧面板中录制 1-30s 目标说话人干净音频，查看到 Ready 状态后即可启用麦克风按钮。
2. 可选填写 voice_traits；可在上方下拉切换语言。
3. 点击麦克风开始采集。断开连接或再次点击麦克风将发送 `stop` 并关闭 WebSocket。
4. 识别结果以聊天气泡方式在左侧滚动显示。
