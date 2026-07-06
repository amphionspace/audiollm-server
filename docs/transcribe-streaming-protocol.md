# 通用流式 ASR API

`/transcribe-streaming` 用于实时语音转写。客户端通过 WebSocket 发送 16 kHz mono s16le PCM 音频流，服务端按 VAD 语音段返回中间结果和最终结果。

可选支持目标说话人识别（TS-ASR）：客户端先通过 `POST /api/asr/enrollment` 上传一段 1-8 秒的目标说话人音频，拿到 `enrollment_id`；后续的 WebSocket `start` / `update_hotwords` 或 REST `/api/asr/upload` 携带该 id 即可进入双音频 prompt（先 enrollment 后 target）的 TS-ASR 模式。不传 `enrollment_id` 时与普通 ASR 完全一致。

REST 上传版本见本文末尾的 `/api/asr/upload`，注册音频接口见 `/api/asr/enrollment`。

## 接口信息

| 项目 | 说明 |
|---|---|
| 协议 | WebSocket |
| 路径 | `/transcribe-streaming` |
| 完整 URL | `ws://172.16.0.3:8082/transcribe-streaming?language=<lang>` |
| 鉴权 | 无内置鉴权，无需自定义请求头 |
| 音频输入 | 二进制 PCM 帧，16 kHz、mono、signed 16-bit little-endian |
| 分段策略 | 服务端 VAD 自动切段 |
| 中间结果 | 支持，受 `enable_pseudo_stream` 与 `pseudo_stream_interval_ms` 影响 |
| 最终结果 | 每个语音段一条；`stop` 后会 flush 尾部残余音频 |

### Query 参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `language` | string | 否 | 语言代码，如 `zh`、`en`、`id`、`th`。不传表示自动检测或使用服务端默认策略 |

## 调用流程

```text
Client                                      Server
  | ---- WebSocket connect --------------> |
  | <---------------- ready -------------- |
  | ---- start --------------------------> |
  | ---- binary PCM chunk ---------------> |
  | <---------------- partial ----------- |
  | ---- binary PCM chunk ---------------> |
  | <---------------- final ------------- |  one speech segment
  | ---- binary PCM chunk ---------------> |
  | ---- stop ---------------------------> |
  | <---------------- final ------------- |  trailing audio, if any
  | ---- close --------------------------> |
```

服务端历史版本可能返回 `partial_asr` / `final_asr`，当前第三方客户端建议同时兼容 `partial` / `partial_asr` 与 `final` / `final_asr`。

## 客户端消息

### start

收到 `ready` 后、发送任何 PCM 前必须先发送 `start`。

```json
{
  "type": "start",
  "format": "pcm_s16le",
  "sample_rate_hz": 16000,
  "channels": 1,
  "language": "zh",
  "hotword_pool_id": "tenant-a",
  "hotwords": ["挚音科技", "张硕"],
  "enrollment_id": "ule8QilVjZql30Q9oy9kiQ",
  "config": {
    "vad_threshold": 0.45
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `type` | string | 是 | 固定为 `start` |
| `format` | string | 是 | 固定为 `pcm_s16le` |
| `sample_rate_hz` | integer | 是 | 固定为 `16000` |
| `channels` | integer | 是 | 固定为 `1` |
| `language` | string | 否 | 语言代码；与 query 参数二选一即可 |
| `hotword_pool_id` | string | 否 | 热词池隔离 ID，默认 `default`；只召回该热词池 |
| `hotwords` | string[] | 否 | 临时请求热词；final 段会把去重后的前 `recall_custom_hotword_limit` 个优先注入 prompt，并覆盖精确重复或整词同音（忽略声调）的 RAG-ASR 召回热词，不写入热词池 |
| `enrollment_id` | string | 否 | 由 `POST /api/asr/enrollment` 返回的目标说话人 id；传 `null` 或省略表示普通 ASR |
| `config` | object | 否 | 当前连接的服务端配置覆写，仅白名单字段生效（见“可覆写配置”） |

### update_hotwords

`hotwords` 字段会被接收为后续 final 段的临时请求热词：服务端会把去重后的前 `recall_custom_hotword_limit` 个优先注入 prompt，并过滤当前热词池中精确重复或整词同音（忽略声调）的召回词，但不会写入热词池。该消息仍可用于切换/清除目标说话人，也可用 `hotword_pool_id` 切换热词池。

```json
{
  "type": "update_hotwords",
  "hotwords": ["产品名", "人名"],
  "src_lang": "zh",
  "hotword_pool_id": "tenant-a",
  "enrollment_id": "ule8QilVjZql30Q9oy9kiQ"
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `type` | string | 是 | 固定为 `update_hotwords` |
| `hotwords` | string[] | 是 | 临时请求热词；空数组表示清空后续 final 段的临时热词，不会改变 RAG-ASR 热词池 |
| `src_lang` | string | 否 | 语言代码或语言名称 |
| `hotword_pool_id` | string | 否 | 切换后续 final 段使用的热词池；非法值会返回 `invalid_hotword_pool_id` 错误并保持原热词池 |
| `enrollment_id` | string \| null | 否 | 切换或清除目标说话人；缺省字段则保持原状，显式传 `null` 表示清除 |

### 二进制音频帧

`start` 后持续发送 PCM bytes。

| 项目 | 要求 |
|---|---|
| 编码 | signed 16-bit little-endian PCM |
| 采样率 | 16000 Hz |
| 声道 | 1 |
| 推荐 chunk | 30-80 ms |
| 80 ms 字节数 | 2560 bytes |

### stop

```json
{"type":"stop"}
```

`stop` 表示本次音频输入结束。服务端会处理剩余音频并返回最终结果。

## 服务端消息

### ready

```json
{"type":"ready"}
```

当服务端 `debug_dump_enabled=true` 时，`ready` 额外带两个调试字段（默认部署不带，契约不变）：

```json
{"type":"ready","session_id":"260629-150812-3f9a","dump_dir":"/abs/path/debug_dumps/260629-150812-3f9a"}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `session_id` | string | 本连接唯一 ID；仅 `debug_dump_enabled=true` 时出现 |
| `dump_dir` | string | 本会话落盘目录绝对路径；仅 `debug_dump_enabled=true` 时出现 |

### partial / partial_asr

```json
{
  "type": "partial",
  "id": "seg-001",
  "text": "你好",
  "language": "zh"
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `type` | string | `partial` 或 `partial_asr` |
| `id` | string | 语音段 ID；可能不存在 |
| `text` | string | 当前语音段的临时转写文本，保持口语形式（不做 ITN/车牌规范化） |
| `language` | string | 检测或传入的语言 |

### final / final_asr

```json
{
  "type": "final",
  "id": "seg-001",
  "text": "你好，欢迎使用语音识别服务。",
  "language": "zh",
  "effective_hotwords": ["挚音科技", "张硕"],
  "duration_sec": 3.42,
  "audio_b64": "UklGR...",
  "dump_id": "260629-150812-3f9a/seg-001"
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `type` | string | `final` 或 `final_asr` |
| `id` | string | 语音段 ID；可能不存在 |
| `text` | string | 最终转写文本，默认已做逆文本规范化（ITN）与车牌规范化，见下方说明 |
| `language` | string | 检测或传入的语言 |
| `effective_hotwords` | string[] | 本段音频经 RAG-ASR/Triton 热词召回得到的热词列表；不包含 `start.hotwords` / `update_hotwords` 里的临时请求热词。召回关闭、失败或无召回结果时为空数组 |
| `duration_sec` | number | 本次推理使用的音频时长；部分流式消息可能不带 |
| `audio_b64` | string | 当前分段音频的 WAV base64，用于前端回放；为空 final 不带。k2 模式下它与送入 LLM ASR 的音频同源，来自 k2 endpoint 对应的本地缓冲，不再经过本地 VAD 段首/段尾二次裁剪 |
| `dump_id` | string | 仅 `debug_dump_enabled=true` 且本段有 final 文本时出现；值为 `<session_id>/<seg_id>`，同时是落盘文件相对路径 stem（`<dump_dir>/../<dump_id>.wav` 与 `.json`） |

文本规范化（仅 final）：服务端默认对 final 文本做两类本地后处理，partial 不变：

- 通用 ITN（`enable_asr_itn`，仅中文）：口语数字转写成阿拉伯形式，如 `六五四三八`→`65438`、`二零二四年`→`2024年`，电话/金额/百分比同理。
- 车牌规范化（`enable_asr_plate_normalize`）：字母大写、去车牌内分隔符、口语数字按位转阿拉伯，并按 GB 车牌形态校验后才改写，如 `辽b二四五零七`→`辽B24507`。
- 已知边界：省份简称被声学误识别成字母（`冀`→`J`）属识别错误，后处理只修数字与字母（`车牌号为JR六五四三八`→`车牌号为JR65438`），不还原省份字。
- 两个开关均为服务端配置（`config.yaml` 的 `defaults.itn` 分组），不在客户端可覆写白名单内；任一处理异常都回退原文，不影响转写主流程。

### error

```json
{
  "type": "error",
  "message": "invalid start message"
}
```

客户端收到 `error` 后应记录完整 payload，并根据业务需要停止发送音频或关闭连接。

## 调试落盘 (debug dump)

运维级开关（不在客户端可覆写白名单内），用于排查“回放/最终结果对不上”一类问题。`config.yaml` 的 `defaults.debug` 分组：

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `debug_dump_enabled` | boolean | false | 开启后每个 final 段落盘音频+元信息；生产建议关闭（磁盘无限增长） |
| `debug_dump_dir` | string | debug_dumps | 落盘根目录；相对路径相对项目根，也可填绝对路径 |

开启后：`ready` 带 `session_id` / `dump_dir`，`final` 带 `dump_id`；每个 final 段写两份文件到 `<dump_dir>/<session_id>/<seg_id>.{wav,json}`。

- `<seg_id>.wav`：送入推理的那段 PCM（与该 final 的 `audio_b64`、即前端回放音频同源同字节）。
- `<seg_id>.json`：final 文本、该段的 partial 历史、主/副模型原始输出、模型与开关快照、时长/时延等。

前端在每个气泡的元信息处显示可点击复制的 `dump_id`，复制后即可直接定位上述文件。该能力只作用于 native `/transcribe-streaming`：AST v3 不下发 `ready`、其 final 也只编码协议字段，故不受影响。

## 可覆写配置

`start.config` 仅对当前连接生效、不落盘，只接受扁平字段名。覆写字段受服务端白名单（`backend/config.py` 的 `CLIENT_OVERRIDABLE_FIELDS`）约束：白名单外字段（如模型地址 `*_vllm_base_url`、密钥、连接池/队列等基础设施项）、未知字段与非法值都会被忽略并保持服务端默认，不会中断连接。完整白名单与 `/tuling/ast/v3` 的 `parameter.asr_config` 共用，按类别速览见 [API 总览](api-reference.md) 的“临时配置覆写”。

当服务端启用 `k2_enabled=true` 时，本端点的 partial 来自外部 k2 流式 ASR，final 仍由本服务 LLM ASR 产生。k2 只做纯识别，不接热词、不接目标说话人、不返回 token timestamps；热词召回、目标说话人过滤与文本规范化只作用于 final。此时切段权威是 k2 endpoint，本服务只保留有界缓冲，并在 partial/final 进入下游前用 `k2_voice_gate_*` 确认有人声证据；voice gate 只做放行/丢弃，不裁剪段首/段尾。VAD 与伪流式间隔类覆写仍会被接受但不再决定切点或首字时机；`enable_pseudo_stream=false` 仍会抑制 partial 下发。

对本端点有效的常用字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| vad_threshold | number | VAD 判定阈值 |
| silence_duration_ms | integer | 静音持续多久后切段 |
| min_segment_duration_ms | integer | 短于该值的语音段会被丢弃 |
| enable_pseudo_stream | boolean | 是否输出伪流式中间结果 |
| pseudo_stream_interval_ms | integer | 伪流式中间结果间隔（仅节流首个之后的刷新，不影响首字） |
| pseudo_stream_first_partial_ms | integer | 每段语音首个 partial（伪流式中间结果）的触发门槛，从 min_segment_duration_ms 解耦（config.yaml 默认 200，已低于 min_segment 350）；与 vad_start_frames 按 max 决定首字延迟 |
| asr_request_timeout | number | 单次 ASR 模型请求超时秒数 |
| enable_primary_asr | boolean | 是否启用主模型 |
| enable_secondary_asr | boolean | 是否启用副模型；关闭后无副模型静音门、无融合 |
| enable_dual_asr_fusion | boolean | final 段是否做主副融合矫正；需副模型开启，否则自动降级为 false |

## Python WebSocket 示例

完整可运行脚本见 [examples/ws_transcribe.py](examples/ws_transcribe.py)。

```bash
pip install websockets numpy

python docs/examples/ws_transcribe.py sample.wav \
  --url ws://172.16.0.3:8082/transcribe-streaming \
  --language zh \
  --hotwords "挚音科技,张硕" \
```

使用 `bash start.sh`（`wss://172.16.0.3:8443/...`）时，示例脚本可加 `--insecure` 跳过自签证书校验。

## REST 上传接口

`POST /api/asr/upload` 适合上传完整音频文件并获得一次最终转写结果。

### 请求

| 表单字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `audio` | file | 是 | WAV 音频文件 |
| `language` | string | 否 | 语言代码 |
| `hotword_pool_id` | string | 否 | 热词池隔离 ID，默认 `default` |
| `hotwords` | string | 否 | 临时请求热词；去重限量后优先进入 prompt，并覆盖精确重复或整词同音（忽略声调）的 RAG-ASR 召回热词，不写入热词池 |
| `enrollment_id` | string | 否 | 由 `POST /api/asr/enrollment` 返回的目标说话人 id；不传或失效时静默回退到普通 ASR |

### 响应

```json
{
  "type": "final",
  "text": "你好，欢迎使用语音识别服务。",
  "language": "zh",
  "duration_sec": 3.42,
  "enrollment_used": false
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `text` | string | 最终转写文本；与流式 final 一致，默认已做 ITN 与车牌规范化（开关见上文“文本规范化（仅 final）”） |
| `enrollment_used` | boolean | 本次推理是否实际使用了目标说话人。`enrollment_id` 不存在、demo 本地缓存已过期、RAG-ASR 落盘文件缺失或 embedding 不兼容时为 `false` |

### Python REST 示例

完整可运行脚本见 [examples/rest_upload.py](examples/rest_upload.py)。

```bash
pip install requests

python docs/examples/rest_upload.py asr sample.wav \
  --base-url http://172.16.0.3:8082 \
  --language zh \
  --hotwords "挚音科技,张硕" \
```

## 目标说话人注册接口

`POST /api/asr/enrollment` 上传一段 1-8 秒的目标说话人音频，支持 WAV、MP3 和 raw PCM（16 kHz mono s16le）。默认配置下服务端把音频规范化为 16 kHz mono WAV、写入 demo 进程内缓存，并返回不透明的 `enrollment_id`；当 `enable_triton_enrollment_store=true` 且配置了 RAG-ASR 管理服务时，新注册音频会转发给 RAG-ASR，RAG-ASR 将 projector frames tensor 和元数据落盘到 `enrollment_store_dir/<enrollment_scope_id>/<enrollment_id>.pt/.json`（默认 `var/enrollments`），本服务和 RAG-ASR 都不保存原始注册音频。后续 WebSocket `start` / `update_hotwords` 或 `/api/asr/upload` 携带该 id 时，主模型 prompt 自动切换为 TS-ASR 双音频形态（先 enrollment 后 target），具体文本位置由服务端 `prompt_template` 随模型选择。

### 请求

| 表单字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `audio` | file | 是 | WAV、MP3 或 raw PCM；raw PCM 必须是 16 kHz mono s16le，建议使用 `.pcm`/`.raw` 后缀或 `audio/pcm` content type；服务端解码为 16 kHz mono 并尾截到 8 秒以内 |

### 成功响应

HTTP 200。

```json
{
  "enrollment_id": "ule8QilVjZql30Q9oy9kiQ",
  "duration_sec": 3.0
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `enrollment_id` | string | 后续请求复用的不透明 id；demo 本地缓存 TTL 默认 3600 秒。RAG-ASR 下沉链路为本地磁盘持久化，无 TTL 自动过期 |
| `duration_sec` | number | 最终注册音频时长（裁剪后）；下沉链路中该时长来自 RAG-ASR 元数据 |

### 错误响应

HTTP 400，`detail` 为结构化对象：

```json
{
  "detail": {
    "code": "too_short",
    "message": "enrollment audio is 0.30s, need at least 1.00s"
  }
}
```

| code | 含义 |
|---|---|
| empty | 上传体为空或解码后没有音频帧 |
| too_short | 音频时长不足 `asr_enrollment_min_sec`（默认 1.0 秒） |
| unsupported_format | 不是 WAV/MP3/raw PCM |
| decode_failed | 容器损坏或解码失败 |

时长超过上限不会拒绝，服务端会自动尾截到 `asr_enrollment_max_sec`（默认 8.0 秒）。

### 删除注册音频

`DELETE /api/asr/enrollment/{enrollment_id}` 立即清除缓存条目。对未知 id 也返回 204，调用方可安全重试。

### 服务端可配置项

| 配置 | 默认 | 说明 |
|---|---|---|
| asr_enrollment_min_sec | 1.0 | 最小时长，低于此值返回 too_short |
| asr_enrollment_max_sec | 8.0 | 最大时长，超出尾截 |
| asr_enrollment_ttl_sec | 3600 | demo 本地进程内缓存 TTL；最近一次 get 后重新计时。RAG-ASR 下沉链路不使用该 TTL 自动删除落盘 embedding |
| asr_enrollment_max_entries | 256 | demo 本地进程内缓存条目上限；RAG-ASR 的 enrollment cache 上限只影响内存热缓存，不删除磁盘文件 |

### 与 fusion 的关系

启用 enrollment 不会改变次级模型（Qwen3-ASR）的调用方式：次级模型始终只看混音/目标音频，不接 enrollment。fusion 仍按原规则比对主/次结果。

### Python 示例

```python
import requests

base = "http://172.16.0.3:8082"
with open("speaker_enroll.wav", "rb") as f:
    r = requests.post(
        f"{base}/api/asr/enrollment",
        files={"audio": ("enroll.wav", f, "audio/wav")},
        timeout=30,
    )
r.raise_for_status()
enrollment_id = r.json()["enrollment_id"]

with open("conversation.wav", "rb") as f:
    r = requests.post(
        f"{base}/api/asr/upload",
        files={"audio": ("conv.wav", f, "audio/wav")},
        data={"enrollment_id": enrollment_id, "hotwords": "北京,清华大学"},
        timeout=60,
    )
print(r.json())
```

## ASR Prompt 模板

后端按主模型 upstream 的 `prompt_template` 选择 prompt 结构。热词偏置来自当前 `hotword_pool_id` 的 RAG-ASR 热词池召回 top-K 结果，以及优先进入 prompt 的少量请求临时 `hotwords`（默认最多 8 个，去重后不写入热词池）；与临时热词精确重复或整词同音（忽略声调）的召回词会被过滤。模板选择是服务端模型配置，不可客户端覆写。两套模板都支持普通 ASR、召回热词、TS-ASR、TS-ASR + 召回热词。

### `amphion_asr`（Amphion 4B）

4B 使用文本和音频混排在 `user` turn 的 swift 风格。启用 enrollment 时，音频顺序为 enrollment 在前、目标音频在后。

```text
普通 ASR:
[user]
Transcribe the following audio.
<audio_target>

热词:
[user]
Transcribe the following audio.
Hotwords: w1,w2,w3
<audio_target>

TS-ASR:
[user]
Given the speaker's voice:
<audio_enroll>
Transcribe what this speaker says in the following audio.
<audio_target>

TS-ASR + 热词:
[user]
Given the speaker's voice:
<audio_enroll>
Transcribe what this speaker says in the following audio.
Hotwords: w1,w2,w3
<audio_target>
```

任务 6 的实际 `messages` 示例：

```json
[
  {
    "role": "user",
    "content": [
      {"type": "text", "text": "Given the speaker's voice:"},
      {"type": "input_audio", "input_audio": {"data": "<ENROLL_B64>", "format": "wav"}},
      {"type": "text", "text": "\nTranscribe what this speaker says in the following audio.\nHotwords: 北京,清华大学"},
      {"type": "input_audio", "input_audio": {"data": "<TARGET_B64>", "format": "wav"}}
    ]
  }
]
```

### `amphion_asr_1.7b`（Amphion 1.7B）

1.7B 继承 Qwen3-ASR 风格：文本只放 `system` turn，`user` turn 只放音频 token。普通 ASR 的 system 内容为空字符串。

```text
普通 ASR:
[system]

[user]
<audio_target>

热词:
[system]
Hotwords: w1,w2,w3

[user]
<audio_target>

TS-ASR:
[system]
Given the speaker's voice in the first audio.

[user]
<audio_enroll>
<audio_target>

TS-ASR + 热词:
[system]
Given the speaker's voice in the first audio.
Hotwords: w1,w2,w3

[user]
<audio_enroll>
<audio_target>
```

TS-ASR + 热词的实际 `messages` 示例：

```json
[
  {
    "role": "system",
    "content": "Given the speaker's voice in the first audio.\nHotwords: 北京,清华大学"
  },
  {
    "role": "user",
    "content": [
      {"type": "input_audio", "input_audio": {"data": "<ENROLL_B64>", "format": "wav"}},
      {"type": "input_audio", "input_audio": {"data": "<TARGET_B64>", "format": "wav"}}
    ]
  }
]
```

final 段 prompt 中实际注入的是去重限量后的临时 `hotwords`（默认最多 `recall_custom_hotword_limit=8` 个），再追加未被精确重复或整词同音（忽略声调）规则过滤的 Triton 召回 top-K（默认 `recall_top_k=50`）热词，格式仍为半角逗号 `,` 分隔无空格。伪流式 partial 不执行召回、不注入热词、也不走 encoder bypass，只使用纯 vLLM raw-audio 推理。`Language:` 行不会出现在 ASR prompt 中；1.7B 若输出 `language Chinese<asr_text>...` 前缀，服务端会剥离前缀并记录模型自检语种。服务端还会折叠超过 20 次的退化重复输出，避免解码 loop 污染 partial / final 文本。

## 相关文档

- [API 总览](api-reference.md)
- [整段情感识别](emotion-streaming-protocol.md)
- [分段情感识别](emotion-segmented-streaming-protocol.md)
