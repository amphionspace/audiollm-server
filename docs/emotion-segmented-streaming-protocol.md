# 分段情感识别 API

`/emotion-segmented-streaming` 用于长连接场景下的分段情感识别。客户端持续发送 PCM 音频流，服务端按 VAD 切段，并在每个语音段结束后返回一条 `final_emotion`。

如果只需要对一段录音做一次情感判断，请使用 [整段情感识别 HTTP API](emotion-streaming-protocol.md)（`POST /api/emotion/jobs` + 轮询）。

## 接口信息

| 项目 | 说明 |
|---|---|
| 协议 | WebSocket |
| 路径 | `/emotion-segmented-streaming` |
| 完整 URL | `ws://172.16.0.3:8082/emotion-segmented-streaming?language=<lang>` |
| 鉴权 | 无内置鉴权，无需自定义请求头 |
| 音频输入 | 二进制 PCM 帧，16 kHz、mono、signed 16-bit little-endian |
| 分段策略 | 服务端 VAD 自动切段 |
| 中间结果 | 不支持 |
| 最终结果 | 每个有效 VAD 语音段一条 `final_emotion` |

### Query 参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `language` | string | 否 | 透传语言字段，如 `zh`、`en`、`id`、`th`；服务端会在结果中回填 |

## 与整段情感接口的区别

| 维度 | 整段 HTTP (`/api/emotion/jobs`) | `/emotion-segmented-streaming` |
|---|---|---|
| 协议 | HTTP 异步任务 | WebSocket |
| VAD | 不启用 | 启用 |
| 结果数量 | 每个任务一条 | 每个有效语音段一条 |
| 空音频 | 任务仍 `succeeded`，空 `final_emotion` | 空会话可能无结果 |
| 适用场景 | 单条录音、离线文件 | 实时对话、直播、长录音 |

## 调用流程

```text
Client                                      Server
  | ---- WebSocket connect --------------> |
  | <---------------- ready -------------- |
  | ---- start --------------------------> |
  | ---- binary PCM chunk ---------------> |
  | ---- binary PCM chunk ---------------> |
  | <---------------- final_emotion ------ |  segment 1
  | ---- binary PCM chunk ---------------> |
  | <---------------- final_emotion ------ |  segment 2
  | ---- stop ---------------------------> |
  | <---------------- final_emotion ------ |  trailing segment, if any
  | ---- close --------------------------> |
```

客户端不应假设 `stop` 后一定有结果。如果整段音频没有被 VAD 判定为有效语音，服务端可能不返回 `final_emotion`。

## 客户端消息

### start

```json
{
  "type": "start",
  "format": "pcm_s16le",
  "sample_rate_hz": 16000,
  "channels": 1,
  "mode": "ser",
  "language": "zh",
  "config": {
    "min_segment_duration_ms": 500
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `type` | string | 是 | 固定为 `start` |
| `format` | string | 否 | 固定为 `pcm_s16le` |
| `sample_rate_hz` | integer | 否 | 固定为 `16000` |
| `channels` | integer | 否 | 固定为 `1` |
| `mode` | string | 否 | `ser` 或 `sec`，不传使用服务端默认 |
| `language` | string | 否 | 透传语言字段；也可通过 query 参数传入 |
| `config` | object | 否 | 当前连接的服务端配置覆写 |

### 二进制音频帧

`start` 后持续发送 PCM bytes。建议每帧 30-80 ms。

### stop

```json
{"type":"stop"}
```

服务端收到后 flush VAD 尾段。如果尾段满足最小时长要求，会返回最后一条 `final_emotion`。

## 服务端消息

### ready

```json
{"type":"ready"}
```

服务端 `debug_dump_enabled=true` 时，`ready` 会额外带 `session_id` / `dump_dir`（与 [实时转写协议](transcribe-streaming-protocol.md) 一致，由共享会话层注入）。但本端点的情感任务不落盘、`final_emotion` 也不带 `dump_id`；这两个字段对情感流仅作连接标识，可忽略。

### final_emotion

payload 与整段情感接口一致：

```json
{
  "type": "final_emotion",
  "mode": "ser",
  "label": "Neutral",
  "text": "Neutral",
  "duration_sec": 2.68,
  "language": "zh"
}
```

`final_emotion` 字段与 [整段情感 HTTP API](emotion-streaming-protocol.md#查询任务) 的 `result` 一致。此处 `duration_sec` 表示**当前 VAD 语音段**的推理时长，不是整段任务的累计时长。

### error

```json
{
  "type": "error",
  "message": "emotion inference failed"
}
```

## 可覆写配置

| 字段 | 类型 | 说明 |
|---|---|---|
| `vad_threshold` | number | VAD 判定阈值 |
| `silence_duration_ms` | integer | 静音持续多久后切段 |
| `min_segment_duration_ms` | integer | 短于该值的语音段会被丢弃 |
| `emotion_task_mode` | string | 默认情感任务模式：`ser` 或 `sec` |
| `emotion_request_timeout` | number | 单次情感模型请求超时秒数 |
| `emotion_max_audio_seconds` | number | 每个语音段最大推理秒数；超出保留尾部 |

`enable_pseudo_stream` 不影响本接口，情感任务不会返回 partial。

## Python WebSocket 示例

完整可运行脚本见仓库内 [tests/test_emotion_ws_client.py](../tests/test_emotion_ws_client.py)。

```bash
pip install websockets numpy

python tests/test_emotion_ws_client.py sample.wav \
  --url ws://172.16.0.3:8082/emotion-segmented-streaming \
  --mode ser \
```

## 相关文档

- [API 总览](api-reference.md)
- [整段情感识别](emotion-streaming-protocol.md)
- [通用流式 ASR](transcribe-streaming-protocol.md)