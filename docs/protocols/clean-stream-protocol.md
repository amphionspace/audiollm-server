# 增强语音识别 WebSocket 协议

## 接口与模型

```text
wss://playground.amphion.top/asr/v1/clean-stream
```

该公开 Playground 接口无需 API Key。它由 AudioLLM Server 直接提供，不连接
Gateway。识别只使用本机 `Qwen3-ASR-1.7B` HTTP 非流式模型，以累计音频快照实现
伪流式；不使用 k2 或副 ASR 模型。refine LLM 和 AmphionSPEC 均为服务端内部依赖，
客户端不能传入 URL、模型或密钥。

音频固定为 16 kHz、单声道、PCM signed 16-bit little-endian，在 JSON 中用 base64
编码。单会话最长 60 秒。

## 消息流程

```text
连接 → session.created
     → session.update → session.updated
     → input_audio_buffer.append × N
     ← transcription.delta × N
     → input_audio_buffer.commit(final=true)
     ← emotion.bucket（可选）
     ← postprocess.delta（可选）
     ← transcription.done
```

客户端应等待 `session.updated` 后再发送音频，每条连接只允许发送一次
`session.update`。

## 客户端消息

### `session.update`

```json
{
  "type": "session.update",
  "language": "zh",
  "translate_mode": false,
  "cleanup": {"level": "light", "text_emotion": true},
  "hotwords": {
    "builtin": ["internet"],
    "custom": ["Amphion", "Qwen3-ASR"]
  }
}
```

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `language` | `auto` | `auto` 或 `zh`、`en`、`ja`、`ko` 等语言码 |
| `translate_mode` | `false` | 开启后执行翻译而非 cleanup |
| `target_language` | 无 | 翻译开启时必填，不能为 `auto` |
| `cleanup.level` | `light` | `off`、`light`、`standard` |
| `cleanup.text_emotion` | `false` | 调用 AmphionSPEC `sec`，并把结果提供给 refine |
| `hotwords.builtin` | `[]` | 最多一个：`finance`、`education`、`internet` |
| `hotwords.custom` | `[]` | 最多取前 100 个自定义术语 |

Qwen3-ASR 本身不接热词。`cleanup.level` 非 `off` 时，热词作为 refine LLM 的
glossary，用于术语纠错；`off` 不承诺热词生效。

### 音频与提交

```json
{"type":"input_audio_buffer.append","audio":"<base64 PCM>"}
```

```json
{"type":"input_audio_buffer.commit","final":true}
```

当前只支持 `final=true`。

## 服务端事件

原始识别累计结果：

```json
{"type":"transcription.delta","delta":"今天天气","text":"今天天气"}
```

情感增强开启时返回（对外 mode 只有 `ser` / `sec`，本接口使用 `sec`）：

```json
{"type":"emotion.bucket","emotion":{"mode":"sec","label":"happy","text":"语气轻快"}}
```

refine 或翻译结果：

```json
{"type":"postprocess.delta","postprocess_mode":"cleanup","delta":"今天天气。","text":"今天天气。"}
```

最终结果：

```json
{
  "type": "transcription.done",
  "session_id": "asr-clean-xxxxxxxxxxxx",
  "text": "今天天气",
  "cleaned_text": "今天天气。",
  "usage": {"type": "duration", "seconds": 4.2},
  "language": "zh",
  "cleanup_status": "completed"
}
```

翻译模式返回 `translated_text` / `translation_status`。refine 失败时仍返回原始
`text`，状态为 `degraded_raw_only`。

错误统一为 JSON：

```json
{"type":"error","session_id":"...","code":"no_speech_detected","message":"No speech detected."}
```

常见 `code`：`invalid_state`、`invalid_request`、`invalid_audio`、`no_audio`、
`no_speech_detected`、`audio_too_long`、`server_error`。
