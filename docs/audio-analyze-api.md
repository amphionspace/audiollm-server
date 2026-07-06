# 非实时音频分析 API

`POST /api/audio/analyze` 面向离线音频处理场景。调用方上传一段 WAV 音频，服务端返回 ASR 识别文本、清洗后的转写文本，以及情感理解结果。情感理解默认同时返回分类标签和文本描述。

热词偏置来自当前 `hotword_pool_id` 对应的 RAG-ASR 热词池对音频召回的 top-K 结果，并让少量表单 `hotwords` 临时热词优先进入 ASR prompt（默认最多 8 个，去重后不写入热词池），同时过滤与临时热词精确重复或整词同音（忽略声调）的召回词。`hotwords` 不会传给文本清洗阶段；文本清洗只负责标点、空格、重复词和明显文本格式问题，不做基于热词的事后替换。

## 接口信息

| 项目 | 说明 |
|---|---|
| 协议 | HTTP |
| 方法 | POST |
| 路径 | `/api/audio/analyze` |
| Content-Type | `multipart/form-data` |
| 鉴权 | AudioLLM 服务本身无内置鉴权 |
## 请求字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `audio` | file | 是 | WAV 音频文件 |
| `language` | string | 否 | 语言代码，如 `zh`、`en` |
| `hotword_pool_id` | string | 否 | 推荐字段，热词池隔离 ID，默认 `default` |
| `hotwords` | string | 否 | 临时请求热词；去重限量后优先进入 ASR prompt，并覆盖精确重复或整词同音（忽略声调）的 RAG-ASR 召回热词，不写入热词池 |

## 调用示例

完整脚本见 [examples/rest_upload.py](examples/rest_upload.py)。

```bash
python docs/examples/rest_upload.py analyze sample.wav \
  --base-url http://172.16.0.3:8082 \
  --language zh \
  --hotwords "挚音科技,张硕" \
```

## 响应结构

```json
{
  "type": "audio_analysis",
  "duration_sec": 8.24,
  "language": "zh",
  "hotwords": ["挚音科技", "张硕"],
  "asr": {
    "text": "融合后的 ASR 文本",
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
      "duration_sec": 8.24,
      "language": "zh"
    },
    "sec": {
      "type": "final_emotion",
      "mode": "sec",
      "label": "Neutral",
      "text": "The speaker sounds calm and neutral.",
      "duration_sec": 8.24,
      "language": "zh"
    }
  }
}
```

## 字段说明

| 字段 | 说明 |
|---|---|
| `asr.text` | 服务端 ASR 最终融合后的文本，用作清洗模型输入 |
| `cleaned_asr.text` | 清洗后的最终文本 |
| `emotion.ser` | 情感分类标签结果 |
| `emotion.sec` | 情感文本描述结果 |

## 错误处理

| 状态码 | 场景 |
|---|---|
| 400 | 音频为空或无法解码 |
| 413 | 上传文件超过服务端大小限制 |
| 422 | multipart 字段缺失或类型不合法 |
| 502 | ASR 模型全部失败 |
| 502 | ASR、情感或文本清洗模型调用失败 |

## 相关文档

- [API 总览](api-reference.md)
- [通用 ASR 上传](transcribe-streaming-protocol.md#rest-上传接口)
- [整段情感识别 HTTP](emotion-streaming-protocol.md)
