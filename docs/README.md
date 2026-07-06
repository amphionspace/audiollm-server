# 文档索引

本目录按文档用途分层：通用服务 API 与协议文档保留在 `docs/` 根目录；合作方对接资料放在 `docs/partners/`。

## 通用 API

| 文档 | 说明 |
| ---- | ---- |
| [API 总览](api-reference.md) | REST 与 WebSocket 能力总览 |
| [音频分析 API](audio-analyze-api.md) | `/api/audio/analyze` 接口 |
| [公开音频分析 API](public-audio-analyze-api.md) | 对外公开音频分析接口 |
| [转写任务 API](transcription-jobs-api.md) | 离线转写任务接口 |

## WebSocket 协议

| 文档 | 说明 |
| ---- | ---- |
| [通用流式 ASR WebSocket](transcribe-streaming-protocol.md) | `/transcribe-streaming` 协议 |
| [情感流式 WebSocket](emotion-streaming-protocol.md) | 情感流式协议 |
| [分段情感识别 WebSocket](emotion-segmented-streaming-protocol.md) | `/emotion-segmented-streaming` 协议 |
| [实时转写 AST v3 WebSocket](tuling-ast-v3-protocol.md) | `/tuling/ast/v3` 讯飞图灵 AST v3 兼容协议 |

## 合作方对接

| 目录 | 说明 |
| ---- | ---- |
| [partners](partners/README.md) | TMGenius、鼎桥（TD Tech）等合作方对接资料 |

## 压测报告

| 文档 | 说明 |
| ---- | ---- |
| [转写任务压测报告](transcription-jobs-benchmark.md) | 离线转写任务压测数据 |
| [AST v3 多并发性能与极限压测报告](tuling-ast-v3-benchmark.md) | AST v3 WebSocket 压测数据 |
