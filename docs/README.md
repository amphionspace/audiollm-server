# 文档索引

本目录按文档用途分层：总览文档保留在 `docs/` 根目录，专题文档分别放在 `api/`、`protocols/`、`benchmarks/`、`partners/` 与 `examples/`。

## 通用 API

| 文档 | 说明 |
| ---- | ---- |
| [API 总览](api-reference.md) | REST 与 WebSocket 能力总览 |
| [音频分析 API](api/audio-analyze-api.md) | `/api/audio/analyze` 接口 |
| [公开音频分析 API](api/public-audio-analyze-api.md) | 对外公开音频分析接口 |
| [转写任务 API](api/transcription-jobs-api.md) | 离线转写任务接口 |

## WebSocket 协议

| 文档 | 说明 |
| ---- | ---- |
| [通用流式 ASR WebSocket](protocols/transcribe-streaming-protocol.md) | `/transcribe-streaming` 协议 |
| [情感流式 WebSocket](protocols/emotion-streaming-protocol.md) | 情感流式协议 |
| [分段情感识别 WebSocket](protocols/emotion-segmented-streaming-protocol.md) | `/emotion-segmented-streaming` 协议 |
| [实时转写 AST v3 WebSocket](protocols/tuling-ast-v3-protocol.md) | `/tuling/ast/v3` 讯飞图灵 AST v3 兼容协议 |

## 合作方对接

| 目录 | 说明 |
| ---- | ---- |
| [partners](partners/README.md) | TMGenius、鼎桥（TD Tech）等合作方对接资料 |

## 压测报告

| 文档 | 说明 |
| ---- | ---- |
| [转写任务压测报告](benchmarks/transcription-jobs-benchmark.md) | 离线转写任务压测数据 |
| [AST v3 多并发性能与极限压测报告](benchmarks/tuling-ast-v3-benchmark.md) | AST v3 WebSocket 压测数据 |
| [Streaming Sortformer 同卡生产压测报告](benchmarks/speaker-diarization-production-load.md) | 单 H20 上 AST v3 与角色分离 sidecar 的 A/B 性能、过载降级和生产水位 |
| [Streaming Sortformer 与讯飞角色分离对比](benchmarks/speaker-diarization-iflytek-comparison.md) | AliMeeting 3×60 秒同音频盲分冒烟对比、评分口径与适用边界 |
