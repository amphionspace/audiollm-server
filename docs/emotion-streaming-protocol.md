# 整段情感识别 API（异步 HTTP）

整段情感识别已改为 **异步 HTTP 任务 API**，不再提供 WebSocket `/emotion-streaming`。

如果需要长连接中按 VAD 语音段持续返回情感结果，请使用 [分段情感识别 WebSocket](emotion-segmented-streaming-protocol.md)。

## 接口信息

| 项目 | 说明 |
|---|---|
| 创建任务 | `POST /api/emotion/jobs` |
| 查询任务 | `GET /api/emotion/jobs/{job_id}` |
| Base URL | `http://172.16.0.3:8082` |
| 鉴权 | 无内置鉴权 |
| 音频输入 | `multipart/form-data` 字段 `audio`（WAV 文件） |
| 中间结果 | 不支持 |
| 最终结果 | 任务 `status=succeeded` 时返回 `result`（`final_emotion`） |

## 模式

| mode | 说明 | 输出 |
|---|---|---|
| `ser` | Speech Emotion Recognition | 8 分类情感标签 |
| `sec` | Speech Emotion Captioning | 自由文本情感描述，并尽量给出匹配标签 |

`mode` 不传时使用服务端配置 `emotion_task_mode`，通常为 `ser`。

### SER 标签集

| 标签 | 含义 |
|---|---|
| `Angry` | 愤怒 |
| `Sad` | 悲伤 |
| `Happy` | 高兴 |
| `Surprise` | 惊讶 |
| `Fear` | 害怕 |
| `Disgust` | 厌恶 |
| `Contempt` | 轻蔑 |
| `Neutral` | 中性 |

## 调用流程

```text
Client                                      Server
  | ---- POST /api/emotion/jobs (WAV) ----> |
  | <---------------- 202 + job_id ------- |
  | ---- GET /api/emotion/jobs/{id} -----> |
  | <---------------- status=queued ------- |
  | ---- GET (poll) ----------------------> |
  | <---------------- status=running ------ |
  | ---- GET (poll) ----------------------> |
  | <---------------- status=succeeded ---- |
```

推荐轮询间隔 300–500 ms，总等待时间不超过 `emotion_request_timeout + 10s`。

## 创建任务

`POST /api/emotion/jobs`

| 表单字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `audio` | file | 是 | WAV 音频文件 |
| `mode` | string | 否 | `ser` 或 `sec` |
| `language` | string | 否 | 透传语言字段 |

### 成功响应（202）

```json
{
  "job_id": "em_8f3c2a1b4e5d6789",
  "status": "queued",
  "poll_url": "/api/emotion/jobs/em_8f3c2a1b4e5d6789"
}
```

### 背压（503）

等待队列满时返回 `503`，响应头含 `Retry-After: 5`。队列与并发上限见配置项 `emotion_job_queue_max`、`emotion_max_concurrent_jobs`。

## 查询任务

`GET /api/emotion/jobs/{job_id}`

### 进行中

```json
{
  "job_id": "em_8f3c2a1b4e5d6789",
  "status": "running",
  "created_at": 1710000000.12,
  "updated_at": 1710000001.05
}
```

### 成功

```json
{
  "job_id": "em_8f3c2a1b4e5d6789",
  "status": "succeeded",
  "created_at": 1710000000.12,
  "updated_at": 1710000002.31,
  "result": {
    "type": "final_emotion",
    "mode": "ser",
    "label": "Happy",
    "text": "Happy",
    "duration_sec": 3.21,
    "language": "zh"
  }
}
```

### 失败

```json
{
  "job_id": "em_8f3c2a1b4e5d6789",
  "status": "failed",
  "error": {
    "message": "emotion model request timed out",
    "code": "inference_timeout"
  }
}
```

无有效音频时仍返回 `succeeded`，`result` 中 `label`/`text` 为空、`duration_sec` 为 `0`。

## 可配置参数（服务端 config.yaml）

| 字段 | 默认 | 说明 |
|---|---|---|
| `emotion_max_concurrent_jobs` | 8 | 同时调用 vLLM 的上限 |
| `emotion_job_queue_max` | 64 | 排队任务上限 |
| `emotion_job_ttl_sec` | 3600 | 已完成任务元数据保留秒数 |
| `emotion_request_timeout` | 30 | 单次 vLLM 请求超时 |
| `emotion_max_audio_seconds` | 20 | 超长音频保留尾部秒数 |

## Python 示例

完整可运行脚本见 [examples/http_emotion_job.py](examples/http_emotion_job.py)。

```bash
pip install requests

python docs/examples/http_emotion_job.py sample.wav \
  --base-url http://172.16.0.3:8082 \
  --mode ser \
  --language zh
```

## 部署说明

- Job 状态保存在 **进程内存** 中；`uvicorn --workers N` 且 N>1 时，创建与轮询必须命中同一 worker，或后续引入 Redis 等共享存储。
- 单 worker systemd 部署（`172.16.0.3:8082`）下可直接使用本 API。

## 相关文档

- [API 总览](api-reference.md)
- [分段情感识别 WebSocket](emotion-segmented-streaming-protocol.md)