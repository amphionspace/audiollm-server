# 长音频离线转写 API（会议纪要）

`POST /api/asr/transcriptions` 面向整段长录音（会议、访谈等）的离线转写：上传完整 WAV，服务端异步完成 VAD 切分与逐段双模型转写，客户端轮询取回带段级时间戳的分段转写稿。

与其他 ASR 入口的分工：

| 入口 | 适用场景 | 局限 |
|---|---|---|
| `POST /api/asr/upload` | 一句话/短录音（≤60 秒），同步返回 | 超长部分被尾截丢弃 |
| `WS /transcribe-streaming` | 实时/准实时流 | 快灌长文件会触发队列背压丢段，且不带段级时间戳 |
| `POST /api/asr/transcriptions` | 会后整段长音频（默认上限 3 小时） | 异步，需轮询 |

## 接口信息

| 项目 | 说明 |
|---|---|
| 协议 | HTTP |
| 方法 | POST（提交）+ GET（轮询） |
| 路径 | `/api/asr/transcriptions`、`/api/asr/transcriptions/{job_id}` |
| Content-Type | `multipart/form-data`（提交） |
| 鉴权 | AudioLLM 服务本身无内置鉴权 |

## 提交转写任务

### 请求字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `audio` | file | 是 | WAV 文件（PCM 8/16/24/32-bit，任意采样率与声道数，服务端重采样到 16 kHz mono）。压缩格式（flac/mp3/m4a）需客户端先转码，例如 `ffmpeg -i in.flac -ac 1 -ar 16000 -sample_fmt s16 out.wav` |
| `language` | string | 否 | 语言提示，如 `zh`、`en`；空为自动检测 |
| `hotword_pool_id` | string | 否 | 推荐字段，热词池隔离 ID，默认 `default`；每段 ASR 都使用该热词池召回 |
| `hotwords` | string | 否 | 临时请求热词；每段 ASR 会把去重后的前 `recall_custom_hotword_limit` 个优先注入 prompt，并覆盖精确重复或整词同音（忽略声调）的 RAG-ASR 召回热词，不写入热词池 |

不支持 `enrollment_id`：目标说话人过滤只保留单一说话人的语音，与多人会议转写语义相反。

### 约束

| 约束 | 默认值 | 超出行为 |
|---|---|---|
| 文件大小 | 512 MB（`transcribe_max_upload_bytes`） | 413 |
| 解码后时长 | 3 小时（`transcribe_max_audio_sec`） | 400 拒绝，不做截断（静默丢内容不可接受，请分割文件） |
| 队列容量 | 8 个活跃任务（`transcribe_job_queue_max`） | 503 + `Retry-After` |

### 调用示例

```bash
curl -X POST http://172.16.0.3:8082/api/asr/transcriptions \
  -F "audio=@meeting.wav" \
  -F "language=zh" \
  -F "hotwords=挚音科技,张硕"
```

受理响应（202）：

```json
{
  "job_id": "tr_6f0c2a8e9b3d41a7c5e21f08",
  "status": "queued",
  "poll_url": "/api/asr/transcriptions/tr_6f0c2a8e9b3d41a7c5e21f08",
  "duration_sec": 1949.076
}
```

## 轮询任务状态

`GET /api/asr/transcriptions/{job_id}`，建议间隔 2-5 秒。

状态机：`queued` → `running` → `succeeded` | `failed`。

运行中响应（`segments_total` 在切分完成前为 `null`）：

```json
{
  "job_id": "tr_6f0c2a8e9b3d41a7c5e21f08",
  "status": "running",
  "created_at": 1781167029.28,
  "updated_at": 1781167040.10,
  "progress": {
    "segments_total": 636,
    "segments_done": 260
  }
}
```

成功响应：

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
      { "id": 0, "start_ms": 21400, "end_ms": 22300, "text": "师傅好啊，师傅好啊！", "language": "zh" },
      { "id": 1, "start_ms": 22800, "end_ms": 24400, "text": "009，我是。", "language": "zh" }
    ]
  }
}
```

### result 字段说明

| 字段 | 说明 |
|---|---|
| `full_text` | 各段文本按时间序以换行拼接 |
| `language` | 请求指定的语言，未指定时取首个检测结果 |
| `duration_sec` | 整段录音时长（秒） |
| `segments[*].id` | 段序号（切分顺序，丢弃的噪声段不补位，序号可能不连续） |
| `segments[*].start_ms` / `end_ms` | 该段在录音内的近似位置（毫秒）。段级精度：含 VAD 起音回填与静音确认窗的偏移，非词级对齐 |
| `segments[*].text` | 该段转写文本，已做 ITN 与车牌规范化（与 `/api/asr/upload` 一致） |
| `segments[*].language` | 该段检测语言（可选） |
| `segments[*].error` | 仅失败段携带：推理错误信息（见下） |
| `failed_segments` | 推理失败的段数 |

### 部分失败语义

单段推理失败会自动重试一次；仍失败时该段以 `error` 字段占位保留在 `segments` 中（`text` 为空），任务整体仍为 `succeeded`——一段失败不应丢弃整场会议。只有所有段都失败时任务才记为 `failed`。模型转写为空的段（VAD 误放行的噪声）不出现在结果中。

失败响应：

```json
{
  "job_id": "tr_…",
  "status": "failed",
  "progress": { "segments_total": 636, "segments_done": 636 },
  "error": {
    "message": "all segments failed ASR inference",
    "code": "inference_failed"
  }
}
```

### 结果生命周期

结果保存在服务进程内存中，保留 `transcribe_job_ttl_sec`（默认 1 小时），过期或服务重启后轮询返回 404。客户端应在 `succeeded` 后立即取走结果。

## 错误码

| 状态码 | 含义 |
|---|---|
| 202 | 已受理，返回 `job_id` 与 `poll_url` |
| 400 | 音频为空 / 无法解码（非 PCM WAV）/ 时长超过 `transcribe_max_audio_sec` |
| 404 | 任务不存在、已过期或服务已重启 |
| 413 | 文件超过 `transcribe_max_upload_bytes` |
| 422 | multipart 字段不符合 FastAPI 校验 |
| 503 | 任务队列已满，按 `Retry-After` 重试 |

## 处理流程与一致性保证

1. 解码并重采样到 16 kHz mono。
2. 用与流式端点同一套 TEN VAD 状态机及参数离线切分，因此同一段录音走 WS 或走本接口得到一致的段边界；连续无停顿语音超过 `transcribe_max_segment_sec`（默认 30 秒）时强制切段兜底。
3. 每段并行（`transcribe_segment_concurrency`）执行与 `/api/asr/upload` 相同的一次性双模型推理（融合开关同全局 `enable_dual_asr_fusion`），失败重试一次。
4. 每段送模前还会执行 `asr_segment_voice_gate_*` / `asr_segment_voice_filter_*` 服务端人声保护：低证据段可能不调用模型，已放行段可能裁去非人声区间。门控返回空文本的段不进入最终 `segments`。
5. 按时间序组装 `segments` 与 `full_text`。

## 服务端配置

均为进程级配置，客户端不可经任何接口覆写，修改后重启生效。

### 推理模型绑定（config.yaml `rest.routes.transcribe`）

`rest.upstreams` 是所有 REST 接口共享的角色到上游绑定（`/api/asr/upload`、`/api/audio/analyze`、情感 jobs）；`rest.routes.<name>` 是单接口覆写。长音频转写用哪个模型、是否双模型融合在 `rest.routes.transcribe` 单独声明，整块或单项省略时回退共享绑定与全局开关：

```yaml
rest:
  upstreams:                         # 共享绑定: 所有 REST 接口的默认上游
    primary: amphion_asr
    secondary: qwen_asr
  routes:
    transcribe:                      # 长音频转写专属覆写（省略 = 跟随上面）
      upstreams:
        primary: amphion_asr         # 转写主模型，换模型只影响本接口
        secondary: qwen_asr          # 仅融合开启时参与
      enable_dual_asr_fusion: false  # 转写是否双模型融合（独立于全局开关）
```

| 键 | 说明 |
|---|---|
| `upstreams.primary` | 转写主模型（`upstreams` 池中的名字）；省略则跟随 `rest.upstreams.primary` |
| `upstreams.secondary` | 融合副模型；仅 `enable_dual_asr_fusion: true` 时被调用 |
| `enable_dual_asr_fusion` | 本接口专属融合开关。离线无延迟压力，质量优先可开 `true`；代价是每段 2 个 vLLM 请求，吞吐约减半。需全局 `enable_secondary_asr: true`，否则自动降级 `false` 并打 WARN |

未知键、未知角色、未知 upstream 名、未知路由名在启动时直接报错（拼写错误静默回退会复现"看不出用了哪个模型"的问题）；旧的扁平 `rest.transcribe` 写法也会被拒绝，不会静默回退。

### 调参（config.yaml `defaults.transcribe`）

默认值经过实测扫参验证（并发、时延、吞吐与容量上界数据见[性能报告](../benchmarks/transcription-jobs-benchmark.md)）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `transcribe_max_concurrent_jobs` | `2` | 同时 running 的任务数 |
| `transcribe_segment_concurrency` | `4` | 单任务内并行推理的段数；总 vLLM 压力为两者乘积（默认 2×4=8） |
| `transcribe_job_queue_max` | `8` | 活跃任务上限（含 queued + running），超出返回 503 |
| `transcribe_job_ttl_sec` | `3600` | 终态结果保留秒数 |
| `transcribe_max_segment_sec` | `30.0` | 连续语音强切上限 |
| `transcribe_max_upload_bytes` | `536870912` | 上传字节上限（512 MB） |
| `transcribe_max_audio_sec` | `10800` | 解码后时长上限（3 小时），超出 400 拒绝 |
| `transcribe_silence_duration_ms` | `800` | 仅本接口生效的切段停顿阈值；`0` = 跟随全局 `silence_duration_ms` |

### 切段停顿调参（`transcribe_silence_duration_ms`)

全局 `silence_duration_ms`（350 ms）是为实时端点的低延迟调的：停顿阈值越短，`final` 出得越快。离线转写没有这个延迟约束，更长的阈值能把被短暂停顿打碎的句子合并成更完整的段落。该参数只作用于本接口的切分，实时端点不受影响；调大全局 `silence_duration_ms` 则会同时拉高所有实时端点的 final 延迟，不要为纪要场景去动它。

实测参考（32.5 分钟 8 麦阵列会议录音，AISHELL-4 风格）：350 ms 下切出 636 段、段长 p50 仅 1.5 秒，对纪要偏碎；800 ms 切 303 段、p50 3.2 秒，且因每段请求开销摊薄，转写还快约 11%（完整扫参见[性能报告](../benchmarks/transcription-jobs-benchmark.md)第 5 节）。多人快节奏讨论本身停顿少，段仍会偏短，属正常现象。

## Python 示例

完整脚本见 [examples/http_transcribe_job.py](../examples/http_transcribe_job.py)（依赖 `pip install requests`）：

```bash
python docs/examples/http_transcribe_job.py meeting.wav \
  --base-url http://172.16.0.3:8082 \
  --language zh \
  --hotwords "挚音科技,张硕" \
  --full-text-only
```

脚本会在 stderr 打印进度（`segments_done/total`），结束后输出 `result` JSON（或 `--full-text-only` 时只输出全文）。

## 已知限制

- 无说话人分离：多人发言在 `segments` 中按时间排列，但不标注说话人。
- 时间戳为段级近似值，不支持词级对齐。
- 任务与结果不持久化（进程内存），服务重启即丢失。
- 仅接受 WAV 容器；压缩格式需客户端先转码。
