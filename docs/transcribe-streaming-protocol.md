# /transcribe-streaming WebSocket 协议文档

> 端点命名约定：本服务按任务一类一个 WebSocket 端点（`/<task>-streaming`）。本文档主体描述的是「个性化语音识别」任务；目标说话人 ASR 使用 `/transcribe-target-streaming`（详见 [docs/tsasr.md](tsasr.md)），整段情感识别使用 `/emotion-streaming`（详见 [docs/emotion-streaming-protocol.md](emotion-streaming-protocol.md)），按段流式情感识别使用 `/emotion-segmented-streaming`（详见 [docs/emotion-segmented-streaming-protocol.md](emotion-segmented-streaming-protocol.md)）。所有端点共享相同的控制消息基础结构（`ready` / `start` / `stop` / `update_hotwords` / `error` / `start.config` 覆写机制），只是任务专属字段与输出语义不同。

## 当前支持的 Demo WS 端点总览

下表列出本服务当前对外提供的任务流式 WebSocket 接入方式，所有端点均共享 `ready / start / 二进制 PCM / stop / error` 的基础时序，差异集中在 `start` 必填字段与服务端推送的结果消息类型上。

| 任务 | 端点路径 | Query 参数 | 是否走 VAD 分段 | partial 输出 | 最终消息类型 | start 关键字段 | 详细协议 |
|---|---|---|---|---|---|---|---|
| 通用流式 ASR | /transcribe-streaming | language（可选） | 是 | 是（伪流式） | final_asr | mode=asr_only, format=pcm_s16le, sample_rate_hz=16000, channels=1 | 见本文档下文 |
| 目标说话人 ASR（TS-ASR） | /transcribe-target-streaming | language（可选） | 是 | 默认关闭，由 tsasr_enable_partial 控制 | final（task=tsasr） | format/sample_rate_hz/channels（可选）+ enrollment_audio（必填，base64 WAV） | docs/tsasr.md |
| 整段情感识别（SER/SEC） | /emotion-streaming | 无 | 否（整段缓存到 stop） | 否 | final_emotion | format=pcm_s16le, sample_rate_hz=16000, channels=1, mode=ser 或 sec（可选） | docs/emotion-streaming-protocol.md |
| 按段流式情感识别（SER/SEC） | /emotion-segmented-streaming | language（可选） | 是 | 否 | final_emotion（每段一条） | format=pcm_s16le, sample_rate_hz=16000, channels=1, mode=ser 或 sec（可选） | docs/emotion-segmented-streaming-protocol.md |

三类端点统一的接入步骤：

1. 建连：ws(s)://host:port/<endpoint>，按需附加 query 参数。
2. 等待服务端首条 ready。
3. 发送一条 start（JSON），声明音频参数与任务专属字段；TS-ASR 还会在校验通过后回一条 enrollment_ok。
4. 持续以二进制帧推送 PCM（统一 16 kHz / mono / s16le，建议每帧 30–80 ms）。
5. 期间按需收取 partial / partial_asr 等中间结果（仅 ASR 端点默认开启）。
6. 发送 {"type":"stop"} 结束本次会话；服务端 flush 后保证回一条最终结果（final_asr / final / final_emotion）再允许 close。
7. 发生错误时服务端推送 {"type":"error",...}，TS-ASR 的注册类错误 code 以 enrollment_ 为前缀，但不会主动断连。

所有端点均允许在 start.config 内平铺覆写服务端 Config 的白名单字段（例如 ASR 的 vad_threshold、TS-ASR 的 tsasr_enable_partial、情感的 emotion_request_timeout 等），仅本次会话生效。

## 概述

`/transcribe-streaming` 是面向上游服务（如 tiro_api）的 ASR WebSocket 接口。客户端通过该接口发送实时音频流，服务端返回增量转写结果（partial）和最终转写结果（final）。

## 连接

### URL

```
ws://<host>:<port>/transcribe-streaming?language=<lang>
wss://<host>:<port>/transcribe-streaming?language=<lang>
```

### Query 参数

| 参数       | 必填 | 说明                                                |
|------------|------|-----------------------------------------------------|
| `language` | 否   | 语言代码，如 `zh`、`en`、`id`、`th`。默认空（自动检测） |

### 请求头

无自定义请求头要求。仅使用 WebSocket 标准握手头。

---

## 消息时序

```
Client                                Server
  |                                      |
  |  ---- WebSocket 连接 (?language=zh) -->
  |                                      |
  |  <--------  ready  ---------------   |
  |                                      |
  |  ----  update_hotwords (可选)  ---->  |
  |  ----  start  -------------------->  |
  |                                      |
  |  ----  binary PCM chunk  --------->  |
  |  ----  binary PCM chunk  --------->  |
  |  <--------  partial_asr  ----------  |  (VAD 检测到语音期间，周期性输出)
  |  ----  binary PCM chunk  --------->  |
  |  <--------  partial_asr  ----------  |
  |  ----  binary PCM chunk  --------->  |
  |                                      |  (VAD 检测到语音结束)
  |  <--------  final_asr  ------------  |
  |                                      |
  |  ----  update_hotwords (可选)  ---->  |  (中途可随时更新热词)
  |                                      |
  |  ----  binary PCM chunk  --------->  |  (下一段语音...)
  |  ...                                 |
  |                                      |
  |  ----  stop  ----------------------> |
  |  <--------  final_asr  ------------  |  (残余音频的最终结果)
  |                                      |
  |  ---- 连接关闭 ---                    |
```

---

## 客户端 -> 服务端消息

### 1. start

建连收到 `ready` 后发送，声明音频参数。必须在发送 PCM 之前发送。

```json
{
  "type": "start",
  "mode": "asr_only",
  "format": "pcm_s16le",
  "sample_rate_hz": 16000,
  "channels": 1
}
```

| 字段             | 类型   | 必填 | 说明                                  |
|------------------|--------|------|---------------------------------------|
| `type`           | string | 是   | 固定 `"start"`                        |
| `mode`           | string | 是   | 固定 `"asr_only"`                     |
| `format`         | string | 是   | 音频编码，固定 `"pcm_s16le"`          |
| `sample_rate_hz` | int    | 是   | 采样率，固定 `16000`                  |
| `channels`       | int    | 是   | 声道数，固定 `1`                      |

### 2. update_hotwords

更新热词和语种。可在 start 之后、音频流期间随时发送。

```json
{
  "type": "update_hotwords",
  "hotwords": ["武新华", "挚音科技", "张硕"],
  "src_lang": "zh"
}
```

| 字段       | 类型     | 必填 | 说明                                                                 |
|------------|----------|------|----------------------------------------------------------------------|
| `type`     | string   | 是   | 固定 `"update_hotwords"`                                             |
| `hotwords` | string[] | 是   | 热词列表，空数组 `[]` 表示清除                                       |
| `src_lang` | string   | 否   | 语种代码或全称，如 `"zh"` / `"Chinese"` / `"en"` / `"English"` 等。不传则保持上一次的值 |

**语种映射表**：

| 短码 | 内部名称      |
|------|---------------|
| `zh` | Chinese       |
| `cn` | Chinese       |
| `en` | English       |
| `id` | Indonesian    |
| `th` | Thai          |

### 3. 二进制 PCM 音频帧

在 `start` 之后持续发送。每帧为原始 PCM bytes，格式要求：

- 编码：pcm_s16le（16-bit 有符号小端整数）
- 采样率：16000 Hz
- 声道：1（mono）
- 每毫秒 bytes 数：`16000 * 1 * 2 / 1000 = 32 bytes/ms`

推荐 chunk 大小：

| chunk 时长 | bytes 数 |
|------------|----------|
| 40 ms      | 1280     |
| 80 ms      | 2560     |
| 100 ms     | 3200     |

服务端对单包大小无严格限制，兼容各种 chunk 大小。

### 4. stop

结束音频流。服务端将 flush VAD 残余音频并返回最后的 `final_asr`。

```json
{
  "type": "stop"
}
```

---

## 服务端 -> 客户端消息

### 1. ready

连接建立后服务端发送的首条消息，表示服务就绪。

```json
{
  "type": "ready"
}
```

### 2. partial_asr

VAD 检测到语音期间，周期性输出的增量转写结果。每条 partial_asr 包含当前已累积语音的最新转写文本（非增量差分，而是当前最佳完整转写）。

```json
{
  "type": "partial_asr",
  "text": "你好世界",
  "language": "zh"
}
```

| 字段       | 类型   | 说明                                 |
|------------|--------|--------------------------------------|
| `type`     | string | `"partial_asr"` 或 `"partial"`（兼容） |
| `text`     | string | 当前转写文本                         |
| `language` | string | 识别语言                             |

**输出频率**：默认每 500ms 输出一次（受 `PSEUDO_STREAM_INTERVAL_MS` 环境变量控制）。主/次 ASR 模型任一启用即生效。

**噪声抑制**：当 secondary ASR 启用时，如果 secondary 输出为空，则 partial 被抑制不输出，防止噪声产生误识别。

**推理方式**：与 final_asr 一致，使用双路 ASR 并行推理 + 融合。

### 3. final_asr

VAD 检测到一段语音结束后，输出该段的最终转写结果。经过双路 ASR 融合。

```json
{
  "type": "final_asr",
  "text": "你好世界",
  "language": "zh"
}
```

| 字段       | 类型   | 说明                                   |
|------------|--------|----------------------------------------|
| `type`     | string | `"final_asr"` 或 `"final"`（兼容）     |
| `text`     | string | 最终转写文本                           |
| `language` | string | 识别语言                               |

**触发时机**：
- 每次 VAD 检测到语音结束时（静音超过阈值）
- 收到 `stop` 后 flush 残余音频时
- 一个 start/stop 周期内可能产生多个 `final_asr`（对应多个 VAD 段）

### 4. error

发生错误时返回。

```json
{
  "type": "error",
  "message": "错误描述"
}
```

---

## 典型调用示例

### Python (websockets)

```python
import asyncio, json, websockets, ssl

async def transcribe(audio_pcm_bytes: bytes):
    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    async with websockets.connect(
        "wss://host:8443/transcribe-streaming?language=zh",
        ssl=ssl_ctx,
    ) as ws:
        # 等待 ready
        ready = json.loads(await ws.recv())
        assert ready["type"] == "ready"

        # 发送热词
        await ws.send(json.dumps({
            "type": "update_hotwords",
            "hotwords": ["武新华", "挚音科技"],
        }))

        # 发送 start
        await ws.send(json.dumps({
            "type": "start",
            "mode": "asr_only",
            "format": "pcm_s16le",
            "sample_rate_hz": 16000,
            "channels": 1,
        }))

        # 发送音频（每 80ms 一个 chunk）
        chunk_size = 2560  # 80ms
        for i in range(0, len(audio_pcm_bytes), chunk_size):
            await ws.send(audio_pcm_bytes[i:i+chunk_size])
            await asyncio.sleep(0.08)

        # 发送 stop
        await ws.send(json.dumps({"type": "stop"}))

        # 接收结果
        async for msg in ws:
            data = json.loads(msg)
            if data["type"] == "partial_asr":
                print(f"[partial] {data['text']}")
            elif data["type"] == "final_asr":
                print(f"[final]   {data['text']}")
            elif data["type"] == "error":
                print(f"[error]   {data['message']}")
```

### 测试客户端

项目自带测试脚本 `tests/test_ws_client.py`：

```bash
# 基本用法
python tests/test_ws_client.py audio.wav

# 指定热词
python tests/test_ws_client.py audio.wav --hotwords "武新华,挚音科技,张硕"

# 指定语言和 chunk 大小
python tests/test_ws_client.py audio.wav --language en --chunk-ms 100

# 自定义服务地址
python tests/test_ws_client.py audio.wav --url ws://10.0.0.1:8907/transcribe-streaming
```

---

## 兼容性说明

本接口设计兼容 tiro_api 的 ASR backend 协议：

| 契约项                 | 支持情况                                     |
|------------------------|----------------------------------------------|
| WS 路径                | `/transcribe-streaming`                      |
| query `language`       | 支持                                         |
| `start` / `stop` 消息  | 完整支持                                     |
| PCM 音频（16k/mono/s16le）| 完整支持                                  |
| `partial_asr` / `partial` | 输出 `partial_asr`                        |
| `final_asr` / `final`    | 输出 `final_asr`                           |
| `error` 消息           | 完整支持                                     |
| 握手鉴权头             | 无要求（内网免鉴权）                         |
| `update_hotwords`      | 扩展支持（tiro_api 当前不使用，可选）        |

tiro_api 侧对接只需修改 `TIRO_API_ASR_WS_BACKENDS` 指向本服务即可。

---

## 环境变量

以下环境变量影响 `/transcribe-streaming` 的行为（均有默认值）：

| 变量                         | 默认值        | 说明                               |
|------------------------------|---------------|------------------------------------|
| `ENABLE_PRIMARY_ASR`         | `1`           | 是否启用主 ASR 模型                |
| `ENABLE_SECONDARY_ASR`       | `1`           | 是否启用次 ASR 模型                |
| `ENABLE_PSEUDO_STREAM`       | `1`           | 是否启用伪流式 partial 输出        |
| `PSEUDO_STREAM_INTERVAL_MS`  | `500`         | partial 输出最小间隔（ms）         |
| `PRIMARY_ASR_TIMEOUT`        | `4.0`         | 主 ASR 单次请求超时（秒）          |
| `ASR_REQUEST_TIMEOUT`        | `120`         | ASR HTTP 请求总超时（秒）          |
| `VAD_THRESHOLD`              | `0.5`         | VAD 语音概率阈值                   |
| `SILENCE_DURATION_MS`        | `200`         | 静音多久判定语音结束（ms）         |
| `MIN_SEGMENT_DURATION_MS`    | `350`         | 最短 VAD 段时长（ms），更短的丢弃  |
