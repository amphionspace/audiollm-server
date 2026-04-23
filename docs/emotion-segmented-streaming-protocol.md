# /emotion-segmented-streaming WebSocket 协议文档

> 端点命名约定：本服务按任务一类一个 WebSocket 端点（`/<task>-streaming`）。本文档描述 VAD 切段的流式情感识别任务，与整段非流式版本 `/emotion-streaming`（详见 [docs/emotion-streaming-protocol.md](emotion-streaming-protocol.md)）并列存在；二者共享同一套控制消息基础结构（`ready` / `start` / `stop` / `error` / `start.config` 覆写机制）以及同一个底层情感模型，仅切段策略与最终消息节奏不同。

## 概述

`/emotion-segmented-streaming` 在客户端持续推送 PCM 流的过程中按 VAD 切段，每段语音结束即触发一次情感推理，按段返回 `final_emotion`。底层模型与 prompt 与 [docs/emotion-streaming-protocol.md](emotion-streaming-protocol.md) 完全一致，支持 `ser` / `sec` 两种任务变体（见原文档的标签集说明）。

适用场景：长时间持续录音、需要按句更新情感判断的对话/直播链路。若上游已自行做了句级切段、只需「整段一次性给出情感」，请使用 `/emotion-streaming`。

与 `/emotion-streaming` 的关键差异：

| 维度 | /emotion-streaming | /emotion-segmented-streaming |
|---|---|---|
| 是否启用 VAD | 否（整段缓存到 stop） | 是（同 /transcribe-streaming） |
| final_emotion 数量 | 每个 start/stop 周期 1 条；空会话兜底返回一条 label="" 的 final_emotion | 每个 VAD 段 1 条；空会话不返回 final_emotion |
| stop 后行为 | 触发整段推理 | flush 残余尾段；若无残余则不再产出 |
| partial 输出 | 否 | 否（情感任务一律不出 partial） |
| 模型 / prompt / mode 字段 | 完全一致 | 完全一致 |
| max_audio_seconds 作用域 | 整段 | 每个 VAD 段独立生效 |

## 连接

```
ws://<host>:<port>/emotion-segmented-streaming?language=<lang>
wss://<host>:<port>/emotion-segmented-streaming?language=<lang>
```

### Query 参数

| 参数     | 必填 | 说明                                          |
|----------|------|-----------------------------------------------|
| language | 否   | 语言代码，如 zh/en/id/th。仅作信息透传（情感模型本身与语言解耦），若提供，服务端会在每条 final_emotion 中回填 language 字段以便上游路由 |

## 消息时序

```
Client                                Server
  |                                      |
  |  ---- WebSocket 连接 -------------->  |
  |  <--------  ready  ---------------   |
  |  ----  start (含 mode) ----------->   |
  |  ----  binary PCM chunk  -------->    |
  |  ----  binary PCM chunk  -------->    |
  |  <--------  final_emotion  -------    |  每个 VAD 段一条
  |  ----  binary PCM chunk  -------->    |
  |  ----  ...                            |
  |  <--------  final_emotion  -------    |
  |  ----  stop  -------------------->    |
  |  <--------  final_emotion  -------    |  仅当存在尾部残余段时
  |  ---- 连接关闭 ---                     |
```

注意：与 `/emotion-streaming` 不同，stop 之后不会强行返回一条空 `final_emotion`。如果整个会话没有任何被 VAD 判定为语音的片段，客户端将收不到任何 `final_emotion`，请按 `stop` + WebSocket close 作为会话结束信号。

## 客户端 -> 服务端消息

### 1. start

建连收到 `ready` 后发送，声明音频参数与任务模式。必须在发送 PCM 之前发送，一次会话只能发一条。

```json
{
  "type": "start",
  "format": "pcm_s16le",
  "sample_rate_hz": 16000,
  "channels": 1,
  "mode": "ser",
  "language": "zh",
  "config": {
    "emotion_request_timeout": 20.0,
    "min_segment_duration_ms": 500
  }
}
```

| 字段           | 类型   | 必填 | 说明                                                     |
|----------------|--------|------|----------------------------------------------------------|
| type           | string | 是   | 固定 "start"                                             |
| format         | string | 否   | 固定 "pcm_s16le"                                         |
| sample_rate_hz | int    | 否   | 16000                                                    |
| channels       | int    | 否   | 1                                                        |
| mode           | string | 否   | 任务变体 "ser" 或 "sec"，缺省读 Config.emotion_task_mode |
| language       | string | 否   | 与 query 参数二选一，仅作元信息透传                      |
| config         | object | 否   | 服务端 Config 字段平铺覆写（同 /emotion-streaming 机制） |

### 2. 二进制 PCM 音频帧

格式与 `/transcribe-streaming` 完全一致：16 kHz、单声道、s16le。建议每帧 30-80 ms。服务端边收边走 VAD，触发段尾时立即排队推理。

### 3. stop

```json
{ "type": "stop" }
```

收到后服务端立即 flush VAD 残余 PCM；若残余构成有效段则再回一条 `final_emotion`，否则直接进入关闭流程。

## 服务端 -> 客户端消息

| 消息                                              | 说明                                                                |
|---------------------------------------------------|---------------------------------------------------------------------|
| `{"type":"ready"}`                                | WebSocket 就绪，可发送 start                                        |
| `{"type":"final_emotion", ...}`                   | 单段语音情感推理结果（payload 结构与 /emotion-streaming 完全一致） |
| `{"type":"error","message":"..."}`                | 推理或控制消息处理中的异常通知；不会主动断开                        |

### final_emotion payload

字段含义、SER / SEC 输出形态、固定 8 分类标签集等均与 [docs/emotion-streaming-protocol.md](emotion-streaming-protocol.md) 一致，此处不再重复。需要注意的点：

- duration_sec 是当前 VAD 段（可能再被 emotion_max_audio_seconds 截尾）的实际推理时长，不是会话累计时长。
- mode 字段在整个会话内固定为 start 时选定的值。
- 如客户端 start 中传入了 language，每条 final_emotion 都会带回 language 字段。

## 限制 / 默认值

| 项                          | 默认值 | 说明                                                                                       |
|-----------------------------|--------|--------------------------------------------------------------------------------------------|
| emotion_vllm_base_url       | http://localhost:8000 | 同 /emotion-streaming                                                       |
| emotion_vllm_model_name     | Amphion/Amphion-3B    | 同 /emotion-streaming                                                       |
| emotion_request_timeout     | 30s                   | 单次推理超时                                                                |
| emotion_max_audio_seconds   | 20s                   | 每个 VAD 段单独生效；超出则保留尾部 20s。长句被切段后通常远低于此上限       |
| emotion_task_mode           | ser                   | 缺省任务变体；可被 start.mode 覆盖                                          |
| min_segment_duration_ms     | 与 ASR 共用           | 短于此阈值的 VAD 段会被丢弃（避免对噪声/咳嗽起推理）                        |
| vad_threshold / silence_duration_ms / vad_*  | 与 ASR 共用 | VAD 灵敏度参数；与 /transcribe-streaming 完全相同                       |
| enable_pseudo_stream        | -                     | 不影响本端点。情感任务始终不输出 partial，VAD 的 partial 快照路径已被显式关闭 |

服务端 `Config` 中所有 `emotion_*` 字段以及 VAD / 切段相关字段均可由 `start.config` 临时覆写。

## 调用示例

```python
import asyncio, json, ssl, websockets

async def stream_emotion(pcm_bytes: bytes, mode: str = "ser"):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    async with websockets.connect(
        "wss://localhost:8443/emotion-segmented-streaming?language=zh", ssl=ctx
    ) as ws:
        ready = json.loads(await ws.recv())
        assert ready["type"] == "ready"

        await ws.send(json.dumps({
            "type": "start",
            "format": "pcm_s16le",
            "sample_rate_hz": 16000,
            "channels": 1,
            "mode": mode,
        }))

        async def feed():
            for i in range(0, len(pcm_bytes), 2560):
                await ws.send(pcm_bytes[i:i + 2560])
                await asyncio.sleep(0.08)
            await ws.send(json.dumps({"type": "stop"}))

        asyncio.create_task(feed())

        async for msg in ws:
            data = json.loads(msg)
            print(data)
            if data.get("type") == "error":
                break
```
