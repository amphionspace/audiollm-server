# /emotion-streaming WebSocket 协议文档

## 概述

`/emotion-streaming` 是面向上游服务的「整段音频情感识别」WebSocket 接口。客户端持续推送一段语音的 PCM 数据，服务端在收到 `stop` 后对完整片段做一次情感推理，返回一条 `final_emotion`。

> 需要按 VAD 切段、长录音内多次返回情感判断的场景，请使用按段流式版本 [`/emotion-segmented-streaming`](emotion-segmented-streaming-protocol.md)。两个端点共享同一套底层模型与 prompt，仅切段策略与 final_emotion 节奏不同。

底层模型与 prompt 与 AmphionASR 项目的多任务模型对齐，支持两种任务变体：

| 模式 | 训练 prompt | 输出形态 |
|---|---|---|
| ser | Classify the emotion of the following audio: | 8 分类标签字符串（见下文标签集） |
| sec | Describe the emotion of the following audio: | 自由文本情感描述 |

通过 `start.mode` 选择，缺省读取 `Config.emotion_task_mode`（默认 `ser`）。

与 `/transcribe-streaming` 的关键差异：

| 维度 | /transcribe-streaming | /emotion-streaming |
|---|---|---|
| 任务 | 流式 ASR | 整段情感识别（SER / SEC） |
| 是否启用 VAD | 是 | 否 |
| partial 输出 | 是 | 否 |
| final 数量 | 每段语音一条 | 每个 start/stop 周期一条 |
| 热词 | 起作用 | 不使用（接收但忽略） |

## 连接

```
ws://<host>:<port>/emotion-streaming
wss://<host>:<port>/emotion-streaming
```

无 query 参数。

## 消息时序

```
Client                                Server
  |                                      |
  |  ---- WebSocket 连接 -------------->  |
  |  <--------  ready  ---------------   |
  |                                      |
  |  ----  start (format/sr/ch/mode)  -> |
  |  ----  binary PCM chunk  ----------> |
  |  ----  binary PCM chunk  ----------> |
  |  ----  binary PCM chunk  ----------> |
  |  ----  ...                           |
  |                                      |
  |  ----  stop  ----------------------> |
  |                                      |  (整段送入情感模型)
  |  <--------  final_emotion  --------  |
  |                                      |
  |  ---- 连接关闭 ---                    |
```

## 客户端 -> 服务端消息

### 1. start

建连收到 `ready` 后发送，声明音频参数与任务模式。必须在发送 PCM 之前发送。

```json
{
  "type": "start",
  "format": "pcm_s16le",
  "sample_rate_hz": 16000,
  "channels": 1,
  "mode": "ser",
  "config": {
    "emotion_request_timeout": 20.0
  }
}
```

| 字段           | 类型   | 必填 | 说明                                                  |
|----------------|--------|------|-------------------------------------------------------|
| type           | string | 是   | 固定 "start"                                          |
| format         | string | 是   | 音频编码，固定 "pcm_s16le"                            |
| sample_rate_hz | int    | 是   | 采样率，固定 16000                                    |
| channels       | int    | 是   | 声道数，固定 1                                        |
| mode           | string | 否   | 任务变体："ser" 或 "sec"，缺省读 Config.emotion_task_mode |
| config         | object | 否   | 服务端 Config 字段平铺覆写（同 ASR 端点机制）         |

### 2. 二进制 PCM 音频帧

格式与 `/transcribe-streaming` 完全一致：16 kHz、单声道、s16le。建议每帧 80 ms（2560 字节）。服务端会一直累积，直到 `stop` 才做一次推理。

### 3. stop

```json
{ "type": "stop" }
```

收到后，服务端把累积的整段 PCM 送入情感模型，并保证返回一条 `final_emotion`。

## 服务端 -> 客户端消息

### 1. ready

```json
{ "type": "ready" }
```

### 2. final_emotion

SER 模式：

```json
{
  "type": "final_emotion",
  "mode": "ser",
  "label": "Happy",
  "text": "Happy",
  "duration_sec": 3.21
}
```

SEC 模式：

```json
{
  "type": "final_emotion",
  "mode": "sec",
  "label": "Happy",
  "text": "The speaker sounds excited and cheerful, speaking at a fast pace with a bright tone.",
  "duration_sec": 3.21
}
```

| 字段          | 类型   | 说明                                                              |
|---------------|--------|-------------------------------------------------------------------|
| type          | string | 固定 "final_emotion"                                              |
| mode          | string | 本次会话所用的任务变体："ser" 或 "sec"                            |
| label         | string | SER 主导情感标签，取自固定 8 分类标签集；SEC 模式下为从文本中匹配到的最佳标签提示，可能为空 |
| text          | string | SER 模式与 label 一致；SEC 模式为模型生成的自由文本描述           |
| raw_text      | string | 模型原始输出，仅当与 text 不一致时回传（如剥离了代码围栏 / JSON 包装） |
| duration_sec  | float  | 实际推理使用的音频时长（秒）                                       |
| language      | string | 仅当客户端 start 中传入 language 时回传，便于上游路由             |

固定 8 分类情感标签集（与 AmphionASR 训练标签集 SER_TAXONOMY 一致）：

```
Neutral, Happy, Sad, Angry, Fear, Disgust, Surprise, Other/Complex
```

注意大小写：模型训练时即采用首字母大写形式，服务端解析后会原样回传。

### 3. error

```json
{
  "type": "error",
  "message": "..."
}
```

## 限制 / 默认值

| 项 | 默认值 | 说明 |
|---|---|---|
| emotion_vllm_base_url | http://localhost:8000 | 情感模型 vLLM 服务地址，默认与主 ASR 共用同一 Amphion 多任务服务 |
| emotion_vllm_model_name | Amphion/Amphion-3B | 模型名，默认与主 ASR 一致 |
| emotion_request_timeout | 30s | 单次推理超时 |
| emotion_max_audio_seconds | 20s | 最长处理音频，超过则保留尾部 20s（贴合 Amphion SER/SEC 训练时的 1-20s utterance 长度上限） |
| emotion_task_mode | ser | 缺省任务变体；可被 start.mode 覆盖 |

服务端 `Config` 中所有 `emotion_*` 字段均可由 `start.config` 临时覆写（与 ASR 端点共用同一套 override 机制）。

## 调用示例

```python
import asyncio, json, ssl, websockets

async def detect_emotion(pcm_bytes: bytes, mode: str = "ser"):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    async with websockets.connect(
        "wss://localhost:8443/emotion-streaming", ssl=ctx
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

        for i in range(0, len(pcm_bytes), 2560):
            await ws.send(pcm_bytes[i:i + 2560])
            await asyncio.sleep(0.08)

        await ws.send(json.dumps({"type": "stop"}))

        async for msg in ws:
            data = json.loads(msg)
            print(data)
            if data["type"] == "final_emotion":
                break
```
