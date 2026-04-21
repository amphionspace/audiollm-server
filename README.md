# AudioLLM Server

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

基于 [Amphion](https://github.com/open-mmlab/Amphion) (vLLM) 的实时语音多任务 Demo，集成 TEN VAD 语音端点检测。
支持三类任务：

- 实时语音转写（双 ASR 模型 Amphion + Qwen 并行推理 + 归一化质量评估 + 风险感知融合）
- 目标说话人识别（TS-ASR，注册音频 + 混合音频双路推理，附带二级语音存在性门控）
- 情感识别（SER 8 分类 / SEC 自由文本描述，整段语音推理）

前端三个 Demo 页面（ASR / TS-ASR / 情感）共享同一套侧边栏导航与 EN / 中文 实时语言切换。

---

## 环境要求

- Python 3.10+
- 已启动的 vLLM 推理服务（兼容 OpenAI API）
- OpenSSL（用于生成自签名证书）
- 可选：用于"长文本热词抽取"功能的外部 LLM（OpenAI 兼容接口），配置文件 `backend/api.json`

## 快速开始

```bash
# 安装依赖（二选一）
pip install -e .
uv sync

# 编辑服务端配置（vLLM 地址、模型名等）
vim backend/config.json

# 可选：配置长文本热词抽取使用的 LLM（仅在前端"从文本抽取热词"功能用到）
cp backend/api.json.example backend/api.json && vim backend/api.json

# 启动服务
bash start.sh
```

浏览器打开 `https://<服务器IP>:8443` 进入实时 ASR Demo，另两个 Demo 入口：

| 页面 | 路径 | 说明 |
|---|---|---|
| 实时语音转写 | / 或 /index.html | 双 ASR 模型并行 + 融合 |
| 目标说话人识别 | /tsasr.html | 录入注册音频后只识别该说话人 |
| 情感识别 | /emotion.html | 整段语音 SER / SEC |

页面右上角的 EN / 中 切换会持久化到浏览器 localStorage，下次访问保持上次的选择。

> 首次访问时浏览器会提示自签名证书不安全，点击 **高级** → **继续访问** 即可。

---

## 系统架构

```mermaid
graph LR
    Browser["浏览器 (麦克风)"] -->|WSS| FastAPI
    FastAPI -->|HTTP| vLLM1["vLLM #1 (Amphion)"]
    FastAPI -->|HTTP| vLLM2["vLLM #2 (Qwen)"]
    FastAPI --- VAD["TEN VAD"]
    VAD --- Fusion["相似度融合"]
```

| 模块 | 说明 |
|---|---|
| **前端** | Web Audio API AudioWorklet 采集 16 kHz PCM，通过 WebSocket 发送 |
| **后端** | FastAPI，每个连接启动两个并发异步任务：VAD 任务（语音检测）+ LLM 任务（ASR 推理），互不阻塞 |
| **热词** | 在浏览器 UI 中管理，通过 WebSocket 实时同步到后端 |

---

## WebSocket 接口

服务暴露四个 WebSocket 端点，按任务一类一个：

| 端点 | 任务 | VAD | 输出 | 协议文档 |
|---|---|---|---|---|
| `/ws/audio` | 浏览器 Demo（ASR + 双模型调试视图） | 是 | response / partial_transcript | 见前端代码 |
| `/transcribe-streaming` | 个性化语音识别 | 是 | partial / final（每段语音一条） | [docs/transcribe-streaming-protocol.md](docs/transcribe-streaming-protocol.md) |
| `/transcribe-target-streaming` | 目标说话人识别（TS-ASR，注册音频 + 混合音频） | 是 | enrollment_ok / final（每段语音一条） | [docs/tsasr.md](docs/tsasr.md) |
| `/emotion-streaming` | 整段情感识别（SER 8 分类 / SEC 自由描述） | 否 | final_emotion（每个 start/stop 周期一条） | [docs/emotion-streaming-protocol.md](docs/emotion-streaming-protocol.md) |

新增任务的命名约定：每个任务一个独立 WebSocket 端点（`/<task>-streaming`），共享同一套 `start` / `stop` / `update_hotwords` 控制消息与 `config` 覆写机制；任务专属字段（如 ASR 的 `language`/`hotwords`、情感的输出标签集）只出现在对应端点的协议文档中。

### `/transcribe-streaming` 协议

通过 WebSocket 连接：

```
wss://<host>:<port>/transcribe-streaming
```

**消息流程：**

```
客户端                                 服务端
  |                                      |
  |  ---- WebSocket 连接 -------------> |
  |  <--------  ready  ---------------  |
  |  ----  start (语种/热词/配置) ----> |
  |  ----  PCM 音频数据  -------------> |
  |  <--------  partial  -------------  |
  |  ----  PCM 音频数据  -------------> |
  |  <--------  final  ---------------  |
  |  ----  stop  ---------------------> |
  |  <--------  final (保证返回) ------  |
```

**客户端 → 服务端：**

| 消息 | 说明 |
|---|---|
| `{"type": "start", ...}` | 声明音频格式、语种、热词和可选配置覆写（见下方示例，发送 PCM 前必须先发） |
| `{"type": "update_hotwords", "hotwords": ["词1", "词2"]}` | 会话中途更新热词列表（可选，随时可发） |
| 二进制 PCM 帧 | 原始音频：16 kHz、单声道、s16le，建议每帧 80 ms（2560 字节） |
| `{"type": "stop"}` | 结束音频流。服务端会处理所有剩余音频并保证返回一条 `final` |

**服务端 → 客户端：**

| 消息 | 说明 |
|---|---|
| `{"type": "ready"}` | 服务端就绪，可以开始发送音频 |
| `{"type": "partial", "text": "...", "language": "zh"}` | 中间结果（语音进行中的实时识别） |
| `{"type": "final", "text": "...", "language": "zh"}` | 最终结果（一段语音结束后，或收到 stop 后） |
| `{"type": "error", "message": "..."}` | 错误通知 |

**`start` 消息完整格式：**

```json
{
  "type": "start",
  "format": "pcm_s16le",
  "sample_rate_hz": 16000,
  "channels": 1,
  "language": "zh",
  "hotwords": ["热词1", "热词2"],
  "config": {
    "enable_primary_asr": true,
    "vad_threshold": 0.3
  }
}
```

- `language` — 源语种代码（`zh`/`en`/`id`/`th`），可选
- `hotwords` — 热词列表，可选。支持万级数量（10K 词约 300KB，无传输压力）
- `config` — 服务端参数覆写，可选。只传需要修改的项，详见 [客户端可配置参数](#客户端可配置参数)

**Python 调用示例：**

```python
import asyncio, json, ssl, websockets

async def transcribe(pcm_bytes: bytes):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    async with websockets.connect(
        "wss://localhost:8443/transcribe-streaming", ssl=ctx
    ) as ws:
        ready = json.loads(await ws.recv())
        assert ready["type"] == "ready"

        await ws.send(json.dumps({
            "type": "start",
            "format": "pcm_s16le",
            "sample_rate_hz": 16000,
            "channels": 1,
            "language": "zh",
            "hotwords": ["挚音科技", "武新华"],
            "config": {"vad_threshold": 0.4},
        }))

        for i in range(0, len(pcm_bytes), 2560):
            await ws.send(pcm_bytes[i:i+2560])
            await asyncio.sleep(0.08)

        await ws.send(json.dumps({"type": "stop"}))

        async for msg in ws:
            data = json.loads(msg)
            print(f"[{data['type']}] {data.get('text', '')}")
            if data["type"] == "final":
                break
```

**测试客户端：**

```bash
python tests/test_ws_client.py audio.wav
python tests/test_ws_client.py audio.wav --hotwords "武新华,挚音科技"
python tests/test_ws_client.py audio.wav --language en --chunk-ms 100
```

完整协议规范见 [docs/transcribe-streaming-protocol.md](docs/transcribe-streaming-protocol.md)。

---

## 启动双 vLLM 推理服务

启动 Amphion（默认端口 8000）：

```bash
MODEL_PATH=/path/to/Amphion-3B bash scripts/start_vllm_amphion.sh
```

在另一个终端启动 Qwen（端口 8001）：

```bash
MODEL_PATH=/path/to/Qwen3-ASR-1.7B bash scripts/start_vllm_qwen.sh
```

---

## 配置说明

服务端默认配置保存在 [`backend/config.json`](backend/config.json)，修改后重启服务生效。

客户端可在 `start` 消息的 `config` 字段中覆写其中任意一项，只需传入要修改的参数，未传入的保持服务端默认值。

### 客户端可配置参数

#### 模型选择

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `vllm_base_url` | string | `http://localhost:8000` | 主 ASR 模型的服务地址 |
| `vllm_model_name` | string | `Amphion/Amphion-3B` | 主 ASR 模型名称 |
| `secondary_vllm_base_url` | string | `http://localhost:8001` | 副 ASR 模型的服务地址 |
| `secondary_vllm_model_name` | string | `Qwen/Qwen3-ASR-1.7B` | 副 ASR 模型名称 |
| `enable_primary_asr` | bool | `true` | 是否启用主模型。关闭后只用副模型 |
| `enable_secondary_asr` | bool | `true` | 是否启用副模型。关闭后只用主模型 |

#### 推理控制

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `primary_asr_timeout` | float | `4.0` | 主模型单次推理的超时秒数，超时则放弃主模型结果 |
| `asr_request_timeout` | float | `120` | 发给模型的 HTTP 请求总超时秒数 |

#### 实时输出

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enable_pseudo_stream` | bool | `true` | 是否开启"伪流式"——说话过程中提前输出中间结果 |
| `pseudo_stream_interval_ms` | int | `500` | 伪流式输出的最小间隔（毫秒），值越小更新越频繁 |

#### 语音检测 (VAD)

控制服务端如何判断"用户开始说话"和"用户说完了"。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `vad_threshold` | float | `0.65` | 语音判定灵敏度（0-1）。值越低越容易触发，但也更容易误判噪音为语音 |
| `silence_duration_ms` | int | `200` | 说话停顿多久算"说完了"（毫秒）。值越大越不容易被短暂停顿打断 |
| `vad_smoothing_alpha` | float | `0.35` | 语音概率的平滑系数（0-1）。值越大波动越小，但响应越慢 |
| `vad_start_frames` | int | `10` | 连续多少帧检测到语音才算"开始说话"。防止瞬间噪音误触发 |
| `vad_pre_speech_ms` | int | `500` | 检测到说话后，往前多保留多少毫秒的音频。避免开头被截掉 |
| `vad_end_frames` | int | `20` | 连续多少帧静默才算"说完了"。和 `silence_duration_ms` 配合使用 |
| `vad_keep_tail_ms` | int | `40` | 语音结束后多保留多少毫秒的尾巴音频 |
| `min_segment_duration_ms` | int | `350` | 低于此时长的语音片段会被丢弃（过滤噪音短脉冲） |

#### 双模型融合

当主副模型同时启用时，系统用以下参数决定采信哪个模型的结果。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `fusion_similarity_threshold` | float | `0.85` | 两个模型结果的文本相似度达到此值时，认为它们"一致"，优先选主模型 |
| `fusion_min_primary_score` | float | `0.55` | 主模型结果的最低质量分。低于此值则不信任主模型 |
| `fusion_max_repetition_ratio` | float | `0.35` | 主模型输出中重复内容的占比上限。超过则判定为"幻觉" |
| `fusion_disagreement_threshold` | float | `0.55` | 两模型结果的分歧度上限。超过则回退到副模型 |
| `fusion_hotword_boost` | float | `0.12` | 主模型命中每个热词时获得的评分加成 |
| `fusion_primary_score_margin` | float | `0.08` | 主模型评分需超过副模型至少这么多才会被选用 |

#### 目标说话人识别（仅 `/transcribe-target-streaming` 使用）

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `tsasr_base_url` | string | `""` | TS-ASR 模型的 vLLM 服务地址；空字符串时回落到 `vllm_base_url` |
| `tsasr_model_name` | string | `""` | TS-ASR 模型名称；空字符串时回落到 `vllm_model_name` |
| `tsasr_request_timeout` | float | `30.0` | 单次推理 HTTP 请求超时（秒） |
| `tsasr_enrollment_min_sec` | float | `1.0` | 注册音频最短时长（秒），小于此值服务端拒绝 |
| `tsasr_enrollment_max_sec` | float | `5.0` | 注册音频最长时长（秒），超过此值服务端用 VAD 抽取有声帧后截断到该上限，而不是拒绝 |
| `tsasr_max_audio_seconds` | float | `30.0` | 单段混合音频时长上限；超过则保留尾部 |
| `tsasr_enable_partial` | bool | `false` | 是否开启伪流式 partial；双音频推理 RTF 较高，默认关闭 |
| `tsasr_enable_hotwords` | bool | `false` | 是否把会话热词注入 Prompt；短期方案未验证，默认关闭 |
| `tsasr_speech_gate_enabled` | bool | `true` | 二级语音存在性门控开关。TS-ASR 模式下副模型被关闭，对 VAD 漏判的键盘敲击等瞬时噪音再做一次过滤 |
| `tsasr_speech_gate_prob_threshold` | float | `0.6` | 门控逐帧判帧的概率阈值，比上层 `vad_threshold` 更严格 |
| `tsasr_speech_gate_min_voiced_ms` | int | `200` | 累计有声时长低于此值的片段会被丢弃，不再下发到 vLLM |

#### 情感识别（仅 `/emotion-streaming` 使用）

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `emotion_vllm_base_url` | string | `http://localhost:8000` | 情感识别模型的 vLLM 服务地址；默认与主 ASR 共用同一 Amphion 多任务服务 |
| `emotion_vllm_model_name` | string | `Amphion/Amphion-3B` | 情感识别模型名称；与 AmphionASR 项目的 SER/SEC 训练模型一致 |
| `emotion_request_timeout` | float | `30.0` | 情感推理 HTTP 请求总超时（秒） |
| `emotion_max_audio_seconds` | float | `20.0` | 单次推理处理的最长音频秒数；超过则保留尾部，贴合 Amphion SER/SEC 训练时 1-20s 的 utterance 上限 |
| `emotion_task_mode` | string | `ser` | 缺省任务变体：`ser` 输出 8 分类标签，`sec` 输出自由文本描述；可被 `start.mode` 覆盖 |

#### 调试

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `debug_show_dual_asr` | bool | `true` | 在 `/ws/audio` 响应中包含双 ASR 调试信息 |

### 服务

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PORT` | `8443` | HTTPS 服务端口（启动脚本参数） |

---

## 项目结构

```
backend/
  main.py                    # FastAPI 入口：把端点映射到 (AudioStream, TaskEngine) 组合
  config.py                  # 配置加载（从 config.json），ASR / Emotion 字段在同一 dataclass 中分组
  config.json                # 服务端默认配置
  http_client.py             # 共享异步 HTTP 客户端
  session.py                 # /ws/audio 浏览器 demo 会话
  streaming/                 # 协议/会话层（任务无关）
    session.py               #   StreamingSession：WS 生命周期 + 控制消息 + 工作队列
    audio_stream.py          #   AudioStream 策略：VadSegmentedStream / WholeUtteranceStream
    events.py                #   SegmentReady / PartialSnapshot
  tasks/                     # 任务推理层（一类任务一个 engine）
    base.py                  #   TaskEngine 协议 + BaseTaskEngine 默认实现
    asr.py                   #   AsrTaskEngine：双模型 + 融合 + 伪流 partial
    emotion.py               #   EmotionTaskEngine：整段情感推理
    ts_asr.py                #   TsAsrTaskEngine：注册音频 + 混合音频双路推理
  audio/                     # 音频信号处理
    utils.py                 #   48→16 kHz 重采样、PCM/WAV 转换
    vad.py                   #   语音端点检测（TEN VAD + 备用方案）
  asr/                       # ASR 模型交互
    client.py                #   vLLM API 调用与输出解析
    fusion.py                #   双模型融合逻辑
    hotword.py               #   热词提取服务
    prompt.py                #   LLM Prompt 模板
  emotion/                   # 情感模型交互
    client.py                #   vLLM API 调用与输出解析
    prompt.py                #   情感识别 Prompt 与标签集
  tsasr/                     # 目标说话人识别（短期方案，独立演化）
    client.py                #   query_tsasr_model：双音频 vLLM 请求
    prompt.py                #   build_tsasr_content：可扩展的 Prompt 构建器
    enrollment.py            #   注册音频解码与时长校验
  api.json                   # 可选：长文本热词抽取使用的外部 LLM 配置（OpenAI 兼容）
frontend/                    # 静态 Web 前端（三个 Demo 页面 + 共享侧边栏 + EN/中文 i18n）
  index.html / app.js        #   实时 ASR 主页
  tsasr.html / tsasr-app.js  #   TS-ASR 演示
  emotion.html / emotion-app.js  # 情感识别演示
  sidebar.js                 #   注入侧边栏导航与 EN/中 语言切换
  i18n.js                    #   极简前端 i18n（data-i18n / data-i18n-attr-* 等）
scripts/                     # vLLM 服务启动脚本
tests/                       # 测试工具（ASR / TS-ASR / 情感各一个客户端脚本）
docs/                        # 协议文档（每个端点一份）
```

> 新增一种任务时，只需新建 `backend/<task>/` 推理客户端 + `backend/tasks/<task>.py` 任务引擎，再在 `main.py` 用对应的 `AudioStream` 策略组装一个新端点即可，不需要改动 `streaming/` 与现有任务的代码。

## 参与贡献

请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解开发环境搭建与贡献指南。

## 开源许可

本项目采用 [Apache License 2.0](LICENSE) 开源许可协议。
