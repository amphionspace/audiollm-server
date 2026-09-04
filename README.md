# AudioLLM Server

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

基于 [Amphion](https://github.com/open-mmlab/Amphion) (vLLM) 的实时语音多任务 Demo，集成 TEN VAD 语音端点检测。
支持以下任务：

- 实时语音转写（双 ASR 模型 Amphion + Qwen 并行推理 + 归一化质量评估 + 风险感知融合，可选在每条转写旁附上情感/语气）
- AST v3 实时角色分离（独立 NVIDIA Streaming Sortformer sidecar，最多 4 位会话内匿名说话人，故障时自动回退普通 ASR）
- AST v3 前端会议模式（最多注册 4 位业务用户，以 TitaNet 声纹将匿名角色绑定为会话内用户 ID）
- 情感识别（SER 8 分类 / SEC 自由文本描述，整段语音推理）

前端四个 Demo 页面（实时 ASR / 增强 ASR / AST v3 测试 / 情感）共享同一套侧边栏导航与 EN / 中文实时语言切换。

---

## 环境要求

- Python 3.10+
- 已启动的 vLLM 推理服务（兼容 OpenAI API）
- OpenSSL（用于生成自签名证书）
- 可选：用于"长文本热词抽取"功能的外部 LLM（OpenAI 兼容接口），在 `config.yaml` 的 `upstreams.hotword_llm` 配置
- 可选：Triton 热词召回服务（默认 `http://localhost:10001` / `rag_asr_retrieve`），用于按 `hotword_pool_id` 隔离的 10 万级热词池召回
- 可选：RAG-ASR HTTP 管理服务（通过 `RAG_ASR_MANAGEMENT_BASE_URL` 配置），用于热词池管理和目标说话人 enrollment embedding 下沉
- 可选：AST v3 Streaming Sortformer 角色分离 sidecar，独立安装方式见 [services/diarization/README.md](services/diarization/README.md)

### AST v3 角色/声纹模式

`WS /tuling/ast/v3` 的 `parameter.asr_config.enable_role_separation` 默认 `true`，并且优先于声纹参数：

| 角色分离 | 声纹设置 | 实际行为 | 最终 `sentence` 的 `cw[].rl` |
|---|---|---|---|
| `true` / 省略 | 任意 | 忽略声纹，执行角色分离；sidecar 故障时回退普通 ASR | 正常为首次/切换 `1..4`、连续同角色 `0`；降级后为 `0` |
| `false` | enrollment 关闭/省略 | 普通 ASR | 不返回 |
| `false` | enrollment 开启且 ID 非空、可用 | TS-ASR | 不返回 |
| `false` | enrollment 开启且 ID 非空、不可用 | 回退普通 ASR并返回声纹失败原因 | 不返回 |
| `false` | enrollment 开启但 ID 为空/省略 | 参数错误并结束会话 | 无识别结果 |

`Progressive` 始终不返回 `cw[].rl`。完整字段和故障矩阵见 [AST v3 WebSocket 协议](docs/protocols/tuling-ast-v3-protocol.md#角色分离与声纹行为矩阵)。

角色分离 sidecar 使用以下服务端配置，均不可由客户端覆写：

| 参数 | 类型 | `config.yaml` 默认值 | 说明 |
|---|---|---|---|
| `diarization_enabled` | bool | `true` | 是否为 AST v3 启用角色分离资源；关闭或 sidecar 故障时会话 fail-open |
| `diarization_target` | string | `localhost:50052` | Sortformer gRPC sidecar 地址 |
| `diarization_connect_timeout_sec` | float | `2.0` | 建立 sidecar 会话的最长等待时间 |
| `diarization_result_timeout_sec` | float | `2.0` | 每个 ASR 段等待 finalized turns 的最长时间；超时后本会话永久降级 |
| `speaker_identity_timeout_sec` | float | `2.0` | 单次 TitaNet embedding RPC 超时；失败不影响 ASR |
| `speaker_identity_min_audio_sec` | float | `3.0` | 会议角色开始声纹匹配前需累计的音频时长 |
| `speaker_identity_max_audio_sec` | float | `10.0` | 单次声纹匹配最多使用的最近音频时长 |
| `speaker_identity_match_threshold` | float | `0.70` | 最佳 cosine similarity 最低门槛 |
| `speaker_identity_match_margin` | float | `0.10` | 多候选时最佳分与第二名的最低差值 |

## 快速开始

如果需要交付包含 Qwen3-ASR-1.7B、AmphionSPEC 和 Gateway 的独立 Docker 部署，
请直接阅读 [Qwen-only 增强 ASR 交付与 Docker 部署](docs/deployment/qwen-only-docker.md)。

```bash
# 安装依赖（二选一）
pip install -e .
uv sync

# 编辑服务端配置（vLLM 地址、模型名等）
vim config.yaml

# 可选：配置长文本热词抽取使用的 LLM（仅在前端"从文本抽取热词"功能用到）
vim config.yaml  # upstreams.hotword_llm

# 启动服务
bash start.sh
```

浏览器打开 `http://172.16.0.3:8082`（systemd 部署）或 `https://172.16.0.3:8443`（`bash start.sh` 自签 HTTPS）进入实时 ASR Demo，另一个 Demo 入口：

| 页面 | 路径 | 说明 |
|---|---|---|
| 实时语音转写 | / 或 /index.html | 双 ASR 模型并行 + 融合；右侧面板可开启"情感识别"开关，在每条 final 转写下附上情绪与语气 |
| 增强语音识别 | /enhanced-asr.html | 通过 Gateway clean-stream 接入 Qwen3-ASR-1.7B，支持热词纠正、文本精修、翻译与情感增强 |
| AST v3 实时语音识别（测试用） | /asr-test.html | 右侧可选择“角色分离 / 会议模式 / 目标说话人 / 普通识别”；会议模式可注册最多 4 个业务用户 ID，并在角色语音足够后更新气泡身份 |
| 情感识别 | /emotion.html | 整段语音 SER / SEC |

页面右上角的 EN / 中 切换会持久化到浏览器 localStorage，下次访问保持上次的选择。

> 首次访问时浏览器会提示自签名证书不安全，点击 **高级** → **继续访问** 即可。

---

## 前端样式重建（Tailwind）

前端四个 Demo 页面共用一份 **预编译** 的 Tailwind 工具类样式 `frontend/tailwind.css`（已入仓），运行时不再依赖 `cdn.tailwindcss.com` 的 JIT 脚本，跨页切换不会再有"重新跑一遍 Tailwind 编译"的卡顿。

仅当你修改了 `frontend/*.html` 或 `frontend/*.js` 中使用的 Tailwind 类名（包括 JS 字符串里拼接出来的 `lg:w-[380px]` 等动态类）后，需要重新生成一次：

```bash
bash scripts/build_tailwind.sh           # 一次性构建
bash scripts/build_tailwind.sh --watch   # 监听文件改动持续构建
```

脚本通过 `npx tailwindcss@3` 调用 Tailwind v3 CLI，按 `frontend/tailwind.config.js` 中声明的 content 范围扫描，并写出压缩后的 `frontend/tailwind.css`。需要本机已安装 Node.js (>= 18) 和 npm。

如果你新增了 Tailwind 类却忘了重建，浏览器只会回退到没有该类的默认样式（不会报错），需要重新运行脚本。

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
| **热词** | Triton 按 `hotword_pool_id` 召回 top-K，召回词注入 ASR prompt；1.7B 可走 encoder bypass；热词池管理可经 RAG-ASR HTTP 管理服务代理 |

---

## HTTP 接口（整段情感）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/emotion/jobs` | 提交 WAV，返回 `202` + `job_id` |
| GET | `/api/emotion/jobs/{job_id}` | 轮询任务状态与 `final_emotion` 结果 |

协议见 [docs/protocols/emotion-streaming-protocol.md](docs/protocols/emotion-streaming-protocol.md)（异步 HTTP，非 WebSocket）。

## WebSocket 接口

服务暴露两个主要 WebSocket 任务端点（流式 / 分段任务），前端实时 ASR 页面使用 `/transcribe-streaming`：

| 端点 | 任务 | VAD | 输出 | 协议文档 |
|---|---|---|---|---|
| `/transcribe-streaming` | 个性化语音识别 | 是 | partial / final（每段语音一条） | [docs/protocols/transcribe-streaming-protocol.md](docs/protocols/transcribe-streaming-protocol.md) |
| `/asr/v1/clean-stream` | Qwen3-ASR 增强识别 | 累计快照伪流式 | transcription.delta / emotion.bucket / postprocess.delta / transcription.done | [docs/protocols/clean-stream-protocol.md](docs/protocols/clean-stream-protocol.md) |
| `/emotion-segmented-streaming` | 按段流式情感识别（同模型，逐段返回） | 是 | final_emotion（每个 VAD 段一条） | [docs/protocols/emotion-segmented-streaming-protocol.md](docs/protocols/emotion-segmented-streaming-protocol.md) |

原生任务的命名约定：每个任务一个独立 WebSocket 端点（`/<task>-streaming`），共享同一套 `start` / `stop` / `update_hotwords` 控制消息与 `config` 覆写机制；任务专属字段只出现在对应端点的协议文档中。`/asr/v1/clean-stream` 为兼容接入方的独立 JSON/base64 协议，不适用这套约定，详见其协议文档。

### `/transcribe-streaming` 协议

通过 WebSocket 连接：

```
ws://172.16.0.3:8082/transcribe-streaming
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
| `{"type": "start", ...}` | 声明音频格式、语种、目标说话人和可选配置覆写（见下方示例，发送 PCM 前必须先发） |
| `{"type": "update_hotwords", "hotwords": ["词1", "词2"]}` | 更新后续 final 段的临时请求热词；去重限量后优先进入 prompt，并覆盖精确重复或整词同音（忽略声调）的 Triton 召回词，也可继续用于携带 `enrollment_id` |
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
- `hotword_pool_id` — 可选的热词池隔离 ID，默认 `default`；不同热词池互相隔离
- `hotwords` — 临时请求热词。服务端不会把它写入热词池；final 段会把去重后的前 `recall_custom_hotword_limit` 个优先注入 prompt，并覆盖精确重复或整词同音（忽略声调）的 RAG-ASR 召回热词
- `config` — 服务端参数覆写，可选。只传需要修改的项，详见 [客户端可配置参数](#客户端可配置参数)

**Python 调用示例：**

```python
import asyncio, json, websockets

async def transcribe(pcm_bytes: bytes):
    async with websockets.connect(
        "ws://172.16.0.3:8082/transcribe-streaming"
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

完整协议规范见 [docs/protocols/transcribe-streaming-protocol.md](docs/protocols/transcribe-streaming-protocol.md)。

---

## vLLM 推理服务

默认开发部署可以调用外部 OpenAI-compatible vLLM 服务；Qwen-only 交付配置则在
本仓库内构建和启动 Qwen3-ASR 与 AmphionSPEC，详见
[Docker 交付说明](docs/deployment/qwen-only-docker.md)。

启动本服务前，请确认 `config.yaml -> upstreams` 中配置的 vLLM 地址已经可访问，例如默认主 ASR `http://localhost:8009`、副 ASR `http://localhost:8001`、情感模型 `http://localhost:8222`。

---

## 运维脚本

systemd 服务（默认单元名 `audiollm-demo`）监听 `172.16.0.3:8082`（HTTP，无 TLS）。对外 REST / WebSocket / 静态页 Base URL 为 `http://172.16.0.3:8082`；WebSocket 使用 `ws://172.16.0.3:8082/<endpoint>`。

如果当前机器还没有安装该 unit，先安装仓库内模板：

```bash
sudo cp deploy/audiollm-demo.service /etc/systemd/system/audiollm-demo.service
sudo systemctl daemon-reload
sudo systemctl enable audiollm-demo
```

模板默认使用 `/home/ubuntu/workspace/audiollm-server/.venv/bin/uvicorn` 在 `8082` 启动 HTTP 服务；如果仓库路径、用户、虚拟环境或端口不同，先改 `deploy/audiollm-demo.service` 再复制。端口冲突时只需要调整模板里的 `Environment=PORT=...`。

安装后可以用 `scripts/restart_service.sh` 一键重启并查看日志，改完后端代码后无需手动敲 `systemctl`：

```bash
scripts/restart_service.sh            # 重启并打印最近 30 行日志
scripts/restart_service.sh -f         # 重启并实时跟随日志（Ctrl+C 退出）
SERVICE=my-demo scripts/restart_service.sh   # 指定其他 systemd 服务名
```

脚本会自动检测是否需要 `sudo`、校验 `systemctl is-active` 状态，并在启动失败时打印近 50 行错误日志后以非零码退出。

---

## 配置说明

服务端默认配置保存在 [`config.yaml`](config.yaml)，修改后重启服务生效。该文件按 `upstreams`、`defaults`、`rest`、`endpoints` 等分组；客户端临时覆写仍只接受扁平字段名（如 `vad_threshold`）。

客户端可在 `start` 消息的 `config` 字段中覆写其中任意一项，只需传入要修改的参数，未传入的保持服务端默认值。

### 配置优先级与覆写逻辑

同一参数最多在三个层次出现，运行时按以下优先级取值（后者覆盖前者）：

1. 代码内置默认：`backend/config.py` 中 `Config` dataclass 的字段默认值。仅当 `config.yaml` 缺少该字段（或文件不存在）时作为兜底，面向 fork/移植场景给出通用值（例如端点默认指向 `localhost:8000`、情感模型默认复用主 ASR 后端）。
2. 服务端默认：`config.yaml` 的值，是本部署实际生效的默认值，修改后重启服务生效。下文各表"默认值"列展示的就是这一层。
3. 客户端临时覆写：`start` 消息 `config` 字段传入的值，仅对当前连接生效、不落盘，连接结束即失效；未传入的字段沿用服务端默认。

因此当 `config.py` 内置默认与 `config.yaml` 不一致时（典型如端点地址、是否启用融合），以 `config.yaml` 为准，内置默认只是文件缺字段时的降级兜底。客户端覆写只接受扁平字段名（如 `vad_threshold`），与 `config.yaml` 是否分组无关；传入非法值或未知字段会被忽略并保持服务端默认。不一致的组合（如 `enable_dual_asr_fusion=true` 但 `enable_secondary_asr=false`）在加载时自动降级为 `false`。

### 客户端可配置参数

#### 模型选择

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `vllm_base_url` | string | `http://localhost:8009` | 主 ASR 模型的服务地址 |
| `vllm_model_name` | string | `AmphionASR-1.7B` | 主 ASR 模型名称 |
| `vllm_prompt_template` | string | `amphion_asr_1.7b` | 主 ASR 请求模板；Amphion 4.3B 使用 `amphion_asr`，Amphion 1.7B 使用 `amphion_asr_1.7b`，官方 Qwen3-ASR 使用 `qwen3_asr`。服务端配置，不可客户端覆写 |
| `astv3_vllm_base_url` | string | `""` | 仅 `/tuling/ast/v3` 端点使用的主模型地址；当前留空，回退全局 `vllm_base_url`（即 `http://localhost:8009`）。服务端配置，不可客户端覆写 |
| `astv3_vllm_model_name` | string | `""` | 仅 `/tuling/ast/v3` 端点使用的主模型名称；当前留空，回退全局 `vllm_model_name`（即 `AmphionASR-1.7B`）。服务端配置，不可客户端覆写 |
| `astv3_vllm_prompt_template` | string | `""` | 仅 `/tuling/ast/v3` 端点使用的主模型 prompt 模板；当前留空，回退全局 `vllm_prompt_template`。服务端配置，不可客户端覆写 |
| `secondary_vllm_base_url` | string | `http://localhost:8001` | 副 ASR 模型的服务地址 |
| `secondary_vllm_model_name` | string | `Qwen/Qwen3-ASR-1.7B` | 副 ASR 模型名称 |
| `enable_primary_asr` | bool | `true` | 是否启用主模型。关闭后只用副模型 |
| `enable_secondary_asr` | bool | `true` | 副模型是否在线。决定 partial 是否有静音门、final 是否可融合 |
| `enable_dual_asr_fusion` | bool | `false` | final 段是否走双模型融合矫正。关闭后 final 只跑主模型（partial 静音门不受影响）。`enable_secondary_asr=false` 时自动降级为 false |

端点级主模型绑定：`/tuling/ast/v3` 恒为 primary-only（强制 `enable_secondary_asr=false`，不调用本地副模型、无 partial 静音门、无融合），其主模型由上面的 `astv3_vllm_*` 指定，当前留空，回退全局 primary（`vllm_base_url`，即 `http://localhost:8009` 的 `AmphionASR-1.7B`），且客户端无法通过 `parameter.asr_config` 重新打开副模型。`/transcribe-streaming` 不受影响，按下表矩阵工作。

三档 ASR 开关组合矩阵（`config.yaml` 默认 `enable_secondary_asr=true`、`enable_dual_asr_fusion=false`，对应下表第二行）：

| `enable_secondary_asr` | `enable_dual_asr_fusion` | Partial 行为 | Final 行为 |
|---|---|---|---|
| true | true | 双调，副模型做静音门，发主模型文本 | 双调 + 融合矫正 |
| true（默认） | false（默认） | 双调，副模型做静音门，发主模型文本 | 仅主模型 |
| false | (自动降级 false) | 仅主模型，无静音门 | 仅主模型 |

#### 推理控制

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `primary_asr_timeout` | float | `4.0` | 主模型单次推理的超时秒数，超时则放弃主模型结果 |
| `asr_request_timeout` | float | `120` | 发给模型的 HTTP 请求总超时秒数 |

#### 实时输出

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enable_pseudo_stream` | bool | `true` | 是否开启"伪流式"——说话过程中提前输出中间结果 |
| `pseudo_stream_interval_ms` | int | `500` | 伪流式输出的最小间隔（毫秒），值越小更新越频繁；仅节流首个之后的刷新，不影响首字 |
| `pseudo_stream_first_partial_ms` | int | `200` | 每段语音首个 partial（伪流式中间结果）的触发门槛，从 `min_segment_duration_ms` 解耦（dataclass 兜底 350，config.yaml 默认设 200 走低延迟）；与 `vad_start_frames` 按 max 决定首字延迟 |

#### k2 流式 ASR partial

`/transcribe-streaming` 与 `/tuling/ast/v3` 可通过 `defaults.k2.k2_enabled=true` 接入外部 k2 gRPC 流式 ASR 服务。k2 只做纯识别：只接收 16 kHz mono PCM_S16LE 音频，不接热词、不接目标说话人注册、不返回 token timestamps。它的 partial 直接下发给客户端；final 仍由本服务的 LLM ASR 路径产生，因此热词召回、目标说话人过滤、ITN、车牌规范化与双模型融合只作用于 final。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `k2_enabled` | bool | `true` | 是否启用 k2 替代本地伪流式 partial；仅影响 `/transcribe-streaming` 与 `/tuling/ast/v3` |
| `k2_target` | string | `localhost:50051` | k2 gRPC 服务地址，服务端配置，不可客户端覆写 |
| `k2_sample_rate` | int | `16000` | k2 期望采样率；启动时会通过 ServerInfo 校验 |
| `k2_include_token_timestamps` | bool | `false` | 当前固定不启用，保留为显式配置 |
| `k2_connect_timeout_sec` | float | `5.0` | k2 ServerInfo / 建连探测超时 |
| `k2_fallback_to_local` | bool | `true` | k2 启动失败时回退本地 VAD + 伪流式 |
| `k2_max_segment_sec` | float | `30.0` | k2 长时间不返回 endpoint 时本地强切上限，防止缓冲无限增长 |
| `k2_idle_keep_ms` | int | `1500` | k2 起音前本地缓冲只保留最近窗口，避免长静音无限累积 |
| `k2_voice_gate_enabled` | bool | `true` | 是否启用本地人声证据门控，抑制 k2 对环境音误 partial / endpoint |
| `k2_voice_gate_threshold` | float | `0.65` | 本地门控的人声概率阈值，只做 accept/drop，不裁剪 k2 段边界 |
| `k2_voice_gate_start_frames` | int | `10` | 需要连续多少帧超过阈值才放行；默认约 160ms 连续人声证据 |

k2 模式下，切段权威是 k2 的 endpoint；本服务只用 `k2_idle_keep_ms` 限制起音前旧静音、用 `k2_max_segment_sec` 防止无 endpoint 时缓冲无限增长，并用 `k2_voice_gate_*` 在 partial/final 进入下游前确认有人声证据。voice gate 只决定放行或丢弃，不再用本地 VAD 裁剪段首/段尾。`silence_duration_ms` / `vad_start_frames` / `pseudo_stream_interval_ms` / `pseudo_stream_first_partial_ms` 不再决定这两个端点的切点或首字时机；`enable_pseudo_stream=false` 仍会抑制 partial 下发。

#### AudioLLM 前人声过滤

每次把音频送入 AudioLLM 前，服务都会重新计算整段人声证据，覆盖本地 VAD、k2 endpoint、REST 上传和离线任务路径。gate 开启时，人声帧占比、累计时长或 RMS 不达标的音频直接丢弃；filter 开启时，已放行音频只保留 VAD 支持的人声区间以及 `asr_segment_voice_filter_pre_ms` / `asr_segment_voice_filter_tail_ms` 上下文。该处理不改变 k2 已经下发的 partial，也不改变 AST v3 的原始 `bg` / `ed` 时间线。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `asr_segment_voice_gate_enabled` | bool | `true` | 是否丢弃低人声证据音频 |
| `asr_segment_voice_gate_threshold` | float | `0.65` | gate 计算人声证据的概率阈值 |
| `asr_segment_voice_gate_min_ratio` | float | `0.05` | 整段中超过阈值的人声帧占比下限 |
| `asr_segment_voice_gate_min_ms` | int | `120` | 整段累计人声证据时长下限（毫秒） |
| `asr_segment_voice_gate_min_rms` | float | `0.001` | 低于该 RMS 的近数字静音直接丢弃 |
| `asr_segment_voice_filter_enabled` | bool | `true` | 是否裁剪已放行音频中 VAD 不支持的区间 |
| `asr_segment_voice_filter_threshold` | float | `0.65` | filter 判定需要保留的人声帧概率阈值 |
| `asr_segment_voice_filter_pre_ms` | int | `160` | 每个人声区间向前保留的上下文（毫秒），默认约 10 个 TEN VAD 帧 |
| `asr_segment_voice_filter_tail_ms` | int | `160` | 每个人声区间向后保留的上下文（毫秒），默认约 10 个 TEN VAD 帧 |
| `asr_silence_removal_threshold_sec` | float | `0.0` | final LLM ASR 前删除连续时长大于等于该值的内部静音；`0` 表示关闭 |

门控与裁剪是两个独立可观察行为，完整组合如下：

| `gate_enabled` | `filter_enabled` | 低证据音频 | 已放行音频 |
|---|---|---|---|
| `true` | `true` | 丢弃 | 只保留人声区间与上下文 |
| `true` | `false` | 丢弃 | 保留原音频 |
| `false` | `true` | 不丢弃 | 有人声证据时裁剪；无证据时 fail-open 保留原音频 |
| `false` | `false` | 不丢弃 | 保留原音频 |

上述 `asr_segment_voice_gate_*` 与 `asr_segment_voice_filter_*` 字段是服务端强制保护项，不接受客户端临时覆写。`asr_silence_removal_threshold_sec` 解决一句话中间夹入长时间无声的问题，例如"打开遮【停顿若干秒】光板"；它先删除满足阈值的内部长静音，随后仍执行人声过滤。

#### 调试落盘 (debug dump)

运维级排查开关（不可客户端覆写），用于核对“前端回放音频 / 最终文本”是否一致这类问题。配置在 `config.yaml` 的 `defaults.debug` 分组：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `debug_dump_enabled` | bool | `false` | 开启后每个 final 段落盘音频+元信息；生产建议关闭（磁盘无限增长） |
| `debug_dump_dir` | string | `debug_dumps` | 落盘根目录；相对路径相对项目根，也可填绝对路径 |

开启后只对 native `/transcribe-streaming` 生效：`ready` 带 `session_id` / `dump_dir`，`final` 带 `dump_id`，并把每段写到 `<dump_dir>/<session_id>/<seg_id>.{wav,json}`。`.wav` 就是送入推理、也是前端回放的那段音频（同源同字节）；`.json` 含 final 文本、该段 partial 历史、主/副模型原始输出、模型与开关快照、时长/时延。前端在每个气泡上显示可点击复制的 `dump_id`，复制后即可定位文件。`debug_dumps/` 已加入 `.gitignore`。

#### 文本规范化 (ITN / 车牌)

final 转写默认做逆文本规范化（口语数字→阿拉伯数字）与车牌格式规范化；partial 保持口语形式不变（避免抖动）。通用 ITN 由本地 wetext 实现（不依赖 pynini、不联网），车牌层为零依赖正则。两者仅作用于 final，且任何异常都回退原文，不影响 ASR 主流程。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enable_asr_itn` | bool | `true` | 是否对 final 做通用 ITN（仅中文）：六五四三八→65438、二零二四年→2024年，以及电话/金额/百分比等 |
| `asr_itn_enable_0_to_9` | bool | `false` | ITN 是否转换孤立个位数字（如单独的"三"）。默认 false，避免"一下/三个"类被误转 |
| `enable_asr_plate_normalize` | bool | `true` | 是否对 final 做车牌规范化：字母大写、去车牌内分隔符、口语数字按位转阿拉伯，并按 GB 车牌形态校验后才改写 |
| `enable_asr_repetition_fix` | bool | `true` | 是否折叠 ASR 解码退化产生的超过 20 次重复字符/短模式；对 partial 与 final 都生效 |

两个开关相互独立，组合矩阵：

| `enable_asr_itn` | `enable_asr_plate_normalize` | 行为 |
|---|---|---|
| true | true | 通用 ITN + 车牌规范化（默认） |
| true | false | 仅通用 ITN（车牌数字串经 ITN 也会转，但不做去分隔/大写/形态校验） |
| false | true | 仅车牌规范化（零依赖），不做通用数字 ITN |
| false | false | 不改写，等于模型原始输出 |

已知边界：省份简称被声学误识别成字母（如"冀"→"J"）属于识别错误而非格式问题，ITN/车牌层不做猜测式还原——会把数字与字母修对（车牌号为JR六五四三八→车牌号为JR65438），但省份位仍是错字。这类问题应通过热词偏置或模型层面解决。

#### 语音检测 (VAD)

控制服务端如何判断"用户开始说话"和"用户说完了"。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `vad_threshold` | float | `0.65` | 语音判定灵敏度（0-1）。值越低越容易触发，但也更容易误判噪音为语音 |
| `silence_duration_ms` | int | `350` | 说话停顿多久算"说完了"（毫秒）。值越大越不容易被短暂停顿打断 |
| `vad_smoothing_alpha` | float | `0.3` | 语音概率的平滑系数（0-1）。值越大波动越小，但响应越慢 |
| `vad_start_frames` | int | `10` | 连续多少帧检测到语音才算"开始说话"。TEN VAD 下约 160ms，兼顾短首词与噪音防抖 |
| `vad_pre_speech_ms` | int | `500` | 检测到说话后，往前多保留多少毫秒的音频。避免开头被截掉 |
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

#### 热词召回与管理

ASR 热词偏置来自 `config.yaml -> services.recall` 指向的 Triton 召回服务。客户端只使用 `hotword_pool_id` 指定热词池；未传时使用配置里的 `hotword_pool_id` 默认值（默认 `default`）。final 段只在该热词池内召回 top-K 热词，再让少量请求临时 `hotwords` 优先进入主 ASR prompt，并过滤精确重复或整词同音（忽略声调）的召回词。当主模型为 `amphion_asr_1.7b` 且未启用目标说话人时，服务端还会把 Triton 返回的 projector 帧作为 `audio_embeds` 发送给 vLLM，避免重复跑音频 encoder。伪流式 partial 不执行召回、不注入热词、也不走 encoder bypass，只使用纯 vLLM raw-audio 推理。

`config.yaml -> services.recall_management` 是可选管理面。若其上游 `RAG_ASR_MANAGEMENT_BASE_URL` 非空，热词池查询、添加、指定删除、整池清空、reload 和 `enable_triton_enrollment_store=true` 的注册写入会转发到 RAG-ASR HTTP 管理服务；RAG-ASR 会把 enrollment projector frames tensor 与元数据落盘到本地 `enrollment_store_dir`（默认 `var/enrollments`），不保存原始注册音频。未配置时保持旧 Triton 兼容路径和 demo 本地进程内 enrollment store，默认不改变现网行为。灰度打开下沉链路前，应先确认 RAG-ASR `scripts/serve_http.sh` 已作为独立服务运行，并使用包含模型依赖、FastAPI/uvicorn 的 Python 3.10 执行环境；否则 demo 会退回旧管理路径或在注册写入时得到上游错误。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enable_hotword_recall` | bool | `true` | 是否启用 Triton 热词池召回。关闭后不查热词池，主 ASR 只使用请求携带的 `hotwords`，`enable_encoder_bypass` 会自动降级为 false |
| `recall_top_k` | int | `50` | final 段从热词池召回的候选热词数量 |
| `hotword_pool_id` | string | `default` | 未显式传 `hotword_pool_id` 时使用的默认热词池 ID |
| `recall_custom_hotword_limit` | int | `8` | 每次请求最多保留多少个临时 `hotwords` 优先进入 prompt；去重、覆盖同音召回词、不写入热词池 |
| `enable_encoder_bypass` | bool | `true` | 是否使用 Triton `AUDIO_EMBEDS_B64` 走 vLLM encoder bypass；仅 1.7B 单音频路径生效 |
| `enable_triton_enrollment_store` | bool | `false` | 是否把新注册音频转发到 RAG-ASR 管理链路保存落盘 embedding tensor 和元数据；默认关闭以保持 demo 本地注册音频缓存行为 |

#### 长音频离线转写（`POST /api/asr/transcriptions`）

接口说明见 [docs/api/transcription-jobs-api.md](docs/api/transcription-jobs-api.md)。任务支持通过 `config` 开启内置/自定义热词、文本精修、翻译和情绪增强；不传增强配置时保持原有纯转写行为。推理用哪个模型、是否双模型融合由 `config.yaml` 的 `rest.routes.transcribe` 块独立声明（省略则跟随共享 `rest.upstreams` 绑定），下表为 `defaults.transcribe` 调参：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `transcribe_max_concurrent_jobs` | int | `2` | 同时 running 的转写任务数 |
| `transcribe_segment_concurrency` | int | `4` | 单任务内并行推理的语音段数；总 vLLM 压力为两者乘积（默认 2×4=8） |
| `transcribe_job_queue_max` | int | `8` | 任务排队上限（含 running），超出返回 503 |
| `transcribe_job_ttl_sec` | float | `3600` | 终态任务结果保留秒数，过期后轮询返回 404 |
| `transcribe_max_segment_sec` | float | `30.0` | 连续无停顿语音的强切上限（VAD 只在静音处切段，此参数兜底超长独白） |
| `transcribe_max_upload_bytes` | int | `536870912` | 上传文件字节上限（512 MB；2 小时 16 kHz mono WAV 约 220 MB） |
| `transcribe_max_audio_sec` | float | `10800` | 解码后时长上限（3 小时）；超出直接 400 拒绝，不做截断 |
| `transcribe_silence_duration_ms` | int | `800` | 仅离线转写生效的切段停顿阈值；`0` 表示跟随全局 `silence_duration_ms`。离线无延迟代价，拉长可让纪要段落更完整，且不影响实时端点 |

#### 情感识别（HTTP jobs + `/emotion-segmented-streaming`）

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `emotion_vllm_base_url` | string | `http://localhost:9001` | 情感识别模型的 vLLM 服务地址；当前部署 AmphionSPEC |
| `emotion_vllm_model_name` | string | `AmphionSPEC` | 服务端内部模型名，不进入公开协议 |
| `emotion_request_timeout` | float | `30.0` | 情感推理 HTTP 请求总超时（秒） |
| `emotion_max_audio_seconds` | float | `20.0` | 单次推理处理的最长音频秒数；超过则保留尾部，贴合 Amphion SER/SEC 训练时 1-20s 的 utterance 上限 |
| `emotion_task_mode` | string | `sec` | 缺省任务变体：`ser` 输出 8 分类标签，`sec` 输出自由文本描述；公开协议只接受这两个值 |
| `emotion_max_concurrent_jobs` | int | `8` | 异步 HTTP 任务同时调 vLLM 的上限 |
| `emotion_job_queue_max` | int | `64` | 异步任务排队上限，超出返回 503 |
| `emotion_job_ttl_sec` | float | `3600` | 已完成任务元数据保留秒数 |

#### 调试

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `debug_show_dual_asr` | bool | `false` | 调试字段，当前主前端 `/transcribe-streaming` 不展示双 ASR 调试视图 |

### 服务

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PORT` | `8443` | `start.sh` 自签 HTTPS 端口；systemd 固定为 `8082`（HTTP） |
| `RAG_ASR_MANAGEMENT_BASE_URL` | 空 | 可选 RAG-ASR HTTP 管理服务根地址；为空时保持旧热词管理/本地 enrollment 行为，灰度时可设为如 `http://127.0.0.1:18080`；启用 `enable_triton_enrollment_store=true` 前必须可达 |

---

## 项目结构

```
backend/
  main.py                    # FastAPI 入口：把端点映射到 (AudioStream, TaskEngine) 组合
  config.py                  # 配置加载与投影，ASR / Emotion 字段在同一 dataclass 中分组
  http_client.py             # 共享异步 HTTP 客户端
  streaming/                 # 协议/会话层（任务无关）
    session.py               #   StreamingSession：WS 生命周期 + 控制消息 + 工作队列
    audio_stream.py          #   AudioStream 策略：VadSegmentedStream / WholeUtteranceStream
    events.py                #   SegmentReady / PartialSnapshot
  tasks/                     # 任务推理层（一类任务一个 engine）
    base.py                  #   TaskEngine 协议 + BaseTaskEngine 默认实现
    asr.py                   #   AsrTaskEngine：双模型 + 融合 + 伪流 partial
    emotion.py               #   EmotionTaskEngine：整段情感推理
  audio/                     # 音频信号处理
    utils.py                 #   48→16 kHz 重采样、PCM/WAV 转换
    vad.py                   #   语音端点检测（TEN VAD + 备用方案）
  asr/                       # ASR 模型交互
    client.py                #   vLLM API 调用与输出解析
    fusion.py                #   双模型融合逻辑
    hotword.py               #   长文本热词抽取服务
    recall.py                #   Triton 热词召回 + 可选 RAG-ASR HTTP 管理代理
    prompt.py                #   LLM Prompt 模板
  emotion/                   # 情感模型交互
    client.py                #   vLLM API 调用与输出解析
    prompt.py                #   情感识别 Prompt 与标签集
frontend/                    # 静态 Web 前端（四个 Demo 页面 + 共享侧边栏 + EN/中文 i18n）
  index.html / app.js        #   实时 ASR 主页
  emotion.html / emotion-app.js  # 情感识别演示
  sidebar.js                 #   注入侧边栏导航与 EN/中 语言切换
  i18n.js                    #   极简前端 i18n（data-i18n / data-i18n-attr-* 等）
scripts/                     # 项目维护、构建与压测辅助脚本
tests/                       # 测试工具（ASR / 情感客户端脚本）
docs/                        # 协议文档（每个端点一份）
```

> 新增一种任务时，只需新建 `backend/<task>/` 推理客户端 + `backend/tasks/<task>.py` 任务引擎，再在 `main.py` 用对应的 `AudioStream` 策略组装一个新端点即可，不需要改动 `streaming/` 与现有任务的代码。

## 参与贡献

请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解开发环境搭建与贡献指南。

## 开源许可

本项目采用 [Apache License 2.0](LICENSE) 开源许可协议。
