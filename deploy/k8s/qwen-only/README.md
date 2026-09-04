# Qwen-only 自包含部署

该部署只对外暴露以下接口：

- `WS /transcribe-streaming`
- `WS /asr/v1/clean-stream`（增强 ASR）
- `POST/GET /api/asr/transcriptions[/<job_id>]`
- `WS /emotion-segmented-streaming`
- `POST/GET /api/emotion/jobs[/<job_id>]`
- 运维探针 `/healthz`、`/readyz`

不部署 AmphionASR、第二 ASR、k2、Sortformer、TitaNet 或其他说话人模型。

## Docker Compose（推荐直接部署）

项目目录包含模型后，可直接构建并启动三个服务。Refine LLM 使用
OpenAI-compatible 接口，密钥只通过环境变量传入：

```bash
cd /path/to/audiollm-server
python deploy/k8s/qwen-only/verify_models.py
cp deploy/docker/qwen-only.env.example /tmp/qwen-only.env
$EDITOR /tmp/qwen-only.env
docker compose --env-file /tmp/qwen-only.env \
  -f deploy/docker/compose.qwen-only.yaml build
docker compose --env-file /tmp/qwen-only.env \
  -f deploy/docker/compose.qwen-only.yaml up -d
docker compose --env-file /tmp/qwen-only.env \
  -f deploy/docker/compose.qwen-only.yaml ps
```

默认暴露 Gateway `8080`、Qwen `8011`、AmphionSPEC `9001`。可通过
`GATEWAY_PORT`、`QWEN3_ASR_PORT`、`AMPHION_SPEC_PORT` 修改宿主机端口。
Compose 默认按单张 H20 的实测值设置 Qwen/SPEC 显存比例为 `0.15`/`0.10`；可通过
`QWEN3_ASR_GPU_MEMORY_UTILIZATION`、`AMPHION_SPEC_GPU_MEMORY_UTILIZATION` 调整。
如果本机已有完全相同的 vLLM 0.18.0 基础镜像，可设置 `VLLM_IMAGE` 避免重新拉取。
网络受限环境也可通过 `APT_DEBIAN_MIRROR` 和 `APT_SECURITY_MIRROR` 指定 Debian 镜像源。

验收：

```bash
.venv/bin/python deploy/k8s/qwen-only/smoke.py \
  --base-url http://127.0.0.1:8080 \
  --audio /path/to/test-16k-mono.wav
```

模型服务需要 NVIDIA Container Toolkit。默认把两套模型暴露给所有可见 GPU；
生产环境可用 `CUDA_VISIBLE_DEVICES` 或容器平台的 GPU 调度分别绑定设备。

## Kubernetes 部署

## 依赖清单

| 依赖 | 固定标识 | 用途 | 是否必需 |
|---|---|---|---|
| Qwen3-ASR | `models/qwen3-asr-1.7b`，served name `Qwen/Qwen3-ASR-1.7B` | 流式伪 partial/final、离线长音频转写 | 是，随项目目录交付 |
| AmphionSPEC | `models/amphion-spec`，served name `AmphionSPEC` | SER、SEC、增强转写的语音情绪 | 是，随项目目录交付 |
| Refine LLM | OpenAI-compatible base URL/model/API key | 文本清洗、翻译、Qwen 热词纠错、SEC 语言兜底 | 是 |
| TEN VAD | Gateway Python 包 `ten-vad` | 本地切段与伪流式 | 是，已打入 Gateway 镜像 |

此部署不包含热词池、RAG 召回或热词管理接口。增强 ASR 仅接受
`session.update.hotwords.builtin`（最多一个内置词表）和
`session.update.hotwords.custom`（请求内自定义词），由 Refine LLM 在 Qwen 转写后
做保守纠错。

## 集群前提

- Kubernetes 1.27+
- NVIDIA device plugin
- nginx ingress controller
- 推荐两张 GPU，Qwen3-ASR 与 AmphionSPEC 各占一张。当前 H20 实测显存分别约
  15.9 GiB 与 11.1 GiB；不同卡型、并发和上下文长度会改变实际占用。
- 构建阶段能拉取基础镜像，运行阶段能访问 Refine LLM

如果用 GPU time-slicing 或其他共享方案把两个模型放在同一张大显存卡上，需要
由集群管理员提供对应的 device-plugin 配置；本清单不隐式假设 GPU 可以共享。

## 1. 构建并推送镜像

先确认收到的项目目录包含完整模型：

```bash
cd /path/to/audiollm-server
python deploy/k8s/qwen-only/verify_models.py
```

模型不会上传 Git，但会从本项目的 `models/` 复制到各自镜像；Pod 启动时不下载权重，
也不需要模型 URL、SHA256、Hugging Face token、PVC 或 initContainer。

```bash
docker build -t REGISTRY/audiollm/server:0.1.0 .
docker build \
  -f deploy/docker/Dockerfile.qwen3-asr \
  -t REGISTRY/audiollm/qwen3-asr-1.7b:0.1.0 .
docker build \
  -f deploy/docker/Dockerfile.amphion-spec \
  -t REGISTRY/audiollm/amphion-spec-vllm:0.1.0 .
docker push REGISTRY/audiollm/server:0.1.0
docker push REGISTRY/audiollm/qwen3-asr-1.7b:0.1.0
docker push REGISTRY/audiollm/amphion-spec-vllm:0.1.0
```

部署前把三个镜像改成实际 registry：

```bash
cd deploy/k8s/qwen-only
kustomize edit set image \
  registry.example.com/audiollm/server:0.1.0=REGISTRY/audiollm/server:0.1.0 \
  registry.example.com/audiollm/qwen3-asr-1.7b:0.1.0=REGISTRY/audiollm/qwen3-asr-1.7b:0.1.0 \
  registry.example.com/audiollm/amphion-spec-vllm:0.1.0=REGISTRY/audiollm/amphion-spec-vllm:0.1.0
```

AmphionSPEC 使用 `AmphionASRForConditionalGeneration` 架构，因此模型服务镜像还
必须安装 vLLM plugin。plugin 已最小化收录在
`deploy/plugins/amphion_spec_vllm`，镜像直接从本项目安装，不依赖
`open-audio-llm` 或在线 clone AmphionASR。

## 2. 创建 Secret

不要提交真实密钥。复制示例、填写值后单独 apply：

```bash
cp secrets.example.yaml /tmp/audiollm-secrets.yaml
$EDITOR /tmp/audiollm-secrets.yaml
kubectl apply -f /tmp/audiollm-secrets.yaml
```

`REFINE_BASE_URL` 应包含兼容服务的版本前缀，例如 `https://host.example/v1`；
Gateway 会在其后追加 `/chat/completions`。

## 3. 修改域名并部署

把 `ingress.yaml` 中的 `audiollm.example.com` 和 TLS Secret 改成实际值，然后：

```bash
kubectl apply -k deploy/k8s/qwen-only
kubectl -n audiollm rollout status deployment/qwen3-asr --timeout=15m
kubectl -n audiollm rollout status deployment/amphion-spec --timeout=15m
kubectl -n audiollm rollout status deployment/audiollm-server --timeout=5m
kubectl -n audiollm get pods,svc,ingress
```

## 4. 验收接口

准备一段有清晰说话声的 16 kHz、mono、s16le WAV：

```bash
.venv/bin/python deploy/k8s/qwen-only/smoke.py \
  --base-url https://audiollm.example.com \
  --audio /path/to/test-16k-mono.wav
```

脚本同时验证：

- Qwen 普通流式 final；
- 与 FourHz 兼容的增强 ASR 会话、请求内热词和整段 cleanup；
- 增强型非流式转写和 cleanup 状态；
- SER 的 `top_emotions` / `best_score`；
- SEC 中文描述；
- 分段情感 WebSocket。
