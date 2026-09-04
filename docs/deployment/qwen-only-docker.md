# Qwen-only 增强 ASR 交付与 Docker 部署

本文面向拿到完整项目目录的部署人员。该部署仅包含 Gateway、
Qwen3-ASR-1.7B 和 AmphionSPEC，不需要 AmphionASR、k2、热词池、说话人模型或
`open-audio-llm` 仓库。

## 交付目录

不能只交付 Git 仓库。模型未提交到 Git，复制项目时必须包含以下目录：

```text
audiollm-server/
├── models/
│   ├── qwen3-asr-1.7b/
│   └── amphion-spec/
├── backend/
├── frontend/
├── deploy/
└── Dockerfile
```

模型和外部依赖如下：

| 组件 | 用途 | 交付方式 |
|---|---|---|
| Qwen3-ASR-1.7B | 普通和增强语音识别 | `models/qwen3-asr-1.7b` |
| AmphionSPEC | SER 情绪分类和 SEC 情感描述 | `models/amphion-spec` |
| Refine LLM | 文本清洗、请求内热词纠错、标点和 emoji 增强 | 部署方提供 OpenAI-compatible API |

## 环境要求

- Linux x86_64；
- Docker 和 Docker Compose v2；
- NVIDIA Driver 和 NVIDIA Container Toolkit；
- 建议至少 32 GiB 可用 GPU 显存、80 GiB 可用磁盘；
- 构建时可拉取 Docker 基础镜像，运行时可访问 Refine LLM。

当前默认参数已在单张 NVIDIA H20 上验证。其他 GPU 应根据实际显存调整模型的
`gpu-memory-utilization`。

## 检查模型

```bash
cd /path/to/audiollm-server
python3 deploy/k8s/qwen-only/verify_models.py
```

预期输出：

```text
qwen3-asr-1.7b: ok (4.38 GiB weights)
amphion-spec: ok (7.40 GiB weights)
```

检查失败表示模型文件不完整，不应继续构建。

## 配置 Refine LLM

```bash
cp deploy/docker/qwen-only.env.example /tmp/qwen-only.env
$EDITOR /tmp/qwen-only.env
```

至少填写：

```dotenv
REFINE_BASE_URL=https://your-openai-compatible-endpoint.example.com/v1
REFINE_MODEL=your-model-name
REFINE_API_KEY=your-api-key
```

`REFINE_BASE_URL` 应包含服务自身要求的 API 版本前缀；Gateway 会追加
`/chat/completions`。密钥只供服务器调用 Refine LLM，接入方不会获得该密钥。
不要把填写了真实密钥的文件提交到 Git。

网络受限时可在该文件中配置镜像源：

```dotenv
APT_DEBIAN_MIRROR=https://mirrors.aliyun.com/debian
APT_SECURITY_MIRROR=https://mirrors.aliyun.com/debian-security
```

## 构建和启动

```bash
docker compose --env-file /tmp/qwen-only.env \
  -f deploy/docker/compose.qwen-only.yaml build

docker compose --env-file /tmp/qwen-only.env \
  -f deploy/docker/compose.qwen-only.yaml up -d
```

构建生成：

```text
audiollm/server:local
audiollm/qwen3-asr-1.7b:local
audiollm/amphion-spec-vllm:local
```

权重直接打入模型镜像，容器启动时不会下载模型。Compose 会先等待 AmphionSPEC
健康，再启动 Qwen，最后启动 Gateway，避免单卡上两个 vLLM 同时做显存 profiling。

默认宿主机端口：

| 服务 | 端口 |
|---|---:|
| Gateway | 8080 |
| Qwen3-ASR | 8011 |
| AmphionSPEC | 9001 |

端口和显存比例可在 `/tmp/qwen-only.env` 中调整。生产环境只需对外暴露 Gateway，
不应直接公开两个模型端口。

## 检查运行状态

```bash
docker compose --env-file /tmp/qwen-only.env \
  -f deploy/docker/compose.qwen-only.yaml ps

curl -f http://127.0.0.1:8011/health
curl -f http://127.0.0.1:9001/health
curl -f http://127.0.0.1:8080/readyz
```

模型名称检查：

```bash
curl http://127.0.0.1:8011/v1/models
curl http://127.0.0.1:9001/v1/models
```

结果应分别包含 `Qwen/Qwen3-ASR-1.7B` 和 `AmphionSPEC`。

## 完整验收

准备一段 16 kHz、单声道、16-bit PCM WAV：

```bash
.venv/bin/python deploy/k8s/qwen-only/smoke.py \
  --base-url http://127.0.0.1:8080 \
  --audio /path/to/test-16k-mono.wav \
  --timeout 180
```

脚本会实际检查：

- 普通伪流式 ASR WebSocket final；
- 增强 ASR WebSocket 和 Refine cleanup；
- 增强型非流式转写；
- SER Top-K 情绪标签和置信度；
- SEC 情感描述；
- 情感 WebSocket。

最终应输出 `"status": "ok"`。

## 对外接口

| 类型 | 接口 |
|---|---|
| 增强 ASR WebSocket | `/asr/v1/clean-stream` |
| 普通伪流式 ASR WebSocket | `/transcribe-streaming` |
| 情感 WebSocket | `/emotion-segmented-streaming` |
| 提交非流式转写 | `POST /api/asr/transcriptions` |
| 查询非流式转写 | `GET /api/asr/transcriptions/{job_id}` |
| 提交非流式情感任务 | `POST /api/emotion/jobs` |
| 查询非流式情感任务 | `GET /api/emotion/jobs/{job_id}` |

情感模式只有 `ser` 和 `sec`，没有 `spec` mode。详细消息格式见：

- [增强 ASR WebSocket 协议](../protocols/clean-stream-protocol.md)
- [普通 ASR WebSocket 协议](../protocols/transcribe-streaming-protocol.md)
- [情感 WebSocket 协议](../protocols/emotion-segmented-streaming-protocol.md)
- [非流式转写接口](../api/transcription-jobs-api.md)

## 日志、停止和更新

```bash
# 查看日志
docker compose --env-file /tmp/qwen-only.env \
  -f deploy/docker/compose.qwen-only.yaml logs -f --tail=200

# 停止并删除容器；模型镜像和项目模型文件不会删除
docker compose --env-file /tmp/qwen-only.env \
  -f deploy/docker/compose.qwen-only.yaml down

# 更新代码后重新构建并启动
docker compose --env-file /tmp/qwen-only.env \
  -f deploy/docker/compose.qwen-only.yaml build
docker compose --env-file /tmp/qwen-only.env \
  -f deploy/docker/compose.qwen-only.yaml up -d
```

## 完全离线交付

模型权重虽已随目录交付，但首次构建仍需基础镜像和 Python 软件包。目标环境完全
离线时，应在已完成构建的机器导出三个最终镜像：

```bash
docker save \
  audiollm/server:local \
  audiollm/qwen3-asr-1.7b:local \
  audiollm/amphion-spec-vllm:local \
  -o audiollm-qwen-only-images.tar
```

目标机器导入后直接启动：

```bash
docker load -i audiollm-qwen-only-images.tar
docker compose --env-file /tmp/qwen-only.env \
  -f deploy/docker/compose.qwen-only.yaml up -d --no-build
```
