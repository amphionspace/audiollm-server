# audiollm 3-Pod 部署（模型走 S3）

在 EKS 上以 **3 个独立工作负载**运行（Argo Rollouts 管理，参考 apihub 的 rollout 模式）：两个 vLLM（Qwen3-ASR、AmphionSPEC）各自独占一个
GPU Pod，Gateway 单独跑 CPU Pod。模型权重不打进镜像，由**两个 vLLM Pod 各自**的 initContainer 从 S3
下载到共享 PVC 后挂载使用；**gateway（audiollm-server-py）不挂载模型 PVC**。

```
┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐
│ amphion-spec-vllm (GPU) │  │ qwen3-asr-vllm (GPU)    │  │ audiollm-server-py (CPU)   │
│ init: aws-cli 拉 spec    │  │ init: aws-cli 拉 qwen    │  │ gateway uvicorn :8080   │
│ vLLM :8001              │  │ vLLM :8000               │  │ 等两个 vLLM 健康后启动   │
│ nvidia.com/gpu: 1       │  │ nvidia.com/gpu: 1        │  │ 无 GPU、无模型 PVC      │
└────────────┬────────────┘  └────────────┬────────────┘  └─────────────────────────┘
             │                            │
             └────── 共享 PVC audiollm-models（RWO，同 GPU 节点）──────┘
```

- 两个 GPU Pod 通过 **GPU Operator Time-Slicing** 共享同一张卡（RTX PRO 4500 32GB，
  1 卡虚拟 2 个可调度单元）；显存划分靠各 vLLM 的 `--gpu-memory-utilization`。
- 集群内访问：Service 均为 ClusterIP，gateway 通过 `qwen3-asr-vllm:8000` /
  `amphion-spec-vllm:8001` 访问后端。

## 目录结构

```
deploy/k8s/s3/
├── base/                     # 基础清单（镜像为占位符，不可直接部署，必须走 overlay）
│   ├── namespace.yaml        # amphion-api 命名空间
│   ├── configmap.yaml        # S3 模型源参数（拿到地址后改这里）
│   ├── pvc.yaml              # 模型缓存卷（20Gi，gp3）
│   ├── rollout-amphion-spec.yaml      # AmphionSPEC vLLM（GPU，安全滚动）
│   ├── rollout-qwen3-asr.yaml         # Qwen3-ASR vLLM（GPU，错峰启动）
│   ├── rollout-audiollm-server-py.yaml   # Gateway（CPU，安全滚动）
│   ├── service.yaml          # 三个 ClusterIP Service
│   ├── ingress.yaml          # nginx Ingress（本部署未启用）
│   ├── config.yaml           # gateway 配置（由 configMapGenerator 生成）
│   └── kustomization.yaml
├── overlay/example/          # 示例 overlay：覆盖 S3 参数、真实镜像
│   ├── kustomization.yaml
│   └── s3-config.yaml        # 环境相关的 S3 地址
├── secrets.example.yaml      # S3 凭证 + Refine LLM 凭证（不提交真实密钥）
└── README.md
```

> **重要**：`base/` 里的镜像名是占位符（`registry.example.com/...`），
> `kubectl apply -k base` 会创建拉不到镜像的 Pod。**部署/更新一律用 overlay**。

## 部署前要填的 4 个地方

### 1. S3 模型源（拿到 S3 地址后填）

改 `overlay/example/s3-config.yaml`（或直接改 `base/configmap.yaml`）：

```yaml
S3_ENDPOINT: ""                # 自定义 endpoint（MinIO 等兼容服务）；AWS S3 留空
S3_REGION: us-west-2           # AWS S3 必填
S3_BUCKET: amphion-models      # 桶名
S3_QWEN_PREFIX: qwen3-asr-1.7b-v1   # qwen 模型在桶内路径（无结尾斜杠）
S3_SPEC_PREFIX: amphion-spec-v1     # amphion-spec 模型在桶内路径
```

模型目录结构需与 vLLM 期望一致：`config.json`、`*.safetensors`、tokenizer 文件等。

### 2. S3 凭证

复制 `secrets.example.yaml`，填 `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`
（临时凭证加 `AWS_SESSION_TOKEN`），然后单独 apply（不要提交到 Git）：

```bash
cp deploy/k8s/s3/secrets.example.yaml /tmp/audiollm-secrets.yaml
$EDITOR /tmp/audiollm-secrets.yaml
kubectl apply -f /tmp/audiollm-secrets.yaml
```

集群在 AWS 且已配 IRSA 时，也可在 Rollout 的 Pod Template 上加 `serviceAccountName` 去掉 AccessKey。

### 3. 镜像

| 服务 | 镜像 | 构建方式 |
|---|---|---|
| qwen3-asr | `vllm/vllm-openai:v0.18.0` | 官方镜像，无需构建 |
| amphion-spec | `REGISTRY/amphion/audiollm-amphion-spec-vllm:0.1.0` | `docker build -f deploy/docker/Dockerfile.amphion-spec-s3 ...`（只装 plugin，不含模型） |
| gateway | `REGISTRY/amphion/audiollm-server:0.1.0` | 仓库根目录现有 Dockerfile，照旧 |

构建 amphion-spec 镜像（模型走 S3，构建一次即可，模型更新不用重新构建）：

```bash
cd /path/to/audiollm-server
docker build -f deploy/docker/Dockerfile.amphion-spec-s3 \
  -t REGISTRY/amphion/audiollm-amphion-spec-vllm:0.1.0 .
docker push REGISTRY/amphion/audiollm-amphion-spec-vllm:0.1.0
```

在 `overlay/example/kustomization.yaml` 里把 `REGISTRY/...` 换成实际仓库。

### 4. 域名与 TLS

本部署仅集群内部访问，`base/ingress.yaml` 未启用；需要对外暴露时再配置域名与 TLS Secret。

## 前置条件：GPU Operator + Time-Slicing

单卡跑两个 GPU Pod 依赖 GPU Operator 的 Time-Slicing（1 卡 → ≥2 个可调度单元）：

```bash
# 部署 GPU Operator（values 见仓库 deploy/gpu-operator-values.yaml）
helm install gpu-operator nvidia/gpu-operator -n gpu-operator \
  --create-namespace -f deploy/gpu-operator-values.yaml

# 启用 Time-Slicing：给 nvidia-device-plugin-daemonset 的 ConfigMap 配置 2 个单元
# 参考：https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html
```

## 部署

```bash
cd /path/to/audiollm-server

# 预检渲染结果（确认镜像已被覆盖为真实 ECR 地址）
kubectl kustomize deploy/k8s/s3/overlay/example | grep 'image:'

# 部署（必须用 overlay；base 镜像是占位符）
kubectl apply -k deploy/k8s/s3/overlay/example

# 等待就绪（模型下载 + 两个 vLLM 串行加载可能 10-20 分钟）
# 方式一：Argo CLI（推荐，装了 kubectl-argo-rollouts 插件）
kubectl argo rollouts status rollout/amphion-spec-vllm -n amphion-api --timeout 30m
kubectl argo rollouts status rollout/qwen3-asr-vllm -n amphion-api --timeout 30m
kubectl argo rollouts status rollout/audiollm-server-py -n amphion-api --timeout 10m

# 方式二：轮询 phase（无插件时）
watch -n 10 'kubectl -n amphion-api get rollout -o custom-columns=NAME:.metadata.name,PHASE:.status.phase'

kubectl -n amphion-api get pods
```

> 注意：`kubectl rollout status` 只支持 Deployment/StatefulSet/DaemonSet，
> **不支持** Argo `Rollout` 资源，请用上面的 `kubectl argo rollouts status` 或轮询 phase。

## 验收

```bash
# 三个 Pod 都 Ready 后（kubectl exec 需要 Pod 名，Deployment 资源已切换为 Rollout）
SPEC_POD=$(kubectl -n amphion-api get pod -l app=amphion-spec-vllm -o jsonpath='{.items[0].metadata.name}')
QWEN_POD=$(kubectl -n amphion-api get pod -l app=qwen3-asr-vllm -o jsonpath='{.items[0].metadata.name}')
GW_POD=$(kubectl -n amphion-api get pod -l app=audiollm-server-py -o jsonpath='{.items[0].metadata.name}')

kubectl -n amphion-api exec "$SPEC_POD" -- curl -s http://127.0.0.1:8001/health
kubectl -n amphion-api exec "$QWEN_POD" -- curl -s http://127.0.0.1:8000/health
kubectl -n amphion-api exec "$GW_POD" -c gateway -- python3 -c \
  "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8080/readyz').read())"
```

再用仓库现有的 smoke 脚本走真实接口（`deploy/k8s/qwen-only/smoke.py`，base-url 指向
`audiollm-server-py` Service）。

## 已知约束与调参

- **单卡无 HA**：两个 GPU Pod 都调度在同一张卡（RTX PRO 4500）上，Time-Slicing 只切
  算力不切显存；任一模型显存超限会影响另一个。扩容方式：新增 g7 节点（GPU Operator
  自动接管驱动）+ 调整 Time-Slicing 单元数。
- **显存划分**：当前实测配置（RTX PRO 4500 32GB）qwen `--gpu-memory-utilization 0.30`
  （≈9.6GiB）、spec `0.25`（≈8.0GiB），共约 17.6GiB，余量充足。其他卡型按 `nvidia-smi`
  实测调整，两值之和不要超过 1。
- **PVC 共享**：**只有两个 GPU Pod** 挂同一 PVC `audiollm-models`（RWO）；gateway
  不挂载。单 GPU 节点下同节点多 Pod 可同时挂载；若未来多 GPU 节点分布部署，
  需改为 ReadWriteMany（如 EFS）或每模型独立 PVC。
- **StorageClass**：`pvc.yaml` 显式指定 `gp3`（provisioner `ebs.csi.aws.com`，EKS 1.36
  已移除 in-tree EBS provisioner，旧 `gp2` 不可用）。
- **Refine LLM**：gateway 依赖外部 OpenAI-compatible 服务，凭证在 `audiollm-refine` Secret。
