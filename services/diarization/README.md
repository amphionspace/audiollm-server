# Streaming Sortformer 角色分离 sidecar

## 调研结论

首期采用 NVIDIA [`diar_streaming_sortformer_4spk-v2.1`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)，固定 revision `fafaab5faa1617a0ca52d38dd3dc4bd636800d3d`。它原生支持最多 4 位说话人和在线 speaker cache，适合 AST v3 的会话内匿名角色编号；不承担跨会话身份识别或声源分离。模型受 NVIDIA Open Model License 约束，上线前必须完成许可证审查。

低延迟参数采用模型官方 streaming 配置：`chunk_len=6`、left context `1`、right context `7`，speaker FIFO/cache 长度 `188`、更新周期 `144`。每帧 80 ms，对当前预测需要约 `6 + 7 = 13` 帧（约 1.04 秒）输入；sidecar 每 480 ms 推进一次 finalized watermark。

## 工作方式

主服务只在 `WS /tuling/ast/v3` 且 `enable_role_separation=true` 时建立双向 gRPC 流。同一份 16 kHz mono PCM S16LE 并行进入 VAD/k2 和 sidecar。VAD/k2 决定 ASR 大段边界，sidecar 返回 `{start_ms,end_ms,speaker_index}` 与 finalized watermark；ASR 最多等待 2 秒，随后按 speaker turn 重切 PCM、串行识别并按时间顺序输出 sentence。

`speaker_index` 在单个 WebSocket 会话内稳定为 `0..3`，协议层映射为讯飞风格的 `cw[].rl`：角色变化返回 `1..4`，同角色连续发言返回 `0`。重叠区归给当前大段内占用更长的角色，短于 `min_segment_duration_ms` 的角色抖动合并到相邻 turn，音频不丢失。

连接失败、结果超时、流中断与队列溢出都只令当前会话剩余部分降级为普通 ASR；AST sentence 返回 `rl=0`，下一会话重新连接。`GET /readyz` 会把 sidecar 状态放进 optional check，但不会让主 ASR readiness 返回 503。

## 独立安装与模型下载

NeMo 不进入主服务 `pyproject.toml`。在仓库根目录执行：

```bash
uv sync --project services/diarization
mkdir -p /home/ubuntu/models
uv run --project services/diarization hf download \
  nvidia/diar_streaming_sortformer_4spk-v2.1 \
  diar_streaming_sortformer_4spk-v2.1.nemo \
  --revision fafaab5faa1617a0ca52d38dd3dc4bd636800d3d \
  --local-dir /home/ubuntu/models
```

先前台验证：

```bash
HF_HUB_OFFLINE=1 \
DIARIZATION_MODEL_PATH=/home/ubuntu/models/diar_streaming_sortformer_4spk-v2.1.nemo \
uv run --project services/diarization python -m services.diarization.server
```

确认启动日志中的 `free_gpu_gib`、`peak_allocated_gpu_gib` 与 `peak_reserved_gpu_gib` 后再安装 `deploy/audiollm-diarization.service`。进程在加载前后都要求至少 10 GiB 空闲显存；低于门槛会以退出码 78 停止，systemd 不重启该资源不足实例，主服务自动降级。主服务配置位于 `config.yaml -> defaults.diarization`，默认目标 `localhost:50052`。

真实 checkpoint 的流式 API smoke test：

```bash
DIARIZATION_GPU_TEST_MODEL=/home/ubuntu/models/diar_streaming_sortformer_4spk-v2.1.nemo \
uv run --project services/diarization --with pytest \
  pytest -q tests/test_diarization_gpu.py
```

## 探索性 H20 诊断（AISHELL-4，非验收）

2026-08-04 使用 `/home/ubuntu/data/testdata/aishell4_work` 中的
AISHELL-4 远场会议录音与 RTTM 金标实测。模型文件固定为上述 revision，
SHA-256 为 `8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8`；
测试脚本为 `scripts/benchmark_diarization.py`，输入以 80 ms PCM 包连续送入真实
`SortformerStream`。下表已改用 `pyannote.metrics` 连续时间 DER；旧版脚本的
80 ms 离散帧估算值不再作为报告口径。

| 30 秒窗口 | 金标/预测人数 | DER（0 collar，含重叠） | DER（250 ms collar，含重叠） | RTF | watermark lag p95 | finish | 峰值 allocated 显存 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `L_R004S02C01 @ 1850s` | 4 / 3 | 11.94% | 10.37% | 0.073 | 560 ms | 76 ms | 0.90 GiB |
| `L_R004S03C01 @ 1000s` | 4 / 4 | 51.84% | 52.93% | 0.054 | 560 ms | 53 ms | 0.90 GiB |

两个窗口另用 NeMo 官方整段 `forward_streaming` 路径复核。无 collar、含重叠的
聚合 DER 与 sidecar 相差 0.09 个绝对百分点，单窗口最大相差 0.16 个百分点，
因此困难窗口的高误差不是 gRPC 或增量分块实现造成。扩展到 6 个人数不超限的
30 秒诊断窗口后，标准 DER 加权为 20.05%，AST 单角色投影代理指标的
250 ms collar、不含重叠 macro DER 为 10.13%，但困难窗口仍达 44.19%。这些窗口
表明 H20 的性能和资源余量充足，但中文远场短窗的质量方差明显，不能只用聚合均值
决定上线。

真实节奏的 30 秒 gRPC 流健康检查 31 ms、建会 1 ms、结束后等待 final watermark
53 ms，无降级，最终 watermark 为 30000 ms。相同音频经完整 AST v3 + k2 + ASR
链路得到 `rl=1,0,3`，验证了首次角色、同角色连续和角色切换语义，并正常收到
terminal frame。瞬时灌入整段音频会触发 2 秒 `result_timeout` 并 fail-open；这属于
非实时突发压测，正常实时发送未触发。

以上 AISHELL-4 两个短窗仅用于本机接线、性能、冷启动和困难样本检查，不能替代下述
AliMeeting Test far 全集验收，也不能与官方 15.60% 直接横向比较。AISHELL-4
完整会议包含 7 位说话人，超出本模型单会话最多 4 人的产品边界；因此只能选取不超过
4 人的诊断窗口，不计入发布门禁。原始 JSON 结果
保存在 `/home/ubuntu/data/testdata/aishell4_work/results/sortformer_v2_1/`。

## 正式评测脚本

`scripts/benchmark_diarization.py` 支持 JSONL manifest，每行至少包含
`audio_filepath` 和 `rttm_filepath`；可选字段为 `uem_filepath`、`offset`、
`duration`、`num_speakers`、`uniq_id`、`rttm_recording_id` 和 `uem_recording_id`。
路径相对 manifest 所在目录解析。截取好的 WAV 与原会议 RTTM 搭配时，使用
`reference_offset`只平移金标；`offset` 则同时裁剪原 WAV 和平移金标。

脚本会拒绝 UEM 评分区域内超过 4 位说话人的会话；
`--allow-over-capacity` 只用于明确的超边界诊断。输出同时包含标准 DER、不重新优化角色
映射的 0–15 / 15–30 / 30 秒后时间分桶，以及不丢时间的 AST 单角色投影代理指标。
先跑 sidecar，再跑官方整段基线并对比：

```bash
uv run --project services/diarization python scripts/benchmark_diarization.py \
  --model /home/ubuntu/models/diar_streaming_sortformer_4spk-v2.1.nemo \
  --manifest /path/to/eval.jsonl --inference-mode sidecar \
  --output /path/to/sidecar.json

uv run --project services/diarization python scripts/benchmark_diarization.py \
  --model /home/ubuntu/models/diar_streaming_sortformer_4spk-v2.1.nemo \
  --manifest /path/to/eval.jsonl --inference-mode official \
  --compare-to /path/to/sidecar.json --output /path/to/official.json
```

## 验收

CI 使用 fake diarizer 跑协议、切段、超时和降级测试。真实 GPU 验收另行执行：AliMeeting Test far 全集标准 DER 不高于 17.60%，sidecar 与 NeMo 官方整段路径差异不超过 0.5 个绝对百分点；AST 单角色投影的 macro DER 不高于 20%，首 30 秒不高于 25%，单会话最差值低于 40%。ASR 与 sidecar 同卡压测不得 OOM，至少保留 10 GiB 显存，角色等待 p95 小于 2 秒，现有 ASR p95 回归不超过 5%。在这些门禁通过前，`config.yaml` 保持 `diarization_enabled=false`。嘈杂执法场景首期灰度，不在没有脱敏金标前调整模型阈值。
