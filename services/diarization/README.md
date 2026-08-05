# Streaming Sortformer 角色分离 sidecar

## 调研结论

首期采用 NVIDIA [`diar_streaming_sortformer_4spk-v2.1`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)，固定 revision `fafaab5faa1617a0ca52d38dd3dc4bd636800d3d`。它原生支持最多 4 位说话人和在线 speaker cache，适合 AST v3 的会话内匿名角色编号；不承担跨会话身份识别或声源分离。模型受 NVIDIA Open Model License 约束；本次部署流程不包含许可证审查步骤。

低延迟参数采用模型官方 streaming 配置：`chunk_len=6`、left context `1`、right context `7`，speaker FIFO/cache 长度 `188`、更新周期 `144`。每帧 80 ms，模型上下文需要约 `6 + 7 = 13` 帧（1.04 秒）；frontend 另保留 20 ms PCM guard，使分窗 Mel 与整段 Mel 对齐。小包输入时有效缓冲约 1.06 秒；使用建议的 80 ms 包时会量化为 1.12 秒。sidecar 每 480 ms 推进一次 finalized watermark，仍低于主服务 2 秒的结果等待上限。

同卡并发使用有界动态 batching：最多等待 12 ms，只合并 PCM 窗口、streaming
offset 和 speaker-cache tensor 形状完全兼容的请求，默认每批最多 8 路。NeMo
以一个 batch 前进后，结果和 cache 分别复制回原会话；不同步会话仍独立执行。
`DIARIZATION_MAX_BATCH_SIZE` 和 `DIARIZATION_BATCH_WAIT_MS` 由 sidecar 的 systemd
环境控制，客户端不能覆写。batch 8 是当前 H20 上兼顾 diarization 吞吐与同卡 ASR
尾部时延的实测值；batch 16 会形成更长 GPU kernel，实测反而轻微抬高 ASR 收尾时延。

## 工作方式

主服务只在 `WS /tuling/ast/v3` 且 `enable_role_separation=true` 时建立双向 gRPC 流。同一份 16 kHz mono PCM S16LE 并行进入 VAD/k2 和 sidecar。VAD/k2 决定 ASR 大段边界，sidecar 返回 `{start_ms,end_ms,speaker_index}` 与 finalized watermark；ASR 最多等待 2 秒，随后按 speaker turn 重切 PCM、串行识别并按时间顺序输出 sentence。

`speaker_index` 在单个 WebSocket 会话内稳定为 `0..3`，协议层映射为讯飞风格的 `cw[].rl`：角色变化返回 `1..4`，同角色连续发言返回 `0`。重叠区归给当前大段内占用更长的角色，短于 `min_segment_duration_ms` 的角色抖动合并到相邻 turn，音频不丢失。

连接失败、结果超时、流中断与队列溢出都只令当前会话剩余部分降级为普通 ASR；AST sentence 返回 `rl=0`，下一会话重新连接。`GET /readyz` 会把 sidecar 状态放进 optional check，但不会让主 ASR readiness 返回 503。

| 客户端/sidecar 状态 | 是否连接 sidecar | AST v3 行为 |
|---|---|---|
| `enable_role_separation=false` | 否 | 普通 ASR 或 TS-ASR；`sentence` / `Progressive` 均不返回 `cw[].rl` |
| 角色分离开启，sidecar 健康 | 是 | 按 speaker turn 重切；最终 `sentence` 返回角色标记，`Progressive` 不返回 |
| sidecar 未启用、启动失败或连接失败 | 尝试后放弃 | 当前会话走普通 ASR；最终 `sentence` 返回 `rl=0` |
| 结果超时、流中断或队列溢出 | 中止当前流 | 已发送结果不变；当前会话剩余部分走普通 ASR并返回 `rl=0`；下一会话重新连接 |

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
DIARIZATION_MAX_BATCH_SIZE=8 \
DIARIZATION_BATCH_WAIT_MS=12 \
uv run --project services/diarization python -m services.diarization.server
```

确认启动日志中的 `free_gpu_gib`、`peak_allocated_gpu_gib` 与 `peak_reserved_gpu_gib` 后再安装 `deploy/audiollm-diarization.service`。进程在加载前后都要求至少 10 GiB 空闲显存；低于门槛会以退出码 78 停止，systemd 不重启该资源不足实例，主服务自动降级。主服务配置位于 `config.yaml -> defaults.diarization`，默认目标 `localhost:50052`。

生产机安装并启动 systemd unit：

```bash
sudo install -o root -g root -m 0644 \
  deploy/audiollm-diarization.service \
  /etc/systemd/system/audiollm-diarization.service
sudo systemctl daemon-reload
sudo systemctl enable --now audiollm-diarization.service
systemctl status audiollm-diarization.service --no-pager
```

确认 gRPC `Healthz`、50052 监听和 GPU 余量后，再将
`config.yaml -> defaults.diarization.diarization_enabled` 设为 `true` 并重启主服务。
sidecar 是可选故障域，不与 `audiollm-demo` 建立 `Requires`：它不可用时主服务仍可启动，
AST v3 按会话 fail-open；unit 自身通过 `WantedBy=multi-user.target` 独立保证开机启动。
生产 unit 显式设置 batch 8 / 12 ms；允许范围分别为 1–32 和 0–100 ms，非法值回退
到默认值并输出 WARN。修改后必须重跑同卡 AST 8/12/16 路压测，不能只依据 sidecar
单独吞吐选择更大的 batch。

真实 checkpoint 的流式 API smoke test：

```bash
DIARIZATION_GPU_TEST_MODEL=/home/ubuntu/models/diar_streaming_sortformer_4spk-v2.1.nemo \
DIARIZATION_GPU_TEST_AUDIO=/path/to/16k-mono-s16le.wav \
uv run --project services/diarization --with pytest \
  pytest -q tests/test_diarization_gpu.py
```

不设置 `DIARIZATION_GPU_TEST_AUDIO` 时使用 2 秒静音做快速 smoke；发布验收应设置真实
会议 WAV，以覆盖 speaker cache 更新/压缩后的单路与 batched 输出等价性。

## 探索性 H20 诊断（AISHELL-4，非验收）

2026-08-04 使用 `/home/ubuntu/data/testdata/aishell4_work` 中的
AISHELL-4 远场会议录音与 RTTM 金标实测。模型文件固定为上述 revision，
SHA-256 为 `8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8`；
测试脚本为 `scripts/benchmark_diarization.py`，输入以 80 ms PCM 包连续送入真实
`SortformerStream`。下表已改用 `pyannote.metrics` 连续时间 DER；旧版脚本的
80 ms 离散帧估算值不再作为报告口径。

| 30 秒窗口 | 金标/预测人数 | DER（0 collar，含重叠） | DER（250 ms collar，含重叠） | RTF | watermark lag p95 | finish | 峰值 allocated 显存 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `L_R004S02C01 @ 1850s` | 4 / 3 | 11.94% | 10.37% | 0.078 | 640 ms | 100 ms | 0.90 GiB |
| `L_R004S03C01 @ 1000s` | 4 / 4 | 52.00% | 53.21% | 0.053 | 640 ms | 50 ms | 0.90 GiB |

两个窗口另用 NeMo 官方整段 `forward_streaming` 路径复核。无 collar、含重叠的
聚合 DER 与两个单窗口 DER 均与 sidecar 一致，
因此困难窗口的高误差不是 gRPC 或增量分块实现造成。扩展到 6 个人数不超限的
30 秒诊断窗口后，标准 DER 加权为 20.12%，AST 单角色投影代理指标的
250 ms collar、不含重叠 macro DER 为 9.06%，困难窗口为 37.46%。这些窗口
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

## AliMeeting Test far 全集实测

2026-08-04 在 H20 上完成 20 场、10.7765 小时全集评测。为复现 NVIDIA 的
15.60%，金标使用模型卡指定的
[`diar-forced-alignment`](https://github.com/nttcslab-sp/diar-forced-alignment)
AliMeeting Test far RTTM；数据包自带的长句级 RTTM 不是同一计分口径。

- sidecar 标准 DER（0 collar、含重叠）为 15.56709%；NeMo 整段
  `forward_streaming` 为 15.57394%。aggregate 相差 0.00686 个百分点，最大单场
  相差 0.18565 个百分点。
- AST 单角色代理在 250 ms collar、不含重叠口径下 macro DER 为 4.251%，最差
  会话为 20.106%；首 30 秒 fixed-mapping macro DER 为 12.151%。
- 80 ms PCM 包下 watermark lag p95 为 640 ms。单进程离群场复测 RTF 为 0.055；
  单进程峰值 allocated 0.897 GiB、reserved 0.941 GiB。

实测还定位并修复了分窗 frontend 的边界误差：直接对每个 1.12 秒窗口计算 Mel
会让 STFT/pre-emphasis 的首尾 context 帧不同于整段特征，个别会话可放大为 11.62
个百分点的 DER 差。sidecar 现在保留并裁掉 20 ms PCM guard，使分窗 Mel 与整段
Mel 对齐；全量复测得到上述最终结果。

## 讯飞盲分角色冒烟对比

2026-08-05 使用同一份 AliMeeting Test far PCM，真实调用讯飞“实时语音转写大模型”
的 `role_type=2` 盲分模式。2、3、4 人各选一个 60 秒窗口；产品代理口径（250 ms
collar、排除重叠）汇总 DER 为：本项目 AST 单角色投影 14.932%，讯飞 31.540%。
严格 0 ms collar、包含重叠口径下，Sortformer 为 23.265%，讯飞词级角色结果为
45.286%。

该结果只证明当前 3×60 秒中文远场样本上的表现，不能外推为整体优于讯飞，也不评价
ASR CER/WER。讯飞 turns 需要从确定性结果的全部词级 `cw.rl` 重建；只读取句首会漏掉
同一 sentence 内的角色切换。完整窗口、解析假设、分项误差和复现边界见
[Streaming Sortformer 与讯飞实时转写大模型角色分离对比](../../docs/benchmarks/speaker-diarization-iflytek-comparison.md)。

## 验收

CI 使用 fake diarizer 跑协议、切段、超时和降级测试。真实 GPU 验收另行执行：AliMeeting Test far 全集标准 DER（0 collar、含重叠）不高于 17.60%，sidecar 与 NeMo 官方整段路径差异不超过 0.5 个绝对百分点；AST 单角色投影使用 250 ms collar、不含重叠口径，macro DER 不高于 20%、单会话最差值低于 40%，首 30 秒 fixed-mapping macro DER 不高于 25%。ASR 与 sidecar 同卡压测不得 OOM，至少保留 10 GiB 显存，角色等待 p95 小于 2 秒，现有 ASR p95 回归不超过 5%。

2026-08-05 初始同卡压测中，1/4/8 路角色会话稳定，12 路无降级但 final lag p95
为 2.381 秒，16 路有 11/16 会话触发 `result_timeout`。根因是每路
`batch_size=1` 并由全局模型锁串行，同相位角色拆段又把 final ASR 数从普通路径的
57 增到 94，导致同卡 GPU 达到 99% 时 watermark 越过 2 秒。

加入 batch 8 / 12 ms 动态调度后的同音频复测中，8/12/16 路均为全量角色、零降级；
16 路直接 gRPC watermark wait p95 从 1.236 秒降到 0.786 秒，完整 AST ASR
16/16 成功。batch 8 下 8/12/16 路 final lag p95 分别为 1.188、2.020、
2.378 秒；16 路吞吐相对同时期普通 ASR 回归 2.715%。batch 16 的 final lag p95
反而升到 2.436 秒，因此生产保留 batch 8。当前单 H20 的角色功能容量已验证到
16 路；若要求完整 final lag p95 严格低于 2 秒，仍按 8 路规划，12 路为边界观察位。
完整方法、指标口径和原始结果位置见
[Streaming Sortformer 同卡生产压测报告](../../docs/benchmarks/speaker-diarization-production-load.md)。
嘈杂执法场景首期灰度，不在没有脱敏金标前调整模型阈值。
