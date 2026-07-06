# 长音频离线转写性能报告与参数调优

日期：2026-06-11
测试脚本：`scripts/bench_transcribe_jobs.py`（每场景独立临时 config + 重启服务、并发提交、轮询计时、段分布统计）
原始数据：`bench_results/transcribe_jobs_h20_fix_tothread_20260611_173037.json`、`bench_results/transcribe_jobs_h20_saturation_20260611_173338.json`
接口文档：[transcription-jobs-api.md](../api/transcription-jobs-api.md)

## 1. 测试环境

| 项目 | 值 |
|---|---|
| 端点 | `POST /api/asr/transcriptions`，127.0.0.1 回环，上传开销可忽略（62 MB 实测 0.3-2.1 s） |
| GPU | 单卡 NVIDIA H20（96 GB），双 vLLM 实例共卡（AmphionASR-4.3B:8009 + Qwen3-ASR-1.7B:8001） |
| 实际推理路径 | `enable_dual_asr_fusion=false` → primary-only，每段恰好 1 个 vLLM 请求（打 Amphion） |
| VAD | ten-vad，离线切分（CPU），`transcribe_silence_duration_ms=800`（除 silence 扫描组外） |
| 测试音频 | 32.5 分钟（1949 s）8 麦阵列会议录音（AISHELL-4 风格），转 16 kHz mono WAV 62 MB，约 57% 有声占比 |
| 服务形态 | uvicorn 单进程单事件循环，每场景重启 |
| 干扰控制 | 压测期间 GPU 无其他流量；同卡双 vLLM 但 Qwen 实例空载 |

注意与 [tuling-ast-v3-benchmark.md](tuling-ast-v3-benchmark.md) 的差异：那份报告是 A800 独占 + 流式 partial 重推理口径，本报告是 H20 共卡 + 离线纯 final 口径，吞吐数字不可直接互比。

## 2. 指标口径

| 指标 | 定义 |
|---|---|
| e2e | 客户端 POST 开始到轮询见 succeeded（轮询间隔 0.5 s，含上传与排队） |
| wall | 批次内全部 job 提交到全部完成的墙钟 |
| 吞吐（xRT） | 成功 job 数 × 音频时长 / wall，跨并发可比 |
| 段长 p50/p90 | result.segments 的 (end_ms-start_ms) 分布 |
| failed | 失败段总数（单段重试一次后仍失败才计入） |

每场景独立重启服务（transcribe_* 是进程级配置）；首场景为 warmup（vLLM 预热）不记录。方差参考：同配置两次 e2e 24.6 s / 23.5 s（±2%）。

## 3. 实验暴露的事件循环阻塞（已修复）

首轮多 job 实验出现反常数据：第 2 个 job 的 POST 上传耗时 13.2 s（本地回环不可能），且批内所有 job 的 e2e 被拉齐到同一时刻。微基准定位：32.5 分钟音频的离线 VAD 切分（`segment_pcm_offline`，纯 CPU）耗时 13.6 s，而它直接跑在事件循环里——切分期间整个服务冻结，包括其他 job 的上传/轮询，以及同进程所有实时 WS 端点。

修复：切分移入工作线程（`asyncio.to_thread`），`backend/asr/transcribe.py`。修复前后对比（jobs=4, conc=2, seg=4）：

| 口径 | 修复前 | 修复后 |
|---|---|---|
| 第 2-4 个 job 的上传耗时 | 13.2-13.6 s（被冻结） | ≤1.0 s |
| 批次 wall | 65.4 s | 27.2 s |
| 总吞吐 | 119 xRT | 286 xRT |

ten-vad 的 C 调用释放 GIL（8 job 场景 109 s 切分 CPU 总量在 41.5 s 墙钟内完成，推断有效并行 ≥2.6 核），多 job 的切分与推理能真正重叠。

## 4. 单任务：段并行度扫描（transcribe_segment_concurrency）

固定 silence=800（303 段，段长 p50 3.18 s），单 job：

| seg_conc | e2e | 吞吐 | 推理段耗时（e2e − 13.6 s VAD） |
|---|---|---|---|
| 1 | 39.6 s | 49 xRT | ~26 s |
| 2 | 30.0 s | 65 xRT | ~16 s |
| 4 | 24.6 s / 23.5 s（复测） | 79-83 xRT | ~11 s |
| 8 | 21.0 s | 93 xRT | ~7.4 s |
| 16 | 20.5 s | 95 xRT | ~6.9 s |

要点：

- 推理部分并行收益在 8 路后基本耗尽（8→16 仅 -0.5 s）；e2e 此后被 13.6 s 的串行 VAD 切分主导（Amdahl 项）。
- 4→8 的 e2e 收益 3.6 s，对轮询间隔 2-5 s 的离线场景体感接近零，但单 job 峰值 vLLM 并发翻倍——默认 2 个并发 job 时峰值压力从 8 升到 16，会挤压同 GPU 实时流量的尾延迟。维持 4。

## 5. 单任务:切段停顿扫描（transcribe_silence_duration_ms）

固定 seg_conc=4，单 job：

| silence_ms | 段数 | 段长 p50 | 段长 p90 | e2e | 吞吐 | 全文字符 |
|---|---|---|---|---|---|---|
| 350（全局值） | 636 | 1.49 s | 4.0 s | 27.5 s | 71 xRT | 7366 |
| 600 | 417 | 2.27 s | 5.6 s | 25.0 s | 78 xRT | 7118 |
| 800（默认） | 303 | 3.18 s | 7.3 s | 24.6 s | 79 xRT | 6933 |
| 1200 | 175 | 4.93 s | 11.6 s | 22.5 s | 87 xRT | 6871 |

要点：

- 段越少越快（每段固定的请求/编码开销摊薄），800 vs 350 提速约 11%，性能与段粒度方向一致。
- 全文字符随段变长缓慢下降（350→1200 约 -7%）。短段倾向保留更多语气词/重复，长段输出更紧凑；未做 WER 评测，不能据此判断准确率优劣，只说明文本量有此趋势。
- 800 ms 维持默认：纪要段粒度（p50 3.2 s）、速度、文本完整性的平衡点。

## 6. 多任务扩展性与容量上界

固定 seg_conc=4、silence=800，同时提交 N 个相同 job：

| 场景 | jobs | conc_jobs | 峰值段并发 | wall | 单 job e2e 范围 | 总吞吐 |
|---|---|---|---|---|---|---|
| 单 job 基线 | 1 | 2 | 4 | 24.6 s | 24.6 s | 80 xRT |
| j2_c2 | 2 | 2 | 8 | 25.2 s | 25.0-25.2 s | 154 xRT |
| j4_c2 | 4 | 2 | 8 | 27.2 s | 27.0-27.2 s | 286 xRT |
| j4_c4 | 4 | 4 | 16 | 27.7 s | 27.1-27.7 s | 281 xRT |
| j2_c2_seg8 | 2 | 2 | 16 | 22.6 s | 22.2-22.6 s | 173 xRT |
| j8_c2 | 8 | 2 | 8 | 41.5 s | 40.7-41.5 s | 376 xRT |
| j8_c4 | 8 | 4 | 16 | 41.1 s | 39.9-41.1 s | 379 xRT |

要点：

- 扩展曲线：1→2→4→8 job 吞吐 80 → 154 → 286 → 377 xRT，4 job 内接近线性（VAD 多核并行 + vLLM 批处理），8 job 时单 job 时延 +69%（24.6→41 s），vLLM 接近饱和（约 58 段/s ≈ 380 xRT 是本音频下的容量上界）。
- conc_jobs 2 与 4 在 4/8 job 下吞吐无差异（±2%）：吞吐由"段级总并发 × vLLM 容量"决定，与 job 级准入无关。conc_jobs 的实际作用是内存驻留（每 running job 持有整段 int16 PCM，32.5 分钟 ≈ 62 MB）与完成顺序（2 更接近 FIFO，先交付先提交的任务）。维持 2。
- 全部 17 个场景 0 失败段、全文字符稳定（6831-7035），高并发未触发 `primary_asr_timeout`（4 s）。

## 7. 强切上限（transcribe_max_segment_sec）

silence=800、seg_conc=4，单 job：

| max_segment_sec | 段数 | 段长 p50 | 段长 max | e2e |
|---|---|---|---|---|
| 30（默认） | 303 | 3.18 s | 21.9 s | 24.6 s |
| 15 | 308 | 3.25 s | ~15 s | 23.5 s |

性能差异在方差内。本音频 800 ms 停顿下最长自然段 21.9 s，15 s 上限仅多切 5 段。维持 30：完整句子优先，性能无代价。

## 8. 推荐配置

实验结论：当前 `config.yaml` 默认值即为甜点，无需修改；本轮的主要性能产出是第 3 节的事件循环阻塞修复。

| 参数 | 推荐值 | 依据 |
|---|---|---|
| `transcribe_segment_concurrency` | 4 | 提到 8 单 job 仅 -3.6 s（离线无感），峰值 vLLM 压力翻倍挤压实时流量（第 4 节） |
| `transcribe_max_concurrent_jobs` | 2 | 提到 4 吞吐无变化，只多占内存、拖慢先到任务的完成（第 6 节） |
| `transcribe_job_queue_max` | 8 | 8 个 32 分钟会议积压实测 41.5 s 清空，容量充裕（第 6 节） |
| `transcribe_silence_duration_ms` | 800 | 段粒度/速度/文本完整性平衡点（第 5 节） |
| `transcribe_max_segment_sec` | 30 | 与 15 性能无差异，段更完整（第 7 节） |

## 9. 容量规划速查

默认配置（seg=4、conc=2）、本音频特征（57% 有声）下：

| 量 | 值 |
|---|---|
| 单任务 e2e | 约音频时长 / 80（32.5 分钟会议 ≈ 25 s；90 分钟 ≈ 70 s 量级） |
| e2e 构成 | VAD 切分 ≈ 0.42 s/分钟音频（串行下界） + 推理（随 seg_conc 并行） |
| 满负荷吞吐 | 约 380 xRT（8 任务并行，单任务时延 +69%） |
| 内存 | 每 running/queued 任务驻留约 2 MB/分钟音频（int16 PCM，切分后段副本短暂翻倍） |

提交端建议：吞吐敏感的批量回灌场景，同时保持 4 个在途任务即可拿到 ~286 xRT（单任务时延仅 +10%）；再加并发只换来排队。

## 10. 局限与已知未知

- 单一测试音频（中文、多人、近讲阵列混缩）；不同语种/信噪比/有声占比下段数与推理耗时会漂移，xRT 数字应理解为该特征下的量级。
- 未测与实时流式端点的混合负载：离线 job 满负荷时对实时端点尾延迟的挤压程度未知，仅从 tuling 报告（不同卡型）推断 vLLM 并发 16 时实时 onset 会有可感退化——这是 seg_conc 维持 4 的保守依据。
- 未做 WER 评测：silence 扫描中的文本量差异（第 5 节）不能等同于准确率结论。
- 双 vLLM 共卡但副模型空载；若开启 `enable_dual_asr_fusion`，每段变为 2 个请求，本报告吞吐数字需除以约 2 重估。
