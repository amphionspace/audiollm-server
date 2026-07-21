# 实时转写 AST v3 WebSocket API

`/tuling/ast/v3` 以讯飞图灵 AST v3 信封协议对外提供实时语音转写。它复用与 `/transcribe-streaming` 相同的 VAD 分段流水线，但按端点策略恒为 primary-only：只调用主模型（由 `astv3_vllm_*` 指定，当前留空，回退全局 primary `vllm_base_url`），不启用本地副模型、partial 静音门与双模型融合。另一处区别在线上协议：音频以 base64 编码放在 JSON 帧的 `payload.audio.audio` 中，`header.status`（0/1/2）驱动开始/中间/结束状态机，结果按 `payload.result` 词图结构返回。

该端点与现有 `/transcribe-streaming` 并存，互不影响。客户端可使用讯飞 `tuling-ast-sdk`（Java）或任意 WebSocket 客户端按本协议对接。

## 接口信息

| 项目 | 说明 |
|---|---|
| 协议 | WebSocket |
| 路径 | `/tuling/ast/v3` |
| 完整 URL | `ws(wss)://[ip]:[port]/tuling/ast/v3` |
| 鉴权 | 无内置鉴权 |
| 音频输入 | base64 编码的 16 kHz、mono、signed 16-bit little-endian PCM；首帧若为带 RIFF/WAVE 头的整段音频会自动剥离文件头 |
| 分段策略 | 服务端 VAD 自动切段 |
| 中间结果 | 支持，msgtype 为 Progressive，受 `enable_pseudo_stream` 影响，可经 `parameter.asr_config` 覆写 |
| 最终结果 | 每个语音段一条，msgtype 为 sentence；尾帧（status=2）后 flush 残余音频 |

## 调用流程

```text
Client                                      Server
  | ---- WebSocket connect --------------> |
  | ---- frame status=0 (首帧 + 音频) ----> |
  | <---------------- Progressive -------- |  中间结果（可选）
  | ---- frame status=1 (音频) ----------> |
  | <---------------- sentence ----------- |  一个语音段
  | ---- frame status=1 (音频) ----------> |
  | ---- frame status=2 (尾帧 + 音频) ----> |
  | <---------------- sentence ----------- |  尾部音频
  | <---------------- status=2 ----------- |  会话结束标志
```

与 `/transcribe-streaming` 不同，AST v3 没有 `ready` 握手和 `start`/`stop` 控制消息：会话生命周期完全由 `header.status` 表达。

## 客户端请求

每一帧都是一个完整的 JSON 信封：

```json
{
  "header": {
    "traceId": "traceId123456",
    "appId": "123456",
    "bizId": "39769795890",
    "status": 0
  },
  "parameter": {
    "asr_config": {
      "hotword_pool_id": "tenant-a",
      "enable_role_separation": false,
      "enrollment_enable": true,
      "enrollment_id": "234567"
    },
    "engine": {
      "wdec_param_LanguageTypeChoice": "3"
    }
  },
  "payload": {
    "audio": {
      "audio": "JiuY3iK9AAB..."
    }
  }
}
```

### header

| 字段 | 类型 | 必传 | 说明 |
|---|---|---|---|
| traceId | String | 是 | 日志追踪 id，原样回显到响应 header.traceId |
| appId | String | 否 | 应用系统 id，仅记录日志 |
| bizId | String | 是 | 业务 id，仅记录日志 |
| resIdList | List<String> | 否 | 已废弃；服务端仅记录并忽略，不作为目标说话人字段 |
| status | int | 是 | 请求状态：0 首帧，1 中间帧，2 尾帧 |

### parameter

| 字段 | 类型 | 必传 | 说明 |
|---|---|---|---|
| engine | Map | 否 | 引擎透传参数，仅记录日志，当前不映射到任何行为（见已知限制）。兼容 SDK 使用的 parameter.service |
| asr_config | Map | 否 | 本服务扩展的 per-connection 配置覆写、热词池 ID 与目标说话人 enrollment ID 承载槽，仅首帧（status=0）读取，仅当前连接生效、不落盘；详见“配置覆写”章节 |

### payload

| 字段 | 类型 | 必传 | 说明 |
|---|---|---|---|
| payload.audio.audio | String | 是 | base64 编码的 PCM 音频分片 |
| payload.text.text | String | 否 | AST v3 临时热词字段；会被解析为临时请求热词，final 段去重限量后优先进入 prompt，并覆盖精确重复或整词同音（忽略声调）的 RAG-ASR 召回热词，不写入热词池 |

### 状态机与音频

| status | 含义 | 服务端处理 |
|---|---|---|
| 0 | 首帧 | 建立会话、捕获 traceId、生成 sid；读取 `payload.text.text` 作为临时请求热词；若本帧带音频则同时送入 VAD |
| 1 | 中间帧 | 解码音频送入 VAD |
| 2 | 尾帧 | 先送本帧音频，再 flush 残余音频结束会话 |

音频建议：每帧约 4096 字节（讯飞 SDK 默认 `32 * 128`），原则上单帧不超过 16 KB，建议至少 40 ms 语音。正常处理过程中客户端不要主动断开。首帧允许携带带 WAV 头的整段音频前缀，服务端会一次性剥离文件头后按裸 PCM 处理。

## 配置覆写（parameter.asr_config）

`parameter.asr_config` 是本服务在 AST v3 信封上的扩展槽位，用于按连接临时调参，也承载 RAG-ASR 热词池隔离 ID、角色分离开关和目标说话人 enrollment 参数。热词池只使用 `hotword_pool_id`；声纹只使用 `enrollment_enable` + `enrollment_id`，且受 `enable_role_separation` 优先级约束；它与讯飞 `parameter.engine`（仅记录日志、不映射行为）并列、互不影响。仅在首帧（status=0）读取，仅对当前连接生效、不落盘；新连接或服务重启都回到服务端默认。

取值优先级（后者覆盖前者）：`backend/config.py` 内置默认 → `config.yaml` 服务端默认 → `parameter.asr_config` 客户端临时覆写。

只接受白名单内的扁平字段名；未知字段、受限字段（模型地址、密钥、连接池/队列等基础设施项）以及非法值会被忽略并保持服务端默认，不会中断连接。`language`、`hotword_pool_id`、`enable_role_separation`、`enrollment_enable` 和 `enrollment_id` 是特例：`language` 会被用作本次会话语言（等价于 `/transcribe-streaming` 的 `start.language`），`hotword_pool_id` 会被用作热词池隔离 ID（默认 `default`），后三者按“目标说话人（TS-ASR）”章节的矩阵决定是否启用声纹；这些字段不会进入普通配置覆写白名单。

可覆写字段与 `/transcribe-streaming` 的 `start.config` 共用同一白名单（`backend/config.py` 的 `CLIENT_OVERRIDABLE_FIELDS`）。下面按类别逐字段说明语义：默认列为服务端 `config.yaml` 当前生效值，本端点列标注该字段在 AST v3 是否产生可观察效果（本端点恒为 primary-only，副模型与融合相关字段即使传入也不生效）。

当服务端启用 `k2_enabled=true` 时，本端点的 `Progressive` 中间结果来自外部 k2 流式 ASR，`sentence` 终稿仍由本服务 LLM ASR 产生。k2 只做纯识别，不接热词、不接目标说话人、不返回 token timestamps；热词召回、目标说话人过滤、ITN 与车牌规范化只作用于 sentence。此时切段权威是 k2 endpoint，本服务只保留有界缓冲，并在 Progressive/sentence 进入下游前用 `k2_voice_gate_*` 确认有人声证据；voice gate 只做放行/丢弃，不裁剪段首/段尾，`bg` / `ed` 仍反映该 k2 段缓冲边界。VAD 与伪流式间隔类覆写仍会被接受但不再决定切点或首字时机；`enable_pseudo_stream=false` 仍会抑制 Progressive 下发。

VAD / 分段（凡按帧计的字段，其帧时长由 VAD 后端 hop 决定：ten-vad 约 16 ms/帧，能量兜底约 10 ms/帧）：

| 字段 | 类型 | 默认 | 含义 | 本端点 |
|---|---|---|---|---|
| vad_threshold | float | 0.65 | 平滑后语音概率高于该阈值才算语音帧；越大越严格、越不易判为语音 | 生效 |
| silence_duration_ms | int | 350 | 进入语音后连续静音达到该时长即切段，是唯一的切段静音阈值（毫秒友好，无需换算帧数） | 生效 |
| vad_smoothing_alpha | float | 0.3 | 语音概率指数平滑系数，smoothed = alpha×上一帧 + (1−alpha)×当前帧；越大越平滑越滞后，0.3 表示当前帧权重 0.7、偏灵敏 | 生效 |
| vad_start_frames | int | 10 | 连续多少语音帧才确认进入语音，用于起始防抖 | 生效 |
| vad_pre_speech_ms | int | 500 | 确认语音起点后向前补回的预缓存音频时长，避免吃掉起音 | 生效 |
| vad_keep_tail_ms | int | 40 | 切段时在结尾保留的静音尾音时长，使收尾更自然，不超过 silence_duration_ms 对应帧数 | 生效 |
| min_segment_duration_ms | int | 350 | 短于该时长的语音段被丢弃、不送 ASR | 生效 |

切段静音阈值由 `silence_duration_ms` 单独决定。上表 VAD 字段经首帧 `parameter.asr_config` 覆写后对本连接即时生效（服务端在 `configure` 时把它们应用到该连接的 VAD 实例）。

伪流式：

| 字段 | 类型 | 默认 | 含义 | 本端点 |
|---|---|---|---|---|
| enable_pseudo_stream | bool | true | 是否输出伪流式中间结果（msgtype=Progressive）；关闭后每段只在结束时发一条 sentence | 生效 |
| pseudo_stream_interval_ms | int | 500 | 相邻中间结果之间的最小间隔（仅节流首个之后的刷新，不影响首字） | 生效 |
| pseudo_stream_first_partial_ms | int | 200 | 每段语音首个 partial 的触发门槛，从 min_segment_duration_ms 解耦（dataclass 兜底 350，config.yaml 默认设 200 走低延迟）；与 vad_start_frames 按 max 决定首字，min_segment_duration_ms 仍独立控制 final 段短噪声过滤 | 生效 |

AST v3 协议兼容：

| 字段 | 类型 | 默认 | 含义 | 本端点 |
|---|---|---|---|---|
| enable_role_separation | bool | true | 客户端角色分离开关；省略等价于 true，且优先级高于声纹 | 生效于协议出参和声纹启用矩阵；true 时 sentence 返回 `cw[].rl=0` 兼容占位，false 时 sentence/Progressive 均不返回 `cw[].rl` |
| enrollment_enable | bool | false | 是否显式启用目标说话人声纹 | 仅当 `enable_role_separation=false && enrollment_enable=true && enrollment_id` 非空时启用 |

ASR 模型组合 / 超时：

| 字段 | 类型 | 默认 | 含义 | 本端点 |
|---|---|---|---|---|
| enable_primary_asr | bool | true | 是否启用主模型（本端点主模型由 astv3_vllm_* 指定，当前留空，回退全局 vllm_base_url）；置 false 则本端点无可用模型、不产出文本 | 生效 |
| enable_secondary_asr | bool | true | 是否启用本地副模型 Qwen3，并关联副模型静音门与融合 | 无效，本端点强制为 false |
| enable_dual_asr_fusion | bool | false | final 段是否做主副融合矫正，依赖副模型 | 无效，本端点强制为 false |
| primary_asr_timeout | float（秒） | 4.0 | 主模型单段调用的整体墙钟超时（asyncio 层），超时即放弃本次主模型结果 | 生效 |
| asr_request_timeout | float（秒） | 120 | 底层 HTTP 请求（httpx）的超时；主模型实际等待时间为 min(primary_asr_timeout, asr_request_timeout) | 生效 |
| debug_show_dual_asr | bool | false | 是否在结果中附带双模型调试信息 | 无效，本端点无副模型且词图响应无调试字段 |

融合阈值（仅在融合开启时参与决策；本端点融合恒关，故全部无效，列出仅为说明字段含义）：

| 字段 | 类型 | 默认 | 含义 | 本端点 |
|---|---|---|---|---|
| fusion_similarity_threshold | float | 0.85 | 主副文本相似度（difflib 比率）不低于此值且主达标时直接采用主结果 | 无效 |
| fusion_min_primary_score | float | 0.55 | 主模型文本质量分达标线，低于则主结果不被单独信任 | 无效 |
| fusion_max_repetition_ratio | float | 0.35 | 主模型 token 重复率上限，超过判为复读/幻觉风险并回退副模型 | 无效 |
| fusion_disagreement_threshold | float | 0.55 | 主副分歧度（1−相似度）上限，超过且主热词不占优时判为幻觉风险 | 无效 |
| fusion_hotword_boost | float | 0.12 | 质量分中每命中一个召回热词的加分系数，累计封顶 0.35 | 无效 |
| fusion_primary_score_margin | float | 0.08 | 主质量分需至少高出副模型该边际，才判定主明显更好 | 无效 |

TS-ASR 注册参数（约束注册接口的时长校验与缓存 TTL）：

| 字段 | 类型 | 默认 | 含义 | 本端点 |
|---|---|---|---|---|
| asr_enrollment_min_sec | float（秒） | 1.0 | 注册音频最小时长，低于返回 too_short | 首帧覆写无可观察效果，见表后说明 |
| asr_enrollment_max_sec | float（秒） | 8.0 | 注册音频最大时长，超出尾截 | 首帧覆写无可观察效果 |
| asr_enrollment_ttl_sec | float（秒） | 3600 | 注册缓存 TTL，最近一次使用后重新计时 | 首帧覆写无可观察效果 |

这三项只在注册接口 `POST /api/asr/enrollment`（独立的 REST 调用，按服务端默认执行）生效。AST v3 首帧经 `parameter.asr_config.enrollment_id` 携带的是已注册 id，本端点既不重新注册、也不消费这些值，因此在 `parameter.asr_config` 里覆写这些生命周期参数不会改变本连接的目标说话人行为。

共享白名单还包含情感类字段（`emotion_*`，如 `emotion_task_mode`、`emotion_request_timeout` 等），仅对情感端点有效，对本 ASR 端点无效。完整白名单与各字段按类别速览见 [API 总览](../api-reference.md) 的“临时配置覆写”。

### 首字延迟优化（低延迟场景）

实时麦克风场景下，首字延迟（首个非空 partial 相对第一帧发送）由两个串联门槛的较大者决定（max）：起音确认（`vad_start_frames` × 每帧约 16 ms）与首个 partial 的音频门槛（`pseudo_stream_first_partial_ms`）。当前默认分别为 10 帧（约 160 ms）和 200 ms，因此首字门槛约 200 ms；通常无需再覆写：

| 参数 | 服务端默认 | 低延迟推荐 | 作用 |
|---|---|---|---|
| vad_start_frames | 10 | 10 | 起音确认帧数；TEN VAD 下约 160 ms，让首字门槛由默认 200 ms partial 门槛决定 |
| pseudo_stream_first_partial_ms | 200 | 200 | 首个 partial 的累积音频门槛；服务端默认已设 200 ms，无需再覆写（再低也会被 vad_start_frames 卡住） |

发送侧建议用较小音频帧（chunk 32-64 ms）而非大块（如 500 ms），让起音确认颗粒更细、边发边出字；首字出来后无需切换大块。

实测参考（kespeech 短句，realtime，chunk 64 ms，20 条）：当前默认 `vad_start_frames=10` + `pseudo_stream_first_partial_ms=200` 时首字 p50 约 223 ms、mean 约 362 ms、全程 0 漏识别；作为对比，旧的全 350/20 默认 p50 约 360 ms。`vad_start_frames` 由 10 再降到 8 的额外收益已饱和（p50 仅再降约 2 ms），不建议更激进，以免在含噪环境抬高误触发率。

首帧 `parameter.asr_config` 的覆写仅对本连接生效、不改变服务端默认；`min_segment_duration_ms` 仍独立控制 final 段的短噪声过滤（保持 350 ms 不变）。注意 `pseudo_stream_first_partial_ms=200` 是服务端全局默认，对所有产 partial 的端点（含 `/transcribe-streaming`）都已生效，并非本端点专属。

低延迟首帧示例：

```json
{
  "header": { "traceId": "demo-low-latency", "status": 0 },
  "parameter": {
    "asr_config": { "vad_start_frames": 10 }
  },
  "payload": { "audio": { "audio": "<base64 PCM>" } }
}
```

首帧示例（指定语言、关闭伪流式中间结果、放宽 VAD 切段）：

```json
{
  "header": {
    "traceId": "traceId123456",
    "bizId": "39769795890",
    "status": 0
  },
  "parameter": {
    "asr_config": {
      "language": "zh",
      "enable_pseudo_stream": false,
      "vad_threshold": 0.45,
      "silence_duration_ms": 300
    }
  },
  "payload": {
    "audio": { "audio": "JiuY3iK9AAB..." }
  }
}
```

## 目标说话人（TS-ASR）

支持只转写指定说话人的语音，复用与 `/transcribe-streaming` 相同的注册机制，分两步：

1. 注册：通过 `POST /api/asr/enrollment` 上传 1-8 秒目标说话人音频，支持 WAV、MP3 和 raw PCM（16 kHz mono s16le），拿到 `enrollment_id`（见 [API 总览](../api-reference.md) 的注册接口）。默认 demo 本地缓存注册音频；打开 `enable_triton_enrollment_store=true` 且配置 RAG-ASR 管理服务后，新注册音频会转发给 RAG-ASR，并由 RAG-ASR 将 embedding tensor 和元数据落盘到本地 `enrollment_store_dir`（默认 `var/enrollments`）。
2. 携带：在首帧（status=0）设置 `parameter.asr_config.enable_role_separation=false`、`enrollment_enable=true`，并把该 id 放进 `parameter.asr_config.enrollment_id`。

```json
{
  "header": {
    "traceId": "traceId123456",
    "bizId": "39769795890",
    "status": 0
  },
  "parameter": {
    "engine": {},
    "asr_config": {
      "enable_role_separation": false,
      "enrollment_enable": true,
      "enrollment_id": "ule8QilVjZql30Q9oy9kiQ"
    }
  },
  "payload": { "audio": { "audio": "JiuY3iK9AAB..." } }
}
```

说明：

- enrollment_id 仅在首帧读取，整段会话沿用同一目标说话人。
- 声纹启用矩阵：
  - `enable_role_separation=true` 或省略：优先角色分离，忽略 `enrollment_enable` 与 `enrollment_id`，响应 `enrollment_applied=false`。
  - `enable_role_separation=false && enrollment_enable=false` 或省略：不启用声纹，即使传入 `enrollment_id` 也忽略。
  - `enable_role_separation=false && enrollment_enable=true && enrollment_id` 非空：启用声纹。
  - `enable_role_separation=false && enrollment_enable=true && enrollment_id` 为空：返回参数错误，不进入普通 ASR。
- 若启用了声纹但 `enrollment_id` 不可用，服务端回退为普通 ASR，结果中 `enrollment_applied=false`，并尽量返回 `enrollment_fallback_reason`（如 `not_found` / `incompatible` / `upstream_unavailable`）。默认 demo 本地缓存下 enrollment_id 有 TTL（默认 3600 秒、每次使用续期）且服务重启即失效；启用 RAG-ASR 管理服务后，embedding tensor 与元数据会落盘到 RAG-ASR 的 `enrollment_store_dir`（默认 `var/enrollments`），重启不丢，但文件缺失或与当前模型/adapter 不兼容时也按同一兼容语义回退。完整生命周期（存储/有效期/容量/删除）见 [API 总览](../api-reference.md) 注册接口的“生命周期”。
- `header.resIdList` 不再作为目标说话人字段；若存在仅记录并忽略。
- 未按矩阵启用声纹时为普通 ASR。

## 服务端响应

每条响应也是一个信封：

```json
{
  "header": {
    "code": 0,
    "message": "success",
    "sid": "AST_MKMZO0WX2SLZ4",
    "traceId": "traceId123456",
    "status": 1
  },
  "payload": {
    "result": { }
  }
}
```

### header

| 字段 | 类型 | 说明 |
|---|---|---|
| code | int | 错误码，0 表示成功，非 0（当前实现为 -1）表示错误 |
| message | String | 描述信息，成功为 success |
| sid | String | 会话唯一标识，服务端按 AST_ 前缀生成 |
| traceId | String | 回显请求的 traceId |
| status | int | 结果状态：1 识别中，2 识别结束（终止帧） |

错误响应只含 header（code 非 0），不含 payload；客户端应在 code 非 0 时停止处理。

### payload.result

| 字段 | 类型 | 说明 |
|---|---|---|
| segId | int | 段 id，从 0 递增 |
| bg | int | 段开始时间，单位 ms，msgtype 为 sentence 时给出 |
| ed | int | 段结束时间，单位 ms，msgtype 为 sentence 时给出 |
| ei | int | 暂未使用，固定 0 |
| ls | Bool | 最后结果标志；逐段结果为 false，仅终止帧为 true |
| metadata | String | 暂未使用，固定空串 |
| msgtype | String | sentence 为最终结果，Progressive 为中间结果 |
| sn | int | 结果序号，从 1 递增，msgtype 为 sentence 时给出 |
| pa | int | 暂未使用，固定 0 |
| enrollment_applied | Bool | 本条结果是否实际携带可用声纹材料进入 ASR；默认 false，不保证模型声学结果一定完全按目标说话人过滤 |
| enrollment_fallback_reason | String | 可选；`enrollment_applied=false` 时的原因，如 `disabled` / `not_found` / `incompatible` / `upstream_unavailable` |
| vad | Object | 句子级 VAD 信息，vad.ws[].bg/ed 为句子起止时间，单位 10 ms 帧 |
| ws | Array | 转写词图，见下表 |

### payload.result.ws

| 字段 | 类型 | 说明 |
|---|---|---|
| ws[].bg | int | 词语开始时间，单位 10 ms 帧 |
| ws[].cw | Array | 词语识别候选 |
| cw[].w | String | 识别文本；msgtype=sentence（最终结果）默认已做 ITN 与车牌规范化，msgtype=Progressive（中间结果）保持口语形式。见“文本规范化”一节 |
| cw[].lg | String | 语种，如 zh |
| cw[].rl | int | 角色编号；当前版本暂不做真实角色分离。仅当 `enable_role_separation=true` 或省略时，msgtype=sentence 中固定返回 0 作为兼容占位；`enable_role_separation=false` 时 sentence/Progressive 均不返回该字段 |
| cw[].wb | int | 词开始位置，单位 10 ms 帧（数值 ×10 为毫秒） |
| cw[].we | int | 词结束位置，单位 10 ms 帧 |
| cw[].wp | String | 顺滑词类型：s 顺滑词，n 普通字符，p 标点，g 语义分段标志 |
| cw[].sc | String | 词置信度 |
| cw[].wc | String | 词置信度 |
| cw[].ng | String | 噪声分 |
| cw[].ph | String | 音素信息 |

result 示例（最终结果）：

```json
{
  "segId": 0,
  "bg": 140,
  "ed": 3230,
  "ei": 0,
  "ls": false,
  "metadata": "",
  "msgtype": "sentence",
  "sn": 1,
  "pa": 0,
  "enrollment_applied": false,
  "enrollment_fallback_reason": "disabled",
  "vad": { "ws": [{ "bg": 14, "ed": 323 }] },
  "ws": [
    {
      "bg": 14,
      "cw": [
        {
          "lg": "zh", "ng": "0.00", "ph": "phone", "sc": "0.00",
          "rl": 0, "w": "你好兄弟", "wb": 14, "wc": "0.00", "we": 323, "wp": "n"
        }
      ]
    }
  ]
}
```

## 文本规范化（仅最终结果）

服务端默认对最终结果（msgtype=sentence）的 cw[].w 文本做两类本地后处理，中间结果（msgtype=Progressive）不变：

- 通用 ITN（enable_asr_itn，仅中文）：口语数字转阿拉伯，如 六五四三八→65438、二零二四年→2024年。
- 车牌规范化（enable_asr_plate_normalize）：字母大写、去车牌内分隔符、口语数字按位转阿拉伯，并按 GB 车牌形态校验后才改写，如 辽b二四五零七→辽B24507。
- 已知边界：省份简称被声学误识别成字母（冀→J）属识别错误，后处理只修数字与字母（车牌号为JR六五四三八→车牌号为JR65438），不还原省份字。
- 两个开关均为服务端配置（config.yaml 的 defaults.itn 分组），不在 parameter.asr_config 可覆写白名单内；任一处理异常都回退原文。

## 降级说明（重要）

当前 ASR 模型只输出整段文本，不产出词级对齐、词级语种或置信度。因此本端点对 ws/cw 词图采用降级填充，集成方需知悉：

| 字段 | 降级行为 |
|---|---|
| ws / cw 结构 | 每个 sentence 段只产生一个 ws、一个 cw，cw.w 为整段文本，不做逐词切分 |
| bg / ed（段级） | 真实值，基于会话累计已消费音频估算，单位 ms |
| vad.ws[].bg/ed | 取段级起止，单位 10 ms 帧 |
| cw.wb / cw.we | 取段级起止帧，并非逐词时间戳 |
| cw.wp | 固定 n |
| cw.sc / cw.wc / cw.ng | 固定字符串 0.00 |
| cw.ph | 固定字符串 phone |
| cw.lg | 取段级检测/传入语种映射的短码 |
| cw.rl | `enable_role_separation=true` 或省略时仅 sentence 返回固定整数 0；`enable_role_separation=false` 时不返回；当前不做真实角色分离 |

段级 bg/ed 为近似值：它基于流累计消费的样本数，会忽略 VAD 静音判定延迟与尾部裁剪，误差通常在百毫秒量级。

## 已知限制

| 限制 | 说明 |
|---|---|
| resIdList | 已废弃，仅记录并忽略；目标说话人请使用 `parameter.asr_config.enrollment_id` |
| parameter.engine | 讯飞引擎透传参数（如 wdec_param_LanguageTypeChoice、wrec_param_language_name）在本服务无对应能力，仅记录日志，不影响识别；如需按连接调参请改用 parameter.asr_config（见配置覆写章节） |
| 角色分离 | 当前版本暂不做真实角色分离；`enable_role_separation` 默认 true 且优先于声纹，仅影响声纹启用矩阵与 `cw[].rl` 是否作为兼容占位返回 |
| 词级时间戳 | 见降级说明，非逐词真实值 |
| 鉴权 | 无内置鉴权，需在网关层实现访问控制 |

## 错误码

| code | 含义 |
|---|---|
| 0 | 成功 |
| -1 | 通用错误，message 携带具体原因 |

AST v3 规范仅约定 code 0 为成功，其余码段交由实现方定义。本服务统一使用 -1 表示可恢复的段处理错误，并在 header.message 给出描述。

## 会话结束语义

- 每个语音段的最终结果以 msgtype=sentence、header.status=1 返回。
- 中间结果以 msgtype=Progressive、header.status=1 返回。
- 整个会话结束时（客户端发送 status=2 或断开连接后），服务端补发一条 header.status=2 的终止帧，其 payload.result 不含 ws（ls=true），作为会话结束标志。
- 若整段会话没有任何有效语音，则只发送终止帧，不发送空的 sentence。

## Python 示例

完整可运行脚本见 [examples/ws_ast_v3.py](../examples/ws_ast_v3.py)，集成测试客户端见 [tests/test_ast_v3_ws_client.py](../../tests/test_ast_v3_ws_client.py)。

```bash
pip install websockets numpy

python docs/examples/ws_ast_v3.py sample.wav \
  --url ws://172.16.0.3:8082/tuling/ast/v3 \
  --hotwords "挚音科技,张硕"
```

目标说话人：先注册拿到 id，再用 `--enrollment-id` 传入（脚本会写进首帧 `parameter.asr_config.enrollment_id`）：

```bash
curl -X POST http://172.16.0.3:8082/api/asr/enrollment -F "audio=@speaker_enroll.wav"
# 也支持 speaker_enroll.mp3；raw PCM 请使用 16 kHz mono s16le。
# {"enrollment_id": "ule8QilVjZql30Q9oy9kiQ", "duration_sec": 3.0}

python docs/examples/ws_ast_v3.py sample.wav \
  --url ws://172.16.0.3:8082/tuling/ast/v3 \
  --enrollment-id "ule8QilVjZql30Q9oy9kiQ"
```

脚本在传入 `--enrollment-id` 时会自动写入 `enable_role_separation=false`、`enrollment_enable=true` 和 `enrollment_id`。

使用 `bash start.sh`（`wss://172.16.0.3:8443/...`）时可加 `--insecure` 跳过自签证书校验。

## Java SDK

可使用讯飞 `tuling-ast-sdk` 直接对接。响应字段名与本文档逐字对齐，可被 `ApiResponse` / `LatticeItem` / `ResultItem` / `ResultWordItem` 直接反序列化。客户端按 `header.status == 2` 判断会话结束、按 `header.code != 0` 判断错误。

## 相关文档

- [API 总览](../api-reference.md)
- [AST v3 多并发性能与极限压测报告](../benchmarks/tuling-ast-v3-benchmark.md)
- [通用流式 ASR WebSocket](transcribe-streaming-protocol.md)
- [分段情感识别 WebSocket](emotion-segmented-streaming-protocol.md)
