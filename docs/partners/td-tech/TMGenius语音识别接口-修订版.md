# TMGenius 语音识别接口（安菲翁评审修订版）

| 属性 | 值 |
| ---- | ---- |
| 文档版本 | V0.4-review |
| 创建日期 | 2026-07-03 |
| 修订日期 | 2026-07-07 |
| 状态 | 安菲翁评审修订建议 |
| 调用方 | CAgent |

---

## 说明

本文档基于对方《TMGenius 语音识别接口》V0.2 及鼎桥《开发指南-实时转写服务AST.3.4.4.1081-增加角色分离》参考文档，经安菲翁评审后整理为建议契约。接口分三个功能域：

1. **实时转写**：WebSocket 流式语音识别（AST v3 协议）。
2. **声纹管理**：目标说话人注册、查询、删除。
3. **热词管理**：按 `hotword_pool_id` 隔离的热词池查询、添加、指定删除、清空及重载。

CAgent 收到小乔端 ASR 请求后作为代理层调用本文档接口；小乔不直接访问 TMGenius。

---

## 接口概要

### 1. 实时转写

| 方法 | 路径 | 作用 | 请求体 / 参数 |
| ---- | ---- | ---- | ---- |
| WebSocket | `/tuling/ast/v3` | 流式实时语音转写（AST v3 协议） | `header`、`parameter.asr_config`、`payload.audio`、`payload.text.text` |

AST v3 是 WebSocket 长连接协议，非 HTTP REST。首帧 `header.status=0` 携带参数，中间帧 `status=1` 推音频，末帧 `status=2` 结束。

### 2. 声纹管理

| 方法 | 路径 | 作用 | 请求体 / 参数 |
| ---- | ---- | ---- | ---- |
| POST | `/api/asr/enrollment` | 注册目标说话人声纹，返回 `enrollment_id` | `multipart/form-data`：`audio` |
| GET | `/api/asr/enrollment/{enrollment_id}` | 查询声纹 ID 当前是否可用 | 路径参数 `enrollment_id` |
| DELETE | `/api/asr/enrollment/{enrollment_id}` | 删除指定声纹注册；未知 id 也返回 204 且无响应体 | 路径参数 `enrollment_id` |

### 3. 热词管理

接口前缀：`/api/asr/hotword-pool`

| 方法 | 路径 | 作用 | 请求体 / 参数 |
| ---- | ---- | ---- | ---- |
| GET | `/api/asr/hotword-pool` | 列出 / 检索指定池热词 | Query：`hotword_pool_id`、`query`、`limit`、`offset` |
| POST | `/api/asr/hotword-pool` | 向指定池增量添加热词 | JSON：`hotword_pool_id`、`hotwords` |
| DELETE | `/api/asr/hotword-pool` | 从指定池删除热词 | JSON：`hotword_pool_id`、`hotwords` |
| POST | `/api/asr/hotword-pool/delete` | 删除指定热词，兼容不稳定支持 DELETE body 的客户端 | JSON：`hotword_pool_id`、`hotwords` |
| POST | `/api/asr/hotword-pool/clear` | 清空指定热词池 | JSON 或 Query：`hotword_pool_id` |
| POST | `/api/asr/hotword-pool/reload` | 重载指定热词池 | JSON 或 Query：`hotword_pool_id` |

不使用也不要求支持 `POST /api/asr/hotword-pool/action`。

---

## 实时转写

### 1. 端点

```text
WebSocket ws(wss)://<host>:<port>/tuling/ast/v3
```

### 2. 请求帧

首帧携带会话参数、声纹参数、会话热词和第一段音频；中间帧继续推送音频；末帧表示结束。

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
      "language": "zh",
      "hotword_pool_id": "default",
      "enable_role_separation": false,
      "enrollment_enable": true,
      "enrollment_id": "enrollment_id_abc"
    }
  },
  "payload": {
    "audio": {
      "audio": "JiuY3iK9AAB..."
    },
    "text": {
      "text": "张维安,新华路派出所,狄志明"
    }
  }
}
```

### 3. 请求字段

| 名称 | 类型 | 必传 | 说明 |
| ---- | ---- | :--: | ---- |
| `header.traceId` | String | 是 | 日志追踪 ID |
| `header.appId` | String | 否 | 应用系统 ID |
| `header.bizId` | String | 是 | 业务 ID |
| `header.status` | Int | 是 | 流式状态：`0` 开始 / `1` 中间 / `2` 结束 |
| `parameter.asr_config.language` | String | 否 | 语种，建议使用 `zh` / `zh_en` 或双方确认的枚举 |
| `parameter.asr_config.hotword_pool_id` | String | 否 | 热词池 ID，缺省为 `default` |
| `parameter.asr_config.enable_role_separation` | Boolean | 否 | 角色分离开关，默认 `true`；省略等价于开启角色分离 |
| `parameter.asr_config.enrollment_enable` | Boolean | 否 | 是否启用声纹，默认 `false` |
| `parameter.asr_config.enrollment_id` | String | 否 | 主讲人声纹 ID，由 `POST /api/asr/enrollment` 返回 |
| `payload.audio.audio` | String | 是 | base64 编码音频数据 |
| `payload.text.text` | String | 否 | 会话热词，逗号分隔，仅当前连接生效，不写入热词池 |

### 4. 角色分离与声纹优先级

角色分离开关继承鼎桥 AST 3.4.4.1081 / 讯飞现有实现语义，优先级高于声纹注册：

- `parameter.asr_config.enable_role_separation` 默认 `true`；客户端省略该字段时等价于 `true`。
- `enable_role_separation=true` 时，服务端优先执行角色分离；即使同时传入 `enrollment_enable=true` 和 `enrollment_id`，也不启用声纹目标说话人过滤。
- `enable_role_separation=false` 时，服务端不执行角色分离，`sentence` 和 `Progressive` 结果均不返回 `cw[].rl` 字段。
- `cw[].rl` 仅在 `enable_role_separation=true` 且 `msgtype=sentence` 的最终结果中返回；`msgtype=Progressive` 的中间结果不返回该字段。
- `sentence` 相对上一条发生角色变化时返回会话内稳定编号 `1..4`；同一角色连续发言返回 `0`；切回旧角色再次返回其原编号。sidecar 故障降级时返回 `0`。
- `cw[].rl` 只表示角色分离编号，不表示声纹是否命中或启用。

角色分离与声纹组合行为矩阵：

| `enable_role_separation` | `enrollment_enable` | `enrollment_id` | 实际模式 | `cw[].rl` 行为 | 声纹状态 | 是否报错 |
|---|---|---|---|---|---|---|
| 省略 | 省略 / `false` | 省略 | 角色分离 | `sentence` 返回角色变化标记 `0..4`；`Progressive` 不返回 | `enrollment_applied=false` | 否 |
| 省略 | `true` | 有效 ID | 角色分离优先，忽略声纹 | `sentence` 返回角色变化标记 `0..4`；`Progressive` 不返回 | `enrollment_applied=false` | 否 |
| 省略 | `true` | 空 / 无效 ID | 角色分离优先，忽略声纹参数 | `sentence` 返回角色变化标记 `0..4`；`Progressive` 不返回 | `enrollment_applied=false` | 否 |
| `true` | 省略 / `false` | 省略 | 角色分离 | `sentence` 返回角色变化标记 `0..4`；`Progressive` 不返回 | `enrollment_applied=false` | 否 |
| `true` | `true` | 有效 ID | 角色分离优先，忽略声纹 | `sentence` 返回角色变化标记 `0..4`；`Progressive` 不返回 | `enrollment_applied=false` | 否 |
| `true` | `true` | 空 / 无效 ID | 角色分离优先，忽略声纹参数 | `sentence` 返回角色变化标记 `0..4`；`Progressive` 不返回 | `enrollment_applied=false` | 否 |
| `false` | 省略 / `false` | 省略 / 任意 | 普通 ASR | 不返回 `cw[].rl` | `enrollment_applied=false` | 否 |
| `false` | `true` | 有效 ID | 声纹目标说话人识别 | 不返回 `cw[].rl` | `enrollment_applied=true` | 否 |
| `false` | `true` | 空 ID | 参数错误 | 无识别结果 | `enrollment_applied=false` | 是 |
| `false` | `true` | 无效 / 过期 ID | 回退普通 ASR | 不返回 `cw[].rl` | `enrollment_applied=false` | 否 |

### 5. 声纹启用规则

- `header.resIdList[0]` 直接废弃，不作为兼容字段继续使用。
- `enrollment_enable` 默认 `false`。
- 声纹仅在 `enable_role_separation=false` 时参与识别；角色分离开启或省略时，声纹参数被忽略。
- `enable_role_separation=false` 且 `enrollment_enable=false` 时，即使传入 `enrollment_id`，也不启用声纹。
- `enable_role_separation=false` 且 `enrollment_enable=true` 且 `enrollment_id` 非空时启用声纹。
- `enable_role_separation=false` 且 `enrollment_enable=true` 但 `enrollment_id` 为空时，返回参数错误，不进入普通 ASR。
- 服务端在 AST v3 识别结果中返回 `enrollment_applied`，表示本条结果是否实际携带可用声纹材料进入 ASR。
- `cw[].rl` 只用于角色分离编号，不表示声纹是否命中或启用。

### 6. 音频格式

`payload.audio.audio` 中 base64 的原始内容建议统一为：

- 16 kHz。
- mono。
- s16le PCM。

如服务端允许首帧携带 WAV header，需要在文档中单独标明；否则按 raw PCM 对接。

### 7. 语种字段

- CAgent 推荐使用 `parameter.asr_config.language` 表示语种。
- `parameter.engine.wdec_param_LanguageTypeChoice` 不作为推荐接入字段。
- 如需保留 `parameter.engine.wdec_param_LanguageTypeChoice`，双方需明确是否实际生效、取值映射和优先级。

### 8. 响应

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
    "result": {
      "segId": 0,
      "bg": 140,
      "ed": 3230,
      "ls": false,
      "msgtype": "sentence",
      "sn": 1,
      "enrollment_applied": true,
      "ws": [
        {
          "bg": 17,
          "cw": [
            {
              "lg": "zh",
              "rl": 0,
              "w": "你好",
              "wp": "n",
              "wb": 17,
              "we": 56
            }
          ]
        }
      ]
    }
  }
}
```

| 名称 | 类型 | 说明 |
| ---- | ---- | ---- |
| `header.code` | Int | 错误码，`0` 为成功 |
| `header.message` | String | 描述信息 |
| `header.sid` | String | 本次会话唯一标识 |
| `header.traceId` | String | 日志追踪 ID |
| `header.status` | Int | 识别状态：`0` 开始 / `1` 识别中 / `2` 结束 |
| `payload.result.msgtype` | String | `sentence` 最终结果 / `Progressive` 中间结果 |
| `payload.result.enrollment_applied` | Boolean | 声纹是否实际携带可用材料进入 ASR；声纹未启用、ID 不可用或回退普通 ASR 时为 `false` |
| `payload.result.enrollment_fallback_reason` | String | 可选；声纹未生效原因，如 `disabled` / `not_found` / `incompatible` / `upstream_unavailable` |
| `payload.result.bg` / `ed` | Int | 句子开始 / 结束时间，单位 ms |
| `payload.result.ws` | Array | 词语列表 |
| `payload.result.ws[].cw[].rl` | Int | 角色变化标记：变化时为稳定编号 `1..4`，连续同角色或 sidecar 降级时为 `0`；仅 sentence 返回，`Progressive` 不返回；`enable_role_separation=false` 时不返回 |

声纹生效状态说明：AST v3 使用 `enrollment_applied` 确认本次识别是否实际携带可用声纹材料；REST 上传响应使用 `enrollment_used`。`cw[].rl` 只用于角色分离编号，不表示声纹状态。角色分离优先级高于声纹注册，角色分离开启或省略时，`enrollment_applied=false`。

---

## 声纹管理

### 1. 注册目标说话人

- **URL**：`POST /api/asr/enrollment`
- **Content-Type**：`multipart/form-data`
- **成功状态码**：`200 OK`

| 字段 | 类型 | 必填 | 说明 |
| ---- | ---- | :--: | ---- |
| `audio` | File | 是 | WAV、MP3 或 raw PCM；服务端解码为 16 kHz mono |

响应示例：

```json
{
  "enrollment_id": "ule8QilVjZql30Q9oy9kiQ",
  "duration_sec": 3.0
}
```

### 2. 注册错误码

| HTTP 状态码 | `detail.code` | 说明 |
| ---- | ---- | ---- |
| 400 | `too_short` | 音频短于 1.0 秒 |
| 400 | `empty` | 上传体为空或解码后无音频 |
| 400 | `unsupported_format` | 上传格式不是 WAV、MP3 或 16 kHz mono s16le PCM |
| 400 | `decode_failed` | 音频损坏或解码失败 |
| 413 | - | 上传文件超过服务端大小限制 |
| 422 | - | multipart 字段类型或必填字段不符合校验 |

### 3. 删除声纹注册

- **URL**：`DELETE /api/asr/enrollment/{enrollment_id}`

| HTTP 状态码 | 说明 |
| ---- | ---- |
| 204 | 删除成功，未知 ID 也返回 204；无响应体 |

### 4. 查询声纹状态

- **URL**：`GET /api/asr/enrollment/{enrollment_id}`

用于让 CAgent 在只保存 `enrollment_id` 的情况下，判断该 ID 当前是否还能直接用于声纹 ASR。接口不返回原始注册音频、PCM、embedding 或其他声纹敏感材料。

该接口也可用于查询已生效声纹、定位数据不一致：如果 CAgent 保存了某个 `enrollment_id`，但查询返回 `available=false`，说明该 ID 在当前服务端不可用，可能是未注册、已删除、落盘文件缺失、embedding 与当前模型/adapter 不兼容，或路由到没有共享同一 RAG-ASR 本地存储的实例导致。若不同实例或管理服务对同一 `enrollment_id` 返回不同 `available`，可以直接暴露实例间存储或路由不一致问题。

需要注意：该接口只负责诊断和暴露状态，不负责同步声纹数据，也不保证查询后实际 ASR 一定使用成功。最终仍需以 AST v3 响应中的 `enrollment_applied` 或 REST 响应中的 `enrollment_used` 确认本次识别是否实际应用了声纹。

响应示例：

```json
{
  "enrollment_id": "ule8QilVjZql30Q9oy9kiQ",
  "available": true,
  "reason": "ok"
}
```

字段说明：

| 字段 | 类型 | 说明 |
| ---- | ---- | ---- |
| `enrollment_id` | String | 本次查询的声纹 ID |
| `available` | Boolean | 是否可直接用于后续 ASR；CAgent 以该字段作为是否需要重新注册的判断依据 |
| `reason` | String | 状态原因；`available=true` 时为 `ok` |

`available=false` 的常见 `reason`：

| reason | 说明 |
| ---- | ---- |
| `not_found` | 服务端找不到该 ID |
| `incompatible` | 落盘 embedding 与当前模型、adapter 或 projector 维度不兼容 |
| `deleted` | 已显式删除 |
| `upstream_unavailable` | 外部 enrollment 管理服务不可用 |

RAG-ASR 管理服务模式下，查询结果以管理服务为准；RAG-ASR 查询/使用会更新元数据 `last_used_at`，但当前不做 TTL 自动过期。若仍使用 demo 进程内 fallback 缓存，则该缓存会受 TTL、重启和 LRU 容量限制。

### 5. 生命周期

| 项目 | 行为 |
| ---- | ---- |
| 存储 | RAG-ASR 管理服务将 projector frames tensor 与 JSON 元数据落盘到 `enrollment_store_dir/<enrollment_scope_id>/<enrollment_id>.pt/.json`，默认目录为 RAG-ASR 的 `var/enrollments`；不保存原始注册音频 |
| 有效期 | RAG-ASR 落盘存储当前不做 TTL 自动过期；查询/使用会更新元数据 `last_used_at`（受 RAG-ASR `enrollment_metadata_touch_interval_sec` 节流） |
| 重启 | RAG-ASR 管理服务重启后仍可从本地落盘文件读取；前提是路由到同一持久化目录或共享存储 |
| 容量 | RAG-ASR `enrollment_cache_max_entries` 只限制内存热缓存，不删除磁盘文件 |
| 删除 | 删除接口会删除对应 `.pt` 与 `.json` 文件；未知 ID 也可按幂等成功处理 |
| 失效 | 文件缺失、显式删除或 embedding 与当前模型/adapter 不兼容时，AST v3 响应应体现 `enrollment_applied=false`，REST 上传响应应体现 `enrollment_used=false`，避免只依赖日志判断 |

若 TMGenius / RAG-ASR 多实例部署，需要说明是否使用共享持久化目录或会话粘滞；否则同一 `enrollment_id` 可能在不同实例不可用。

---

## 热词管理

### 1. 热词池作用域

- 热词池按 `hotword_pool_id` 隔离。
- 缺省热词池为 `default`。
- 查询、添加、删除、`POST /delete`、清空、reload 均只作用于指定热词池。
- 会话热词仅当前 WebSocket 连接生效，不写入热词池。
- 会话热词与热词池同时存在时，客户端会话热词优先级高于热词池召回词。
- 当两者存在同音、近音或语义冲突时，优先采用客户端显式传入的会话热词。例如客户端传入“王惠”时，应优先于热词库中的“王慧”。

### 2. 查询热词

- **URL**：`GET /api/asr/hotword-pool`

| 参数 | 类型 | 必填 | 说明 |
| ---- | ---- | :--: | ---- |
| `hotword_pool_id` | String | 否 | 热词池 ID，缺省 `default` |
| `query` | String | 否 | 子串匹配；非 ASCII 需 URL encode |
| `limit` | Int | 否 | 每页条数，上限 1000，默认 100 |
| `offset` | Int | 否 | 分页偏移，默认 0 |

### 3. 添加热词

- **URL**：`POST /api/asr/hotword-pool`
- **Content-Type**：`application/json`

```json
{
  "hotword_pool_id": "default",
  "hotwords": ["张维安", "新华路派出所", "狄志明"]
}
```

响应应体现实际生效情况：

```json
{
  "action": "add",
  "status": "ok",
  "hotwords": ["张维安", "新华路派出所"],
  "added_count": 2,
  "duplicate_count": 1,
  "invalid_count": 0,
  "ignored_hotwords": ["狄志明"],
  "total_count": 150
}
```

### 4. 删除热词

提供两个等价路径：

- `DELETE /api/asr/hotword-pool`
- `POST /api/asr/hotword-pool/delete`

```json
{
  "hotword_pool_id": "default",
  "hotwords": ["张维安", "狄志明"]
}
```

响应应体现实际生效情况：

```json
{
  "action": "delete",
  "status": "ok",
  "hotwords": ["张维安"],
  "deleted_count": 1,
  "missing_count": 1,
  "missing_hotwords": ["狄志明"],
  "total_count": 149
}
```

### 5. 清空热词池

- **URL**：`POST /api/asr/hotword-pool/clear`
- **Content-Type**：`application/json`

```json
{
  "hotword_pool_id": "default"
}
```

也可支持 query：

```text
POST /api/asr/hotword-pool/clear?hotword_pool_id=default
```

若 query 和 body 同时存在且不一致，应返回参数错误。清空只作用于指定 `hotword_pool_id`，会删除该池当前所有热词并刷新运行时词池与嵌入缓存；不影响其他热词池。`DELETE /api/asr/hotword-pool` 和 `POST /api/asr/hotword-pool/delete` 仍只表示删除请求体中指定的 `hotwords`，空数组不得解释为清空。

响应示例：

```json
{
  "action": "clear",
  "status": "ok",
  "hotwords": [],
  "cleared": 149,
  "total_count": 0
}
```

### 6. 重载热词池

- **URL**：`POST /api/asr/hotword-pool/reload`

建议支持 JSON body：

```json
{
  "hotword_pool_id": "default"
}
```

也可支持 query：

```text
POST /api/asr/hotword-pool/reload?hotword_pool_id=default
```

若 query 和 body 同时存在且不一致，应返回参数错误。reload 只作用于指定 `hotword_pool_id`，不应默认影响所有热词池。

大批量导入和 reload 使用的文件或管理服务存储也必须按 `hotword_pool_id` 隔离，不能让多个池共享同一个 `hotword_pool.txt`。

### 7. 热词校验规则

| 项目 | 规则 |
| ---- | ---- |
| 规范化 | 去首尾空白，内部连续空白压成单个空格；保留原始大小写与书面形态 |
| 长度 | 中文按字：有效 2-32 字；拉丁按词：有效 1-32 词；越界丢弃 |
| 去重 | 中文按原样去重；非中文 casefold 后去重 |

非法词、重复词、不存在词可以不报错，但响应中必须体现统计或明细，便于 CAgent 和后台页面展示真实生效情况。

热词添加 / 删除响应以本节列出的 `*_count`、`ignored_hotwords`、`missing_hotwords`
字段为唯一对外契约；上游内部兼容字段如 `added`、`skipped_duplicates`、`duplicates`、
`deleted`、`missing`、`invalid` 不再透出给 CAgent。

---

## 错误语义、鉴权和审计

### 1. REST 错误

| 状态码 | 含义 |
| ---- | ---- |
| 200 | 成功 |
| 204 | 删除成功；无响应体 |
| 400 | 请求字段缺失、参数冲突、音频为空、无法解码、注册音频时长校验失败 |
| 413 | 上传文件超过服务端大小限制 |
| 422 | multipart 字段类型或必填字段不符合校验 |
| 502 | 后端模型推理或管理服务调用失败 |

### 2. WebSocket 错误

参数错误应明确返回错误。AST v3 中 `enable_role_separation=false && enrollment_enable=true` 但缺少 `enrollment_id` 会返回 error 并结束本次会话。

角色分离例外：`parameter.asr_config.enable_role_separation=true` 或省略时不视为参数错误；该模式优先执行角色分离，并忽略声纹参数校验。只有在 `enable_role_separation=false` 且 `enrollment_enable=true` 但 `enrollment_id` 为空时，才因声纹参数缺失返回参数错误。

示例：

```json
{
  "header": {
    "code": 1001,
    "message": "model inference failed",
    "sid": "AST_MKMZO0WX2SLZ4"
  }
}
```

声纹不可用导致回退普通 ASR 时，应在响应字段中体现 `enrollment_applied=false`，并尽量返回 `enrollment_fallback_reason`。

### 3. 鉴权和审计

声纹属于敏感生物特征，热词管理会影响识别结果。管理类接口需要补充：

- 服务间鉴权机制。
- 操作审计日志。
- `traceId` / `requestId` 贯穿。
- 记录调用方、操作类型、`hotword_pool_id`、`enrollment_id` 和操作结果。

---

## 变更记录

| 版本 | 日期 | 说明 |
| ---- | ---- | ---- |
| V0.5-review | 2026-08-04 | 接入 Streaming Sortformer 实时角色分离；明确 `rl=0..4`、speaker turn 重切与 sidecar fail-open 语义 |
| V0.4-review | 2026-07-07 | 确认角色分离默认开启；`enable_role_separation` 优先级高于 `enrollment_enable`；关闭角色分离时不返回 `cw[].rl` |
| V0.3-review | 2026-07-06 | 增加鼎桥 AST 3.4.4.1081 角色分离字段兼容口径：接受 `enable_role_separation`，`sentence` 返回 `cw[].rl=0`，`Progressive` 不返回 `cw[].rl` |
| V0.2-review | 2026-07-04 | 补充声纹管理、热词池清空和重载、错误语义等评审建议 |
