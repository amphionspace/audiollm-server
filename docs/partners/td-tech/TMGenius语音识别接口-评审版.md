# TMGenius 语音识别接口

> 本文档为历史评审材料，不再作为当前实现契约。角色分离、声纹与 sidecar 降级的完整行为矩阵以 [TMGenius 语音识别接口（修订版）](TMGenius语音识别接口-修订版.md#4-角色分离与声纹优先级) 和 [AST v3 WebSocket 协议](../../protocols/tuling-ast-v3-protocol.md#角色分离与声纹行为矩阵) 为准。

| 属性     | 值                                                                                                                                                                               |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 文档版本 | V0.2                                                                                                                                                                             |
| 创建日期 | 2026-07-03                                                                                                                                                                       |
| 状态     | 评审中                                                                                                                                                                           |
| 调用方   | CAgent                                                                                                                                                                           |
| 关联设计 | 外部资料：`小乔语音识别-热词方案设计.md`、`小乔语音识别第二期-方案概述.md` |
| 原始参考 | 外部归档：`三方接口文档归档/安菲翁接口/api-reference.md`、`ASR热词库使用手册` |

***

## 说明

本文档定义 **TMGenius** 对外暴露给 **CAgent** 的接口，分三个功能域：

1. **实时转写**：WebSocket 流式语音识别（AST v3 协议）
2. **声纹管理**：目标说话人注册、删除
3. **热词管理**：按 `hotword_pool_id` 隔离的热词池查询、添加、指定删除、清空及重载

CAgent 收到小乔端的 ASR 请求后作为代理层调用本文档接口；小乔不直接访问 TMGenius。

---

## 接口概要

### 1. 实时转写

| 方法      | 路径             | 作用                            | 请求体 / 参数                                       |
| --------- | ---------------- | ------------------------------- | --------------------------------------------------- |
| WebSocket | `/tuling/ast/v3` | 流式实时语音转写（AST v3 协议） | `header`、`parameter.asr_config`、`payload.audio`、`payload.text.text`（会话热词） |

> AST v3 是 WebSocket 长连接协议，非 HTTP REST。首帧 `header.status=0` 携带参数（含 `parameter.asr_config` 中的声纹启用矩阵、会话热词 `payload.text.text`），中间帧 `status=1` 推音频，末帧 `status=2` 结束。

---

### 2. 声纹管理

> 来源：外部归档 `三方接口文档归档/安菲翁接口/api-reference.md` — REST 上传接口 / 目标说话人注册

| 方法   | 路径                               | 作用                                         | 请求体 / 参数                                   |
| ------ | ---------------------------------- | -------------------------------------------- | ----------------------------------------------- |
| POST   | `/api/asr/enrollment`              | 注册目标说话人声纹，返回 `enrollment_id`     | `multipart/form-data`：`audio`（WAV/MP3/PCM，1~8 秒） |
| DELETE | `/api/asr/enrollment/{enrollment_id}` | 删除指定声纹注册；未知 id 也返回 204 且无响应体 | 路径参数 `enrollment_id`                        |

**声纹生命周期关键约束**：

| 项目   | 行为                                                                 |
| ------ | -------------------------------------------------------------------- |
| 存储   | RAG-ASR 管理服务将 projector frames tensor 与 JSON 元数据落盘到 `enrollment_store_dir/<scope>/<enrollment_id>.pt/.json`，默认 `var/enrollments`；不保存原始注册音频 |
| 有效期 | RAG-ASR 落盘存储当前不做 TTL 自动过期；查询/使用会更新 `last_used_at` |
| 容量   | RAG-ASR `enrollment_cache_max_entries` 只限制内存热缓存，不删除磁盘文件 |
| 失效   | 文件缺失、显式删除或 embedding 与当前模型/adapter 不兼容后，服务端回退为普通 ASR；AST v3 结果返回 `enrollment_applied=false`，并尽量返回 `enrollment_fallback_reason` |

> **CAgent 职责**：CAgent 需维护 `userId → enrollmentId` 映射；RAG-ASR 本地落盘存在时，TMGenius/RAG-ASR 重启后通常无需重新注册。若路由到未共享同一 `enrollment_store_dir` 的实例、文件被删除或模型/adapter 不兼容，则需重新注册。

---

### 3. 热词管理

> 来源：[`ASR热词库使用手册`](file:///D:/WelinkFile/ReceiveFiles/ASR热词库使用手册.md) — 三、管理接口（REST）

接口前缀：`/api/asr/hotword-pool`

热词添加 / 删除响应以 `*_count` 统计字段为唯一对外契约，不透出上游内部兼容字段 `added`、`skipped_duplicates`、`duplicates`、`deleted`、`missing`、`invalid`。`status="ok"` 为成功；`message` 为可读说明。

| 方法   | 路径                             | 作用                                             | 请求体 / 参数                                                           |
| ------ | -------------------------------- | ------------------------------------------------ | ----------------------------------------------------------------------- |
| GET    | `/api/asr/hotword-pool`          | 列出 / 检索热词（支持分页）                      | Query：`query`（子串匹配，非 ASCII 需 URL-encode）、`limit`（≤1000）、`offset` |
| POST   | `/api/asr/hotword-pool`          | 增量添加热词（重复 / 非法自动过滤，不报错）      | `{"hotwords": [...]}`                                                   |
| DELETE | `/api/asr/hotword-pool`          | 删除指定热词                                     | `{"hotwords": [...]}`                                                   |
| POST   | `/api/asr/hotword-pool/delete`   | 删除指定热词（兼容不支持 body 的 DELETE 场景）   | `{"hotwords": [...]}`                                                   |
| POST   | `/api/asr/hotword-pool/clear`    | 清空指定热词池                                   | JSON 或 Query：`hotword_pool_id`                                        |
| POST   | `/api/asr/hotword-pool/reload`   | 从指定热词池文件重载并重建嵌入缓存（无需重启服务） | JSON 或 Query：`hotword_pool_id` |

**热词格式与校验规则**（服务端自动过滤，非法词静默丢弃不报错）：

| 项目   | 规则                                                                              |
| ------ | --------------------------------------------------------------------------------- |
| 规范化 | 去首尾空白，内部连续空白压成单个空格；保留原始大小写与书面形态                    |
| 长度   | 中文按字计：有效 2~32 字；拉丁按词计：有效 1~32 词；越界直接丢弃                  |
| 去重   | 中文按原样去重；非中文 casefold 后去重（`IBM` 与 `ibm` 视为同一条）               |

**两类热词的关系**（需在接口设计中明确调用路径）：

| 类型 | 作用范围 | 传入方式 |
| ---- | -------- | -------- |
| 按 `hotword_pool_id` 隔离的热词池（本接口） | 使用同一热词池 ID 的请求生效 | 管理员通过 CAgent 配置后台写入 |
| 会话热词 | 仅当前连接、首帧生效 | AST v3 首帧 `payload.text.text`（逗号分隔） |

> 两者同时存在时，客户端显式传入的会话热词优先进入 ASR prompt，并覆盖精确重复或整词同音（忽略声调）的热词池召回词。

**大批量装载方式**（10 万级）：

| 方式 | 说明 |
| ---- | ---- |
| 方式 A（推荐） | `docker cp` 覆盖容器内 `hotword_pool.txt`，再调用 `POST .../reload` 触发重载；reload 已有缓存时约 15 秒 |
| 方式 B | 分批 POST（每批 1000~5000 条）增量写入，适合无容器操作权限时使用 |

---

---

## 详细设计

### 1. 实时转写

#### 1.1 端点

```
WebSocket  ws(wss)://<host>:<port>/tuling/ast/v3
```

#### 1.2 请求

首帧（`header.status = 0`）携带全部参数，中间帧（`status = 1`）持续推送音频，末帧（`status = 2`）结束会话。

**请求帧示例**

```json
{
    "header": {
        "traceId": "traceId123456",
        "appId": "123456",
        "bizId": "39769795890",
        "status": 0
    },
    "parameter": {
        "engine": {
            "wdec_param_LanguageTypeChoice": "1"
        },
        "asr_config": {  //  基于讯飞
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

**参数说明**

| 名称 | 类型 | 必传 | 说明 |
| :--- | :--- | :---: | :--- |
| header | Object | 是 | 服务相关参数 |
| header.traceId | String | 是 | 日志追踪 ID |
| header.appId | String | 否 | 应用系统 ID |
| header.bizId | String | 是 | 业务 ID |
| header.status | int | 是 | 流式状态：`0` 开始 / `1` 中间 / `2` 结束 |
| header.resIdList | List\<String\> | 否 | 已废弃；服务端仅记录并忽略，不作为声纹字段 |
| parameter.engine | Map | 否 | 引擎透传参数 |
| parameter.engine.wdec_param_LanguageTypeChoice | String | 否 | 语种：`"1"` 中文 / `"3"` 中英混合 |
| parameter.asr_config.enable_role_separation | Boolean | 否 | 角色分离开关，默认 `true`；开启时优先角色分离并忽略声纹参数 |
| parameter.asr_config.enrollment_enable | Boolean | 否 | 是否显式启用主讲人声纹门控，默认 `false` |
| parameter.asr_config.enrollment_id | String | 否 | 主讲人声纹 ID；仅当 `enable_role_separation=false && enrollment_enable=true` 时使用 |
| payload.audio.audio | String | 是 | base64 编码的音频数据；每帧建议 4096 字节，不超过 16 KB，至少覆盖 40ms 语音 |
| payload.text.text | String | 否 | 会话热词，逗号分隔；仅对当前连接的 VAD 最终句生效；与按 `hotword_pool_id` 隔离的热词池独立 |

#### 1.3 响应

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
            "enrollment_applied": false,
            "enrollment_fallback_reason": "disabled",
            "ws": [
                {
                    "bg": 17,
                    "cw": [
                        {
                            "lg": "zh",
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

**响应字段说明**

| 名称 | 类型 | 说明 |
| :--- | :--- | :--- |
| header.code | Int | 错误码，`0` 为成功 |
| header.message | String | 描述信息 |
| header.sid | String | 本次会话唯一标识 |
| header.traceId | String | 日志追踪 ID |
| header.status | Int | 识别状态：`0` 开始 / `1` 识别中 / `2` 结束 |
| payload.result.msgtype | String | `sentence` 最终结果 / `Progressive` 中间结果 |
| payload.result.bg | Int | 句子开始时间（ms） |
| payload.result.ed | Int | 句子结束时间（ms） |
| payload.result.ls | Bool | 是否最后一段结果 |
| payload.result.sn | Int | 结果序号 |
| payload.result.enrollment_applied | Bool | 本条识别是否实际携带可用声纹材料进入 ASR；默认 false |
| payload.result.enrollment_fallback_reason | String | 可选；声纹未生效原因，如 `disabled` / `not_found` / `incompatible` / `upstream_unavailable` |
| payload.result.ws | ResultItem[] | 词语列表 |
| payload.result.ws.bg | Int | 词语开始时间（单位 10ms） |
| payload.result.ws.cw | ResultWordItem[] | 词语识别结果 |
| cw.lg | String | 语种 |
| cw.w | String | 词内容 |
| cw.wp | String | 词性：`n` 普通 / `s` 顺滑词 / `p` 标点 / `g` 语义分段 |
| cw.wb / cw.we | Int | 词起止位置（帧数，×10 为毫秒） |

---

### 2. 声纹管理

#### 2.1 注册目标说话人

- **URL**：`POST /api/asr/enrollment`
- **Content-Type**：`multipart/form-data`
- **成功状态码**：`200 OK`

**请求字段**

| 字段 | 类型 | 必填 | 说明 |
| ---- | ---- | :--: | ---- |
| audio | File | 是 | WAV 、MP3、PCM文件；服务端解码为 16 kHz mono。短于 1.0 秒返回 400；长于 8.0 秒尾截不拒绝 |

**响应示例**

```json
{
  "enrollment_id": "ule8QilVjZql30Q9oy9kiQ",
  "duration_sec": 3.0
}
```

**响应字段**

| 字段 | 类型 | 说明 |
| ---- | ---- | ---- |
| enrollment_id | String | 声纹 ID；AST v3 后续注入 `parameter.asr_config.enrollment_id`，不再使用 `header.resIdList[0]` |
| duration_sec | Float | 实际注册音频时长（秒） |

**错误响应**

```json
{
  "detail": {
    "code": "too_short",
    "message": "enrollment audio is 0.30s, need at least 1.00s"
  }
}
```

| HTTP 状态码 | detail.code | 说明 |
| ----------- | ----------- | ---- |
| 400 | `too_short` | 音频短于 1.0 秒 |
| 400 | `empty` | 上传体为空或解码后无音频 |
| 400 | `decode_failed` | 音频损坏或解码失败 |

***

#### 2.2 删除声纹注册

- **URL**：`DELETE /api/asr/enrollment/{enrollment_id}`

**路径参数**

| 字段 | 类型 | 必填 | 说明 |
| ---- | ---- | :--: | ---- |
| enrollment_id | String | 是 | 待删除的声纹 ID；未知 ID 同样返回 204 且无响应体，可安全重试 |

**响应**

| HTTP 状态码 | 说明 |
| ----------- | ---- |
| 204 | 删除成功（含未知 ID）；无响应体 |

> `enrollment_id` 不可用（未注册、已删除、落盘文件缺失或 embedding 与当前模型/adapter 不兼容）后再被 AST v3 使用时，服务端回退为普通 ASR，并在结果中返回 `enrollment_applied=false` 与可用时的 `enrollment_fallback_reason`。CAgent 应据此重新调用注册接口。

---

### 3. 热词管理

接口前缀：`/api/asr/hotword-pool`

**响应字段**

```json
{
  "action": "add",
  "status": "ok",
  "message": "added 2 hotwords",
  "hotwords": ["张维安", "新华路派出所"],
  "added_count": 2,
  "duplicate_count": 0,
  "invalid_count": 0,
  "ignored_hotwords": [],
  "total_count": 150
}
```

| 字段 | 类型 | 说明 |
| ---- | ---- | ---- |
| action | String | 操作类型：`add` / `delete` / `list` / `reload` |
| status | String | `ok` 成功；非 ok 为失败 |
| message | String | 文字说明，如 `added N hotwords` / `deleted N hotwords` / `reloaded hotword pool: X -> Y` |
| hotwords | Array | 本次操作涉及的热词列表 |
| added_count / deleted_count | Int | 添加 / 删除实际生效的热词数 |
| duplicate_count / invalid_count | Int | 添加请求中的重复词 / 非法词数量 |
| missing_count | Int | 删除请求中不存在的热词数量 |
| ignored_hotwords / missing_hotwords | Array | 添加时被忽略的非法词；删除时未命中的词 |
| cleared / reloaded | Int | 清空或 reload 实际影响的热词数 |
| matched_count | Int | 查询命中的热词数 |
| total_count | Int | 操作后热词池内总数 |

***

#### 3.1 列出 / 检索热词

- **URL**：`GET /api/asr/hotword-pool`

**Query 参数**

| 参数 | 类型 | 必填 | 说明 |
| ---- | ---- | :--: | ---- |
| query | String | 否 | 子串匹配；非 ASCII 需 URL-encode；不传则返回全部 |
| limit | Int | 否 | 每页条数，上限 1000；默认 100 |
| offset | Int | 否 | 分页偏移；默认 0 |

**响应示例**

```json
{
  "action": "list",
  "status": "ok",
  "message": "returned 2 hotwords",
  "hotwords": ["张维安", "新华路派出所"],
  "matched_count": 2,
  "total_count": 150
}
```

***

#### 3.2 增量添加热词

- **URL**：`POST /api/asr/hotword-pool`
- **Content-Type**：`application/json`

**请求 Body**

```json
{
  "hotwords": ["张维安", "新华路派出所", "狄志明"]
}
```

| 字段 | 类型 | 必填 | 说明 |
| ---- | ---- | :--: | ---- |
| hotwords | Array\<String\> | 是 | 待添加热词列表；非法词 / 重复词自动过滤，不报错 |

**热词校验规则**（服务端自动执行）

| 项目 | 规则 |
| ---- | ---- |
| 规范化 | 去首尾空白，内部连续空白压成单个空格；保留原始大小写与书面形态 |
| 长度 | 中文按字：有效 2~32 字；拉丁按词：有效 1~32 词；越界丢弃 |
| 去重 | 中文按原样去重；非中文 casefold 后去重（`IBM` 与 `ibm` 同一条） |

**响应示例**

```json
{
  "action": "add",
  "status": "ok",
  "message": "added 2 hotwords",
  "hotwords": ["张维安", "新华路派出所", "狄志明"],
  "added_count": 2,
  "duplicate_count": 1,
  "invalid_count": 0,
  "ignored_hotwords": [],
  "total_count": 153
}
```

***

#### 3.3 删除热词

提供两个等价路径，语义完全相同，兼容不同 HTTP 客户端：

**路径 A**

- **URL**：`DELETE /api/asr/hotword-pool`
- **Content-Type**：`application/json`

**路径 B（兼容不支持带 body 的 DELETE）**

- **URL**：`POST /api/asr/hotword-pool/delete`
- **Content-Type**：`application/json`

**请求 Body**

```json
{
  "hotword_pool_id": "default",
  "hotwords": ["张维安", "狄志明"]
}
```

| 字段 | 类型 | 必填 | 说明 |
| ---- | ---- | :--: | ---- |
| hotword_pool_id | String | 否 | 热词池 ID，缺省 `default` |
| hotwords | Array\<String\> | 是 | 待删除热词列表；不在池中的词忽略，不报错 |

> `DELETE /api/asr/hotword-pool` 和 `POST /api/asr/hotword-pool/delete` 只删除请求体中指定的 `hotwords`；空数组不得解释为清空整池。整池清空必须使用 `POST /api/asr/hotword-pool/clear`。

**响应示例**

```json
{
  "action": "delete",
  "status": "ok",
  "message": "deleted 1 hotwords",
  "hotwords": ["张维安", "狄志明"],
  "deleted_count": 1,
  "missing_count": 1,
  "missing_hotwords": ["狄志明"],
  "total_count": 151
}
```

***

#### 3.4 清空热词池

清空指定 `hotword_pool_id` 的运行时热词池并刷新嵌入缓存；不会影响其他热词池。

- **URL**：`POST /api/asr/hotword-pool/clear`
- **Content-Type**：`application/json`

**请求 Body**

```json
{
  "hotword_pool_id": "default"
}
```

也可使用 query：

```text
POST /api/asr/hotword-pool/clear?hotword_pool_id=default
```

若 query 和 body 同时存在且不一致，应返回参数错误。

**响应示例**

```json
{
  "action": "clear",
  "status": "ok",
  "message": "cleared 151 hotwords",
  "hotwords": [],
  "cleared": 151,
  "total_count": 0
}
```

***

#### 3.5 重载热词池

从指定 `hotword_pool_id` 对应的池文件重建内存池并刷新嵌入缓存，无需重启服务。适用于通过 `docker cp` 写入大批量词表后触发热加载，或运维手动刷新。

- **URL**：`POST /api/asr/hotword-pool/reload`

**请求参数**

支持 Query 或 JSON body 指定 `hotword_pool_id`；缺省为 `default`。若 Query 和 Body 同时存在且不一致，返回参数错误。

```json
{
  "hotword_pool_id": "tenant-a"
}
```

**响应示例**

```json
{
  "action": "reload",
  "status": "ok",
  "message": "reloaded hotword pool: 0 -> 100000",
  "hotwords": [],
  "reloaded": 100000,
  "total_count": 100000
}
```

> **性能参考**（10 万级池，交付镜像实测）：reload 有缓存时约 15 秒；首次冷装载无缓存时明显更慢，建议安排在部署预热或维护窗口执行。

***

#### 3.6 不提供统一 action 入口

本版本不提供 `POST /api/asr/hotword-pool/action`。CAgent 只对接上文列出的标准 REST 接口：查询、添加、指定删除、`POST /delete`、清空和 reload。

---

### 4. 错误码

**HTTP 状态码（REST 接口通用）**

| 状态码 | 含义 |
| ------ | ---- |
| 200 | 成功 |
| 204 | 删除成功（`DELETE /api/asr/enrollment/{id}`，无响应体） |
| 400 | 请求字段缺失、音频为空 / 无法解码、注册音频时长校验失败 |
| 413 | 上传文件超过服务端大小限制 |
| 422 | multipart 字段类型或必填字段不符合校验 |
| 502 | 后端模型推理失败 |

**WebSocket 错误（AST v3）**

服务端遇到可恢复错误时发送 `error` 帧：

```json
{
  "header": {
    "code": 1001,
    "message": "model inference failed",
    "sid": "AST_MKMZO0WX2SLZ4"
  }
}
```

> `enrollment_id` 失效不触发 error，服务端回退普通 ASR，并在 AST v3 结果中返回 `enrollment_applied=false` 与可用时的 `enrollment_fallback_reason`。`rl` 是角色变化标记：变化时返回会话内稳定编号 `1..4`，连续同角色或 sidecar 降级时返回 `0`；它不能用于判断声纹是否生效。CAgent 可结合声纹状态查询接口判断是否需要重新注册。

---

## 安菲翁评审意见

整体接口方向可以接受，实时转写、声纹注册删除、热词标准 REST 管理接口可进入对接。以下几点需要修改或补充后再作为最终接口契约。

### 1. AST v3 声纹参数

声纹参数统一放在 `parameter.asr_config` 中：

```json
{
  "parameter": {
    "asr_config": {
      "enrollment_enable": true,
      "enrollment_id": "xxx"
    }
  }
}
```

评审结论：

- 支持 `parameter.asr_config.enrollment_id`。
- 新增 `parameter.asr_config.enrollment_enable`。
- 新增 `parameter.asr_config.enable_role_separation`；默认 `true`，省略等价于开启，且优先级高于声纹。
- `enrollment_enable` 默认 `false`。
- 只有 `enable_role_separation=false && enrollment_enable=true && enrollment_id` 非空时启用声纹。
- `enable_role_separation=false && enrollment_enable=false` 时，即使传入 `enrollment_id`，也不启用声纹。
- `enable_role_separation=false && enrollment_enable=true` 但 `enrollment_id` 为空时，返回参数错误并结束本次会话。
- `header.resIdList[0]` 直接废弃，不作为兼容字段继续使用。

### 2. 声纹生效状态

文档和实现已补充声纹是否实际生效的返回字段。

AST v3 服务端在识别结果中返回：

```json
{
  "enrollment_applied": true
}
```

AST v3 当前采用 `enrollment_applied`；当声纹 ID 不存在、被删除、embedding 不兼容或服务端回退普通 ASR 时返回 `false`，并尽量返回 `enrollment_fallback_reason`，避免 CAgent 只能依赖日志判断。

### 3. 热词管理接口

不使用：

```http
POST /api/asr/hotword-pool/action
```

CAgent 仅对接标准 REST 接口：

```http
GET    /api/asr/hotword-pool
POST   /api/asr/hotword-pool
DELETE /api/asr/hotword-pool
POST   /api/asr/hotword-pool/delete
POST   /api/asr/hotword-pool/clear
POST   /api/asr/hotword-pool/reload
```

其中 `POST /api/asr/hotword-pool/delete` 保留，用于兼容不稳定支持 DELETE body 的客户端、网关或代理。

### 4. clear / reload 热词池

`POST /api/asr/hotword-pool/clear` 必须支持 `hotword_pool_id`。

要求：

- clear 只作用于指定 `hotword_pool_id`。
- 缺省时使用 `default` 池。
- 不应默认影响所有热词池。
- 建议支持 JSON body：

```json
{
  "hotword_pool_id": "default"
}
```

如同时支持 query 和 body，需要明确冲突处理规则。

`POST /api/asr/hotword-pool/reload` 必须支持 `hotword_pool_id`。

要求：

- reload 只作用于指定 `hotword_pool_id`。
- 缺省时使用 `default` 池。
- 不应默认影响所有热词池。
- 建议支持 JSON body：

```json
{
  "hotword_pool_id": "default"
}
```

如同时支持 query 和 body，需要明确冲突处理规则。

### 5. 热词池作用域

- 热词池按 `hotword_pool_id` 隔离。
- 缺省热词池为 `default`。
- 会话热词仅当前 WebSocket 连接生效。
- 会话热词不写入热词池。
- 会话热词与热词池同时存在时，客户端会话热词优先级高于热词池召回词。
- 当两者存在同音、近音或语义冲突时，优先采用客户端显式传入的会话热词。例如客户端传入“王惠”时，应优先于热词库中的“王慧”。

### 6. 热词增删响应

目前文档描述非法词、重复词自动过滤且不报错。该行为可以保留，但响应中需要体现实际生效情况。

添加热词建议返回：

- 实际新增数量。
- 重复数量。
- 非法数量。
- 被忽略词列表或统计。
- 操作后总数。

删除热词建议返回：

- 实际删除数量。
- 未命中数量。
- 未命中词列表或统计。
- 操作后总数。

这样 CAgent 和后台页面可以准确展示哪些词真正生效。

### 7. 鉴权和审计

声纹属于敏感生物特征，热词管理会影响识别结果。管理类接口需要补充：

- 服务间鉴权机制。
- 操作审计日志。
- `traceId` / `requestId` 贯穿。
- 记录调用方、操作类型、`hotword_pool_id`、`enrollment_id` 和操作结果。

### 8. 错误语义

文档已明确 WebSocket 和 REST 的错误行为：

- 参数错误明确返回错误；例如 AST v3 中 `enrollment_enable=true` 但缺少 `enrollment_id` 会返回 error 并结束本次会话。
- 声纹不可用导致回退普通 ASR 时，AST v3 通过 `enrollment_applied=false` 和 `enrollment_fallback_reason` 体现。
- WebSocket 参数错误会结束本次 AST v3 会话；已启用但不可用的声纹 ID 不触发 error。
- reload 失败时，需要说明是否保留旧热词池。

### 9. 声纹查询接口

已新增声纹查询接口：

```http
GET /api/asr/enrollment/{enrollment_id}
```

该接口用于让 CAgent 在只保存 `enrollment_id` 的情况下，判断该 ID 当前是否还能直接用于声纹 ASR。接口不返回原始注册音频、PCM、embedding 或其他声纹敏感材料。

该接口也可用于查询已生效声纹、定位数据不一致：如果 CAgent 保存了某个 `enrollment_id`，但查询返回 `available=false`，说明该 ID 在当前服务端不可用，可能是未注册、已删除、落盘文件缺失、embedding 与当前模型/adapter 不兼容，或路由到没有共享同一 RAG-ASR 本地存储的实例导致。若不同实例或管理服务对同一 `enrollment_id` 返回不同 `available`，可以直接暴露实例间存储或路由不一致问题。

需要注意：该接口只负责诊断和暴露状态，不负责同步声纹数据，也不保证查询后实际 ASR 一定使用成功。最终仍需以 AST v3 响应中的 `enrollment_applied` 或 REST 响应中的 `enrollment_used` 确认本次识别是否实际应用了声纹。

响应结构建议固定为：

```json
{
  "enrollment_id": "xxx",
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

### 10. 热词管理接口的 hotword_pool_id

`hotword_pool_id` 不应只用于 reload。所有热词管理 REST 接口都需要支持 `hotword_pool_id`：

```http
GET    /api/asr/hotword-pool?hotword_pool_id=default
POST   /api/asr/hotword-pool
DELETE /api/asr/hotword-pool
POST   /api/asr/hotword-pool/delete
POST   /api/asr/hotword-pool/clear
POST   /api/asr/hotword-pool/reload
```

要求：

- 缺省时使用 `default` 池。
- 添加、删除、查询、清空、reload 均只作用于指定热词池。
- 大批量导入和 reload 使用的文件或管理服务存储也必须按 `hotword_pool_id` 隔离，不能让多个池共享同一个 `hotword_pool.txt`。

### 11. AST v3 音频格式

文档需要明确 `payload.audio.audio` 中 base64 的原始音频格式。

建议约定为：

- 16 kHz。
- mono。
- s16le PCM。
- base64 编码后放入 `payload.audio.audio`。

如服务端允许首帧携带 WAV header，需要在文档中单独标明，否则按 raw PCM 对接。

### 12. 语种字段

当前文档中 `parameter.engine.wdec_param_LanguageTypeChoice` 与 `parameter.asr_config` 的职责边界不清晰。

建议：

- CAgent 推荐使用 `parameter.asr_config.language` 表示语种。
- `parameter.engine.wdec_param_LanguageTypeChoice` 如需保留，需要明确是否实际生效、取值映射和优先级。
- 若仅作为历史透传字段，应在文档中标明“不作为推荐接入字段”。

### 13. 注册错误码

注册接口详细设计中支持 WAV、MP3、PCM，但错误码表缺少不支持格式的场景。

建议补充：

| HTTP 状态码 | detail.code | 说明 |
| ----------- | ----------- | ---- |
| 400 | `unsupported_format` | 上传格式不是 WAV、MP3 或 16 kHz mono s16le PCM |

### 14. 最终契约文档建议

当前文档正文已按本轮实现同步修订。若要作为双方最终接口契约，仍建议另出无评审意见附录的正式修订版：

- 保留“按 `hotword_pool_id` 隔离的热词池”、声纹生效状态、错误语义、音频格式和注册错误码。
- 鉴权审计仍待双方确认机制后单独补充。
