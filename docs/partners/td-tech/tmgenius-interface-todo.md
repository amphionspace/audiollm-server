# TMGenius 接口对接我方 TODO

## 内部状态和待办

- 已完成：`/tuling/ast/v3` 从 `parameter.asr_config.enrollment_id` 读取声纹 ID，`header.resIdList[0]` 不再作为兼容字段使用。
- 已完成：角色分离兼容字段 `parameter.asr_config.enable_role_separation`，当前不做角色分离，`sentence` 的 `cw[].rl` 固定返回 `0`，`Progressive` 不返回 `rl`。
- 已完成：热词池按 `hotword_pool_id` 隔离，支持查询、添加、指定删除、`POST /delete`、清空和 reload。
- 已完成：RAG-ASR HTTP 管理面提供 `GET /enrollments/{enrollment_id}`，并将 embedding tensor 与 JSON 元数据落盘到本地 `enrollment_store_dir`。
- 待补：`audiollm-demo` 尚未对外代理 `GET /api/asr/enrollment/{enrollment_id}`。
- 待补：AST v3 响应尚未返回 `enrollment_used` / `enrollment_applied` / `enrollment_fallback_reason`。
- 待补：`audiollm-demo` 尚未实现 `parameter.asr_config.enrollment_enable` 声纹显式启用开关；当前只要首帧携带非空 `enrollment_id` 就启用目标说话人。

## 1. AST v3 声纹参数改造

- 已完成 `/tuling/ast/v3` 首帧解析逻辑：
  - 不再读取 `header.resIdList[0]`。
  - 只从 `parameter.asr_config.enrollment_id` 读取声纹 ID。
- 待补读取 `parameter.asr_config.enrollment_enable`。
  - `enrollment_enable` 默认值为 `false`。
- 待补声纹启用规则：
  - `enrollment_enable=false`：不启用声纹，即使传了 `enrollment_id` 也忽略。
  - `enrollment_enable=true && enrollment_id 非空`：启用声纹。
  - `enrollment_enable=true && enrollment_id 为空`：返回参数错误，不进入静默普通 ASR。
- 待补响应声纹生效状态：
  - 在 AST v3 识别结果中返回 `enrollment_used` 或 `enrollment_applied`。
  - 当声纹 ID 不存在、被删除、embedding 不兼容或不可用时，返回 `false`。
  - 如有条件，补充 `enrollment_fallback_reason`，例如 `not_found` / `incompatible` / `disabled` / `upstream_unavailable`。

## 2. AST v3 文档和测试同步

- 已更新我方 AST v3 协议文档：
  - 标明 `header.resIdList[0]` 已废弃，不再支持。
  - 标明声纹只通过 `parameter.asr_config.enrollment_id` 传入。
- 已更新测试：
  - 覆盖 `resIdList[0]` 不再生效。
  - 覆盖 `parameter.asr_config.enrollment_id` 映射到目标说话人。
- 待补 `enrollment_enable` 文档、测试客户端和单元测试：
  - 标明 `enrollment_enable` 默认 `false`。
  - 标明 `enrollment_enable=true` 但缺少 `enrollment_id` 的错误行为。
  - 覆盖 `enrollment_enable=false` 忽略声纹。
  - 覆盖 `enrollment_enable=true + enrollment_id` 正常启用。
  - 覆盖 `enrollment_enable=true + enrollment_id 缺失` 返回错误。
- 若补充 `enrollment_used` / `enrollment_applied`，需要同步更新 AST v3 协议文档和单元测试。

## 3. 热词删除兼容接口

- 已保留并实现 `POST /api/asr/hotword-pool/delete`。
- 语义与 `DELETE /api/asr/hotword-pool` 完全一致。
- 用于兼容不稳定支持 DELETE body 的 HTTP 客户端、网关或代理。

## 4. 不实现统一 action 入口

- 不新增 `/api/asr/hotword-pool/action`。
- CAgent 只对接标准 REST 接口：
  - `GET /api/asr/hotword-pool`
  - `POST /api/asr/hotword-pool`
  - `DELETE /api/asr/hotword-pool`
  - `POST /api/asr/hotword-pool/delete`
  - `POST /api/asr/hotword-pool/clear`
  - `POST /api/asr/hotword-pool/reload`

## 5. 清空热词池接口

- 已新增并实现 `POST /api/asr/hotword-pool/clear`。
- 只清空指定 `hotword_pool_id`，不影响其他池。
- 建议同时支持 query 和 JSON body：
  - query：`?hotword_pool_id=xxx`
  - JSON body：`{"hotword_pool_id": "xxx"}`
- 若 query 和 body 同时存在且不一致，返回参数错误。
- `DELETE /api/asr/hotword-pool` 和 `POST /api/asr/hotword-pool/delete` 仍只删除指定 `hotwords`，空数组不得解释为清空。

## 6. reload 支持 hotword_pool_id

- `POST /api/asr/hotword-pool/reload` 必须支持 `hotword_pool_id`。
- 建议同时支持 query 和 JSON body：
  - query：`?hotword_pool_id=xxx`
  - JSON body：`{"hotword_pool_id": "xxx"}`
- 若 query 和 body 同时存在且不一致，返回参数错误。
- 缺省时使用 `default` 热词池。
- reload 只作用于指定热词池，不影响其他池。
- 所有热词管理 REST 接口都应支持 `hotword_pool_id`，包括查询、添加、删除、`POST /delete`、清空和 reload。
- 大批量导入和 reload 使用的文件或管理服务存储也必须按 `hotword_pool_id` 隔离，不能让多个池共享同一个 `hotword_pool.txt`。

## 7. 热词增删响应增强

- 添加热词响应建议包含：
  - `added_count`
  - `duplicate_count`
  - `invalid_count`
  - `ignored_hotwords` 或 `invalid_hotwords`
  - `total_count`
- 删除热词响应建议包含：
  - `deleted_count`
  - `missing_count`
  - `missing_hotwords`
  - `total_count`
- 目标是避免非法词、重复词、未命中词被静默过滤后，后台无法感知真实生效结果。

## 8. 热词池作用域修正

- 文档和接口语义中避免使用“全局热词池”作为唯一描述。
- 应改为：
  - 热词池按 `hotword_pool_id` 隔离。
  - 缺省池为 `default`。
  - 会话热词仍只作用于当前连接，不写入热词池。
- 明确会话热词与热词池同时存在时的优先级：
  - 客户端会话热词优先级高于热词池召回词。
  - 当两者存在同音、近音或语义冲突时，优先采用客户端显式传入的会话热词。
  - 示例：客户端会话热词传入“王惠”时，应优先于热词库中的“王慧”。

## 9. 鉴权、审计和错误语义

- 管理类 REST 接口需补充服务间鉴权。
- 热词管理和声纹管理需记录审计日志：
  - 调用方
  - `traceId` / `requestId`
  - `action`
  - `hotword_pool_id`
  - `enrollment_id`
  - 操作结果
- WebSocket error 需明确：
  - 哪些错误关闭连接。
  - 哪些错误可以继续连接。
  - 参数错误应返回明确错误，不静默回退。

## 10. 声纹查询接口

- 待补 `audiollm-demo` 对外代理接口：`GET /api/asr/enrollment/{enrollment_id}`。
- RAG-ASR HTTP 管理面已提供原生接口：`GET /enrollments/{enrollment_id}?enrollment_scope_id=...`。
- 用于让 CAgent 查询已保存的 `enrollment_id` 当前是否还能直接用于声纹 ASR。
- 可用于查询已生效声纹、定位数据不一致：不同实例或管理服务对同一 `enrollment_id` 返回不同 `available` 时，说明落盘存储、模型兼容性或路由存在不一致。
- 该接口只负责诊断和暴露状态，不负责同步声纹数据，也不保证查询后实际 ASR 一定使用成功；若后续补充 AST v3 声纹生效字段，最终仍以本次 ASR 响应中的 `enrollment_used` / `enrollment_applied` 为准。
- 响应字段固定为简化结构：

```json
{
  "enrollment_id": "xxx",
  "available": true,
  "reason": "ok"
}
```

- 字段语义：
  - `enrollment_id`：本次查询的声纹 ID。
  - `available`：是否可直接用于后续 ASR；CAgent 以该字段作为是否需要重新注册的判断依据。
  - `reason`：状态原因，`available=true` 时为 `ok`。
- `available=false` 的常见 `reason`：
  - `not_found`：服务端找不到该 ID。
  - `incompatible`：落盘 embedding 与当前模型、adapter 或 projector 维度不兼容。
  - `deleted`：已显式删除。
  - `upstream_unavailable`：外部 enrollment 管理服务不可用。
- 查询接口不返回原始注册音频、PCM、embedding 或其他声纹敏感材料。
- RAG-ASR 管理服务模式下，查询结果以管理服务为准；RAG-ASR 将 projector frames tensor 和 JSON 元数据落盘到 `enrollment_store_dir/<scope>/<enrollment_id>.pt/.json`（默认 `var/enrollments`），查询/使用会更新 `last_used_at`，当前不做 TTL 自动过期。
- 若仍使用 demo 进程内 fallback 缓存，则该缓存会受 TTL、重启和 LRU 容量限制。

## 11. AST v3 音频和语种字段

- 明确 `payload.audio.audio` 的音频格式：
  - base64 编码内容为 16 kHz、mono、s16le PCM。
  - 如允许首帧携带 WAV header，需要在文档中单独标明。
- 语种字段建议统一使用 `parameter.asr_config.language`。
- `parameter.engine.wdec_param_LanguageTypeChoice` 不作为推荐接入字段；如对方要求保留，需要双方确认映射关系和生效范围。

## 12. 文档、示例和错误码补齐

- 同步更新：
  - [`../../tuling-ast-v3-protocol.md`](../../tuling-ast-v3-protocol.md)
  - [`../../api-reference.md`](../../api-reference.md)
  - `tests/test_ast_v3_ws_client.py`
- 测试客户端的 `--enrollment-id` 已写入 `parameter.asr_config.enrollment_id`；待实现 `enrollment_enable` 后需要同步设置 `enrollment_enable=true`。
- 注册接口错误码补充 `unsupported_format`，用于 WAV/MP3/PCM 之外的格式。
