# 鼎桥（TD Tech）对接资料

本目录归档与鼎桥（TD Tech）实时转写 AST v3 对接相关的外部参考文档和内部修订文档索引。

| 文档 | 说明 |
| ---- | ---- |
| [开发指南-实时转写服务AST.3.4.4.1081-增加角色分离.doc](开发指南-实时转写服务AST.3.4.4.1081-增加角色分离.doc) | 鼎桥提供的 AST 3.4.4.1081 参考文档，包含角色分离字段说明 |
| [TMGenius 语音识别接口（修订版）](TMGenius语音识别接口-修订版.md) | 面向 CAgent/TMGenius 对接的评审修订版契约 |
| [TMGenius 语音识别接口（评审版）](TMGenius语音识别接口-评审版.md) | TMGenius 对接评审版历史资料 |
| [TMGenius 接口对接 TODO](tmgenius-interface-todo.md) | 对接口径、待确认项和实现 TODO |
| [实时转写 AST v3 WebSocket API](../../protocols/tuling-ast-v3-protocol.md) | 本服务当前 `/tuling/ast/v3` 实现文档 |

## AST v3 角色/声纹行为矩阵

| `enable_role_separation` | 声纹设置 | 实际行为 | `sentence` 的 `cw[].rl` |
|---|---|---|---|
| `true` / 省略 | 任意 | 忽略声纹参数，执行最多 4 人的会话内角色分离 | 正常为首次/切换 `1..4`、连续同角色 `0`；sidecar 降级后为 `0` |
| `false` | `enrollment_enable=false` / 省略 | 普通 ASR；即使传入 `enrollment_id` 也忽略 | 不返回 |
| `false` | `enrollment_enable=true` 且 ID 非空、可用 | TS-ASR | 不返回 |
| `false` | `enrollment_enable=true` 且 ID 非空、不可用 | 回退普通 ASR，`enrollment_applied=false` 并返回原因 | 不返回 |
| `false` | `enrollment_enable=true` 且 ID 为空/省略 | 返回参数错误并结束会话 | 无识别结果 |

`Progressive` 始终不返回 `cw[].rl`。AST v3 响应中的 `enrollment_applied` 表示本条结果是否实际携带可用声纹材料；REST 上传响应使用 `enrollment_used`。`GET /api/asr/enrollment/{enrollment_id}` 可查询声纹 ID 当前是否可用。完整契约以 [实时转写 AST v3 WebSocket API](../../protocols/tuling-ast-v3-protocol.md#角色分离与声纹行为矩阵) 为准，内部实现状态统一维护在 TODO 文档。
