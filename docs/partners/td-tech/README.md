# 鼎桥（TD Tech）对接资料

本目录归档与鼎桥（TD Tech）实时转写 AST v3 对接相关的外部参考文档和内部修订文档索引。

| 文档 | 说明 |
| ---- | ---- |
| [开发指南-实时转写服务AST.3.4.4.1081-增加角色分离.doc](开发指南-实时转写服务AST.3.4.4.1081-增加角色分离.doc) | 鼎桥提供的 AST 3.4.4.1081 参考文档，包含角色分离字段说明 |
| [TMGenius 语音识别接口（修订版）](TMGenius语音识别接口-修订版.md) | 面向 CAgent/TMGenius 对接的评审修订版契约 |
| [TMGenius 语音识别接口（评审版）](TMGenius语音识别接口-评审版.md) | TMGenius 对接评审版历史资料 |
| [TMGenius 接口对接 TODO](tmgenius-interface-todo.md) | 对接口径、待确认项和实现 TODO |
| [实时转写 AST v3 WebSocket API](../../protocols/tuling-ast-v3-protocol.md) | 本服务当前 `/tuling/ast/v3` 实现文档 |

当前角色分离对接口径：接受 `parameter.asr_config.enable_role_separation`，默认 `true`、省略等价于开启，并且优先级高于声纹。开启时最多区分 4 位会话内匿名角色；`sentence` 在角色变化时返回稳定 `cw[].rl=1..4`，同一角色连续发言返回 `0`，`Progressive` 不返回 `cw[].rl`。sidecar 故障时本会话继续普通 ASR 并返回 `rl=0`；关闭角色分离时两类结果均不返回 `cw[].rl`。

声纹对外契约口径：声纹仅在 `enable_role_separation=false && enrollment_enable=true && enrollment_id` 非空时启用，`enrollment_enable` 默认 `false`；AST v3 响应返回 `enrollment_applied` 表示本次是否实际携带可用声纹材料，REST 上传响应使用 `enrollment_used`；`GET /api/asr/enrollment/{enrollment_id}` 用于查询声纹 ID 当前是否可用。内部实现状态统一维护在 TODO 文档。
