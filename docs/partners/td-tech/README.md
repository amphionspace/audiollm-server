# 鼎桥（TD Tech）对接资料

本目录归档与鼎桥（TD Tech）实时转写 AST v3 对接相关的外部参考文档和内部修订文档索引。

| 文档 | 说明 |
| ---- | ---- |
| [开发指南-实时转写服务AST.3.4.4.1081-增加角色分离.doc](开发指南-实时转写服务AST.3.4.4.1081-增加角色分离.doc) | 鼎桥提供的 AST 3.4.4.1081 参考文档，包含角色分离字段说明 |
| [TMGenius 语音识别接口（修订版）](TMGenius语音识别接口-修订版.md) | 面向 CAgent/TMGenius 对接的评审修订版契约 |
| [TMGenius 语音识别接口（评审版）](TMGenius语音识别接口-评审版.md) | TMGenius 对接评审版历史资料 |
| [TMGenius 接口对接 TODO](tmgenius-interface-todo.md) | 对接口径、待确认项和实现 TODO |
| [实时转写 AST v3 WebSocket API](../../tuling-ast-v3-protocol.md) | 本服务当前 `/tuling/ast/v3` 实现文档 |

当前角色分离对接口径：本版本暂不支持角色分离，但接受 `parameter.asr_config.enable_role_separation` 字段；无论客户端传 `true` 还是 `false`，服务端均正常识别，`sentence` 结果中的 `cw[].rl` 固定返回整数 `0`，`Progressive` 结果不返回 `cw[].rl`。
