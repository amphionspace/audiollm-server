# AGENTS.md

本文件汇总原 `.cursor/rules/` 中的仓库级代理规则。目标是让后续 agent 在调试、配置设计、接口变更和重构时，先处理根因，再做最小且完整的代码改动。

## Git 与 PR 收尾纪律

开始任何文件改动前，必须先确认当前分支；禁止在 `main` 上直接修改文件。若当前在 `main`，先创建或切换到符合本仓库命名规则的工作分支（例如 `chore/...`、`fix/...`、`feat/...`），再进行编辑。用户明确要求在主分支操作时除外。

PR 合入后，必须切回 `main`，拉取远端最新改动，并清除已合入的无用本地分支；如果远端同名工作分支仍存在且不再需要，也应一并删除。

## 调试与重构纪律

### 先抓运行现场，再读代码

当用户反馈“慢”“不工作”“没响应”“长时间没有 final”等运行时问题时，第一步必须抓 runtime evidence，例如日志、监控、可复现请求或实际 WebSocket 事件流。不要先从配置或代码 grep 开始。

示例：

```bash
journalctl -u audiollm-demo --since "5 minutes ago" --no-pager | rg "session|vad|httpx" | tail -50
```

原因：真实事故中，日志已经显示 32 秒内只有 partial、没有 final；先看现场能直接定位问题层。

### 拆分布尔开关前先列完整矩阵

任何把一个开关拆成多个开关的改动，都必须先列出“配置组合 × 期望行为”的完整矩阵。矩阵覆盖率是评审硬指标，不能漏掉合法组合。

模板：

| 开关 A | 开关 B | 期望行为 | 当前实现 |
|---|---|---|---|
| true | true | ... | ... |
| true | false | ... | ... |
| false | true | ... | ... |
| false | false | ... | ... |

只要有一行没有被代码路径或测试覆盖，就不算改完。

### 同语义双路径先去重

如果两条代码路径实现同一个语义，例如 `session.py` 与 `tasks/asr.py` 各自实现 partial 调度，应先抽出公共函数或统一入口，再做功能改动。

判断信号：搜索某个函数或语义时，出现两个以上独立实现且参数面相似，就应停下来先消除重复。否则每次改动都要同步多处，容易漏掉 legacy 或新路径。

### 多文件改动前自检

提交多文件改动前，回答这三个问题：

1. 改动涉及的每一种合法配置组合，我是否都跑过或读过期望路径？
2. 如果存在 legacy/new 双路径，我是否只改了熟悉的那条？
3. 自我批判中暴露的根因，是不是被新代码真正修掉，而不是绕过去了？

## 配置与参数设计纪律

### 数据类不变量下沉到构造器

Python `dataclass` 的字段间依赖应在 `__post_init__` 中强制执行，不要只依赖 `load_config()`、`override()` 等外部函数。外部函数覆盖不到测试或调用方直接 `Config(...)` 构造的路径。

示例：

```python
@dataclass(frozen=True)
class Config:
    enable_secondary_asr: bool = True
    enable_dual_asr_fusion: bool = True

    def __post_init__(self) -> None:
        if self.enable_dual_asr_fusion and not self.enable_secondary_asr:
            object.__setattr__(self, "enable_dual_asr_fusion", False)
```

`load_config` 层可以保留 WARN 日志，方便运维发现错误配置；降级逻辑只应保留一份，并放在数据源头。

### 禁止两个参数表达同一概念后取 max/min

如果两个配置项表达同一物理量或同一语义，不允许用 `max(a, b)` 或 `min(a, b)` 让“更严格者获胜”。这会让用户修改某个参数时看不到效果，形成配置欺骗。

正确做法：

- 删除其中一个参数，保留单一事实来源；
- 或明确其中一个只是 fallback，并在另一个被忽略时输出 WARN。

### 配置项命名必须匹配作用面

开关名不能暗示一个语义，实际却控制多个可观察行为。每个独立可观察行为应有独立开关，并在文档中给出完整行为矩阵。

例如：一个“副模型是否在线”的开关不应同时控制 partial 通道、partial 静音门和 final 融合。

### 新增或拆分配置时同步更新

修改 `backend/config.py` 字段时，必须同步检查并更新：

- `backend/config.json` 默认值；
- `.env.example` 环境变量注释；
- `README.md` 与 `docs/api-reference.md` 中的字段表；
- 相关行为矩阵；
- 至少一个覆盖新不变量的 pytest 用例。

## 前后端接口契约纪律

### 前后端只通过协议通信

本项目的前端 `frontend/` 与后端 `backend/` 只通过 HTTP/WebSocket 协议通信。

禁止：

- 后端模板渲染前端页面，例如引入 Jinja/SSR；
- 前端 JS 通过相对路径直接读取 `backend/` 内文件；
- 前后端分别手写同一套约定字段名、enum 值或错误码字符串；如必须共享，使用 OpenAPI 自动生成或显式文档登记；
- 后端在错误响应中返回 HTML，错误必须是 JSON，例如 `{message, code, detail}` 结构。

允许：

- 前端通过 `/api/...` REST 接口调用后端；
- 前端通过 `/ws/...`、`/transcribe-streaming`、`/emotion-segmented-streaming` 等 WebSocket 调用后端；
- 前端通过 `fetch('/openapi.json')` 等公开端点拉取 schema。

### 后端接口修改必须同步文档

以下变化都算后端接口修改，必须同步对应文档：

- 新增、删除或重命名路由；
- 请求或响应字段增删改，包括可选字段语义变化；
- WebSocket 协议中任意 `type` 取值或字段语义变化，包括 `partial`、`final`、`vad_event` 等；
- HTTP 状态码或错误结构变化；
- 客户端可在 `start.config` 临时覆写的字段集合变化；
- 默认值改变且对外可观察。

以下变化不算接口修改：

- 内部 Python 函数签名；
- 模块重组；
- 私有辅助函数调整。

### 路由与文档映射

修改对应后端路由时，必须同步更新映射文档。

| 后端路由 | 文档 |
|---|---|
| `@app.websocket("/transcribe-streaming")` | `docs/protocols/transcribe-streaming-protocol.md` + `docs/api-reference.md` 速览 |
| `@app.websocket("/tuling/ast/v3")` | `docs/protocols/tuling-ast-v3-protocol.md` + `docs/api-reference.md` 速览 |
| `@app.websocket("/astv3-test-proxy")` | `docs/api-reference.md` |
| `@app.websocket("/emotion-segmented-streaming")` | `docs/protocols/emotion-segmented-streaming-protocol.md` + `docs/api-reference.md` 速览 |
| `@app.post("/api/asr/upload")` | `docs/api-reference.md` |
| `@app.post("/api/asr/transcriptions")` + `@app.get("/api/asr/transcriptions/{job_id}")` | `docs/api/transcription-jobs-api.md` + `docs/api-reference.md` 速览 |
| `@app.post("/api/asr/enrollment")` | `docs/api-reference.md` + `docs/protocols/transcribe-streaming-protocol.md` 注册接口章节 |
| `@app.delete("/api/asr/enrollment/{id}")` | 同上 |
| `@app.get("/api/asr/hotword-pool")` + `@app.post("/api/asr/hotword-pool")` + `@app.delete("/api/asr/hotword-pool")` + `@app.post("/api/asr/hotword-pool/reload")` | `docs/api-reference.md` |
| `@app.post("/api/audio/analyze")` | `docs/api/audio-analyze-api.md` + `docs/api/public-audio-analyze-api.md` |
| `@app.post("/api/emotion/jobs")` + `@app.get("/api/emotion/jobs/{job_id}")` | `docs/api-reference.md` |
| `@app.post("/api/emotion-spec/jobs")` + `@app.get("/api/emotion-spec/jobs/{job_id}")` | 已知漂移，见下节 |

新增路由时，必须至少在 `docs/api-reference.md` 添加一节，并更新本表。

### 接口改动提交前清单

接口改动提交前必须确认：

1. 改动是否属于“后端接口修改”；
2. 对应文档是否补足路由、方法、请求字段、响应字段、错误码与 `detail` 结构；
3. 如果前端消费该改动，`frontend/app.js`、`frontend/streaming-text.js` 等前端代码是否同步修改；
4. `tests/` 中是否至少有一个用例覆盖新行为；
5. 文档中的 `curl` 或 JSON 示例是否可以原样跑通。

任一项空缺都表示改动未完成。

### 接口漂移自检

提交前可用以下命令快速对照代码和文档：

```bash
rg "@app\.(websocket|get|post|put|delete)" backend/ -n
rg "POST|GET|DELETE|WebSocket" docs/ -n
```

如果代码中存在文档未描述的路由，或文档字段在代码中不存在，就是契约漂移。

### 已知存量漂移

`POST /api/emotion-spec/jobs` 与 `GET /api/emotion-spec/jobs/{job_id}` 在 `docs/` 中没有对应描述。下次修改这两个路由时，必须先在 `docs/api-reference.md` 补充对应章节，再做功能改动。
