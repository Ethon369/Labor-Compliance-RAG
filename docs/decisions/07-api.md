# 07 · FastAPI 流式 API（SSE + 路由设计）

> 阶段 6 决策记录。对应 `app/main.py`、`app/api/models.py`、`app/api/routes.py`、`tests/test_api.py`。
> 格式约定（CLAUDE.md 第 6 条）：选了什么、备选、为什么、已知取舍。

## 1. 一句话结论

**问答接口用 SSE（Server-Sent Events）单向推送阶段进度，不用 WebSocket 双向通道。** 原因：问答场景天然是 client→server（发一个问题）→ server→client（推回答），没有客户端多次推送的需求——WebSocket 的双向通道在这里是过度设计。SSE 比 WebSocket 简单一个数量级：不需要握手升级协议、不需要心跳保活、HTTP/1.1 原生支持、用 `fetch + ReadableStream` 即可在浏览器消费。

## 2. SSE vs WebSocket 选型依据

| 维度 | SSE | WebSocket |
|---|---|---|
| 方向 | 单向：server→client | 双向：client↔server |
| 协议 | HTTP/1.1 长连接，`Content-Type: text/event-stream` | 独立 ws:// 协议，需 HTTP Upgrade 握手 |
| 客户端实现 | `fetch()` 或 `EventSource`，浏览器原生支持 | 需 `new WebSocket()` + `onmessage` |
| 代理/负载均衡 | HTTP 代理天然支持，无特殊配置 | 需要代理也升级到 ws（Nginx 需额外配置） |
| 断线重连 | `EventSource` 自动重连（内建 Last-Event-ID） | 需手动实现重连逻辑 |
| 心跳保活 | 不需要——HTTP 长连接由 TCP keepalive 负责 | 需应用层心跳（ping/pong），否则中间代理可能断开空闲连接 |
| 调试 | `curl --no-buffer` 即可，标准 HTTP 工具都能看 | 需 `wscat` / Postman / 专用 ws 工具 |

**本案为什么选 SSE：**

- 问答场景只有两个动作：客户端发一个问题 + 服务端推回答进度。客户端**没有持续推送新消息**的需求——用户问完一句后等答案，不存在"提问中途再补充信息"的场景。
- 流式回答是单向推送——server 把进度帧（rewriting→searching→verifying→answering→done）推给 client，client 只消费、不回复。SSE 天生就是干这个的。
- 验证/演示场景下，`curl --no-buffer` 就能看效果，不需要额外工具——与项目"离线可演示"的思路一致。

**什么时候会用 WebSocket：** 如果后续加"对话式反问"——比如 Agent 追问用户"你说的'补偿'是指经济补偿金还是工伤赔偿？"——这时才需要双向通道。但那是 v2 的考量。

## 3. SSE 具体设计

### 3.1 帧格式

```
data: {"stage":"rewriting","content":"正在理解您的问题..."}

data: {"stage":"searching","content":"正在检索相关法律条文..."}

data: {"stage":"verifying","content":"正在核验条文与问题的相关性..."}

data: {"stage":"answering","content":"正在整理回答..."}

data: {"stage":"done","content":"<ChatDonePayload JSON 字符串>"}
```

- 每条一行 `data: <json>\n\n`
- `stage` 字段让客户端知道当前在哪一步（控制 UI 进度条/阶段指示器）
- 最终帧 `stage="done"` 时 `content` 是完整 `ChatDonePayload`（答案文本 + refuse 标记 + 引用列表）
- 异常帧 `stage="error"` 时 `content` 是 `ErrorResponse` JSON

### 3.2 流式生成器

当前 v1 用 `async def _stream_chat()` 生成器在单个线程内**模拟**进度推送（发帧→invoke→发帧），不是 LangGraph 的 `astream_events` 逐节点事件。原因：

- 当前 Graph 输出的是同步 `invoke`，无原生节点事件流
- 改写-检索-核验-回答 四步在 `invoke` 内部完成，对外是黑盒
- v2 改用 `astream` 后可实现真正的逐节点推送（每完成一个节点发一帧），无需手动硬编码进度顺序

### 3.3 响应头

```python
StreamingResponse(
    media_type="text/event-stream",
    headers={
        "Cache-Control": "no-cache",     # SSE 不能被缓存
        "Connection": "keep-alive",       # 复用 HTTP 连接
        "X-Accel-Buffering": "no",        # 告知 nginx 不缓冲（否则 SSE 变一次性返回）
    },
)
```

## 4. /chat/history 设计

- **GET /chat/history?limit=50**：返回最近 N 条会话记录，倒序（最新在前）
- **存储**：内存列表，最大 200 条。服务重启后清空——个人展示项目不需要持久化，演示场景重启丢失是可接受的
- **备选方案（已拒绝）**：存 PostgreSQL——对展示项目过度，增加部署复杂度；存 SQLite——引入新依赖；存文件——并发读写要加锁。当前内存方案对"展示"场景是最轻的

## 5. 异常处理分层

### 5.1 路由级：FastAPI 全局异常处理

```python
@app.exception_handler(ValueError)  → 400（客户端参数错误）
@app.exception_handler(Exception)   → 500（服务器内部错误）
```

两种 handler 都返回 `ErrorResponse` Pydantic 模型的 JSON——错误响应不裸奔，始终结构化。

### 5.2 流式内部的异常处理

SSE 流一旦开始，HTTP 状态码已发（200），不能再改为 4xx/5xx。所以流内的异常**嵌入为 error 帧**：

```
data: {"stage":"error","content":"<ErrorResponse JSON>"}
```

客户端消费 SSE 时遇到 `stage="error"` 即知出错了——与 done 帧对称。这是**流式场景异常处理的核心模式**：不能改 HTTP status → 改在数据流里带错误帧。

### 5.3 参数校验：Pydantic 自动拦截

`ChatRequest(question=..., min_length=1, max_length=2000)` —— FastAPI 在请求体到达路由处理函数之前就校验完了，不合规的自动返回 422 Unprocessable Entity + 详细的 validation error 列表。

## 6. lifespan 装配

`@asynccontextmanager lifespan` 在 FastAPI 启动时做一次性装配：

```
Settings() → build_retriever(settings) → build_agent_from_config(retriever, settings)
```

三个对象挂到 `app.state` 上，路由通过 `request.app.state.agent` 获取。这样做的好处：

- 不用引入 FastAPI `Depends()` 依赖注入——保持路由函数签名简单
- 装配逻辑集中在一处，不散落在各路由
- 离线模式缺 key/缺 DB 时在启动阶段快速失败，不在第一次请求时才暴露

## 7. 已知取舍

1. **SSE 帧是手动发的，不是 LangGraph astream_events**。当前 invoke 是同步黑盒，四步进度是硬编码顺序。v2 用 `astream_events` 替换可实现真正的逐节点事件推送——不需要改 SSE 帧格式，只改生成器内部。
2. **回答是整段文本，不是逐 token**。当前 LLM 调用得到全文后一次性塞进 done 帧，不是每个 token 一发 SSE。逐 token 流式是 v2 优化方向——需要在 `ToolCallingAnswerer` 内部替换为流式请求（`stream: true`）并逐 token yield。
3. **历史存内存，不持久化**。演示场景足够；生产环境需替换为数据库存储。换存储不改接口——GET /chat/history 返回格式不变。
4. **CORS 全开**。本地开发安全范围宽松；部署时需收紧为前端域名。
5. **只有 chat 路由**。系统为单页面问答场景设计，没有鉴权/用户管理——展示项目不需要。