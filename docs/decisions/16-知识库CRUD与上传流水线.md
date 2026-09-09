# 16 · 知识库 CRUD 与上传流水线

> V2.3 决策记录。对应 `app/kb/store.py`、`app/kb/pipeline.py`、`app/api/kb_routes.py`、
> `app/api/deps.py`、`app/main.py`、`tests/test_kb.py`、`tests/test_api_kb.py`。
> 格式约定（CLAUDE.md 第 6 条）。

## 1. 一句话结论

**知识库是三层模型（kb → documents → chunks）落成的一整套 CRUD；文档上传只"登记
pending + 流式落盘 + 派发后台任务"，解析/切分/向量化交给 `BackgroundTasks` 异步完成，
前端轮询状态直到 ready/failed。流水线"永不抛异常"——任何失败都落进
`documents.error + status='failed'`，让用户能看见原因而不是永远卡在 pending。**

## 2. 目标

- 知识库的建/改/删/列、文档的上传/列/删/状态轮询、重建索引，全部走 API。
- 上传要"秒回"，不让用户盯着转圈等解析 + 向量化（大文档可能几十秒）。
- 越权语义与决策 14 一致：他人私有库 404、公开库非 owner 写 403、普通用户调 admin 接口 403，
  并写成端到端测试守住。
- 内容与向量同事务写入，不留"内容已入库但还没向量"的中间窗口。

## 3. 选中的方案

### 存储层（`app/kb/store.py`）

| 能力 | 关键实现 |
|---|---|
| kb CRUD | `create_kb` / `update_kb` / `delete_kb` / `fetch_kb` / `list_kbs`（分页 + 计数） |
| 文档 CRUD | `create_document`（登记 pending）/ `get_document` / `list_documents` / `delete_document` / `set_document_status` |
| 切片写入 | `replace_chunks`：**单事务**内"删尾片 → upsert → 同步计数" |
| 计数同步 | `_sync_counts`：doc_count / chunk_count 每次写后重算，保证列表数字真实 |

**为什么 `replace_chunks` 要"先删 seq 超出的尾片"**：文档重传后变短时，`ON CONFLICT
(doc_id, seq)` 只会覆盖前 N 片，多出来的旧片会永远留在库里被检索到——这是"越用越脏"
的典型来源。删尾片保证重传后库里恰好是新文档的切片集合。

**为什么计数"每次写后重算"而不是 `+1/-1` 维护**：增量维护在并发上传、删除、重传
交织时会漂移；重算一次 `count(*)` 简单且绝对正确。个人展示规模下成本可忽略。

### 流水线（`app/kb/pipeline.py`）

`run(dsn, doc_id, path, source_type, kb_row)` 的硬性契约：**永不抛异常**。

它是 BackgroundTasks 里的后台任务——抛出去的异常没人接、没人看，文档会永远卡在
pending。所以：

- `UnsupportedContentError`（扫描件）→ `failed` + 用户可读提示；
- 解析后切不出片 → `failed`（"没有可入库的内容片段"），不留 `chunk_count=0` 的 ready；
- 其它任何异常 → `failed` + `type(exc).__name__: exc`；
- 临时文件在 `finally` 里清理（上传端点写盘、后台任务消费，职责成对才不会漏）。

**为什么 embedding 用批量**：`embedder.embed([...])` 一次请求处理整篇切片，比逐片调用
少 N-1 次网络往返，也便于 provider 端批处理。

### API 层（`app/api/kb_routes.py`）

| 方法 | 路径 | 作用 | 鉴权 |
|---|---|---|---|
| GET | `/kb` | 我可见的库（分页） | 登录 |
| POST | `/kb` | 新建库 | 登录 |
| GET | `/kb/admin/all` | 平台全部库 | `require_admin` |
| GET/PATCH/DELETE | `/kb/{kb_id}` | 单库查/改/删 | `require_kb_access(write=...)` |
| GET/POST | `/kb/{kb_id}/documents` | 文档列表 / 上传 | 读 / `write=True` |
| GET/DELETE | `/kb/documents/{doc_id}` | 单文档状态 / 删除 | 反查所属库判定 |
| POST | `/kb/{kb_id}/reindex` | 重建向量 | `write=True` |

**上传三件事（不等待）**：① 校验扩展名（`detect_source_type` 不认 → 400）；② 流式读盘
（`await file.read(1MB)` 循环），超过 10MB 当场 `413`，**不整体 `await file.read()`**
——否则等于把内存上限交给客户端，10MB × 并发上传足以顶翻进程；③ 插 pending 行 +
`background.add_task(pipeline.run, ...)`，立即返回 `doc_id` 供轮询。

**路由顺序是正确性的前提**：`/kb/admin/all` 与 `/kb/documents/{doc_id}` 必须排在
`/kb/{kb_id}` 之前，否则 "admin"/"documents" 会被 `{kb_id}` 先吃掉（转不成 int）。
这条在 `main.py` 挂载顺序 + 文件内定义顺序上双保险。

**文档级端点的权限**（`_doc_access`）：只有 `doc_id`，先取文档、反查所属库、
再走与 `/kb/{kb_id}` 同一套 `assert_kb_access` 语义——保证"删单文档"与"删库"的越权
判定完全一致，不会出现两套规则。

### 越权端到端测试（`tests/test_api_kb.py`）

把决策 14 的"404 vs 403"分界落成可执行的用例：他人私有库 → 404、公开库非 owner 写 →
403、普通用户调 `/kb/admin/all` → 403。权限从"口头承诺"变成回归测试守得住的东西。

## 4. 备选及为什么不选

| 备选 | 为什么不选 |
|---|---|
| 上传同步处理完再返回 | 大文档解析 + 向量化几十秒，用户干等；且阻塞事件循环的线程模型不堪并发 |
| 用 Celery / RQ 做队列 | 单机演示规模引入 Redis + worker 进程是过度设计；BackgroundTasks 够用，换队列时只需替换任务提交处（决策 06 同款判断） |
| `await file.read()` 整体读盘 | 内存上限交给客户端，并发上传可打爆进程；流式读 + 大小上限才是安全的 |
| 计数 `+1/-1` 增量维护 | 并发交织下会漂移；重算简单且绝对正确 |
| 文档删除只删 documents 行 | chunks 会成孤儿数据；靠外键 `ON DELETE CASCADE` 级联删，BM25 指纹自动失效 |
| pipeline 抛异常让 FastAPI 记日志 | 后台任务无人接异常，文档永远卡 pending；落 failed 才可观测 |

## 5. 为什么这样最好

**把"后台任务"和"可观测性"绑在一起**：上传接口返回 doc_id 后，前端轮询状态直到
终态（ready/failed），失败原因在 `error` 字段里、tooltip 可见。用户不必理解后台在
干什么，但任何失败都能看到一句话原因，而不是沉默的 pending。同时**存储层返回纯
dict、SQL 全参数绑定**，与检索层风格一致（决策 12），单元测试用真实临时库跑 CRUD、
API 测试用 TestClient 跑端到端越权，两层都不需要 mock 数据库。

## 6. 已知取舍

1. **BackgroundTasks 在响应返回后同步执行**，不是独立 worker：单个超大文档入库会占用
   一个请求周期的线程。演示规模够用；真要隔离要用 Celery/RQ（明确不做，见 §4）。
2. **临时文件落在系统 temp 目录**（`labor_kb_uploads/`），进程崩溃会残留；`finally`
   覆盖正常路径，异常退出才可能漏，属于可接受的运维噪音。
3. **reindex 只重算向量、不重新解析原文件**：原文件入库后就删了。要"换切分参数重切"
   需重新上传（前端提示已说明）。
4. **`list_kbs` 与 `visible_kb_ids` 各自维护同一套可见性 WHERE**：两处若漂移会看到
   "列表里有但检索不到"的库。当前用同一构造约束，但仍是隐患（注释已标明）。
