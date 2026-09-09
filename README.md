# 劳动争议智能合规助手 · Labor Compliance RAG

面向中国劳动法体系的 **检索增强生成（RAG）问答系统**。输入一个劳动争议场景问题
（如"公司拖欠加班费怎么维权"），系统从《劳动法》《劳动合同法》原文中检索相关条款，
给出**带准确条文引用的回答**。

> **定位**：个人求职展示项目，读者是技术面试官。定位为"辅助参考工具"，**不是法律
> 咨询系统**——回答不构成任何法律意见。

---

## 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│       客户端（浏览器聊天页 / curl）                                │
│   GET / → app/static/index.html（静态页，POST /chat SSE 流式）   │
│   /chat/history（会话历史）  /docs（Swagger）                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │  text/event-stream
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI · app 容器 :8000                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              LangGraph StateGraph (Agent 编排)              │  │
│  │                                                             │  │
│  │  START → rewrite ──→ retrieve ──→ verify ──┬─→ answer → END│  │
│  │     (LLM改写)    (hybrid+rerank)  (核验)   │  (LLM回答)     │  │
│  │                                           └─→ refuse → END │  │
│  │                                              (拒答兜底)     │  │
│  │                                                             │  │
│  │  answer 节点内置 function calling 闭环：                     │  │
│  │      LLM tool_calls → ToolRegistry.call() → 结果回传 LLM   │  │
│  │      当前注册：经济补偿金计算器（《劳动合同法》第47/87条）     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                   │
│  能力对象（Protocol 注入，离线/线上可切换）：                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ embedder │  │ reranker │  │ rewriter      │  │ answerer    │  │
│  │ bge-m3   │  │ bge-     │  │ PassThrough / │  │ Template /  │  │
│  │ / Hash   │  │ reranker │  │ LLMRewriter   │  │ LLMAnswerer │  │
│  │          │  │ / Term   │  │               │  │ / ToolCall  │  │
│  └──────────┘  └──────────┘  └──────────────┘  └─────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │  psycopg3
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              PostgreSQL + pgvector · postgres 容器 :5432         │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  三层知识库模型（V2）：kb → documents → chunks              │  │
│  │  ┌──────────┐  ┌──────────────┐  ┌──────────────────────┐  │  │
│  │  │ kb       │  │ documents    │  │ chunks               │  │  │
│  │  │ 默认库=   │  │ 上传/法条迁移 │  │ 条款级切片 + 向量      │  │  │
│  │  │ 205法条   │  │ status/error  │  │ embedding (1024d)     │  │  │
│  │  └──────────┘  └──────────────┘  └──────────────────────┘  │  │
│  │  + users(角色/禁用) + chat_sessions/chat_messages(多轮)      │  │
│  └────────────────────────────────────────────────────────────┘  │
│  检索：pgvector 余弦近邻 (<=>) + BM25 倒排索引（进程内缓存，作用域按 kb） │
│  连接池：psycopg_pool（按 dsn 缓存，替换短连接）                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 项目阶段（V1 十步 + V2 升级，266 passed + 5 conditionally-skipped）

### V1 · 单语料 RAG 内核（10 步，155 测试）

| 阶段 | 内容 | 测试数 | 状态 |
|---|---|---|---|
| 1 | 法条全文获取 + 条款级切分入库（PostgreSQL + pgvector） | 32 | ✅ |
| 2 | 混合检索：pgvector 向量路 + BM25 稀疏路 + RRF 融合 | 14 (离线) + 5 (DB) | ✅ |
| 3 | bge-reranker 精排 + 50 问评测框架（Recall@k / HitRate@k） | 14 (eval) + 10 (rerank) | ✅ |
| 4 | LangGraph Agent（5 节点 StateGraph + 条件路由 + 引用核验 + 拒答分支）| 20 | ✅ |
| 5 | 经济补偿金计算器 + ToolRegistry + ToolCallingAnswerer（function calling）| 28 | ✅ |
| 6 | FastAPI + SSE 流式 /chat + /chat/history + 统一异常处理 | 18 | ✅ |
| 7 | Dockerfile + docker-compose 双容器部署 + README 架构文档 | — | ✅ |
| 8 | 静态聊天前端 + LLM 真接入 + 拒答边界重构 | 143 | ✅ |
| 9 | 前端 B-2 美化：浅灰侧边栏 + 白色消息区 + 引用中文名 | — | ✅ |
| 10 | 认证与会话历史：登录/注册 + 持久化多轮会话（同名会话续问带上下文） | 12 | ✅ |

### V2 · 从单语料 RAG 演进到多知识库平台（266 passed + 5 skipped）

| 阶段 | 内容 | 测试数 | 状态 |
|---|---|---|---|
| V2.1 | 数据层三层模型（kb→documents→chunks）+ 205 条法条迁移为默认库 + 引用片段/角标 | — | ✅ |
| V2.2 | 角色权限（user/admin）+ 改密码 + admin 初始化 + 越权语义（404/403 分界） | 15 (admin) | ✅ |
| V2.3 | 知识库管理全链路：文档解析（pdf/docx/txt/md）→ 递归切分 → 向量化 → 单事务落库 + CRUD/上传/轮询 API + 前端四件套（登录/知识库工作区/首页 KB 选择器） | 78 | ✅ |
| V2.4 | 性能工程收口：连接池（测试耗时 21.6s→6.6s）、HNSW 索引、GZip/静态缓存、令牌桶限流、分页、X-Request-Id 观测、线程池、嵌入去重 | 14 | ✅ |

**全量测试：`pytest tests/ -v` → 266 passed + 5 skipped（真模型库下离线检索断言自动跳过，见决策 19）**

> 核心资产 100% 复用：RAG 内核（混合检索 + RRF + rerank + Agent + 评测框架）一行不动，
> 只把"数据层"从固定法条换成可上传知识库、再加权限层——这是"单语料 RAG → 多知识库
> RAG"的演进，而非推倒重来。详见各阶段决策记录。

---

## 评测结果（50 问 · top-8，真模型基线）

> 默认评测已切换到**真模型**（硅基流动 bge-m3 embedding + bge-reranker-v2-m3）。
> 需要真 key（`.env` 配 `SILICONFLOW_API_KEY` + `EMBEDDING_MODE/RERANK_MODE=siliconflow`）
> 并重灌默认库向量后复跑；无 key 时回退离线占位（数字见决策 19 对比表）。

| 方法 | Recall@8 | HitRate@8 |
|---|---|---|
| 纯向量 | 0.969 | 1.000 |
| 纯BM25 | 0.789 | 0.830 |
| 混合(RRF) | 0.928 | 0.962 |
| **混合+rerank** | **0.965** | **0.981** |

> 真模型下的核心结论：**语义向量把纯向量召回拉到 0.97（BM25 的 1.2 倍），RRF 融合
> 两条互补的路（语义+词面）后 0.93，rerank 精排再提到 0.97 全场最高**。
> 早期"混合 < 纯BM25"的结论是**离线占位向量的假象**（占位向量≈词面匹配、带噪），
> 真模型下已不成立——对比细节见 [决策 19](docs/decisions/19-评测真实化.md)。

复跑评测：`python eval/runner.py --top-k 8`

---

## 技术栈（锁定清单）

| 层次 | 技术选型 |
|---|---|
| 语言/框架 | Python 3.11+ · FastAPI 0.115+ · Pydantic v2 |
| Agent 编排 | LangGraph 1.2+（StateGraph，仅用编排层，不用 LangChain 组件） |
| 向量库 | PostgreSQL 16 + pgvector（HNSW 索引，规模化预留） |
| 稠密检索 | bge-m3（硅基流动 API / 离线 HashEmbedder 占位） |
| 稀疏检索 | rank-bm25（进程内索引，作用域按知识库缓存） |
| 融合 | RRF（Reciprocal Rank Fusion，按排序位置融合） |
| 精排 | bge-reranker-v2-m3（硅基流动 API / 离线 TermOverlapReranker 占位） |
| LLM | OpenAI 兼容接口（DeepSeek / Qwen 均可）· urllib 直连 |
| 文档解析 | pypdf（PDF 文字层）+ python-docx（Word 段落） |
| 连接池 | psycopg_pool（按 dsn 缓存，替换每次查询短连接） |
| 部署 | Docker Compose（app + postgres 双容器） |

---

## 快速上手

### 方式一：Docker Compose（推荐，无需装 Python/PostgreSQL）

```bash
# 1) 克隆
git clone https://github.com/Ethon369/Labor-Compliance-RAG.git
cd Labor-Compliance-RAG

# 2) 配置（可选——离线模式可跳过）
cp .env.example .env
# 编辑 .env：填 API key（siliconflow / LLM），不改就默认离线

# 3) 启动系统（app + postgres 双容器）
docker compose up -d

# 4) 首次入库（只需一次）
docker compose exec app python scripts/ingest_laws.py

# 5) 冒烟
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"公司延长我工作时间，加班费怎么算"}' \
  --no-buffer

# 打开浏览器：http://localhost:8000 → 聊天界面（SSE 流式问答）
#            http://localhost:8000/docs → Swagger 交互式文档

# 6) 停止（数据不丢）
docker compose down
```

### 管理员账号与多用户（V2）

- **默认管理员**：`admin / 123456`（可经 `.env` 的 `ADMIN_USERNAME` / `ADMIN_PASSWORD`
  修改；首次启动自动创建，**绝不覆盖已改过的密码**）。
- **普通用户**：打开浏览器 `http://localhost:8000/login.html` 注册即可。
- **权限模型**：普通用户管理自己的知识库、对公开库问答；**admin 额外有"平台管理"
  视图**（`/kb.html?view=admin`）可管全部库与用户。每个管理接口后端独立校验角色，
  前端隐藏入口 ≠ 安全。
- **知识库问答**：登录后在首页 header 的知识库选择器切换库；问答回答会带
  `[n]` 角标 + 可展开的引用片段卡片（文档名 / 第 N 段 / 相关度）。
- **上传格式**：pdf / docx / txt / md，单文件 ≤ 10MB。上传后自动解析 → 切分 →
  向量化（后台异步，前端轮询进度），期间可离开页面。**扫描版 PDF 无文字层不支持**。

### 方式二：本机直连（已有 PostgreSQL + Python 3.11+）

```bash
# 1) 依赖
pip install -e ".[dev]"

# 2) 起数据库
docker compose up -d postgres  # 或用本地 PG

# 3) 配置
cp .env.example .env

# 4) 入库
python scripts/ingest_laws.py

# 5) 跑测试
pytest tests/ -v

# 6) 启动服务
uvicorn app.main:app --reload

# 7) 冒烟
curl http://localhost:8000/docs
```

---

## 目录结构

```
app/                    # FastAPI 应用
  main.py               # 入口：lifespan 装配 + GZip/静态缓存/X-Request-Id 中间件 + 线程池
  api/                  # 路由层
    auth.py             # 注册/登录/改密码 + 签名 token + PBKDF2 哈希 + 登录防爆破
    admin_routes.py     # 管理员：用户列表/升降级/重置密码/禁用
    kb_routes.py        # 知识库 CRUD + 文档上传(流式落盘/10MB)/轮询/reindex
    routes.py           # POST /chat (SSE) + 会话/历史(分页)
    deps.py             # require_admin / require_kb_access（越权 404/403 语义）
    models.py           # ChatRequest / AuthRequest / KbBrief / CitationBrief / ...
  static/               # 前端四件套（纯静态 HTML+CSS+JS，零构建）
    index.html          # 问答页：SSE 流式 + KB 选择器 + 引用卡片/角标
    login.html          # 登录/注册/改密码
    kb.html             # 知识库工作区 + 平台管理视图（admin）
    common.js           # 多页共用的 token/fetch 封装
  kb/                   # 知识库领域（V2）
    schema.py           # ensure_kb_schema（三层表 DDL）
    parser.py           # pdf/docx/txt/md → Block 流（页码+标题标记）
    splitter.py         # 递归切分：段落→句子→硬切，不劈句子
    store.py            # kb/documents/chunks CRUD + 计数重算 + 尾片清理
    pipeline.py         # 解析→切分→向量化→单事务落库（异常落 failed + 嵌入去重）
  agent/                # LangGraph 流程编排
    graph.py / nodes.py / state.py / protocol.py / verify.py
  rag/                  # 检索
    embedder.py         # Embedder 协议 + build_embedder 工厂 + Hash/SiliconFlow
    bm25.py / vectorstore.py / retriever.py / fusion.py / reranker.py / tokenizer.py / models.py
  tools/                # Function Calling 工具（经济补偿金计算器）
    compensation.py / registry.py
  core/                 # 基础设施
    config.py           # 环境变量 + Pydantic Settings
    db.py               # psycopg_pool 连接池（按 dsn 缓存）
    ratelimit.py        # 内存令牌桶限流 + 登录失败小窗口

scripts/                # 数据管道（纯标准库 / 独立脚本）
  fetch_laws.py         # 从 npc.gov.cn 抓取→解析→自校验→输出 JSON
  ingest_laws.py        # JSON → 建表 → upsert 入库（法条迁移为默认库）
  backfill_embeddings.py # 给切片灌 embedding 向量（幂等）
  ensure_hnsw.py        # 幂等建 HNSW 向量索引（建前建后跑 eval 对比）

data/raw/               # 解析产物（git 提交）
  labor_law.json / labor_contract_law.json

eval/                   # 评测
  questions.py / metrics.py / runner.py / compat.py

tests/                  # 266 用例 + 5 条件跳过，离线可跑（DB 集成测试标 @pytest.mark.db）
  test_fetch_laws.py / test_ingest.py / test_rag_offline.py / test_rag_db.py
  test_reranker.py / test_eval.py / test_agent.py / test_tools.py / test_api.py
  test_auth.py / test_admin.py / test_kb.py / test_kb_schema.py
  test_parser.py / test_splitter.py / test_pipeline.py / test_api_kb.py
  test_db.py / test_ratelimit.py

docs/
  PRD.md                # 产品需求文档
  decisions/            # 技术决策记录（面试弹药库，18 篇）
    01-法条数据获取与条款切分.md ... 18-性能工程收口.md

Dockerfile / docker-compose.yml / .env.example / pyproject.toml / CLAUDE.md
```

---

## 技术决策清单（19 篇，面试前必读）

| # | 标题 | 核心决策 |
|---|---|---|
| 01 | [法条数据获取与条款切分](docs/decisions/01-法条数据获取与条款切分.md) | 选 npc.gov.cn 为源 + 哨兵词硬断言现行性防线 |
| 02 | [入库表结构与条款级切分](docs/decisions/02-入库表结构与条款级切分.md) | 条款级切分（非固定长度 chunk）+ (law_id, article_no) 双键 |
| 03 | [混合检索与RRF融合](docs/decisions/03-混合检索与RRF融合.md) | 双路各取候选 → RRF 按排序位置融合（非分数融合）|
| 04 | [rerank与评测](docs/decisions/04-rerank与评测.md) | 初筛+精排两段式 + Recall/HitRate 双指标 |
| 05 | [Agent编排](docs/decisions/05-agent.md) | Graph 不用 Chain（条件分叉） + 纯规则核验 + urllib 直连 |
| 06 | [Function Calling](docs/decisions/06-tools.md) | 手写 JSON Schema + ToolCallingAnswerer 内循环 |
| 07 | [流式API](docs/decisions/07-api.md) | SSE 不用 WebSocket + 流内异常用 error 帧 |
| 08 | [Docker部署](docs/decisions/08-docker部署.md) | single-stage 构建 + compose 双容器 + 环境变量注入 |
| 09 | [拒答边界与LLM接入](docs/decisions/09-拒答边界与LLM接入.md) | verify_mode 命中即答 + LLM 自己判断拒答，替换规则 top-1 阈值拦截 |
| 10 | [认证与会话历史](docs/decisions/10-认证与会话历史.md) | 自制 HMAC token + PBKDF2 哈希 + PostgreSQL 持久化 |
| 11 | [多知识库三层模型与默认库迁移](docs/decisions/11-多知识库三层模型与默认库迁移.md) | kb→documents→chunks + 205 法条迁移为默认库，检索只走 chunks 一套路径 |
| 12 | [检索层chunks化与BM25作用域缓存](docs/decisions/12-检索层chunks化与BM25作用域缓存.md) | 检索按 kb 作用域 + BM25 指纹缓存失效 |
| 13 | [引用编号与snippet字段](docs/decisions/13-引用编号与snippet字段.md) | `[n]` 角标 + snippet(截断300字)/score/doc_title |
| 14 | [角色权限与越权语义](docs/decisions/14-角色权限与越权语义.md) | 两级角色 + 越权 404（私有）vs 403（公开非owner） |
| 15 | [文档解析与切分](docs/decisions/15-文档解析与切分.md) | 两阶段职责（parser 判标题 / splitter 只消费）+ 不劈句子 |
| 16 | [知识库CRUD与上传流水线](docs/decisions/16-知识库CRUD与上传流水线.md) | 上传只登记+后台任务，异常落 failed 不抛 |
| 17 | [连接池与HNSW向量索引](docs/decisions/17-连接池与HNSW向量索引.md) | psycopg_pool 替换短连接 + HNSW 用 EXPLAIN 验证"当前不生效" |
| 18 | [性能工程收口](docs/decisions/18-性能工程收口.md) | GZip/静态缓存/限流/分页/观测/线程池/嵌入去重 |
| 19 | [评测真实化](docs/decisions/19-评测真实化.md) | 占位 vs 真 bge-m3 对比：纯向量 0.64→0.97，"混合<BM25"是占位假象 |

---

## 三个贯穿全链路的设计原则

面试官问我"你项目怎么做技术选型的"，我按这三个原则回答：

1. **能力对象从参数进，不从代码里 new**——检索/改写/回答/Rerank 都以 Protocol 注入，
   离线占位和真 API 在同一接口下切换，Graph 对"能力是谁"无感知。

2. **离线可复现是底线**——零 key、零 DB 时全链路仍可跑、可测。求职演示与 CI 都靠它。
   嵌入层用确定性哈希投影、rerank 用词元覆盖率、LLM 用模板回答——真能力填 key 即换。

3. **每层都有自己的防错边界**——数据层有哨兵词现行性断言 + 自校验；检索层有参数绑定 +
   Pydantic 校验；Agent 层有纯规则核验 + 拒答兜底；API 层有全局异常 handler + SSE error 帧。
   每一层不依赖下一层的"假设正确"。

---

## 沟通约定

- **对话语言**：中文。代码注释、文档、技术决策记录以中文为主，关键术语保留英文。
- 项目作者会追问设计细节——回答时讲清楚"为什么"，不敷衍。

## 工程规范

- 配置一律环境变量 + `.env.example`，仓库无硬编码 key。
- SQL 全部参数绑定；模型数据一律 Pydantic 校验。
- 建表幂等、入库 upsert，整文重跑安全。
- **每个核心模块配 pytest + 一篇技术决策记录**。

## 免责声明

本项目为技术展示，数据与回答仅供学习参考，不构成法律意见；涉及具体纠纷请咨询执业律师。