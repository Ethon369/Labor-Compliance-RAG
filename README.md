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
│                        客户端（浏览器 / curl）                     │
│                    POST /chat (SSE 流式)  GET /chat/history      │
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
│  ┌─────────────────────┐  ┌──────────────────────────────────┐  │
│  │ laws (法律元信息)     │  │ articles (205条条款级向量)        │  │
│  │ 《劳动法》107条       │  │ law_id │ article_no │ text       │  │
│  │ 《劳动合同法》98条    │  │        │ embedding (1024d)       │  │
│  └─────────────────────┘  └──────────────────────────────────┘  │
│  检索：pgvector 余弦近邻 (<=>) + BM25 倒排索引（进程内缓存）       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 项目阶段（7 步交付，141 测试全绿）

| 阶段 | 内容 | 测试数 | 状态 |
|---|---|---|---|
| 1 | 法条全文获取 + 条款级切分入库（PostgreSQL + pgvector） | 32 | ✅ |
| 2 | 混合检索：pgvector 向量路 + BM25 稀疏路 + RRF 融合 | 14 (离线) + 5 (DB) | ✅ |
| 3 | bge-reranker 精排 + 50 问评测框架（Recall@k / HitRate@k） | 14 (eval) + 10 (rerank) | ✅ |
| 4 | LangGraph Agent（5 节点 StateGraph + 条件路由 + 引用核验 + 拒答分支）| 20 | ✅ |
| 5 | 经济补偿金计算器 + ToolRegistry + ToolCallingAnswerer（function calling）| 28 | ✅ |
| 6 | FastAPI + SSE 流式 /chat + /chat/history + 统一异常处理 | 18 | ✅ |
| 7 | Dockerfile + docker-compose 双容器部署 + README 架构文档 | — | ✅ |

**全量测试：`pytest tests/ -v` → 141 passed**

---

## 评测结果（阶段 3，50 问 · top-8，离线占位）

| 方法 | Recall@8 | HitRate@8 |
|---|---|---|
| 纯向量 | 0.635 | 0.679 |
| 纯BM25 | 0.789 | 0.830 |
| 混合(RRF) | 0.742 | 0.793 |
| **混合+rerank** | **0.815** | **0.849** |

> 混合(RRF) < 纯BM25，但加 rerank 后全场最高——这是"召回 vs 排序"分离的教科书演示。
> 原因：离线占位向量路带入噪音 → RRF 位置被噪音挤占 → rerank 按语义覆盖率把好条文
> 从深水位拉回可见区。详见 [docs/decisions/04](docs/decisions/04-rerank与评测.md)。

复跑评测：`python eval/runner.py --top-k 8`

---

## 技术栈（锁定清单）

| 层次 | 技术选型 |
|---|---|
| 语言/框架 | Python 3.11+ · FastAPI 0.115+ · Pydantic v2 |
| Agent 编排 | LangGraph 1.2+（StateGraph，仅用编排层，不用 LangChain 组件） |
| 向量库 | PostgreSQL 16 + pgvector |
| 稠密检索 | bge-m3（硅基流动 API / 离线 HashEmbedder 占位） |
| 稀疏检索 | rank-bm25（进程内索引，205 条语料够用） |
| 融合 | RRF（Reciprocal Rank Fusion，按排序位置融合） |
| 精排 | bge-reranker-v2-m3（硅基流动 API / 离线 TermOverlapReranker 占位） |
| LLM | OpenAI 兼容接口（DeepSeek / Qwen 均可）· urllib 直连 |
| 部署 | Docker Compose（app + postgres 双容器） |

---

## 快速上手

### 方式一：Docker Compose（推荐，无需装 Python/PostgreSQL）

```bash
# 1) 克隆
git clone https://gitee.com/ren-youwen/Labor-Compliance-RAG.git
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

# 打开浏览器：http://localhost:8000/docs → Swagger 交互式文档

# 6) 停止（数据不丢）
docker compose down
```

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
  main.py               # 入口：lifespan 装配 + CORS + 异常 handler
  api/                  # 路由层（阶段 6）
    routes.py           # POST /chat (SSE) + GET /chat/history
    models.py           # ChatRequest / SseChunk / ErrorResponse
  agent/                # LangGraph 流程编排（阶段 4）
    graph.py            # build_agent / build_agent_from_config
    nodes.py            # 5 节点 + 条件边路由
    state.py            # AgentState (TypedDict) + AgentAnswer (Pydantic)
    protocol.py         # 三大协议 + 离线实现 + LLMRewriter + LLMAnswerer
                        # + ToolCallingAnswerer（function calling 闭环）
    verify.py           # CitationVerifier 纯规则核验
  rag/                  # 检索（阶段 2-3）
    embedder.py         # Embedder 协议 + HashEmbedder + SiliconFlowEmbedder
    bm25.py             # rank-bm25 封装
    vectorstore.py      # PgVectorStore：建表/灌向量/余弦近邻
    retriever.py        # HybridRetriever：两路检索 + RRF 融合 + rerank 精排
    fusion.py           # RRF 融合算法
    reranker.py         # Reranker 协议 + TermOverlapReranker + SiliconFlowReranker
    tokenizer.py        # 中文双字粒度分词（BM25 + 词元重叠共用）
    models.py           # ArticleRef / FusedHit / RetrievalResult (Pydantic)
  tools/                # Function Calling 工具（阶段 5）
    compensation.py     # 经济补偿金/赔偿金计算器（第47条+第87条）
    registry.py         # ToolRegistry：注册 + 导出 Schema + 按名分发
  core/                 # 配置（环境变量 + Pydantic Settings）
    config.py

scripts/                # 数据管道（阶段 1，解析脚本纯标准库零依赖）
  fetch_laws.py         # 从 npc.gov.cn 抓取→解析→自校验→输出 JSON
  ingest_laws.py        # JSON → 建表 → upsert 入库
  backfill_embeddings.py # 给 articles 灌 embedding 向量（幂等）

data/raw/               # 解析产物（git 提交）
  labor_law.json        # 《劳动法》2018修正，107条
  labor_contract_law.json # 《劳动合同法》2012修正，98条

eval/                   # 评测（阶段 3）
  questions.py          # 50 条劳动争议场景问答集 + 多标注
  metrics.py            # Recall@k / HitRate@k 双指标
  runner.py             # python eval/runner.py --top-k 8 复跑对比表

tests/                  # 141 用例，全离线可跑（DB 集成测试标 @pytest.mark.db）
  test_fetch_laws.py    # 数据解析 + 自校验
  test_ingest.py        # 入库（DB 集成）
  test_rag_offline.py   # 检索离线全链路 + 融合 + RRF
  test_rag_db.py        # 检索真库验证（DB 集成）
  test_reranker.py      # rerank 两实现 + 精排管道
  test_eval.py          # 评测集自校验 + 指标计算
  test_agent.py         # Agent 5 节点 + 拒答 + LLM mock
  test_tools.py         # 补偿金计算器 + Registry + tool_calling mock
  test_api.py           # SSE 帧解析 + history + 异常处理

docs/
  PRD.md                # 产品需求文档
  decisions/            # 技术决策记录（面试弹药库）
    01-法条数据获取与条款切分.md
    02-入库表结构与条款级切分.md
    03-混合检索与RRF融合.md
    04-rerank与评测.md
    05-agent.md
    06-tools.md
    07-api.md
    08-docker部署.md

Dockerfile              # 应用镜像
docker-compose.yml      # app + postgres 双容器编排
.env.example            # 环境变量模板（.env 不入库）
```

---

## 技术决策清单（7 篇，面试前必读）

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