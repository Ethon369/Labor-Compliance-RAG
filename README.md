

# 劳动与社会保障 RAG 问答平台 · Labor Compliance RAG

面向中国**劳动与社会保障**法律域的检索增强生成（RAG）问答平台，内置 13 部法律法规、805 条条文，并支持上传任意法律文档构建私有知识库。输入一个真实的劳动争议场景（如“公司拖欠加班费怎么维权”），系统从法律原文中检索相关条款，给出**带条款级引用、可溯源**的回答；当检索证据不足以支撑结论时，主动**拒答**而不是编造。

> **定位**：辅助参考工具，不构成法律意见——涉及具体纠纷请咨询执业律师。

---

## 核心特性

* **混合检索引擎**：融合 pgvector 语义向量（bge-m3）与 BM25 词面匹配，通过 RRF 排序与 bge-reranker 精排，确保高召回与高准确率。
* **智能 Agent 编排**：基于 LangGraph StateGraph 的多阶段流程（意图改写 -> 检索 -> 核验 -> 生成/拒答），内置引用合规性校验与越界拒答机制。
* **函数调用工具**：内置《劳动合同法》第 47/87 条经济补偿金计算器，支持 LLM 主动调用工具进行数值推理。
* **多知识库平台 (V2)**：支持用户自定义知识库（PDF/Word/Markdown），文档解析、切片、向量化全链路自动化。
* **完整对话体验**：SSE 流式输出、登录注册、多轮会话上下文、角色权限管理（普通用户/管理员）。
* **工程健壮性**：离线零依赖可跑评测与单元测试；Docker 一键部署；PostgreSQL + pgvector 成熟存储。

---

## 技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         客户端层                                  │
│   浏览器聊天页 (SSE 流式) / curl / Swagger (/docs)               │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP POST / SSE
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FastAPI 应用层 (:8000)                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │            LangGraph StateGraph (Agent 编排)               │  │
│  │  START → rewrite ──→ retrieve ──→ verify ──┬─→ answer     │  │
│  │     (LLM改写)    (混合检索)     (纯规则核验) │  (LLM回答)    │  │
│  │                                      └─→ refuse (拒答)    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                   │
│  核心 Protocol 能力（可插拔实现）：                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Embedder │  │ Reranker │  │ Rewriter │  │ Answerer │       │
│  │ bge-m3   │  │ bge-v2   │  │ LLM/Pass │  │ LLM/Tool │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└───────────────────────────┬─────────────────────────────────────┘
                            │ psycopg3 / psycopg_pool
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                数据持久层 (PostgreSQL 16 + pgvector)             │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  三层知识库模型 (V2)：                                        │  │
│  │  kb (知识库) → documents (文档元数据) → chunks (切片+向量)    │  │
│  │  + users + chat_sessions/chat_messages                      │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 技术栈清单

| 层级 | 技术选型 |
| :--- | :--- |
| **语言 / 框架** | Python 3.11+ · FastAPI 0.115+ · Pydantic v2 |
| **Agent 编排** | LangGraph 1.2+ (StateGraph，仅使用编排层) |
| **向量库** | PostgreSQL 16 + pgvector (HNSW 索引) |
| **稠密检索** | bge-m3 (SiliconFlow API / 离线 Hash 占位) |
| **稀疏检索** | rank-bm25 (进程内索引，作用域按 KB 缓存) |
| **排序融合** | RRF (Reciprocal Rank Fusion) |
| **精排** | bge-reranker-v2-m3 (SiliconFlow API / 离线 Term 占位) |
| **LLM 接口** | OpenAI 兼容 (DeepSeek / Qwen) · urllib 直连 |
| **文档解析** | pypdf (PDF) + python-docx (Word) |
| **连接池** | psycopg_pool (连接复用) |
| **部署** | Docker Compose (双容器) |

---

## 快速上手

### 方式一：Docker Compose（推荐）

无需本地安装 Python 或 PostgreSQL，一条命令启动。

```bash
# 1) 克隆仓库
git clone https://gitee.com/ren-youwen/Labor-Compliance-RAG.git
cd Labor-Compliance-RAG

# 2) 配置环境 (可选 - 不配置则使用离线模式)
cp .env.example .env
# 编辑 .env 填入 SiliconFlow API Key 和 LLM Key

# 3) 启动服务 (app + postgres)
docker compose up -d

# 4) 初始化法条数据 (首次运行只需一次)
docker compose exec app python scripts/ingest_laws.py

# 5) 验证与对话
# 浏览器访问：http://localhost:8000
# 命令行测试：
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"公司延长我工作时间，加班费怎么算"}'
```

### 方式二：本地开发环境

已安装 Python 3.11+ 和 PostgreSQL。

```bash
# 1) 安装依赖
pip install -e ".[dev]"

# 2) 启动数据库 (或使用本地已有实例)
docker compose up -d postgres

# 3) 配置
cp .env.example .env

# 4) 初始化数据
python scripts/ingest_laws.py

# 5) 运行测试
pytest tests/ -v

# 6) 启动服务
uvicorn app.main:app --reload
```

---

## 项目结构

```text
Labor-Compliance-RAG/
├── app/                       # FastAPI 应用主目录
│   ├── main.py                # 入口 (中间件、静态文件、线程池配置)
│   ├── api/                   # 路由层
│   │   ├── auth.py            # 注册/登录/Token/PBKDF2 哈希
│   │   ├── admin_routes.py    # 管理员接口 (用户管理)
│   │   ├── kb_routes.py       # 知识库 CRUD + 文档上传
│   │   ├── routes.py          # /chat (SSE) + 历史记录
│   │   └── deps.py            # 依赖注入 (CurrentUser, require_admin)
│   ├── agent/                 # LangGraph Agent 编排
│   │   ├── graph.py           # StateGraph 构建
│   │   ├── nodes.py           # 节点逻辑 (改写/检索/核验/回答/拒答)
│   │   └── protocol.py        # 能力接口 (AnswerGenerator, Retriever)
│   ├── rag/                   # 检索增强模块
│   │   ├── retriever.py       # 混合检索 (HybridRetriever)
│   │   ├── embedder.py        # 向量化协议与实现
│   │   ├── bm25.py            # BM25 索引
│   │   └── fusion.py          # RRF 融合算法
│   ├── kb/                    # 知识库领域 (V2)
│   │   ├── schema.py          # 表结构初始化
│   │   ├── parser.py          # PDF/DOCX/TXT 解析
│   │   ├── splitter.py        # 递归切分逻辑
│   │   └── store.py           # 知识库与文档的 CRUD
│   ├── tools/                 # 函数调用工具
│   │   ├── compensation.py    # 经济补偿金计算器
│   │   └── registry.py        # 工具注册表
│   └── static/                # 前端页面 (纯静态)
│       ├── index.html         # 问答界面
│       ├── kb.html            # 知识库管理
│       └── login.html         # 登录注册
├── scripts/                   # 数据管道脚本
│   ├── fetch_laws.py          # 从官网抓取法条
│   ├── ingest_laws.py         # 法条入库
│   └── migrate_default_kb.py  # 默认库迁移脚本
├── eval/                      # 评测框架
│   ├── runner.py              # 评测入口
│   └── questions.py           # 50问评测集
├── tests/                     # 测试用例 (266 passed)
├── docs/                      # 技术决策记录 (19篇) & PRD
├── Dockerfile                 # 应用容器构建
└── docker-compose.yml         # 开发/演示环境编排
```

---

## 评测结果

在 50 问评测集（Top-8）上，使用 SiliconFlow 真模型（bge-m3 + bge-reranker-v2-m3）、语料为全部 805 条条文的成绩：

| 检索方法 | Recall@8 | HitRate@8 |
| :--- | :---: | :---: |
| 纯 BM25 | 0.733 | 0.793 |
| 纯向量 (bge-m3) | 0.921 | 0.962 |
| 混合 (RRF) | 0.893 | 0.943 |
| **混合 + Rerank** | **0.937** | **0.981** |

> **结论**：语料扩到 800+ 条后，单路检索的召回压力明显上升（BM25 0.733），而 **rerank 精排的收益更加突出**——混合 + rerank 以 0.937 Recall@8 领先纯向量与单路 RRF，HitRate 稳定在 0.981。
> 评测命令：`python eval/runner.py --top-k 8`

---

## 核心设计原则

1.  **能力对象注入 (Protocol Oriented)**：检索、改写、回答、Rerank 均通过 Protocol 注入。Graph 对“能力是谁”无感知，离线占位和真 API 无缝切换。
2.  **离线可复现 (Offline Reproducible)**：零 API Key、零 DB 时全链路仍可跑测试。关键路径必须覆盖离线场景，确保 CI/Gitpod 快速演示可行。
3.  **分层防错 (Layered Defensive)**：数据层（哨兵词校验）、检索层（参数绑定）、Agent 层（规则核验）、API 层（SSE error 帧），每层不依赖下一层的“假设正确”。

---

## 交流与贡献

* 本项目遵循 MIT 许可证。
* 提交代码前请运行 `pytest tests/` 确保全量用例通过。
* 如有疑问，欢迎在 Issues 中讨论技术细节。

**免责声明**：本项目为技术展示，数据与回答仅供学习参考，不构成法律意见；涉及具体纠纷请咨询执业律师。