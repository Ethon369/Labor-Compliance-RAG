# 劳动争议智能合规助手 · Labor Compliance RAG

面向中国劳动法体系的 **检索增强生成（RAG）问答系统**。输入一个劳动争议场景问题
（如"公司拖欠加班费怎么维权"），系统从《劳动法》《劳动合同法》原文中检索相关条款，
给出**带准确条文引用的回答**。

> **定位**：个人求职展示项目，读者是技术面试官。定位为"辅助参考工具"，**不是法律
> 咨询系统**——回答不构成任何法律意见。

## 现在能做到哪一步

项目按阶段推进。已完成：**阶段 1 数据管道**、**阶段 2（混合检索）**、**阶段 3（rerank + 评测）**：

| 阶段 | 内容 | 状态 |
|---|---|---|
| 1 | 法条全文获取 + 条款级切分入库（PostgreSQL） | ✅ 完成 |
| 2a | 混合检索：pgvector 向量路 + BM25 稀疏路 + RRF 融合 | ✅ 完成 |
| 2b | 检索编排完整化（reranker 接入 + 精排管道） | ✅ 完成 |
| 3 | bge-reranker 精排 + 评测框架（50 问 + 召回率/命中率对比） | ✅ 完成 |
| 4 | FastAPI 服务 / LangGraph Agent 编排（问题理解 → 检索 → 引用核验） | 🔜 |

阶段 1 交付：把《劳动法》（2018 修正现行版，107 条）与《劳动合同法》（2012 修正
现行版，98 条）的全文解析成 `法名 / 条号 / 条文` 结构化数据，**一条法律条文 = 一行**
入库 PostgreSQL，管线全自动自校验。

阶段 2 交付：`app/rag/` 混合检索——同一查询走两条路各取候选，再按**排序位置**（而非
分数）做 RRF 融合：pgvector `<=>` 余弦近邻（稠密、抓语义）+ rank-bm25（稀疏、抓词面）。
全链路离线可跑可测：embedding 用确定性占位向量打通 pgvector/SQL/融合，接口预留真
bge-m3（硅基流动 API），填 key 重灌即换，表结构不动。

阶段 3 交付：**rerank 精排 + 评测框架**——RRF 融合出一个更大候选池（默认 30）后交给
reranker 成对语义精排（离线占位 / 真 bge-reranker 二选一），再把 top-k 可见结果排出来。
同时新增 `eval/` 评测套件：50 条劳动争议场景问答集，用 **Recall@k 与 Hit Rate@k** 双指标
输出「纯向量 / 纯BM25 / 混合 / 混合+rerank」对比表，实测（离线占位，top-8）：

| 方法 | Recall@8 | HitRate@8 |
|---|---|---|
| 纯向量 | 0.635 | 0.679 |
| 纯BM25 | 0.789 | 0.830 |
| 混合(RRF) | 0.742 | 0.793 |
| 混合+rerank | **0.815** | **0.849** |

`pytest` 全绿（含新增 rerank / eval 用例）；评测可 `python eval/runner.py --top-k 8` 复跑。
指标定义与「rerank 收益从哪来」的分析见 [docs/decisions/04](docs/decisions/04-rerank与评测.md)。

## 技术栈

> 锁定清单，不引入清单外依赖。详见 [docs/decisions](docs/decisions)。

- **语言/服务**：Python 3.11+ · FastAPI · Pydantic v2 · Docker Compose
- **Agent 编排**：LangGraph；仅用 LangChain 基础组件
- **存储/检索**：PostgreSQL + pgvector · rank-bm25 · bge-reranker / bge-m3
  （本地或硅基流动 API）· LLM 走 OpenAI 兼容接口（DeepSeek/Qwen）

## 三个在阶段 1 就想清楚的设计点

面试时我会从这里讲起：

1. **条款级切分，而不是固定长度切块**
   法律文本的检索单元天然是一条法条：语义原子、自带唯一地址 `(law_id, article_no)`、
   长度适配 embedding 窗口。"chunk size 怎么定"的正解是**先找内容边界再谈数值**，
   所以仓库里没有一个 chunk 字符常数。论证见 [docs/decisions/02](docs/decisions/02-入库表结构与条款级切分.md)。

2. **数据现行性防线（容易翻车的地方）**
   法律数据最大的坑不是爬不到，而是**爬到旧版镜像**——各地政府官网大量存在 2018
   修正前的《劳动法》旧文本。方案：选源 + 用**修正决定公布的条文措辞做哨兵词**
   硬断言（如劳动法第 15(2)/69/94 条必须是 2018 措辞），页面回落旧版立即红灯。
   详见 [docs/decisions/01](docs/decisions/01-法条数据获取与条款切分.md)。

3. **解析用状态机，拒绝全文正则**
   法律文本条款跨多段、末条贴着页脚噪音、条号形态不一（`第 一 条` / `第一百零七条本法自…`）。
   按块级标签切成段落流后逐段分类，每条规则可单测、可对着页面审。

## 快速上手

前置：Python 3.11+、Docker。

```bash
# 1) 依赖（psycopg 驱动 + pydantic；解析脚本本身零第三方依赖）
pip install -e ".[dev]"

# 2) 起 PostgreSQL（pgvector 镜像）
docker compose up -d

# 3) 配置连接串（默认值已与 compose 对齐）
cp .env.example .env

# 4) 用已提交的解析结果入库（会建表 + upsert，可重复执行）
python scripts/ingest_laws.py

# 5) 跑测试（75 个用例；无 DB 时 @pytest.mark.db 集成测试自动跳过，仍全绿）
pytest tests/ -v
```

如需重新从源站抓取全文：

```bash
python scripts/fetch_laws.py              # 抓《劳动法》+《劳动合同法》→ data/raw/*.json
python scripts/fetch_laws.py --check      # 只做离线自校验；不带路径默认扫 data/raw/*.json
python scripts/ingest_laws.py --dry-run   # 只打印入库计划，不碰数据库
```

## 目录结构

```
app/            # FastAPI 应用骨架；rag/ 已实现混合检索 + rerank 精排，api/core/agent/tools 待填充
  rag/          # 混合检索（阶段 2a/3）：pgvector 向量路 + BM25 + RRF 融合 + rerank 精排
scripts/
  fetch_laws.py # 抓取 + 解析 + 自校验（现行性哨兵硬断言）→ data/raw/*.json（纯标准库）
  ingest_laws.py# 校验 JSON → 建表 → 条款级 upsert 入库
  backfill_embeddings.py  # 给 articles 灌 embedding 向量（幂等；--force 全量重灌）
data/raw/       # 解析产物（git 提交）：labor_law.json(107) / labor_contract_law.json(98)
eval/           # 评测框架（阶段 3）：50 问问答集 + Recall@k/HitRate@k 指标 + 对比表
                #   python eval/runner.py --top-k 8   复跑「纯向量 vs 混合 vs 混合+rerank」
tests/          # 75 用例；fetch/rag/eval 单测离线，检索真库验收标 @pytest.mark.db
docs/
  PRD.md
  decisions/    # 01 数据获取与切分 / 02 入库表结构 / 03 混合检索与RRF融合 / 04 rerank与评测 —— 每个取舍都写了"备选 + 为什么"
```

## 沟通约定

- **对话语言**：与 Claude Code 交互尽量使用中文。代码注释、文档、技术决策记录也以中文为主（关键术语保留英文）。
- 项目作者会追问设计细节——回答时像导师带学生一样讲清楚"为什么"，不敷衍。

## 工程规范

- 配置一律环境变量 + `.env.example`，仓库里没有任何硬编码 key。
- SQL 全部参数绑定；模型数据一律 Pydantic 校验。
- 建表幂等（`CREATE TABLE IF NOT EXISTS`）、入库 upsert（`ON CONFLICT`），整文重跑安全。
- 每个核心模块配 pytest + 一篇技术决策记录（[docs/decisions](docs/decisions)）。

## 免责声明

本项目为技术展示，数据与回答仅供学习参考，不构成法律意见；涉及具体纠纷请咨询执业律师。
