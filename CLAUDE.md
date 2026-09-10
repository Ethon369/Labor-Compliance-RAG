# 项目：劳动与社会保障 RAG 问答平台（Labor Compliance RAG）

## 定位
面向中国劳动法体系的辅助参考工具，不是法律咨询系统；全链路支持离线可复现。

## 技术栈（锁定，不得擅自替换）
- Python 3.11+ / FastAPI / Pydantic v2
- LangGraph（Agent 编排）、LangChain（仅用其基础组件）
- PostgreSQL + pgvector（向量库）
- BM25：rank-bm25；Rerank：bge-reranker（本地）或硅基流动 API
- Embedding：bge-m3（本地）或 API
- LLM：OpenAI 兼容接口（DeepSeek/Qwen 均可），key 从环境变量读取
- 部署：Docker Compose（app + postgres）
- 文档解析（V2.3 新增，已获同意）：pypdf（PDF 取文字层）、python-docx（Word 段落）
- 连接池（V2.4 新增，已获同意）：psycopg_pool（替换每次查询开短连接）
- 文件上传（V2.3 新增，FastAPI 官方配套件）：python-multipart（UploadFile 解析必需）

## 硬性规则
1. 禁止引入上述清单之外的依赖；确有必要，先说明理由并征得我同意
2. 所有配置走环境变量 + .env.example，禁止硬编码任何 key
3. 每完成一个模块必须附带 pytest 测试，跑通后才算完成
4. 数据库操作一律参数绑定；API 输出一律 Pydantic 校验
5. 中文注释只写在关键逻辑处，解释"为什么"，不解释"是什么"
6. 每个核心模块完成后，输出一份 docs/decisions/NN-模块名.md 技术决策记录：
   选了什么方案、备选方案、为什么、已知取舍

## 目录结构（保持整洁）
app/            # FastAPI 应用
  api/          # 路由
  core/         # 配置、依赖注入
  rag/          # 切分、检索、rerank
  agent/        # LangGraph 流程
  tools/        # Function Calling 工具
scripts/        # 数据下载、解析、入库脚本
eval/           # 评测集与评测脚本
tests/
docs/
  decisions/    # 技术决策记录（选型理由、备选方案与已知取舍）
  PRD.md

## 验证命令
- 跑测试：pytest tests/ -v
- 启动服务：uvicorn app.main:app --reload
- 冒烟：curl http://localhost:8000/docs

## 我的工作模式
我是项目作者，需要理解每一行核心代码。写完代码后请主动讲解关键设计，
我会随时追问，请像导师带学生一样解释，不要敷衍。
