# 08 · Docker 部署与交付

> 阶段 7 决策记录。对应 `Dockerfile`、`docker-compose.yml`、`.dockerignore`、`README.md`。
> 格式约定（CLAUDE.md 第 6 条）：选了什么、备选、为什么、已知取舍。

## 1. 一句话结论

**用 Docker Compose 编排 app + postgres 双容器，一句 `docker compose up -d` 拉起整个系统。** 不引入 Kubernetes（过度设计）、不在 Dockerfile 里跑数据库初始化脚本（数据库建表/入库是应用层职责，容器只负责进程启动）。

## 2. 双容器编排设计

```
docker-compose.yml
├── app (labor-app)
│   ├── build: .          ← 从 Dockerfile 构建
│   ├── depends_on: postgres (service_healthy)
│   ├── ports: 8000:8000
│   └── environment:     ← 覆盖 DATABASE_URL 指向 postgres 容器
└── postgres (labor-pg)
    ├── image: pgvector/pgvector:pg16
    ├── ports: 5432:5432
    ├── volumes: pgdata:/var/lib/postgresql/data  ← 持久化
    └── healthcheck: pg_isready
```

关键设计点：

- **depends_on + condition: service_healthy**：不是简单的"先启动 postgres 再启动 app"，而是等 PG 的 `pg_isready` 返回成功后才启动 app——避免 app 启动时数据库还没准备好接受连接。
- **DATABASE_URL 覆盖**：compose 内将 host 从 localhost 改为 postgres（服务名=Docker 内 DNS）。Pydantic-settings 的优先级链是 shell 环境变量 > .env 文件 > 代码默认值，compose 的 environment 属于最高优先级。
- **pgdata volume 持久化**：容器删了数据还在——`docker compose down` 不删 volume（除非加 -v 参数）。

## 3. Dockerfile 决策

### 3.1 为什么单阶段，不用多阶段构建

| 方案 | 镜像大小 | 复杂度 |
|---|---|---|
| single-stage（选中） | ~400MB | Dockerfile 20 行，一眼看懂 |
| multi-stage + slim + 清缓存 | ~250MB | 构建/运行两阶段，调试麻烦 |

本项目是纯 Python 应用（无前端构建产物），多阶段能省约 150MB（主要是编译工具链 gcc/libpq-dev），但对展示项目不划算——镜像 400MB 在个人展示场景完全合理，多阶段增加的维护成本（需要理解两个 FROM 的关系）不值得。

### 3.2 为什么用 pip install . 不逐条 pip install

pyproject.toml 的 `[project]` table 已列出全部依赖，`pip install .` 读这份清单一次性安装——保持依赖声明单一源（DRY）。Dockerfile 里不复述依赖列表，避免 pyproject.toml 加了东西、Dockerfile 忘了加。

需要添加 `[build-system]` 配置让 pip 能识别。在 pyproject.toml 末尾加：

```toml
[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["app*", "eval*"]
```

### 3.3 为什么用 uvicorn 不用 gunicorn

- gunicorn 默认 sync worker 与 SSE 流式不兼容（会缓冲整个响应再发送）→ 需要 uvicorn worker，等于绕了一圈还是用 uvicorn
- 个人展示项目并发量低，单 worker 足够
- 部署运维简单：一个进程、一个端口、一句 CMD

### 3.4 为什么不把数据库初始化（建表+入库）做进容器启动

有些项目在 entrypoint.sh 里做 `python scripts/ingest_laws.py`。本项目刻意不做：

- 入库依赖数据文件（`data/raw/*.json`），这些文件随镜像打包——但如果数据更新，又要重建镜像，不合理
- 数据库初始化是**一次性操作**，放在容器启动脚本里每次启动都跑，是浪费
- 正确做法：容器启动后用 `docker compose exec app python scripts/ingest_laws.py` 手动入库一次

## 4. 环境变量流（`.env → docker compose → 容器`）

```
.env.example ──(用户复制为 .env 填写 key)──→ .env
                                                │
                          ┌─────────────────────┘
                          ▼
                  docker compose up
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
    app 容器 environment       postgres 容器
    (compose 注入 + .env       (基础用户名/密码
     自动读取的变量)            从 .env 或默认值)
```

Docker Compose 自动从项目根目录的 `.env` 文件读取环境变量。compose 文件内 `${VAR:-default}` 语法：先取 shell/`.env` 的值，没有才用 default。

关键理解：**DATABASE_URL 不在 .env 里改**——compose 直接在 environment 里硬覆盖为 `postgresql://labor:labor@postgres:5432/labor`。因为 Docker 内服务间通信用 service name，这个值是固定的、与用户环境无关。

## 5. 冒烟验证步骤

```bash
# 1) 启动
docker compose up -d

# 2) 确认两个容器都在
docker compose ps
# 应看到 labor-app (running) + labor-pg (healthy)

# 3) 首次入库（只需一次）
docker compose exec app python scripts/ingest_laws.py

# 4) API 冒烟（离线模式——零 key 可跑）
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"公司延长我工作时间，加班费怎么算"}' \
  --no-buffer

# 5) Swagger 交互式文档
# 浏览器打开 http://localhost:8000/docs

# 6) 停止（数据不丢）
docker compose down

# 7) 彻底清理（删数据库——谨慎）
docker compose down -v
```

## 6. 已知取舍

1. **单阶段构建，镜像约 400MB**。开发依赖（gcc/libpq-dev）留在了最终镜像里。用 multi-stage 可减到 250MB，但对展示项目不值得——Dockerfile 可读性优先。
2. **单 worker，无反向代理**。生产环境应在前面挂 Nginx（SSL 终止 + 静态文件 + 限流），容器间通信加 network isolation。这些是运维层面的事，不在展示项目范围。
3. **app 容器以 root 运行**。安全最佳实践是创建非 root 用户（useradd + USER 指令）。展示项目简化，生产部署必须加。
4. **不包含向量初始化脚本**。首次启动后需手动执行 `ingest_laws.py` + `backfill_embeddings.py`。不做自动化的原因见 §3.4。
5. **Compose 文件暴露了数据库端口到主机**（5432:5432）。本地调试方便（本机 Python 直连 PG），但生产部署应移除 ports 只保留容器内通信。