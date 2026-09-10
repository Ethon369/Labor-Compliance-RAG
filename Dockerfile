# ---- 劳动与社会保障 RAG 问答平台 · Docker 镜像 ----
# 基于 Python 3.11 slim，安装项目依赖 + 应用代码，uvicorn 启动 FastAPI。
#
# 为什么不用多阶段构建：项目是纯 Python（无前端构建产物），依赖全在 pip install 阶段，
# 多阶段节省的空间有限（~50MB），但增加维护理解成本——对展示项目不划算。见决策 08。

FROM python:3.11-slim

WORKDIR /app

# 国内构建加速：Debian apt 源换阿里云镜像（python:3.11-slim 的源文件存在两种格式，都替换）
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list 2>/dev/null; \
    sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null; true

# psycopg 需 libpq；gcc 用于编译部分二进制包（如 rank-bm25 的 numpy 传递依赖）
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq-dev gcc && \
    rm -rf /var/lib/apt/lists/*

# 先复制依赖清单——Docker 层缓存：依赖不变时不用重装
COPY pyproject.toml .

# 安装项目依赖：pip 源同样换阿里云镜像（pypi.org 国内直连慢）
# 不使用 -e ".[dev]"——容器内不需要 pytest 等开发依赖
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ .

# 复制运行时需要的目录
COPY app/ app/
COPY data/raw/ data/raw/
# scripts/ 在首次入库时需要；后续 Docker 内可手动执行
COPY scripts/ scripts/

EXPOSE 8000

# 为什么用 uvicorn 而非 gunicorn：gunicorn 对 async/SSE 流式的兼容需要额外配置
# （uvicorn worker），且本项目的并发量（个人展示）不需要多 worker。单 worker + uvicorn
# 是简化运维的最优解。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]