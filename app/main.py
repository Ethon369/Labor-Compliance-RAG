"""FastAPI 应用入口：lifespan 初始化配置/检索器/Agent，挂载 chat 路由，统一异常处理。

启动：uvicorn app.main:app --reload
冒烟：curl http://localhost:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.admin_routes import router as admin_router
from app.api.auth import ensure_admin, ensure_user_schema, router as auth_router
from app.api.kb_routes import router as kb_router
from app.api.models import ErrorResponse
from app.api.routes import router as chat_router
from app.agent.graph import build_agent_from_config
from app.core.config import Settings
from app.core.db import close_pools
from app.kb.schema import ensure_kb_schema
from app.rag.retriever import build_retriever


class CacheStaticFiles(StaticFiles):
    """给静态资源按扩展名加 Cache-Control，静态资产可被浏览器长缓存。

    只对"内容可指纹化"的扩展名缓存：.js/.css/.png/.svg 等改版时要么换文件名、
    要么由开发者手动刷新；html 与其它文件不缓存，保证页面与接口始终最新。
    """
    _CACHEABLE = {".js", ".css", ".png", ".svg", ".ico", ".woff2", ".woff", ".ttf"}
    _MAX_AGE = 86400  # 24h：本地演示够用，太长会阻碍开发期改版

    def file_response(self, full_path, stat_result, scope, status_code=200):
        resp = super().file_response(full_path, stat_result, scope, status_code)
        if Path(full_path).suffix.lower() in self._CACHEABLE:
            resp.headers["Cache-Control"] = f"public, max-age={self._MAX_AGE}"
        return resp


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期：启动时一次性组装 settings/retriever/agent，关闭无需清理。"""
    settings = Settings()
    # 建表顺序有依赖，不能调换：users 先于 kb（kb.owner_id 外键指向 users），
    # kb 先于 build_retriever（后者 ensure_ready 要给 chunks 加 embedding 列）。
    ensure_user_schema(settings.database_url)
    ensure_kb_schema(settings.database_url)
    # 管理员账号幂等确保存在（用户名已存在则只补 role，不覆盖已改密码）
    ensure_admin(settings.database_url, settings.admin_username, settings.admin_password)
    retriever = build_retriever(settings)
    agent = build_agent_from_config(retriever, settings)
    _app.state.settings = settings
    _app.state.retriever = retriever
    _app.state.agent = agent
    # 会话历史：内存列表，最大保留 200 条（个人展示项目不需要持久化）
    _app.state.history: list[dict] = []
    yield
    close_pools()


app = FastAPI(
    title="劳动争议智能合规助手",
    description="基于中国劳动法律体系的 RAG 问答系统（检索增强生成）",
    version="0.1.0",
    lifespan=lifespan,
)

# 允许前端跨域调试（本地开发安全范围宽松）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# GZip 压响应：只压 >1KB 的，小响应压缩收益低还费 CPU。
# 中间件是栈式，后 add 的在外层——GZip 放 CORS 之后，压缩的是最终响应体。
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def add_request_id(request, call_next):
    """给每个请求挂 X-Request-Id：透传上游 id，没有则生成一个。

    观测用途：日志里带着同一个 id，能把"一条请求"在检索/LLM/DB 各段的耗时串起来
    定位。响应头回显同一个 id，前端报障时贴出来即可定位到具体请求。
    """
    rid = request.headers.get("x-request-id") or uuid4().hex[:12]
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    return response

# 顺序：auth → admin → chat → kb，最后才挂静态目录。
# StaticFiles 挂在最后是因为它匹配 "/" 前缀，放前面会把 /docs、/chat、/kb 全吃掉。
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(chat_router)
app.include_router(kb_router)

# 挂载静态前端：放在路由之后，避免抢占 /docs、/chat 等路径。
# index.html 由 StaticFiles(html=True) 作为默认首页提供。
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/", CacheStaticFiles(directory=static_dir, html=True), name="static")


# ---- 统一异常处理 ----


@app.exception_handler(ValueError)
async def handle_value_error(_request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(error="请求参数有误", detail=str(exc)).model_dump(),
    )


@app.exception_handler(Exception)
async def handle_general(_request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="服务器内部错误", detail=str(exc)).model_dump(),
    )