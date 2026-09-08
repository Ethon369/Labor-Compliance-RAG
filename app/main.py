"""FastAPI 应用入口：lifespan 初始化配置/检索器/Agent，挂载 chat 路由，统一异常处理。

启动：uvicorn app.main:app --reload
冒烟：curl http://localhost:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.auth import ensure_user_schema, router as auth_router
from app.api.models import ErrorResponse
from app.api.routes import router as chat_router
from app.agent.graph import build_agent_from_config
from app.core.config import Settings
from app.rag.retriever import build_retriever


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期：启动时一次性组装 settings/retriever/agent，关闭无需清理。"""
    settings = Settings()
    retriever = build_retriever(settings)
    agent = build_agent_from_config(retriever, settings)
    # 认证 + 登录用户历史两张表（幂等，启动可反复执行）
    ensure_user_schema(settings.database_url)
    _app.state.settings = settings
    _app.state.retriever = retriever
    _app.state.agent = agent
    # 会话历史：内存列表，最大保留 200 条（个人展示项目不需要持久化）
    _app.state.history: list[dict] = []
    yield


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

app.include_router(auth_router)
app.include_router(chat_router)

# 挂载静态前端：放在路由之后，避免抢占 /docs、/chat 等路径。
# index.html 由 StaticFiles(html=True) 作为默认首页提供。
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


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