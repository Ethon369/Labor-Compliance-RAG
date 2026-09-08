"""API 层的 Pydantic 请求/响应模型。

与 agent 层的 AgentAnswer 分层：agent 层承载内部状态数据（含 Citation 引用列表），
API 层只暴露给客户端需要的字段——不要把内部 Citation 对象直接泄到 API 响应里。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HistoryMsg(BaseModel):
    """多轮上下文里的一轮消息（登录用户会话由后端读取，游客由前端上传）。"""
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """POST /chat 请求体：一条劳动争议问题。

    session_id：登录用户目标会话（不传则自动新建会话）。游客置空。
    history：可选的多轮上下文，游客前端上传用；登录用户传了也以服务端会话为准。
    """
    question: str = Field(..., min_length=1, max_length=2000, description="劳动争议问题")
    session_id: int | None = Field(default=None, description="登录用户目标会话 id；空则新建")
    history: list[HistoryMsg] = Field(default_factory=list, description="游客端多轮上下文")


class SseChunk(BaseModel):
    """单帧 SSE 事件推送的数据结构。

    stream stage 命名（客户端据此决策 UI 展示）：
    - rewriting  → 正在改写查询
    - searching  → 正在检索
    - verifying  → 正在核验引用
    - answering  → 正在生成回答（LLM 流式逐段到达时用此 stage）
    - refusing   → 命中不足，正在生成拒答
    - done       → 全部完成，content 为最终答案文本，citations 带引用列表
    - error      → 异常，content 为错误描述
    """
    stage: str
    content: str = ""


class CitationBrief(BaseModel):
    """API 暴露的简化引用：仅含定位信息，不返回完整条文文本（客户端可自行查阅）。"""
    law_id: str
    article_no: int
    chapter: str | None = None


class ChatDonePayload(BaseModel):
    """SSE done 帧的完整载荷：答案 + 拒答标记 + 引用列表 + 会话定位。"""
    refuse: bool = False
    answer: str
    citations: list[CitationBrief] = []
    session_id: int | None = None


class ErrorResponse(BaseModel):
    """统一错误响应。"""
    error: str
    detail: str = ""


class AuthRequest(BaseModel):
    """注册/登录请求：用户名 + 密码。"""
    username: str = Field(..., min_length=2, max_length=32, description="用户名")
    password: str = Field(..., min_length=4, max_length=64, description="密码")


class AuthResponse(BaseModel):
    """认证成功响应：token 由前端存 localStorage，请求时放 Authorization 头。"""
    token: str
    username: str