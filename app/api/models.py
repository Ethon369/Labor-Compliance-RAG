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
    kb_id: int | None = Field(default=None, description="目标知识库；空则用默认库/会话已绑定的库")


class SseChunk(BaseModel):
    """单帧 SSE 事件推送的数据结构。

    stream stage 命名（客户端据此决策 UI 展示）：
    - rewriting  → 正在改写查询
    - searching  → 正在检索
    - verifying  → 正在核验引用
    - token      → LLM 回答文本的一段增量：客户端应把连续 token 累积成完整回答
                  （打字机效果的数据帧；回答流式期间没有 answering 帧）
    - answering  → 无 LLM（离线模板瞬时整段生成）时的过渡提示，回答不逐 token 推
    - refusing   → 命中不足，正在生成拒答
    - done       → 全部完成，content 为最终答案文本，citations 带引用列表
    - error      → 异常，content 为错误描述
    """
    stage: str
    content: str = ""


class CitationBrief(BaseModel):
    """API 暴露的引用：定位信息 + 截断片段 + 相关度分，供前端可展开卡片与 [n] 角标。

    snippet 截断到 300 字：卡片只需让人判断"是不是这条"，不必把整段（甚至整页）
    塞进响应与历史记录。law_id/article_no/chapter 三个老字段保留为可空——
    改造前落库的历史消息仍是旧结构，回读时要能降级渲染（见决策 13）。
    """
    kb_id: int | None = None
    doc_id: int | None = None
    doc_title: str | None = None
    seq: int | None = None
    page: int | None = None
    heading: str = ""
    snippet: str = ""
    score: float | None = None
    source_law_id: str | None = None
    # ---- 旧格式兼容（V2.1 之前的 citations 只有这三个字段）----
    law_id: str | None = None
    article_no: int | None = None
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