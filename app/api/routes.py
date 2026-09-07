"""聊天路由：POST /chat SSE 流式 + GET /chat/history 会话历史。

SSE 设计（与 WebSocket 对比见决策 07）：
- 单向推送（server→client），客户端只需 fetch + ReadableStream 即可接收
- 每条事件是一行 data: <json>\n\n，stage 字段让客户端知道当前在哪个阶段
- 最终帧 stage="done"，附带完整答案 + 引用列表

为什么不直接用 LangGraph 的 astream_events：本项目回答不是 LLM 流式生成（v1 是
一次 LLM 调用拿到全文），SSE 主要展示"阶段进度"——改写→检索→核验→回答→完成——而非
逐 token 推送。真正的逐 token 流式 v2 再做（见决策 07 取舍）。
"""

from __future__ import annotations

import json
import traceback
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.models import (
    ChatRequest,
    ChatDonePayload,
    CitationBrief,
    SseChunk,
    ErrorResponse,
)

router = APIRouter(prefix="/chat", tags=["chat"])

_MAX_HISTORY = 200


def _sse_frame(data: dict) -> str:
    """把 dict/Pydantic 对象编码为一条 SSE frame（data: json\n\n）。"""
    if hasattr(data, "model_dump"):
        d = data.model_dump()
    else:
        d = data
    return f"data: {json.dumps(d, ensure_ascii=False)}\n\n"


async def _stream_chat(app, question: str) -> AsyncIterator[str]:
    """SSE 流式生成器：逐阶段推送进度事件，最后推送 done 帧。

    为什么不用 async for 在节点间插桩：当前 Graph compile 后 invoke 是同步的，
    流式推送是在单个线程内"发帧→执行一步→发帧"模拟的。v2 换成 astream_events
    可实现真正的逐节点 yield。见决策 07。
    """
    try:
        # 阶段 1：改写
        yield _sse_frame(SseChunk(stage="rewriting", content="正在理解您的问题..."))
        # 阶段 2：检索
        yield _sse_frame(SseChunk(stage="searching", content="正在检索相关法律条文..."))
        # 阶段 3：核验
        yield _sse_frame(SseChunk(stage="verifying", content="正在核验条文与问题的相关性..."))

        # 执行全链路——所有阶段在 invoke 内部完成，SSE 只发"进度提示"
        result = app.invoke({"original": question})
        answer = result["answer"]

        if answer.refuse:
            yield _sse_frame(SseChunk(stage="refusing", content="检索到的条文不足以支持明确结论"))
        else:
            yield _sse_frame(SseChunk(stage="answering", content="正在整理回答..."))

        # 完成帧：携带最终答案 + 引用
        citations = [
            CitationBrief(
                law_id=c.law_id,
                article_no=c.article_no,
                chapter=c.chapter,
            )
            for c in answer.citations
        ]
        done = ChatDonePayload(
            refuse=answer.refuse,
            answer=answer.answer,
            citations=citations,
        )
        yield _sse_frame(SseChunk(stage="done", content=json.dumps(done.model_dump(), ensure_ascii=False)))

    except Exception as exc:
        yield _sse_frame(SseChunk(
            stage="error",
            content=json.dumps(
                ErrorResponse(error="处理请求时出错", detail=str(exc)).model_dump(),
                ensure_ascii=False,
            ),
        ))


@router.post("")
async def chat(req: ChatRequest, request: Request):
    """POST /chat：接收劳动争议问题，返回 SSE 流式响应。

    调用示例：
        curl -X POST http://localhost:8000/chat \\
          -H "Content-Type: application/json" \\
          -d '{"question":"公司拖欠我工资怎么办"}' \\
          --no-buffer

    客户端解析：每行 data: <json> 对应一个 SseChunk；
    当 stage=="done" 时 content 字段是 ChatDonePayload 的 JSON 字符串。
    """
    app = request.app.state.agent
    if app is None:
        raise HTTPException(status_code=503, detail="Agent 未就绪，请检查配置")
    stream = _stream_chat(app, req.question)

    # 保存历史
    history: list = request.app.state.history
    history.append({"question": req.question, "timestamp": None})
    if len(history) > _MAX_HISTORY:
        history.pop(0)

    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 告知 nginx 不做缓冲（见决策 07）
        },
    )


@router.get("/history")
async def chat_history(request: Request, limit: int = 50):
    """GET /chat/history：返回最近 limit 条会话记录（倒序，最新在前）。

    仅返回存于内存的历史；服务重启后清空。
    个人展示项目不持久化——面试演示场景重启丢失是可接受的。
    """
    history: list = request.app.state.history
    if limit < 1:
        limit = 50
    if limit > _MAX_HISTORY:
        limit = _MAX_HISTORY
    return {"history": list(reversed(history[-limit:])), "total": len(history)}