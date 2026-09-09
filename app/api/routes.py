"""聊天路由：POST /chat SSE 流式 + GET /chat/history 会话历史。

SSE 设计（与 WebSocket 对比见决策 07）：
- 单向推送（server→client），客户端只需 fetch + ReadableStream 即可接收
- 每条事件是一行 data: <json>\n\n，stage 字段让客户端知道当前在哪个阶段
- 回答文本不是整段收尾：LLM 生成期间逐 token 推 stage="token"（打字机效果），
  前端边收边渲染；最终帧 stage="done" 携带完整答案 + 引用列表（供落库/历史/收尾）
- 阶段进度帧（rewriting→searching→verifying）先于回答推，让用户在等待时看到"走到哪一步"
"""

from __future__ import annotations

import asyncio
import json
import threading
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
from app.api.auth import (
    _bearer_user_id,
    create_session,
    list_user_sessions,
    load_session_messages,
    load_user_history,
    save_chat_history,
    save_session_message,
)
from app.agent.protocol import answer_sink_reset, answer_sink_set

router = APIRouter(prefix="/chat", tags=["chat"])

_MAX_HISTORY = 200
_MAX_CONTEXT_TURNS = 6  # 多轮上下文：取最近 N 轮问答喂给 LLM，防 prompt 过长


def _sse_frame(data: dict) -> str:
    """把 dict/Pydantic 对象编码为一条 SSE frame（data: json\n\n）。"""
    if hasattr(data, "model_dump"):
        d = data.model_dump()
    else:
        d = data
    return f"data: {json.dumps(d, ensure_ascii=False)}\n\n"


async def _stream_chat(app, question: str, history: list[dict], session_id: int | None = None,
                       settings=None, uid: int | None = None) -> AsyncIterator[str]:
    """SSE 流式生成器：进度帧 + 逐 token 回答帧 + done 收尾帧。

    为什么"跑图"要放后台线程 + asyncio.Queue 桥接：Graph compile 后 invoke 是同步阻塞，
    直接在本 async 生成器里调用会把事件循环堵死，前端收到的帧会"攒到结尾一次性到达"。
    于是把完整链路丢进一个守护线程去跑；回答器每产出一段文本就回调 on_token，
    on_token 用 call_soon_threadsafe 把片段跨线程投进 asyncio.Queue；本生成器 await
    queue.get() 逐段取出、逐段 yield——既真逐 token 打字机，事件循环也一直空闲。
    （LangGraph astream_events 逐节点事件留待后续，见决策 07 取舍。）
    """
    try:
        # 阶段 1-3：改写/检索/核验的"进度提示"。这些步骤短，先发帧占位；
        # 耗时在 answer 节点的 LLM 上，那里才需要真正的逐 token 推送。
        yield _sse_frame(SseChunk(stage="rewriting", content="正在理解您的问题..."))
        yield _sse_frame(SseChunk(stage="searching", content="正在检索相关法律条文..."))
        yield _sse_frame(SseChunk(stage="verifying", content="正在核验条文与问题的相关性..."))

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def on_token(text: str) -> None:
            # 后台线程里被回答器回调 → 线程安全地把文本片段投给主事件循环消费
            loop.call_soon_threadsafe(queue.put_nowait, ("token", text))

        def run_graph() -> None:
            """守护线程：注入 token 接收器后同步跑完整链路，把片段/结果投进队列。"""
            ctx = answer_sink_set(on_token)  # answer 节点据此拿到本请求的接收器
            try:
                result = app.invoke({"original": question, "history": history})
                answer = result["answer"]
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
                return
            finally:
                answer_sink_reset(ctx)
            # 关键顺序：token 都在 invoke 期间投递，result 一定在最后 → 队列天然有序
            loop.call_soon_threadsafe(queue.put_nowait, ("result", answer))

        threading.Thread(target=run_graph, daemon=True).start()

        # 消费队列：逐 token 推给前端；遇错误/最终结果进入收尾
        answer = None
        streamed = False  # 是否推过 token 帧（区分离线模板的"瞬时整段"与真流式）
        while True:
            kind, payload = await queue.get()
            if kind == "token":
                streamed = True
                yield _sse_frame(SseChunk(stage="token", content=payload))
            elif kind == "error":
                yield _sse_frame(SseChunk(
                    stage="error",
                    content=json.dumps(
                        ErrorResponse(error="处理请求时出错", detail=payload).model_dump(),
                        ensure_ascii=False,
                    ),
                ))
                return
            else:  # "result"
                answer = payload
                break

        if answer.refuse:
            yield _sse_frame(SseChunk(stage="refusing", content="检索到的条文不足以支持明确结论"))
        elif not streamed:
            # 无 token 流（离线模板瞬时整段生成）：补一个过渡帧收尾进度条；
            # 有 token 流时前端已经边收边渲染，不需要它。
            yield _sse_frame(SseChunk(stage="answering", content="正在整理回答..."))

        # 完成帧：携带最终答案 + 引用 + 会话 id（前端据此收尾、落历史、挂引用）
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
            session_id=session_id,
        )
        yield _sse_frame(SseChunk(stage="done", content=json.dumps(done.model_dump(), ensure_ascii=False)))

        # 登录用户会话：这条问答作为两条消息写进 chat_messages。
        # 落库失败静默忽略——数据库抖动不该阻断回答本身。
        if uid and session_id and settings:
            try:
                save_session_message(settings.database_url, session_id, "user", question)
                save_session_message(
                    settings.database_url,
                    session_id,
                    "assistant",
                    answer.answer,
                    [c.model_dump() for c in answer.citations],
                )
            except Exception:
                pass

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

    session_id（登录用户）传了 → 续该会话（带历史上下文）；没传 → 新建会话。
    游客：不落库，多轮靠请求里自带 history（前端内存维护，关页即没）。
    """
    settings = request.app.state.settings
    app = request.app.state.agent
    if app is None:
        raise HTTPException(status_code=503, detail="Agent 未就绪，请检查配置")

    uid = _bearer_user_id(request, settings.auth_secret)
    session_id: int | None = req.session_id

    if uid:
        # 登录用户：指定会话 → 校验归属并取其历史；否则新建会话（标题=问题首截）
        if session_id is not None:
            msgs = load_session_messages(settings.database_url, session_id, uid)
            if not msgs:
                raise HTTPException(status_code=404, detail="会话不存在或无权访问")
        else:
            title = req.question[:20]
            session_id = create_session(settings.database_url, uid, title)
            msgs = []
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in msgs[-_MAX_CONTEXT_TURNS:]
        ]
        stream = _stream_chat(app, req.question, history, session_id=session_id,
                              settings=settings, uid=uid)
    else:
        # 游客：历史由前端上传（内存），不落库
        history = [m.model_dump() for m in req.history[-_MAX_CONTEXT_TURNS:]]
        stream = _stream_chat(app, req.question, history)
        # 游客历史：仍进内存列表（见决策 07 取舍）
        mem: list = request.app.state.history
        mem.append({"question": req.question, "timestamp": None})
        if len(mem) > _MAX_HISTORY:
            mem.pop(0)

    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 告知 nginx 不做缓冲（见决策 07）
        },
    )


@router.get("/sessions")
async def chat_sessions(request: Request, limit: int = 50):
    """GET /chat/sessions：登录用户的会话列表（倒序，最新在前）。"""
    settings = request.app.state.settings
    uid = _bearer_user_id(request, settings.auth_secret)
    if uid is None:
        raise HTTPException(status_code=401, detail="未登录或登录已过期")
    if limit < 1 or limit > _MAX_HISTORY:
        limit = 50
    return {"sessions": list_user_sessions(settings.database_url, uid, limit)}


@router.get("/sessions/{session_id}/messages")
async def chat_session_messages(session_id: int, request: Request):
    """GET /chat/sessions/{id}/messages：某会话的全部多轮消息。"""
    settings = request.app.state.settings
    uid = _bearer_user_id(request, settings.auth_secret)
    if uid is None:
        raise HTTPException(status_code=401, detail="未登录或登录已过期")
    msgs = load_session_messages(settings.database_url, session_id, uid)
    if not msgs:
        raise HTTPException(status_code=404, detail="会话不存在或无权访问")
    return {"messages": msgs}


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


@router.get("/history/mine")
async def chat_history_mine(request: Request, limit: int = 50):
    """GET /chat/history/mine：登录用户自己的问答历史（DB 持久化，重启不丢）。"""
    settings = request.app.state.settings
    uid = _bearer_user_id(request, settings.auth_secret)
    if uid is None:
        raise HTTPException(status_code=401, detail="未登录或登录已过期")
    if limit < 1:
        limit = 50
    if limit > _MAX_HISTORY:
        limit = _MAX_HISTORY
    records, total = load_user_history(settings.database_url, uid, limit)
    return {"history": records, "total": total}