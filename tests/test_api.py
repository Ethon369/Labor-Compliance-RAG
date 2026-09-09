"""阶段 6 API 单测：SSE 流式 /chat + /chat/history + 异常处理 + 数据模型校验。

用 FastAPI TestClient（无需启动 uvicorn），注入离线 agent（FakeRetriever + 占位组件）——
全链路 SSE 帧解析 + 历史记录验证，零 key 零 DB 可跑。

重要：TestClient 会触发 FastAPI lifespan，lifespan 内部执行 build_agent_from_config(settings)
并用真实 HybridRetriever 覆盖 app.state.agent。因此所有 state 注入必须在 with TestClient() 内部
（lifespan 运行之后）进行，否则会被覆盖。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.agent.graph import build_agent
from app.api.models import ChatRequest, ChatDonePayload, ErrorResponse, SseChunk
from app.main import app as fastapi_app
from app.rag.models import FusedHit, RetrievalResult

# ---- 测试用命中数据 ----

OVERTIME_HIT = FusedHit(
    law_id="labor_law",
    article_no=44,
    text="安排劳动者延长工作时间的，支付不低于工资的百分之一百五十的工资报酬。",
    lanes=["bm25"],
    rrf_score=1.0,
)

UNRELATED_HIT = FusedHit(
    law_id="labor_law",
    article_no=36,
    text="工时制度。",
    lanes=["bm25"],
    rrf_score=1.0,
)


class FakeRetriever:
    def __init__(self, hits: list[FusedHit]):
        self.hits = hits

    def search(self, query: str, top_k: int = 8, use_rerank: bool = True) -> RetrievalResult:
        return RetrievalResult(query=query, hits=self.hits)


# ---- Fixture：在 lifespan 之后注入 agent ----

def _inject_agent(hits: list[FusedHit], verify_mode: str = "pass"):
    """每次创建 TestClient 后调用：用 FakeRetriever agent 替换 lifespan 创建的 agent。"""
    fastapi_app.state.agent = build_agent(FakeRetriever(hits), verify_mode=verify_mode)
    fastapi_app.state.history = []


@pytest.fixture
def client_with_agent():
    """构造正常回答链路的 TestClient。"""
    with TestClient(fastapi_app) as client:
        _inject_agent([OVERTIME_HIT])
        yield client


@pytest.fixture
def client_refuse():
    """构造拒答链路的 TestClient：检索命中无关条文。

    显式用 verify_mode="rule"：拒答是规则核验（真 LLM 配套兜底）的语义，
    离线模板模式（pass）命中即答，不走拒答。见 graph.build_agent 的 verify_mode。"""
    with TestClient(fastapi_app) as client:
        _inject_agent([UNRELATED_HIT], verify_mode="rule")
        yield client


# ---- SSE 帧解析工具 ----

def _parse_sse_frames(response_text: str) -> list[dict]:
    """从 SSE 响应文本中提取每条 data: 的 JSON 对象。"""
    frames = []
    for line in response_text.strip().split("\n\n"):
        for sub in line.split("\n"):
            if sub.startswith("data: "):
                frames.append(json.loads(sub[6:]))
    return frames


# ---- POST /chat SSE 流式 ----

def test_chat_sse_happy_path(client_with_agent):
    """正常链路：应经过 rewriting→searching→verifying→answering→done。"""
    resp = client_with_agent.post("/chat", json={"question": "延长工作时间 加班费 工资报酬"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"

    frames = _parse_sse_frames(resp.text)
    stages = [f["stage"] for f in frames]
    assert "rewriting" in stages
    assert "searching" in stages
    assert "verifying" in stages
    assert "answering" in stages
    assert stages[-1] == "done"


def test_chat_sse_done_frame_has_answer(client_with_agent):
    """done 帧的 content 应包含答案文本和引用。"""
    resp = client_with_agent.post("/chat", json={"question": "延长工作时间 加班费 工资报酬"})
    frames = _parse_sse_frames(resp.text)
    done_frame = frames[-1]
    assert done_frame["stage"] == "done"
    done_payload = json.loads(done_frame["content"])
    assert done_payload["refuse"] is False
    assert len(done_payload["answer"]) > 0
    assert len(done_payload["citations"]) > 0
    assert done_payload["citations"][0]["law_id"] == "labor_law"


def test_chat_sse_refuse_path(client_refuse):
    """拒答链路：无关条文 → 经过 refusing → done 的 refuse=True。"""
    resp = client_refuse.post("/chat", json={"question": "加班费怎么计算"})
    frames = _parse_sse_frames(resp.text)
    stages = [f["stage"] for f in frames]
    assert "refusing" in stages
    done_frame = frames[-1]
    assert done_frame["stage"] == "done"
    payload = json.loads(done_frame["content"])
    assert payload["refuse"] is True
    assert payload["citations"] == []


def test_chat_sse_error_frame_on_bad_agent():
    """Agent 未就绪（state.agent = None）→ 503。注入在 lifespan 之后。"""
    with TestClient(fastapi_app) as client:
        fastapi_app.state.agent = None
        resp = client.post("/chat", json={"question": "测试"})
    assert resp.status_code == 503


def test_chat_sse_correct_headers(client_with_agent):
    """验证 SSE 必需的响应头。"""
    resp = client_with_agent.post("/chat", json={"question": "延长工作时间 工资报酬"})
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["connection"] == "keep-alive"
    assert "X-Accel-Buffering" in resp.headers


# ---- GET /chat/history ----

def test_history_starts_empty():
    """新会话历史应为空（lifespan 初始化 history=[]）。"""
    with TestClient(fastapi_app) as client:
        fastapi_app.state.history = []
        resp = client.get("/chat/history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["history"] == []


def test_history_records_questions(client_with_agent):
    """每次 POST /chat 应在 history 中追加一条记录。"""
    # fixture 已清空 history（_inject_agent 做了 history=[]）
    client_with_agent.post("/chat", json={"question": "问题A"})
    client_with_agent.post("/chat", json={"question": "问题B"})
    resp = client_with_agent.get("/chat/history")
    data = resp.json()
    assert data["total"] == 2
    assert data["history"][0]["question"] == "问题B"
    assert data["history"][1]["question"] == "问题A"


def test_history_limit():
    """limit 参数应限制返回数量。注入在 lifespan 之后。"""
    with TestClient(fastapi_app) as client:
        fastapi_app.state.history = [{"question": f"Q{i}"} for i in range(10)]
        resp = client.get("/chat/history", params={"limit": 3})
    data = resp.json()
    assert len(data["history"]) == 3
    assert data["total"] == 10


def test_history_limit_min_clamped():
    """limit < 1 应被 clamp 到 50。注入在 lifespan 之后。"""
    with TestClient(fastapi_app) as client:
        fastapi_app.state.history = [{"question": "Q"}]
        resp = client.get("/chat/history", params={"limit": 0})
    assert resp.status_code == 200


def test_history_invalid_limit():
    """非数字 limit 被 FastAPI 校验拦截。"""
    with TestClient(fastapi_app) as client:
        resp = client.get("/chat/history", params={"limit": "abc"})
    assert resp.status_code == 422


# ---- 请求/响应模型 ----

def test_chat_request_rejects_empty():
    with pytest.raises(Exception):
        ChatRequest(question="")


def test_chat_request_rejects_too_long():
    with pytest.raises(Exception):
        ChatRequest(question="x" * 2001)


def test_chat_request_accepts_valid():
    req = ChatRequest(question="公司拖欠工资怎么办")
    assert req.question == "公司拖欠工资怎么办"


def test_sse_chunk_serialize():
    chunk = SseChunk(stage="rewriting", content="正在改写...")
    d = chunk.model_dump()
    assert d["stage"] == "rewriting"
    assert d["content"] == "正在改写..."


def test_error_response_serialize():
    err = ErrorResponse(error="测试错误", detail="这是一次测试")
    d = err.model_dump()
    assert d["error"] == "测试错误"


# ---- 异常处理路由级测试 ----

def test_404_unknown_route():
    with TestClient(fastapi_app) as client:
        resp = client.get("/nonexistent")
    assert resp.status_code == 404


def test_405_method_not_allowed():
    with TestClient(fastapi_app) as client:
        resp = client.put("/chat")
    assert resp.status_code == 405


def test_get_chat_no_params():
    """GET /chat 无参数 → 404。本路由只注册了 POST（路由层返回 405），
    GET 未命中任何路由（方法路由不匹配时不占用路径），被静态根挂载兜底为 404。
    这也验证了静态挂载不会抢占已有的 API 路由。"""
    with TestClient(fastapi_app) as client:
        resp = client.get("/chat")
    assert resp.status_code == 404


# ---- 流式回答集成：token 帧逐段到达，done 帧收尾 ----

class StreamingFakeAnswerer:
    """带 on_token 的回答器：把回答拆成片段逐段回调（模拟 LLM 逐 token 到达）。"""

    def __init__(self, pieces=None):
        self.pieces = pieces or ["根据", "《劳动法》第44条", "，应支付", "不低于150%的报酬。"]

    def generate(self, query, hits, history=None, on_token=None):
        for p in self.pieces:
            if on_token:
                on_token(p)
        return "".join(self.pieces)


def test_chat_sse_streams_token_frames_then_done():
    """注入流式回答器后：回答以 token 帧逐段推送，拼起来=done 帧完整答案。"""
    with TestClient(fastapi_app) as client:
        fastapi_app.state.agent = build_agent(
            FakeRetriever([OVERTIME_HIT]),
            answerer=StreamingFakeAnswerer(),
            verify_mode="pass",
        )
        fastapi_app.state.history = []
        resp = client.post("/chat", json={"question": "延长工作时间 加班费 工资报酬"})

    assert resp.status_code == 200
    frames = _parse_sse_frames(resp.text)
    stages = [f["stage"] for f in frames]
    token_texts = [f["content"] for f in frames if f["stage"] == "token"]
    assert len(token_texts) > 0
    assert stages[0] == "rewriting"
    assert stages[-1] == "done"
    assert "answering" not in stages  # 真流式走 token 帧，不需要"正在整理"过渡帧

    done_payload = json.loads(frames[-1]["content"])
    assert "".join(token_texts) == done_payload["answer"]  # 前端拼的=权威完整答案
    assert len(done_payload["citations"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])