"""阶段 9 认证测试：密码哈希 + 签名 token 纯函数（离线可跑）+ 注册/登录/历史 DB 集成。

纯函数部分零依赖；路由部分标 @pytest.mark.db，需要 PostgreSQL 在线（docker compose）。
沿用 test_api.py 模式：TestClient 触发 lifespan（建表 + 装配 agent），随后注入
FakeRetriever 的离线 agent 避免真 LLM 调用。
"""
from __future__ import annotations

import secrets

import pytest
from fastapi.testclient import TestClient

from app.api.auth import hash_password, make_token, verify_password, verify_token
from app.agent.graph import build_agent
from app.main import app as fastapi_app
from app.rag.models import FusedHit, RetrievalResult


TEST_HIT = FusedHit(
    law_id="labor_law",
    article_no=21,
    text="劳动合同期限三个月以上不满一年的，试用期不得超过一个月。",
    lanes=["bm25"],
    rrf_score=1.0,
)


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query: str, top_k: int = 8, use_rerank: bool = True) -> RetrievalResult:
        return RetrievalResult(query=query, hits=self.hits)


def _inject_offline_agent():
    """用离线 agent 替换 lifespan 建的真 LLM agent（认证测试不碰真 LLM API）。"""
    fastapi_app.state.agent = build_agent(FakeRetriever([TEST_HIT]), verify_mode="pass")
    fastapi_app.state.history = []


# ---- 纯函数：密码哈希（离线可跑） ----

def test_hash_roundtrip():
    stored = hash_password("hello123")
    assert stored.count("$") == 1
    assert verify_password("hello123", stored)


def test_hash_wrong_password():
    stored = hash_password("hello123")
    assert not verify_password("wrong", stored)


def test_hash_malformed_stored():
    assert not verify_password("x", "not-a-hash$format")


# ---- 纯函数：签名 token（离线可跑） ----

def test_token_roundtrip():
    token = make_token(42, "secret")
    assert verify_token(token, "secret") == 42


def test_token_wrong_secret():
    token = make_token(42, "secret")
    assert verify_token(token, "other") is None


def test_token_expired():
    token = make_token(42, "secret", ttl=-10)
    assert verify_token(token, "secret") is None


def test_token_tampered():
    token = make_token(42, "secret")
    assert verify_token(token + "x", "secret") is None


def test_token_garbage():
    assert verify_token("not.a.token", "secret") is None


# ---- 路由（DB 集成）----

def _new_user(client) -> str:
    """注册一个随机用户，返回 token。"""
    username = "tester_" + secrets.token_hex(4)
    r = client.post("/auth/register", json={"username": username, "password": "pass1234"})
    assert r.status_code == 200
    return r.json()["token"], username


@pytest.mark.db
def test_register_and_login():
    with TestClient(fastapi_app) as client:
        _inject_offline_agent()
        token, username = _new_user(client)
        assert token.count(".") == 1

        # 重复注册 → 409
        r2 = client.post("/auth/register", json={"username": username, "password": "pass1234"})
        assert r2.status_code == 409

        # 登录正确 → 200；密码错 → 401
        r3 = client.post("/auth/login", json={"username": username, "password": "pass1234"})
        assert r3.status_code == 200
        r4 = client.post("/auth/login", json={"username": username, "password": "wrong"})
        assert r4.status_code == 401

        # /auth/me 带 token → 用户名；不带 → 401
        h = {"Authorization": "Bearer " + token}
        r5 = client.get("/auth/me", headers=h)
        assert r5.status_code == 200 and r5.json()["username"] == username
        r6 = client.get("/auth/me")
        assert r6.status_code == 401


@pytest.mark.db
def test_login_chat_saves_session():
    """登录提问 → 自动建会话；会话列表出现；消息回读含问答 + session_id 透传。"""
    with TestClient(fastapi_app) as client:
        _inject_offline_agent()
        token, _ = _new_user(client)
        h = {"Authorization": "Bearer " + token}

        # 登录后问一条 → SSE 正常
        r = client.post("/chat", json={"question": "试用期最长可以约定多久"}, headers=h)
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")
        # done 帧应带新建的 session_id
        sid = None
        for line in r.text.split("\n"):
            if '"done"' in line:
                import json as _j
                payload = _j.loads(line[len("data: "):])
                sid = _j.loads(payload["content"])["session_id"]
        assert sid is not None

        # 会话列表里有这条
        r = client.get("/chat/sessions", headers=h)
        assert r.status_code == 200
        sessions = r.json()["sessions"]
        assert len(sessions) >= 1

        # 该会话消息含 user 问题 + assistant 回答
        r = client.get(f"/chat/sessions/{sid}/messages", headers=h)
        assert r.status_code == 200
        msgs = r.json()["messages"]
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant"]
        assert msgs[0]["content"] == "试用期最长可以约定多久"
        assert len(msgs[1]["content"]) > 0

        # 续问同一会话 → 追加消息（多轮）
        r = client.post("/chat", json={"question": "超过六个月怎么办", "session_id": sid}, headers=h)
        assert r.status_code == 200
        r = client.get(f"/chat/sessions/{sid}/messages", headers=h)
        msgs = r.json()["messages"]
        assert len(msgs) == 4  # 已累计两轮问答


@pytest.mark.db
def test_login_session_isolation():
    """会话归属隔离：他人无权读别人的会话消息。"""
    with TestClient(fastapi_app) as client:
        _inject_offline_agent()
        token_a, _ = _new_user(client)
        token_b, _ = _new_user(client)

        ha = {"Authorization": "Bearer " + token_a}
        r = client.post("/chat", json={"question": "试用期最长可以约定多久"}, headers=ha)
        # 从 done 帧取 session_id
        sid = None
        for line in r.text.split("\n"):
            if '"done"' in line:
                import json as _j
                payload = _j.loads(line[len("data: "):])
                sid = _j.loads(payload["content"])["session_id"]

        # 用户 B 访问 A 的会话 → 404
        hb = {"Authorization": "Bearer " + token_b}
        r = client.get(f"/chat/sessions/{sid}/messages", headers=hb)
        assert r.status_code == 404


@pytest.mark.db
def test_anonymous_chat_not_in_db():
    """游客可问，但其问答不进任何登录用户的会话。"""
    with TestClient(fastapi_app) as client:
        _inject_offline_agent()
        token, _ = _new_user(client)
        h = {"Authorization": "Bearer " + token}

        # 匿名问一条（不带 header），/chat 本身可用
        r = client.post("/chat", json={"question": "公司拖欠我工资怎么办"})
        assert r.status_code == 200

        # 登录用户会话里不应出现匿名那条
        r = client.get("/chat/sessions", headers=h)
        assert r.status_code == 200
        assert r.json()["sessions"] == []


@pytest.mark.db
def test_anonymous_chat_not_in_db():
    with TestClient(fastapi_app) as client:
        _inject_offline_agent()
        token, _ = _new_user(client)

        # 匿名问一条（不带 header），/chat 本身可用
        r = client.post("/chat", json={"question": "公司拖欠我工资怎么办"})
        assert r.status_code == 200

        # 登录用户历史里不应出现匿名那条（匿名不落库）
        h = {"Authorization": "Bearer " + token}
        r = client.get("/chat/history/mine", headers=h)
        assert r.status_code == 200
        for rec in r.json()["history"]:
            assert rec["question"] != "公司拖欠我工资怎么办"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
