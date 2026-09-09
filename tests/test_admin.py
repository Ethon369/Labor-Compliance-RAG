"""V2.2 权限测试（@pytest.mark.db）：admin 初始化、角色鉴权、改密码、越权语义。

越权是本文件的重点：
- 普通用户调 /admin/* → 403（角色不够）
- 未登录 → 401
- 管理员不能改自己的角色 / 不能禁用自己（自锁保护）
连不上库整模块 skip。
"""
from __future__ import annotations

import secrets

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.agent.graph import build_agent
from app.api.auth import ensure_admin
from app.core.config import Settings
from app.main import app as fastapi_app
from app.rag.models import FusedHit, RetrievalResult

pytestmark = pytest.mark.db


def _hit() -> FusedHit:
    return FusedHit(
        chunk_id=44, kb_id=1, doc_id=2, doc_title="劳动法", seq=44, heading="第四章",
        text="安排劳动者延长工作时间的，支付不低于工资的百分之一百五十的工资报酬。",
        source_law_id="labor_law", lanes=["bm25"], rrf_score=1.0,
    )


class _FakeRetriever:
    def search(self, query, top_k=8, use_rerank=True, kb_ids=None):
        return RetrievalResult(query=query, hits=[_hit()])


@pytest.fixture(scope="module")
def client():
    s = Settings()
    try:
        with psycopg.connect(s.database_url):
            pass
    except psycopg.OperationalError as e:
        pytest.skip(f"labor-pg 不可达（docker compose up -d 后重跑）：{e}")
    with TestClient(fastapi_app) as c:      # lifespan 会幂等建 admin
        fastapi_app.state.agent = build_agent(_FakeRetriever(), verify_mode="pass")
        fastapi_app.state.history = []
        yield c


def _register(client) -> tuple[str, str, str]:
    """注册一个随机普通用户，返回 (token, username, password)。"""
    username = "u_" + secrets.token_hex(4)
    password = "pass1234"
    r = client.post("/auth/register", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["role"] == "user"           # 注册出来的永远是普通用户
    return body["token"], username, password


def _auth(token: str) -> dict:
    return {"Authorization": "Bearer " + token}


def _admin_headers(client) -> dict:
    s = Settings()
    r = client.post("/auth/login", json={
        "username": s.admin_username, "password": s.admin_password,
    })
    assert r.status_code == 200, r.text
    assert r.json()["role"] == "admin"
    return _auth(r.json()["token"])


# ---- admin 初始化 ----

def test_admin_login_and_me(client):
    headers = _admin_headers(client)
    me = client.get("/auth/me", headers=headers)
    assert me.status_code == 200
    assert me.json()["role"] == "admin"
    assert me.json()["username"] == Settings().admin_username


def test_ensure_admin_does_not_overwrite_changed_password(client):
    """幂等初始化的关键约定：已存在的管理员，重复调用不得把密码打回默认值。"""
    s = Settings()
    headers = _admin_headers(client)
    new_password = "newpass123"
    assert client.post("/auth/change-password", headers=headers, json={
        "old_password": s.admin_password, "new_password": new_password,
    }).status_code == 200

    # 再跑一次初始化（模拟重启）
    uid = ensure_admin(s.database_url, s.admin_username, s.admin_password)
    assert uid > 0

    # 新密码仍然有效，旧默认密码失效
    assert client.post("/auth/login", json={
        "username": s.admin_username, "password": new_password}).status_code == 200
    assert client.post("/auth/login", json={
        "username": s.admin_username, "password": s.admin_password}).status_code == 401

    # 复原，避免影响其他用例
    client.post("/auth/change-password", headers=headers, json={
        "old_password": new_password, "new_password": s.admin_password,
    })


# ---- 越权 ----

def test_normal_user_gets_403_on_admin_endpoints(client):
    token, _, _ = _register(client)
    assert client.get("/admin/users", headers=_auth(token)).status_code == 403


def test_anonymous_gets_401_on_admin_endpoints(client):
    assert client.get("/admin/users").status_code == 401


def test_admin_can_list_users(client):
    token, username, _ = _register(client)
    r = client.get("/admin/users?page=1&page_size=50", headers=_admin_headers(client))
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 1
    assert any(u["username"] == username for u in body["users"])


def test_admin_cannot_change_own_role(client):
    headers = _admin_headers(client)
    me = client.get("/auth/me", headers=headers).json()
    r = client.patch(f"/admin/users/{me['id']}/role", headers=headers, json={"role": "user"})
    assert r.status_code == 400


def test_admin_cannot_disable_self(client):
    headers = _admin_headers(client)
    me = client.get("/auth/me", headers=headers).json()
    r = client.post(f"/admin/users/{me['id']}/disabled", headers=headers,
                    json={"disabled": True})
    assert r.status_code == 400


# ---- 用户管理动作 ----

def test_admin_reset_password_then_user_logs_in(client):
    _, username, _ = _register(client)
    headers = _admin_headers(client)
    uid = next(u["id"] for u in client.get("/admin/users?page_size=100", headers=headers)
               .json()["users"] if u["username"] == username)

    assert client.post(f"/admin/users/{uid}/reset-password", headers=headers,
                       json={"new_password": "reset123"}).status_code == 200
    assert client.post("/auth/login", json={
        "username": username, "password": "reset123"}).status_code == 200


def test_admin_promote_and_demote(client):
    token, username, _ = _register(client)
    admin_h = _admin_headers(client)
    uid = next(u["id"] for u in client.get("/admin/users?page_size=100", headers=admin_h)
               .json()["users"] if u["username"] == username)

    assert client.patch(f"/admin/users/{uid}/role", headers=admin_h,
                        json={"role": "admin"}).status_code == 200
    # 升为 admin 后可以访问管理接口
    assert client.get("/admin/users", headers=_auth(token)).status_code == 200

    assert client.patch(f"/admin/users/{uid}/role", headers=admin_h,
                        json={"role": "user"}).status_code == 200
    assert client.get("/admin/users", headers=_auth(token)).status_code == 403


def test_admin_disable_user_blocks_login_and_existing_token(client):
    token, username, password = _register(client)
    admin_h = _admin_headers(client)
    uid = next(u["id"] for u in client.get("/admin/users?page_size=100", headers=admin_h)
               .json()["users"] if u["username"] == username)

    assert client.post(f"/admin/users/{uid}/disabled", headers=admin_h,
                       json={"disabled": True}).status_code == 200
    # 登录被拒（与密码错同一句话，不暴露"账号被禁"）
    r = client.post("/auth/login", json={"username": username, "password": password})
    assert r.status_code == 401
    # 旧 token 也失效
    assert client.get("/auth/me", headers=_auth(token)).status_code == 401

    assert client.post(f"/admin/users/{uid}/disabled", headers=admin_h,
                       json={"disabled": False}).status_code == 200
    assert client.post("/auth/login", json={
        "username": username, "password": password}).status_code == 200


def test_admin_reset_password_404_for_unknown_user(client):
    r = client.post("/admin/users/99999999/reset-password", headers=_admin_headers(client),
                    json={"new_password": "whatever1"})
    assert r.status_code == 404


# ---- 改密码 ----

def test_change_password_wrong_old_is_400(client):
    token, _, _ = _register(client)
    r = client.post("/auth/change-password", headers=_auth(token), json={
        "old_password": "wrongold", "new_password": "newpass123"})
    assert r.status_code == 400


def test_change_password_then_login_with_new(client):
    token, username, _ = _register(client)
    assert client.post("/auth/change-password", headers=_auth(token), json={
        "old_password": "pass1234", "new_password": "brandnew1"}).status_code == 200
    assert client.post("/auth/login", json={
        "username": username, "password": "brandnew1"}).status_code == 200
    assert client.post("/auth/login", json={
        "username": username, "password": "pass1234"}).status_code == 401


def test_change_password_too_short_is_422(client):
    token, _, _ = _register(client)
    r = client.post("/auth/change-password", headers=_auth(token), json={
        "old_password": "pass1234", "new_password": "12345"})   # 少于 6 位
    assert r.status_code == 422


def test_change_password_anonymous_is_401(client):
    r = client.post("/auth/change-password", json={
        "old_password": "x", "new_password": "newpass123"})
    assert r.status_code == 401
