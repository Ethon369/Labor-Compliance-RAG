"""知识库接口端到端测试（@pytest.mark.db）。

重点不是"接口能不能调通"，而是**权限边界**：他人私有库 404（不泄露存在性）、
公开库非 owner 写 403、普通用户调管理员接口 403、游客一律 401。
前端隐藏入口只是体验，这些用例才是安全边界的证明。
"""
from __future__ import annotations

import secrets

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.agent.graph import build_agent
from app.api.auth import ensure_admin
from app.core.config import Settings
from app.kb import store
from app.kb.schema import ensure_kb_schema
from app.main import app as fastapi_app

pytestmark = pytest.mark.db


class _FakeRetriever:
    """最小检索器替身：让 lifespan 不必连真库也能装配 Agent。"""

    def search(self, query, top_k=8, use_rerank=True, kb_ids=None):
        from app.rag.models import RetrievalResult
        return RetrievalResult(query=query, hits=[])


@pytest.fixture
def client():
    s = Settings()
    try:
        with psycopg.connect(s.database_url):
            pass
    except psycopg.OperationalError as e:
        pytest.skip(f"labor-pg 不可达（docker compose up -d 后重跑）：{e}")
    ensure_kb_schema(s.database_url)
    with TestClient(fastapi_app) as c:
        fastapi_app.state.agent = build_agent(_FakeRetriever(), verify_mode="pass")
        fastapi_app.state.history = []
        yield c


def _register(client) -> tuple[str, int]:
    username = "u_" + secrets.token_hex(4)
    r = client.post("/auth/register", json={"username": username, "password": "pass1234"})
    assert r.status_code == 200, r.text
    body = r.json()
    # 注册响应只有 token/username/role；uid 从 /auth/me 取
    me = client.get("/auth/me", headers=_auth(body["token"])).json()
    return body["token"], me["id"]


def _auth(token: str) -> dict:
    return {"Authorization": "Bearer " + token}


def _admin_headers(client) -> dict:
    s = Settings()
    r = client.post("/auth/login", json={
        "username": s.admin_username, "password": s.admin_password,
    })
    assert r.status_code == 200, r.text
    return _auth(r.json()["token"])


def _new_kb(client, token, **kw) -> int:
    r = client.post("/kb", json={"name": "kb_" + secrets.token_hex(3), **kw},
                    headers=_auth(token))
    assert r.status_code == 200, r.text
    return r.json()["id"]


# ---- 未认证 ----

def test_anonymous_access_is_401(client):
    assert client.get("/kb").status_code == 401
    assert client.get("/kb/1").status_code == 401
    assert client.get("/kb/admin/all").status_code == 401


# ---- 建库 / 列表 / 详情 ----

def test_create_then_list_then_get(client):
    token, _ = _register(client)
    kb_id = _new_kb(client, token, description="我的库")

    listed = client.get("/kb", headers=_auth(token)).json()
    assert kb_id in {k["id"] for k in listed["kbs"]}

    detail = client.get(f"/kb/{kb_id}", headers=_auth(token))
    assert detail.status_code == 200
    assert detail.json()["description"] == "我的库"
    assert detail.json()["chunk_size"] == 500          # 默认切分参数


def test_update_and_delete_own_kb(client):
    token, _ = _register(client)
    kb_id = _new_kb(client, token)

    assert client.patch(f"/kb/{kb_id}", json={"name": "改名了", "is_public": True},
                        headers=_auth(token)).status_code == 200
    assert client.get(f"/kb/{kb_id}", headers=_auth(token)).json()["name"] == "改名了"

    assert client.delete(f"/kb/{kb_id}", headers=_auth(token)).status_code == 200
    assert client.get(f"/kb/{kb_id}", headers=_auth(token)).status_code == 404


# ---- 越权语义（本文件的重点）----

def test_others_private_kb_is_404_not_403(client):
    """他人私有库：看不到就是不存在——404，不泄漏资源存在性。"""
    owner_token, _ = _register(client)
    other_token, _ = _register(client)
    kb_id = _new_kb(client, owner_token)

    assert client.get(f"/kb/{kb_id}", headers=_auth(other_token)).status_code == 404
    assert client.delete(f"/kb/{kb_id}", headers=_auth(other_token)).status_code == 404
    assert client.patch(f"/kb/{kb_id}", json={"name": "x"},
                        headers=_auth(other_token)).status_code == 404
    # 别人的列表里也不该出现
    assert kb_id not in {k["id"] for k in
                         client.get("/kb", headers=_auth(other_token)).json()["kbs"]}


def test_public_kb_readable_but_not_writable_by_others(client):
    """公开库：他人可读（200）、不可写（403，因为资源可见，拒的是权限）。"""
    owner_token, _ = _register(client)
    other_token, _ = _register(client)
    kb_id = _new_kb(client, owner_token, is_public=True)

    assert client.get(f"/kb/{kb_id}", headers=_auth(other_token)).status_code == 200
    assert client.patch(f"/kb/{kb_id}", json={"name": "篡改"},
                        headers=_auth(other_token)).status_code == 403
    assert client.delete(f"/kb/{kb_id}", headers=_auth(other_token)).status_code == 403
    # 文档列表可读，上传不可
    assert client.get(f"/kb/{kb_id}/documents",
                      headers=_auth(other_token)).status_code == 200


def test_admin_sees_and_manages_others_private_kb(client):
    token, _ = _register(client)
    kb_id = _new_kb(client, token)
    headers = _admin_headers(client)

    assert client.get(f"/kb/{kb_id}", headers=headers).status_code == 200
    assert kb_id in {k["id"] for k in client.get("/kb/admin/all", headers=headers).json()["kbs"]}
    assert client.patch(f"/kb/{kb_id}", json={"name": "管理员改名"},
                        headers=headers).status_code == 200


def test_normal_user_cannot_call_admin_kb_view(client):
    token, _ = _register(client)
    assert client.get("/kb/admin/all", headers=_auth(token)).status_code == 403


# ---- 上传 ----

def test_upload_rejects_unsupported_extension(client):
    token, _ = _register(client)
    kb_id = _new_kb(client, token)
    r = client.post(f"/kb/{kb_id}/documents", headers=_auth(token),
                    files={"file": ("恶意.exe", b"MZ\x00\x00", "application/octet-stream")})
    assert r.status_code == 400


def test_upload_returns_pending_then_pipeline_finishes(client):
    """上传立即返回 pending，后台任务把它推到终态（轮询接口能看到）。"""
    token, _ = _register(client)
    kb_id = _new_kb(client, token, chunk_size=120, chunk_overlap=0)

    r = client.post(f"/kb/{kb_id}/documents", headers=_auth(token),
                    files={"file": ("手册.txt", ("甲" * 300 + "。").encode("utf-8"),
                                    "text/plain")})
    assert r.status_code == 200, r.text
    doc_id = r.json()["id"]
    assert r.json()["status"] == "pending"

    # BackgroundTasks 在 TestClient 里是同步跑完的，这里直接看终态
    doc = client.get(f"/kb/documents/{doc_id}", headers=_auth(token)).json()
    assert doc["status"] in ("ready", "failed")
    assert doc["status"] == "ready"
    assert doc["chunk_count"] > 1


def test_upload_to_others_kb_is_rejected(client):
    owner_token, _ = _register(client)
    other_token, _ = _register(client)
    kb_id = _new_kb(client, owner_token)
    r = client.post(f"/kb/{kb_id}/documents", headers=_auth(other_token),
                    files={"file": ("x.txt", "内容".encode("utf-8"), "text/plain")})
    assert r.status_code == 404


def test_document_status_of_others_doc_is_404(client):
    owner_token, _ = _register(client)
    other_token, _ = _register(client)
    kb_id = _new_kb(client, owner_token)
    doc_id = store.create_document(Settings().database_url, kb_id, "私有.txt", "txt")
    assert client.get(f"/kb/documents/{doc_id}",
                      headers=_auth(other_token)).status_code == 404


# ---- 删除文档与重建索引 ----

def test_delete_document_removes_chunks(client):
    token, _ = _register(client)
    kb_id = _new_kb(client, token)
    r = client.post(f"/kb/{kb_id}/documents", headers=_auth(token),
                    files={"file": ("d.txt", ("乙" * 200 + "。").encode("utf-8"),
                                    "text/plain")})
    doc_id = r.json()["id"]
    assert client.delete(f"/kb/documents/{doc_id}", headers=_auth(token)).status_code == 200
    assert client.get(f"/kb/documents/{doc_id}", headers=_auth(token)).status_code == 404


def test_reindex_requires_write_access(client):
    owner_token, _ = _register(client)
    other_token, _ = _register(client)
    kb_id = _new_kb(client, owner_token, is_public=True)

    assert client.post(f"/kb/{kb_id}/reindex", headers=_auth(owner_token)).status_code == 200
    assert client.post(f"/kb/{kb_id}/reindex", headers=_auth(other_token)).status_code == 403
