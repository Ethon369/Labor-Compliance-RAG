"""知识库存储层 CRUD 集成测试（@pytest.mark.db）。

验的是"写进去的东西和读出来的东西是不是一回事"：三层 CRUD、计数同步、重传变短时的
尾片清理、可见性规则与 visible_kb_ids 不漂移。连不上库整模块 skip。
"""
from __future__ import annotations

import secrets

import psycopg
import pytest

from app.api.auth import ensure_user_schema, hash_password
from app.core.config import Settings
from app.kb.schema import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    ensure_kb_schema,
)
from app.kb import store
from app.rag.embedder import HashEmbedder
from app.rag.vectorstore import PgVectorStore

pytestmark = pytest.mark.db


@pytest.fixture
def conn():
    s = Settings()
    try:
        c = psycopg.connect(s.database_url)
    except psycopg.OperationalError as e:
        pytest.skip(f"labor-pg 不可达（docker compose up -d 后重跑）：{e}")
    ensure_user_schema(s.database_url)
    ensure_kb_schema(s.database_url)
    PgVectorStore(s.database_url, s.embedding_dim).ensure_ready()  # chunks.embedding 列
    yield c
    c.close()


@pytest.fixture
def dsn():
    return Settings().database_url


def _new_user(conn, role: str = "user") -> int:
    name = "u_" + secrets.token_hex(4)
    row = conn.execute(
        "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s) RETURNING id",
        (name, hash_password("x"), role),
    ).fetchone()
    conn.commit()
    return row[0]


def _chunks(n: int, prefix: str = "内容"):
    return [
        {"seq": i, "content": f"{prefix}{i}", "heading": "章", "page": None,
         "char_count": len(prefix) + 1, "content_hash": f"h{i}"}
        for i in range(1, n + 1)
    ]


# ---- kb CRUD ----

def test_create_kb_uses_default_chunk_params(conn, dsn):
    uid = _new_user(conn)
    kb_id = store.create_kb(dsn, uid, "默认参数库")
    kb = store.fetch_kb(dsn, kb_id)
    assert kb["chunk_size"] == DEFAULT_CHUNK_SIZE
    assert kb["chunk_overlap"] == DEFAULT_CHUNK_OVERLAP
    assert kb["is_public"] is False
    assert kb["doc_count"] == 0 and kb["chunk_count"] == 0
    assert kb["owner_id"] == uid


def test_update_kb_ignores_fields_outside_whitelist(conn, dsn):
    uid = _new_user(conn)
    kb_id = store.create_kb(dsn, uid, "改名库")
    assert store.update_kb(dsn, kb_id, name="新名字", is_public=True) is True
    # doc_count 不在白名单：流水线托管的字段不允许上层顺手改，避免计数被写歪
    assert store.update_kb(dsn, kb_id, doc_count=999) is False
    kb = store.fetch_kb(dsn, kb_id)
    assert kb["name"] == "新名字" and kb["is_public"] is True
    assert kb["doc_count"] == 0


def test_delete_kb_cascades_documents_and_chunks(conn, dsn):
    uid = _new_user(conn)
    kb_id = store.create_kb(dsn, uid, "待删库")
    doc_id = store.create_document(dsn, kb_id, "a.txt", "txt")
    store.replace_chunks(dsn, kb_id, doc_id, _chunks(3))
    assert store.delete_kb(dsn, kb_id) is True

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents WHERE kb_id = %s", (kb_id,))
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT count(*) FROM chunks WHERE kb_id = %s", (kb_id,))
        assert cur.fetchone()[0] == 0


def test_list_kbs_visibility_matches_visible_kb_ids(conn, dsn):
    owner = _new_user(conn)
    other = _new_user(conn)
    admin = _new_user(conn, role="admin")
    private = store.create_kb(dsn, owner, "私有库")
    public = store.create_kb(dsn, owner, "公开库", is_public=True)

    own, _ = store.list_kbs(dsn, uid=owner)
    assert {k["id"] for k in own} >= {private, public}

    others, _ = store.list_kbs(dsn, uid=other)
    ids = {k["id"] for k in others}
    assert public in ids and private not in ids

    # 列表可见性与检索可见性必须一致，否则会出现"列表有、检索不到"的库
    assert set(store.visible_kb_ids(dsn, other)) >= {public}
    assert private not in set(store.visible_kb_ids(dsn, other))

    all_ids = {k["id"] for k in store.list_kbs(dsn, uid=admin)[0]}
    assert {private, public} <= all_ids


def test_list_kbs_pagination_and_owner_name(conn, dsn):
    uid = _new_user(conn)
    for i in range(3):
        store.create_kb(dsn, uid, f"分页库{i}")
    page1, total = store.list_kbs(dsn, uid=uid, page=1, page_size=2)
    page2, _ = store.list_kbs(dsn, uid=uid, page=2, page_size=2)
    assert len(page1) == 2 and len(page2) >= 1 and total >= 3
    assert {p["id"] for p in page1}.isdisjoint({p["id"] for p in page2})
    assert page1[0]["owner_name"]     # JOIN users 带出所有者用户名


# ---- documents ----

def test_create_document_starts_pending(conn, dsn):
    uid = _new_user(conn)
    kb_id = store.create_kb(dsn, uid, "文档库")
    doc_id = store.create_document(dsn, kb_id, "手册.pdf", "pdf", file_size=2048)
    doc = store.get_document(dsn, doc_id)
    assert doc["status"] == "pending" and doc["error"] is None
    assert doc["source_type"] == "pdf" and doc["file_size"] == 2048


def test_set_document_status_clears_previous_error(conn, dsn):
    uid = _new_user(conn)
    kb_id = store.create_kb(dsn, uid, "状态库")
    doc_id = store.create_document(dsn, kb_id, "x.txt", "txt")
    store.set_document_status(dsn, doc_id, "failed", error="扫描件需要 OCR")
    assert store.get_document(dsn, doc_id)["error"] == "扫描件需要 OCR"

    store.set_document_status(dsn, doc_id, "ready", error=None, chunk_count=7)
    doc = store.get_document(dsn, doc_id)
    assert doc["status"] == "ready" and doc["error"] is None
    assert doc["chunk_count"] == 7


def test_list_documents_is_paginated_newest_first(conn, dsn):
    uid = _new_user(conn)
    kb_id = store.create_kb(dsn, uid, "列表库")
    ids = [store.create_document(dsn, kb_id, f"d{i}.txt", "txt") for i in range(3)]
    docs, total = store.list_documents(dsn, kb_id, page=1, page_size=2)
    assert total == 3 and len(docs) == 2
    assert docs[0]["id"] > docs[1]["id"]     # id DESC
    assert docs[0]["id"] == ids[-1]


# ---- chunks ----

def test_replace_chunks_truncates_stale_tail_on_reupload(conn, dsn):
    """重传后文档变短：只 upsert 会让多出来的旧片永远留在库里被检索到。"""
    uid = _new_user(conn)
    kb_id = store.create_kb(dsn, uid, "重传库")
    doc_id = store.create_document(dsn, kb_id, "v.txt", "txt")

    store.replace_chunks(dsn, kb_id, doc_id, _chunks(5))
    assert store.fetch_kb(dsn, kb_id)["chunk_count"] == 5

    store.replace_chunks(dsn, kb_id, doc_id, _chunks(3, prefix="新"))
    with conn.cursor() as cur:
        cur.execute("SELECT seq, content FROM chunks WHERE doc_id = %s ORDER BY seq", (doc_id,))
        rows = cur.fetchall()
    assert [r[0] for r in rows] == [1, 2, 3]
    assert all(r[1].startswith("新") for r in rows)
    kb = store.fetch_kb(dsn, kb_id)
    assert kb["chunk_count"] == 3


def test_replace_chunks_writes_embeddings_and_counts(conn, dsn):
    uid = _new_user(conn)
    kb_id = store.create_kb(dsn, uid, "向量库")
    doc_id = store.create_document(dsn, kb_id, "e.txt", "txt")
    chunks = _chunks(2)
    embedder = HashEmbedder(Settings().embedding_dim)
    vecs = [embedder.embed_one(c["content"]) for c in chunks]

    assert store.replace_chunks(dsn, kb_id, doc_id, chunks, embeddings=vecs) == 2

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks WHERE doc_id = %s AND embedding IS NOT NULL",
                    (doc_id,))
        assert cur.fetchone()[0] == 2
    doc = store.get_document(dsn, doc_id)
    assert doc["chunk_count"] == 2
    kb = store.fetch_kb(dsn, kb_id)
    assert kb["doc_count"] == 1 and kb["chunk_count"] == 2


def test_replace_chunks_without_embeddings_leaves_them_pending(conn, dsn):
    """不传向量时整篇标记待回填，由 backfill 脚本补——不能留下"假装有向量"的行。"""
    uid = _new_user(conn)
    kb_id = store.create_kb(dsn, uid, "待回填库")
    doc_id = store.create_document(dsn, kb_id, "p.txt", "txt")
    store.replace_chunks(dsn, kb_id, doc_id, _chunks(2))
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks WHERE doc_id = %s AND embedding IS NULL",
                    (doc_id,))
        assert cur.fetchone()[0] == 2


def test_delete_document_syncs_kb_counts(conn, dsn):
    uid = _new_user(conn)
    kb_id = store.create_kb(dsn, uid, "计数库")
    d1 = store.create_document(dsn, kb_id, "one.txt", "txt")
    d2 = store.create_document(dsn, kb_id, "two.txt", "txt")
    store.replace_chunks(dsn, kb_id, d1, _chunks(4))
    store.replace_chunks(dsn, kb_id, d2, _chunks(6))
    kb0 = store.fetch_kb(dsn, kb_id)
    assert kb0["doc_count"] == 2 and kb0["chunk_count"] == 10

    assert store.delete_document(dsn, d1) is True
    kb = store.fetch_kb(dsn, kb_id)
    assert kb["doc_count"] == 1 and kb["chunk_count"] == 6
    assert store.delete_document(dsn, 999999) is False
