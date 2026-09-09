"""三层知识库模型的 DDL 集成测试（@pytest.mark.db）。

验的是"建表这件事本身"：幂等、列/约束/索引到位、既有表增量列正确。
连不上库整模块 skip（沿用 test_rag_db.py 的约定），离线环境不影响全绿。
"""
from __future__ import annotations

import psycopg
import pytest

from app.api.auth import ensure_user_schema
from app.core.config import Settings
from app.kb.schema import DEFAULT_KB_ID, ensure_kb_schema

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
def conn():
    s = Settings()
    try:
        c = psycopg.connect(s.database_url)
    except psycopg.OperationalError as e:
        pytest.skip(f"labor-pg 不可达（docker compose up -d 后重跑）：{e}")
    ensure_user_schema(s.database_url)  # kb.owner_id 外键依赖 users
    ensure_kb_schema(s.database_url)
    ensure_kb_schema(s.database_url)    # 幂等：第二次调用不得报错
    yield c
    c.close()


def _columns(conn, table: str) -> dict[str, str]:
    rows = conn.execute(
        "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s",
        (table,),
    ).fetchall()
    return {name: dtype for name, dtype in rows}


def test_three_tables_exist(conn):
    names = {r[0] for r in conn.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
    ).fetchall()}
    assert {"kb", "documents", "chunks"} <= names


def test_chunks_columns(conn):
    cols = _columns(conn, "chunks")
    # embedding 列由 PgVectorStore.ensure_ready 后加，此处只验静态 DDL 的列
    assert {"id", "kb_id", "doc_id", "seq", "page", "heading", "content",
            "char_count", "content_hash"} <= set(cols)


def test_documents_columns_and_status_check(conn):
    cols = _columns(conn, "documents")
    assert {"source_type", "source_law_id", "status", "error", "chunk_count"} <= set(cols)
    # 用约束定义断言 CHECK 存在，而不是插非法值试探——插值即使回滚也会推进
    # sequence（id 被永久消耗），让文档 id 变得不可预期
    defs = " ".join(
        r[0] or ""
        for r in conn.execute(
            "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
            "WHERE conrelid = 'documents'::regclass AND contype = 'c'"
        ).fetchall()
    )
    assert "status" in defs and "pending" in defs and "failed" in defs


def test_users_role_column_defaults_to_user(conn):
    cols = _columns(conn, "users")
    assert "role" in cols
    default = conn.execute(
        "SELECT column_default FROM information_schema.columns "
        "WHERE table_name = 'users' AND column_name = 'role'"
    ).fetchone()[0]
    assert "user" in (default or "")


def test_chat_sessions_kb_id_is_set_null_on_delete(conn):
    """删库不能被历史会话阻断：外键规则必须是 SET NULL 而不是默认 NO ACTION。"""
    rule = conn.execute(
        """
        SELECT rc.delete_rule
        FROM information_schema.referential_constraints rc
        JOIN information_schema.table_constraints tc
          ON tc.constraint_name = rc.constraint_name
        WHERE tc.table_name = 'chat_sessions' AND tc.constraint_type = 'FOREIGN KEY'
        """
    ).fetchall()
    assert "SET NULL" in {r[0] for r in rule}


def test_chunks_unique_doc_seq(conn):
    """(doc_id, seq) 唯一：重传同一文档靠 ON CONFLICT 原地更新而不是堆重复片。"""
    idx = conn.execute(
        "SELECT indexdef FROM pg_indexes WHERE tablename = 'chunks'"
    ).fetchall()
    assert any("UNIQUE" in d[0] and "doc_id" in d[0] and "seq" in d[0] for d in idx)


def test_expected_indexes_exist(conn):
    idx = {r[0] for r in conn.execute(
        "SELECT indexname FROM pg_indexes WHERE schemaname = 'public'"
    ).fetchall()}
    assert {"idx_kb_owner", "idx_doc_kb", "idx_chunk_kb", "idx_chunk_doc",
            "idx_sessions_user_updated", "idx_messages_session"} <= idx


def test_default_kb_id_constant():
    """默认库身份是跨模块常量（迁移脚本/检索缺省值/测试共用），固定为 1。"""
    assert DEFAULT_KB_ID == 1
