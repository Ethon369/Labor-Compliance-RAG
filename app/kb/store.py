"""知识库数据访问层：可见性解析 + kb/documents/chunks 的 CRUD。

SQL 一律参数绑定；返回给上层的是纯数据（dict/list），不暴露连接。

计数（doc_count / chunk_count）一律用 COUNT 重算而不是增量累加：增量要在"上传失败
回滚、删文档、重传变短"等每条路径上都记得加减，漏一处就永久漂移；重算多一次往返，
但它是自证的——任何时候查出来的数都是真的。
"""
from __future__ import annotations

from app.core.db import pool_conn

from app.kb.schema import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_KB_ID
from app.rag.vectorstore import format_vector

# kb 可更新字段白名单：SET 子句的列名只能来自这里（列名无法参数绑定，值才绑定）
_KB_UPDATABLE = ("name", "description", "is_public", "chunk_size", "chunk_overlap")


def fetch_kb(dsn: str, kb_id: int) -> dict | None:
    """取一个知识库的基本信息；不存在返回 None。"""
    with pool_conn(dsn) as conn:
        row = conn.execute(
            "SELECT id, name, description, owner_id, is_public, chunk_size, chunk_overlap, "
            "doc_count, chunk_count, created_at FROM kb WHERE id = %s",
            (kb_id,),
        ).fetchone()
    if row is None:
        return None
    return {"id": row[0], "name": row[1], "description": row[2], "owner_id": row[3],
            "is_public": row[4], "chunk_size": row[5], "chunk_overlap": row[6],
            "doc_count": row[7], "chunk_count": row[8], "created_at": row[9].isoformat()}


def visible_kb_ids(dsn: str, uid: int | None) -> list[int]:
    """该用户可检索的知识库 id 集合：自己的 + 公开的；admin 是全部库。

    uid=None（游客）只给公开库。为什么用一条 JOIN 而不是先查角色再分支：
    少一次往返，且"admin 看全部"这条规则与其它条件同处一个 WHERE，不会漏改。
    """
    with pool_conn(dsn) as conn:
        if uid is None:
            rows = conn.execute("SELECT id FROM kb WHERE is_public ORDER BY id").fetchall()
        else:
            rows = conn.execute(
                """
                SELECT k.id
                FROM kb k LEFT JOIN users u ON u.id = %s
                WHERE u.role = 'admin' OR k.owner_id = %s OR k.is_public
                ORDER BY k.id
                """,
                (uid, uid),
            ).fetchall()
    return [r[0] for r in rows]


def resolve_kb_ids(dsn: str, uid: int | None, requested: int | None) -> list[int]:
    """解析本次检索范围：在可见集合内收窄到 requested；不可见时返回空列表。

    返回空列表代表"请求的库不可见"（调用方按 404 处理，不泄露资源是否存在）；
    没有 requested 时若可见集合为空（极端情况），回退默认库，保证问答不至于全空。
    """
    visible = visible_kb_ids(dsn, uid)
    if requested is not None:
        return [requested] if requested in visible else []
    return visible or [DEFAULT_KB_ID]


# ---------------------------------------------------------------------------
# kb CRUD
# ---------------------------------------------------------------------------

def create_kb(dsn: str, owner_id: int, name: str, description: str = "",
              is_public: bool = False,
              chunk_size: int = DEFAULT_CHUNK_SIZE,
              chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> int:
    """建库，返回 kb_id。切分参数随库存储（不同语料适合的片长不同，见 schema 注释）。"""
    with pool_conn(dsn) as conn:
        row = conn.execute(
            "INSERT INTO kb (name, description, owner_id, is_public, chunk_size, chunk_overlap)"
            " VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
            (name, description, owner_id, is_public, chunk_size, chunk_overlap),
        ).fetchone()
    return row[0]


def update_kb(dsn: str, kb_id: int, **fields) -> bool:
    """更新库的可变字段；只认白名单内的键，返回是否命中行。

    error/chunk_count 这类由流水线托管的字段不在这里，避免上层误改导致状态不一致。
    """
    cols = [c for c in fields if c in _KB_UPDATABLE]
    if not cols:
        return False
    # 列名来自 _KB_UPDATABLE 常量（非用户输入），值全部走 %s
    set_sql = ", ".join(f"{c} = %s" for c in cols)
    params = [fields[c] for c in cols] + [kb_id]
    with pool_conn(dsn) as conn:
        cur = conn.execute(
            f"UPDATE kb SET {set_sql}, updated_at = now() WHERE id = %s", params
        )
        return cur.rowcount > 0


def delete_kb(dsn: str, kb_id: int) -> bool:
    """删库。documents/chunks 靠 ON DELETE CASCADE 一起清；历史会话的 kb_id 置 NULL。"""
    with pool_conn(dsn) as conn:
        cur = conn.execute("DELETE FROM kb WHERE id = %s", (kb_id,))
        return cur.rowcount > 0


def list_kbs(dsn: str, uid: int | None = None, page: int = 1,
             page_size: int = 20) -> tuple[list[dict], int]:
    """分页列出用户可见的知识库：自己的 + 公开的，admin 是全部（游客只看公开）。

    可见性规则与 visible_kb_ids 保持一致——两处若漂移，用户会看到"列表里有但检索
    检索不到"的库，所以这里复用同一条 WHERE 构造。
    """
    page = max(page, 1)
    page_size = min(max(page_size, 1), 100)
    if uid is None:
        base = "FROM kb k JOIN users u ON u.id = k.owner_id"
        where, params = "WHERE k.is_public", []
    else:
        # viewer 用 LEFT JOIN 带进来，避免"先查 role 再分支"多一次往返
        base = ("FROM kb k JOIN users u ON u.id = k.owner_id"
                " LEFT JOIN users v ON v.id = %s")
        where = "WHERE (v.role = 'admin' OR k.owner_id = %s OR k.is_public)"
        params = [uid, uid]
    with pool_conn(dsn) as conn:
        total = conn.execute(f"SELECT count(*) {base} {where}", params).fetchone()[0]
        rows = conn.execute(
            "SELECT k.id, k.name, k.description, k.owner_id, k.is_public,"
            " k.chunk_size, k.chunk_overlap, k.doc_count, k.chunk_count,"
            " k.created_at, k.updated_at, u.username "
            f"{base} {where} ORDER BY k.id DESC LIMIT %s OFFSET %s",
            params + [page_size, (page - 1) * page_size],
        ).fetchall()
    return [_kb_row(r) for r in rows], total


def _kb_row(r) -> dict:
    return {"id": r[0], "name": r[1], "description": r[2], "owner_id": r[3],
            "is_public": r[4], "chunk_size": r[5], "chunk_overlap": r[6],
            "doc_count": r[7], "chunk_count": r[8],
            "created_at": r[9].isoformat(), "updated_at": r[10].isoformat(),
            "owner_name": r[11]}


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------

def create_document(dsn: str, kb_id: int, title: str, source_type: str,
                    file_size: int | None = None,
                    source_law_id: str | None = None) -> int:
    """登记一个待处理文档（status='pending'），返回 doc_id。

    先插行再交给后台任务：客户端立刻拿到 doc_id 可轮询，不必等解析+向量化走完。
    """
    with pool_conn(dsn) as conn:
        row = conn.execute(
            "INSERT INTO documents (kb_id, title, source_type, file_size, source_law_id)"
            " VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (kb_id, title, source_type, file_size, source_law_id),
        ).fetchone()
    return row[0]


def set_document_status(dsn: str, doc_id: int, status: str, error: str | None = None,
                        chunk_count: int | None = None) -> None:
    """更新处理状态。error 每次都覆盖（成功时传 None 清掉上一次失败的残留）。"""
    sets = ["status = %s", "error = %s", "updated_at = now()"]
    params: list = [status, error]
    if chunk_count is not None:
        sets.append("chunk_count = %s")
        params.append(chunk_count)
    params.append(doc_id)
    with pool_conn(dsn) as conn:
        conn.execute(
            f"UPDATE documents SET {', '.join(sets)} WHERE id = %s", params
        )


def get_document(dsn: str, doc_id: int) -> dict | None:
    with pool_conn(dsn) as conn:
        row = conn.execute(
            "SELECT id, kb_id, title, source_type, source_law_id, file_size, status,"
            " error, chunk_count, created_at, updated_at FROM documents WHERE id = %s",
            (doc_id,),
        ).fetchone()
    return None if row is None else _doc_row(row)


def list_documents(dsn: str, kb_id: int, page: int = 1,
                   page_size: int = 20) -> tuple[list[dict], int]:
    page = max(page, 1)
    page_size = min(max(page_size, 1), 100)
    with pool_conn(dsn) as conn:
        total = conn.execute(
            "SELECT count(*) FROM documents WHERE kb_id = %s", (kb_id,)
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT id, kb_id, title, source_type, source_law_id, file_size, status,"
            " error, chunk_count, created_at, updated_at FROM documents"
            " WHERE kb_id = %s ORDER BY id DESC LIMIT %s OFFSET %s",
            (kb_id, page_size, (page - 1) * page_size),
        ).fetchall()
    return [_doc_row(r) for r in rows], total


def _doc_row(r) -> dict:
    return {"id": r[0], "kb_id": r[1], "title": r[2], "source_type": r[3],
            "source_law_id": r[4], "file_size": r[5], "status": r[6],
            "error": r[7], "chunk_count": r[8],
            "created_at": r[9].isoformat(), "updated_at": r[10].isoformat()}


def delete_document(dsn: str, doc_id: int) -> bool:
    """删文档（chunks 级联删）并同步所属库的计数。返回是否命中。"""
    with pool_conn(dsn) as conn:
        row = conn.execute(
            "SELECT kb_id FROM documents WHERE id = %s", (doc_id,)
        ).fetchone()
        if row is None:
            return False
        kb_id = row[0]
        conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        _sync_counts(conn, kb_id)
        return True


# ---------------------------------------------------------------------------
# chunks
# ---------------------------------------------------------------------------

def replace_chunks(dsn: str, kb_id: int, doc_id: int, chunks: list[dict],
                   embeddings: list[list[float]] | None = None) -> int:
    """整篇替换某文档的切片：单事务内删尾片 + upsert + 同步计数。

    为什么先删 seq 超出的尾片：文档重传后变短时，ON CONFLICT 只会覆盖前 N 片，
    多出来的旧片会永远留在库里被检索到——这是"越用越脏"的典型来源。

    embeddings 与 chunks 等长时一并写入；不传则整篇标记待回填（embedding=NULL），
    由 backfill 脚本补。传了向量就没有"写了内容但还没向量"的中间窗口。
    """
    with pool_conn(dsn) as conn:
        if chunks:
            max_seq = max(int(c["seq"]) for c in chunks)
            conn.execute(
                "DELETE FROM chunks WHERE doc_id = %s AND seq > %s", (doc_id, max_seq)
            )
            rows = []
            for i, c in enumerate(chunks):
                vec = embeddings[i] if embeddings is not None else None
                rows.append((
                    kb_id, doc_id, int(c["seq"]), c.get("page"),
                    c.get("heading", ""), c["content"], c.get("char_count"),
                    c.get("content_hash"),
                    format_vector(vec) if vec is not None else None,
                ))
            with conn.cursor() as cur:     # psycopg3 的 executemany 在游标上，不在连接上
                cur.executemany(
                    "INSERT INTO chunks (kb_id, doc_id, seq, page, heading, content,"
                    " char_count, content_hash, embedding)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)"
                    " ON CONFLICT (doc_id, seq) DO UPDATE SET page = EXCLUDED.page,"
                    " heading = EXCLUDED.heading, content = EXCLUDED.content,"
                    " char_count = EXCLUDED.char_count, content_hash = EXCLUDED.content_hash,"
                    " embedding = EXCLUDED.embedding, created_at = now()",
                    rows,
                )
        else:
            conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
        _sync_counts(conn, kb_id, doc_id)
    return len(chunks)


def _sync_counts(conn, kb_id: int, doc_id: int | None = None) -> None:
    """重算文档与库的计数。必须在调用方的事务里执行（失败随事务一起回滚）。"""
    if doc_id is not None:
        conn.execute(
            "UPDATE documents SET chunk_count ="
            " (SELECT count(*) FROM chunks WHERE doc_id = %s) WHERE id = %s",
            (doc_id, doc_id),
        )
    conn.execute(
        "UPDATE kb SET doc_count = (SELECT count(*) FROM documents WHERE kb_id = %s),"
        " chunk_count = (SELECT count(*) FROM chunks WHERE kb_id = %s),"
        " updated_at = now() WHERE id = %s",
        (kb_id, kb_id, kb_id),
    )
