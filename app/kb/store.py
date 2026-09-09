"""知识库数据访问层：可见性解析与（后续）kb/documents/chunks 的 CRUD。

SQL 一律参数绑定；返回给上层的是纯数据（dict/list），不暴露连接。
"""
from __future__ import annotations

import psycopg

from app.kb.schema import DEFAULT_KB_ID


def fetch_kb(dsn: str, kb_id: int) -> dict | None:
    """取一个知识库的基本信息；不存在返回 None。"""
    with psycopg.connect(dsn) as conn:
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
    with psycopg.connect(dsn) as conn:
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
