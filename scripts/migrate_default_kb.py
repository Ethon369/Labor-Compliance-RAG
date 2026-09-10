"""把 laws/articles 里的内置法条迁移成"默认知识库"（kb_id=1）。

为什么要有这一步：V2 起检索层统一走 chunks 表（多知识库共用一套代码），内置法条不再是
特殊路径，而是"默认库里的若干个文档"。迁移规则：
- 一部法 = 一个 document（短名如"劳动法"，source_law_id 记来源）
- 一条法条 = 一个 chunk（seq=条号，可逆；heading=章名；page 为空）
- articles 表保留不删，继续作为迁移数据源（见决策 11）

用法：
    python scripts/migrate_default_kb.py                # 幂等迁移 + 补 embedding
    python scripts/migrate_default_kb.py --skip-embed   # 只写文本片，不灌向量（测试/离线）
    python scripts/migrate_default_kb.py --force-embed  # 全量重算向量（换 embedding 提供方后）
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# 让脚本能以"项目根包"方式 import app.*（与 backfill_embeddings.py 同款做法）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg  # noqa: E402

from app.api.auth import ensure_admin, ensure_user_schema  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.kb.schema import DEFAULT_KB_ID, ensure_kb_schema  # noqa: E402
from app.rag.vectorstore import PgVectorStore, format_vector  # noqa: E402
from backfill_embeddings import _make_embedder  # noqa: E402

DEFAULT_KB_NAME = "劳动与社会保障法律法规库"
DEFAULT_KB_DESC = "内置劳动与社会保障领域 13 部法律法规全文，条款级切分。所有用户可问答。"

UPSERT_KB_SQL = """
INSERT INTO kb (id, name, description, owner_id, is_public)
VALUES (%s, %s, %s, %s, TRUE)
ON CONFLICT (id) DO UPDATE SET
    name = EXCLUDED.name,
    description = EXCLUDED.description
"""

UPSERT_CHUNK_SQL = """
INSERT INTO chunks (kb_id, doc_id, seq, page, heading, content, char_count, content_hash)
VALUES (%s, %s, %s, NULL, %s, %s, %s, %s)
ON CONFLICT (doc_id, seq) DO UPDATE SET
    heading      = EXCLUDED.heading,
    content      = EXCLUDED.content,
    char_count   = EXCLUDED.char_count,
    content_hash = EXCLUDED.content_hash
"""


def short_title(law_name: str) -> str:
    """法名取短名：《中华人民共和国劳动法》→《劳动法》。引用卡片上更易读。"""
    return law_name.removeprefix("中华人民共和国") or law_name


def content_hash(text: str) -> str:
    """内容指纹：重跑迁移时未变的片可跳过重算 embedding。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _find_or_create_doc(cur, kb_id: int, law_id: str, title: str, chunk_count: int) -> int:
    """按 (kb_id, source_law_id) 找文档，没有就建。没有唯一约束，所以先查后插。"""
    row = cur.execute(
        "SELECT id FROM documents WHERE kb_id = %s AND source_law_id = %s",
        (kb_id, law_id),
    ).fetchone()
    if row is not None:
        return row[0]
    created = cur.execute(
        "INSERT INTO documents (kb_id, title, source_type, source_law_id, status, chunk_count) "
        "VALUES (%s, %s, 'text', %s, 'ready', %s) RETURNING id",
        (kb_id, title, law_id, chunk_count),
    ).fetchone()
    return created[0]


def migrate(dsn: str, owner_id: int, *, embed: bool, force_embed: bool) -> dict[str, int]:
    """写默认库/文档/切片，返回统计。整个迁移在一个事务里，中途失败不留半套数据。"""
    stats = {"documents": 0, "chunks": 0, "embedded": 0}
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(UPSERT_KB_SQL, (DEFAULT_KB_ID, DEFAULT_KB_NAME, DEFAULT_KB_DESC, owner_id))

        laws = cur.execute(
            "SELECT l.law_id, l.name, count(a.article_no) "
            "FROM laws l LEFT JOIN articles a USING (law_id) "
            "GROUP BY l.law_id, l.name ORDER BY l.law_id"
        ).fetchall()

        doc_ids: list[int] = []
        for law_id, law_name, chunk_count in laws:
            doc_id = _find_or_create_doc(cur, DEFAULT_KB_ID, law_id, short_title(law_name), chunk_count)
            doc_ids.append(doc_id)
            articles = cur.execute(
                "SELECT article_no, chapter, text FROM articles WHERE law_id = %s ORDER BY article_no",
                (law_id,),
            ).fetchall()
            cur.executemany(
                UPSERT_CHUNK_SQL,
                [
                    (DEFAULT_KB_ID, doc_id, no, chapter, text, len(text), content_hash(text))
                    for no, chapter, text in articles
                ],
            )
            cur.execute(
                "UPDATE documents SET chunk_count = %s, status = 'ready', error = NULL, "
                "updated_at = now() WHERE id = %s",
                (len(articles), doc_id),
            )
            stats["documents"] += 1
            stats["chunks"] += len(articles)

        cur.execute(
            "UPDATE kb SET doc_count = (SELECT count(*) FROM documents WHERE kb_id = %s), "
            "chunk_count = (SELECT count(*) FROM chunks WHERE kb_id = %s), updated_at = now() "
            "WHERE id = %s",
            (DEFAULT_KB_ID, DEFAULT_KB_ID, DEFAULT_KB_ID),
        )

        if embed:
            where = "" if force_embed else "AND embedding IS NULL"
            pending = cur.execute(
                f"SELECT id, content FROM chunks WHERE kb_id = %s {where} ORDER BY id",
                (DEFAULT_KB_ID,),
            ).fetchall()
            if pending:
                embedder = _make_embedder(Settings())
                vectors = embedder.embed([text for _, text in pending])
                cur.executemany(
                    "UPDATE chunks SET embedding = %s::vector WHERE id = %s",
                    [(format_vector(v), cid) for (cid, _), v in zip(pending, vectors)],
                )
                stats["embedded"] = len(pending)
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-embed", action="store_true", help="只写文本片，不灌向量")
    ap.add_argument("--force-embed", action="store_true", help="全量重算向量")
    args = ap.parse_args(argv)

    settings = Settings()
    if not settings.database_url:
        print("[fail] 未设置 DATABASE_URL（.env 或环境变量）", file=sys.stderr)
        return 1

    try:
        ensure_user_schema(settings.database_url)   # 依赖 users 表
        ensure_kb_schema(settings.database_url)     # 建 kb/documents/chunks + role 列
        if not args.skip_embed:
            # 建 chunks.embedding 列（幂等），否则下面的向量回填无处可写
            PgVectorStore(settings.database_url, settings.embedding_dim).ensure_ready()
        owner_id = ensure_admin(
            settings.database_url, settings.admin_username, settings.admin_password
        )
        stats = migrate(
            settings.database_url, owner_id,
            embed=not args.skip_embed, force_embed=args.force_embed,
        )
    except Exception as e:
        print(f"[fail] 迁移失败: {e}", file=sys.stderr)
        return 1

    print(f"[ok] 默认库 kb_id={DEFAULT_KB_ID}：{stats['documents']} 个文档 / "
          f"{stats['chunks']} 个切片，本次写入向量 {stats['embedded']} 条")

    # 自校验：默认库的文档/切片数必须与 articles 表一致（一部法=一文档，一条法=一片）。
    # 期望值从数据源实时推导而不硬编码——扩充法域后无需同步改这里。
    with psycopg.connect(settings.database_url) as conn:
        docs, chunks = conn.execute(
            "SELECT count(*) FROM documents WHERE kb_id = %s", (DEFAULT_KB_ID,)
        ).fetchone()[0], conn.execute(
            "SELECT count(*) FROM chunks WHERE kb_id = %s", (DEFAULT_KB_ID,)
        ).fetchone()[0]
        exp_docs, exp_chunks = conn.execute(
            "SELECT count(DISTINCT law_id), count(*) FROM articles"
        ).fetchone()
    if (docs, chunks) != (exp_docs, exp_chunks):
        print(f"[fail] 自校验失败：articles 有 {exp_docs} 部法 / {exp_chunks} 条，"
              f"默认库实际 {docs} 文档 / {chunks} 片", file=sys.stderr)
        return 1
    print(f"[ok] 自校验通过：{docs} 文档 / {chunks} 切片")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
