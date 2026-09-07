"""pgvector 访问层：就绪（扩展+列）、语料全量、写向量、余弦近邻检索。

向量以文本字面量 + `%s::vector` 显式转型传入——刻意不引 pgvector 的 Python 适配
包：psycopg 的参数绑定已足够，且少一个锁定清单外的依赖；转型交给 PG 端完成。
每次调用开短连接：检索频次不高（见决策 03 已知取舍），换取代码零状态、易测试。
"""
from __future__ import annotations

import psycopg

from app.rag.models import ArticleRef, LaneHit

_DIM_ALTER_SQL = "embedding vector({dim})"  # 占位拼接只拼维度数字常量，不拼用户输入


def format_vector(v: list[float]) -> str:
    """向量 → PG 的 vector 文本字面量（形如 [0.1,0.2,...]），供 %s::vector 转型。

    放在模块级公开：回填脚本灌库时复用同一格式化，保证写/查两侧格式一致。"""
    return "[" + ",".join(f"{x:.8g}" for x in v) + "]"


class PgVectorStore:
    def __init__(self, dsn: str, dim: int):
        self._dsn = dsn
        self._dim = dim

    def ensure_ready(self) -> None:
        """幂等迁移：启 pgvector 扩展、给 articles 加 embedding 列。可反复调用。"""
        with psycopg.connect(self._dsn) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"ALTER TABLE articles ADD COLUMN IF NOT EXISTS {_DIM_ALTER_SQL.format(dim=self._dim)}"
            )

    def all_articles(self) -> list[ArticleRef]:
        """拉全量语料，供 BM25 建内存倒排。排序固定，保证每次索引顺序一致。"""
        with psycopg.connect(self._dsn) as conn:
            rows = conn.execute(
                "SELECT law_id, article_no, chapter, text FROM articles ORDER BY law_id, article_no"
            ).fetchall()
        return [ArticleRef(law_id=law_id, article_no=no, chapter=ch, text=t) for law_id, no, ch, t in rows]

    def count_embedded(self) -> int:
        with psycopg.connect(self._dsn) as conn:
            return conn.execute(
                "SELECT count(*) FROM articles WHERE embedding IS NOT NULL"
            ).fetchone()[0]

    def pending_rows(self, force: bool = False) -> list[tuple[str, int, str]]:
        """待回填 (law_id, article_no, text)；force=True 全量，否则仅 embedding 为空。

        WHERE 分支由调用方布尔决定、非用户输入，仅此两态，不构成注入面。"""
        where = "" if force else "WHERE embedding IS NULL"
        with psycopg.connect(self._dsn) as conn:
            return conn.execute(
                f"SELECT law_id, article_no, text FROM articles {where} "
                "ORDER BY law_id, article_no"
            ).fetchall()

    def set_embeddings(self, rows: list[tuple[str, int, list[float]]]) -> None:
        """批量写向量。205 条一次 executemany 比逐行省一半网络往返。"""
        with psycopg.connect(self._dsn) as conn, conn.cursor() as cur:
            cur.executemany(
                "UPDATE articles SET embedding = %s::vector WHERE law_id = %s AND article_no = %s",
                [(format_vector(v), law_id, no) for law_id, no, v in rows],
            )

    def vector_topk(self, qvec: list[float], top_k: int) -> list[LaneHit]:
        """余弦近邻：`<=>` 是余弦距离（越小越近），用 1-d 转回相似度便于展示。

        `%s::vector` 里的 %s 仍是参数绑定，禁止把 qvec 拼进 SQL 字符串。"""
        q = format_vector(qvec)
        with psycopg.connect(self._dsn) as conn:
            rows = conn.execute(
                """
                SELECT law_id, article_no, 1 - (embedding <=> %s::vector) AS cosine
                FROM articles
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (q, q, top_k),
            ).fetchall()
        return [LaneHit(law_id=l, article_no=n, score=float(s)) for l, n, s in rows]
