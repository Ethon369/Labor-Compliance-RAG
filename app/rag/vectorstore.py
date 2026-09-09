"""pgvector 访问层：就绪（扩展+列）、切片语料全量、写向量、余弦近邻检索。

向量以文本字面量 + `%s::vector` 显式转型传入——刻意不引 pgvector 的 Python 适配
包：psycopg 的参数绑定已足够，且少一个锁定清单外的依赖；转型交给 PG 端完成。
每次调用开短连接：检索频次不高（见决策 03 已知取舍），换取代码零状态、易测试。

知识库作用域：检索一律带 kb_id 过滤（`kb_id = ANY(%s)`，list 参数由 psycopg
绑定成 int[]，仍是参数绑定，不拼 SQL）。
"""
from __future__ import annotations

from app.core.db import pool_conn

from app.rag.models import ChunkRef, LaneHit

_DIM_ALTER_SQL = "embedding vector({dim})"  # 占位拼接只拼维度数字常量，不拼用户输入


def format_vector(v: list[float]) -> str:
    """向量 → PG 的 vector 文本字面量（形如 [0.1,0.2,...]），供 %s::vector 转型。

    放在模块级公开：回填/迁移脚本灌库时复用同一格式化，保证写/查两侧格式一致。"""
    return "[" + ",".join(f"{x:.8g}" for x in v) + "]"


def parse_vector(text: str) -> list[float]:
    """PG vector 文本字面量 → float 列表。是 format_vector 的逆操作。

    供嵌入去重复用旧向量用：读回 `embedding::text` 再解析，省一次 embedding 调用。"""
    return [float(x) for x in text.strip("[]").split(",") if x.strip()]


class PgVectorStore:
    def __init__(self, dsn: str, dim: int):
        self._dsn = dsn
        self._dim = dim

    def ensure_ready(self) -> None:
        """幂等迁移：启 pgvector 扩展、给 chunks 加 embedding 列。可反复调用。"""
        with pool_conn(self._dsn) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS {_DIM_ALTER_SQL.format(dim=self._dim)}"
            )

    def all_chunks(self, kb_ids: list[int]) -> list[ChunkRef]:
        """拉指定知识库的全量切片，供 BM25 建内存倒排。排序固定，保证索引顺序一致。"""
        with pool_conn(self._dsn) as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.kb_id, c.doc_id, d.title, c.seq, c.page, c.heading,
                       c.content, d.source_law_id
                FROM chunks c JOIN documents d ON d.id = c.doc_id
                WHERE c.kb_id = ANY(%s)
                ORDER BY c.doc_id, c.seq
                """,
                (kb_ids,),
            ).fetchall()
        return [
            ChunkRef(chunk_id=cid, kb_id=kb, doc_id=doc, doc_title=title, seq=seq,
                     page=page, heading=head, text=text, source_law_id=law_id)
            for cid, kb, doc, title, seq, page, head, text, law_id in rows
        ]

    def chunk_signature(self, kb_ids: list[int]) -> tuple[int, int, str]:
        """作用域内语料的廉价指纹：片数 / 最大片 id / 文档最近更新时间。

        为什么这三项够用：新增片 → 片数+最大 id 变；删除片/整库 → 片数变；
        同文档重传（seq 相同、内容变、id 不变）→ documents.updated_at 变。
        三条写路径全覆盖，且只是一次聚合查询，可每次检索都探一次。
        """
        with pool_conn(self._dsn) as conn:
            row = conn.execute(
                """
                SELECT count(c.id), COALESCE(max(c.id), 0),
                       COALESCE(max(d.updated_at)::text, '')
                FROM chunks c JOIN documents d ON d.id = c.doc_id
                WHERE c.kb_id = ANY(%s)
                """,
                (kb_ids,),
            ).fetchone()
        return (int(row[0]), int(row[1]), str(row[2]))

    def count_embedded(self, kb_ids: list[int]) -> int:
        with pool_conn(self._dsn) as conn:
            return conn.execute(
                "SELECT count(*) FROM chunks WHERE embedding IS NOT NULL AND kb_id = ANY(%s)",
                (kb_ids,),
            ).fetchone()[0]

    def pending_rows(self, kb_ids: list[int], force: bool = False) -> list[tuple[int, str]]:
        """待回填 (chunk_id, content)；force=True 全量，否则仅 embedding 为空。

        WHERE 分支由调用方布尔决定、非用户输入，仅此两态，不构成注入面。"""
        where = "" if force else "AND embedding IS NULL"
        with pool_conn(self._dsn) as conn:
            return conn.execute(
                f"SELECT id, content FROM chunks WHERE kb_id = ANY(%s) {where} ORDER BY id",
                (kb_ids,),
            ).fetchall()

    def set_embeddings(self, rows: list[tuple[int, list[float]]]) -> None:
        """批量写向量。executemany 比逐行省一半网络往返。"""
        with pool_conn(self._dsn) as conn, conn.cursor() as cur:
            cur.executemany(
                "UPDATE chunks SET embedding = %s::vector WHERE id = %s",
                [(format_vector(v), cid) for cid, v in rows],
            )

    def vector_topk(self, qvec: list[float], top_k: int, kb_ids: list[int]) -> list[LaneHit]:
        """余弦近邻：`<=>` 是余弦距离（越小越近），用 1-d 转回相似度便于展示。

        `%s::vector` 里的 %s 仍是参数绑定，禁止把 qvec 拼进 SQL 字符串。"""
        q = format_vector(qvec)
        with pool_conn(self._dsn) as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.doc_id, c.seq, 1 - (c.embedding <=> %s::vector) AS cosine
                FROM chunks c
                WHERE c.embedding IS NOT NULL AND c.kb_id = ANY(%s)
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (q, kb_ids, q, top_k),
            ).fetchall()
        return [
            LaneHit(chunk_id=cid, doc_id=doc_id, seq=seq, score=float(s))
            for cid, doc_id, seq, s in rows
        ]
