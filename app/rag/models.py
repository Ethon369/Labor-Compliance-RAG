"""检索相关的数据模型。跨层返回一律 Pydantic，供后续 API 层直接序列化。"""
from __future__ import annotations

from pydantic import BaseModel


class ArticleRef(BaseModel):
    """语料里一条法条的稳定引用。内存与 SQL 两侧共用 (law_id, article_no) 这把钥匙。"""

    law_id: str
    article_no: int
    chapter: str | None = None
    text: str


class LaneHit(BaseModel):
    """单条检索路的命中。score 语义随路而异（BM25 得分 / 余弦相似度），
    不跨路比较——融合只吃排序位置，见 fusion.py 为何。"""

    law_id: str
    article_no: int
    score: float


class FusedHit(BaseModel):
    """RRF 融合后的结果：命中它的路、融合分、以及定位回语料的完整条文。

    rrf_score 始终保留原始 RRF 分（用于排序的可解释性）；rerank_score 仅
    在精排路径有值——保留两列便于评测时分析 rerank 对排序的改变幅度。
    """

    law_id: str
    article_no: int
    chapter: str | None = None
    text: str
    lanes: list[str]  # 例 ["bm25", "vector"]：哪些路把它带进了候选
    rrf_score: float
    rerank_score: float | None = None


class RetrievalResult(BaseModel):
    """一次查询的完整返回：融合排序的命中列表。"""

    query: str
    hits: list[FusedHit]
