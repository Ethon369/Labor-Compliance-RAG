"""检索相关的数据模型。跨层返回一律 Pydantic，供后续 API 层直接序列化。

身份键是 chunk_id（chunks 表主键）：内置法条与上传文档统一是"切片"，
检索/融合/精排都不关心片来自哪部法还是哪个 PDF——只关心它稳定、可哈希、可回查。
对外命中另带文档定位字段（doc_title/seq/page/heading），供引用卡片与角标使用。
"""
from __future__ import annotations

from pydantic import BaseModel


class ChunkRef(BaseModel):
    """语料里一个切片的稳定引用（内存 BM25 语料与 SQL 两侧共用 chunk_id）。"""

    chunk_id: int
    kb_id: int
    doc_id: int
    doc_title: str
    seq: int                  # 片在文档内的顺序（内置法条=条号）
    page: int | None = None   # PDF 页码，其它来源为空
    heading: str = ""         # 最近的上级标题
    text: str
    # 内置法律迁移来的文档才有值：引用展示时用它区分"第 N 条"（法条）与"第 N 段"（长文档）
    source_law_id: str | None = None


class LaneHit(BaseModel):
    """单条检索路的命中。score 语义随路而异（BM25 得分 / 余弦相似度），
    不跨路比较——融合只吃排序位置，见 fusion.py 为何。

    doc_id/seq 是给评测与调试用的：单路结果也要能直接对到"哪个文档的哪一段"，
    否则评测只能拿 chunk_id 去猜，且两路之间的身份无法对齐。
    """

    chunk_id: int
    doc_id: int
    seq: int
    score: float


class FusedHit(BaseModel):
    """RRF 融合后的结果：命中它的路、融合分、以及定位回语料的完整切片。

    rrf_score 始终保留原始 RRF 分（用于排序的可解释性）；rerank_score 仅
    在精排路径有值——保留两列便于评测时分析 rerank 对排序的改变幅度。
    """

    chunk_id: int
    kb_id: int
    doc_id: int
    doc_title: str
    seq: int
    page: int | None = None
    heading: str = ""
    text: str
    source_law_id: str | None = None
    lanes: list[str]  # 例 ["bm25", "vector"]：哪些路把它带进了候选
    rrf_score: float
    rerank_score: float | None = None


class RetrievalResult(BaseModel):
    """一次查询的完整返回：融合排序的命中列表。"""

    query: str
    hits: list[FusedHit]
