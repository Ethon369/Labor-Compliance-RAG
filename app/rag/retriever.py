"""混合检索编排：一路 pgvector 余弦近邻 + 一路 BM25，RRF 融合成最终排序。

两路索引性质互补（稠密 vs 稀疏、语义 vs 词面），各自只当"候选生成器"，融合后才排序，
这正是 hybrid 优于单路的原因——见决策 03。语料首次用到时从 PG 拉全量建 BM25 倒排
并缓存进程内；205 条规模这是合理取舍，语料变更需重建（本阶段数据静态）。
"""
from __future__ import annotations

from app.core.config import Settings
from app.rag.bm25 import Bm25Index
from app.rag.embedder import Embedder, HashEmbedder, SiliconFlowEmbedder
from app.rag.fusion import rrf_fuse
from app.rag.models import ArticleRef, FusedHit, LaneHit, RetrievalResult
from app.rag.vectorstore import PgVectorStore


class HybridRetriever:
    def __init__(
        self,
        store: PgVectorStore,
        embedder: Embedder,
        candidate_k: int = 20,  # 每路取多少候选交给融合，而不是直接把 top-1 当答案
    ):
        self._store = store
        self._embedder = embedder
        self._candidate_k = candidate_k
        self._corpus: list[ArticleRef] | None = None  # 进程内缓存，见模块 docstring
        self._by_key: dict[tuple[str, int], ArticleRef] = {}
        self._bm25: Bm25Index | None = None

    def _load_corpus(self) -> None:
        if self._corpus is None:
            corpus = self._store.all_articles()
            self._corpus = corpus
            self._by_key = {(a.law_id, a.article_no): a for a in corpus}
            self._bm25 = Bm25Index(corpus)

    def search_vector(self, query: str, top_k: int | None = None) -> list[LaneHit]:
        """稠密路：查询文本 embedding → pgvector 余弦近邻。"""
        qvec = self._embedder.embed_one(query)
        return self._store.vector_topk(qvec, top_k or self._candidate_k)

    def search_bm25(self, query: str, top_k: int | None = None) -> list[LaneHit]:
        """稀疏路：BM25 词面匹配。"""
        self._load_corpus()
        k = top_k or self._candidate_k
        return [LaneHit(law_id=l, article_no=n, score=s) for l, n, s in self._bm25.search(query, k)]

    def search(self, query: str, top_k: int = 8) -> RetrievalResult:
        """两路各取候选 → RRF 融合 → 取前 top_k 组装带出处的命中。"""
        self._load_corpus()
        vec_hits = self.search_vector(query)
        bm_hits = self.search_bm25(query)
        vec_keys = [(h.law_id, h.article_no) for h in vec_hits]
        bm_keys = [(h.law_id, h.article_no) for h in bm_hits]
        vec_set, bm_set = set(vec_keys), set(bm_keys)

        fused = rrf_fuse([vec_keys, bm_keys])
        hits: list[FusedHit] = []
        for key, rrf in fused[:top_k]:
            ref = self._by_key[key]
            lanes = [name for name, ks in (("vector", vec_set), ("bm25", bm_set)) if key in ks]
            hits.append(
                FusedHit(
                    law_id=ref.law_id,
                    article_no=ref.article_no,
                    chapter=ref.chapter,
                    text=ref.text,
                    lanes=lanes,
                    rrf_score=round(rrf, 6),
                )
            )
        return RetrievalResult(query=query, hits=hits)


def build_retriever(settings: Settings) -> HybridRetriever:
    """组装点：按配置选 embedding 提供方并确保向量库就绪（幂等），供脚本/API 复用。

    siliconflow 模式缺 key 直接报错——与其静默降级到占位向量，不如让配置错误当场暴露。
    """
    store = PgVectorStore(settings.database_url, settings.embedding_dim)
    store.ensure_ready()
    if settings.embedding_mode == "siliconflow":
        embedder: Embedder = SiliconFlowEmbedder(
            api_key=settings.siliconflow_api_key or "",
            model=settings.siliconflow_model,
            base_url=settings.siliconflow_base_url,
            dim=settings.embedding_dim,
        )
    else:
        embedder = HashEmbedder(settings.embedding_dim)
    return HybridRetriever(store=store, embedder=embedder)
