"""混合检索编排：一路 pgvector 余弦近邻 + 一路 BM25，RRF 融合成候选排序，可选 rerank 精排。

检索是两段式：RRF 只负责把两路候选"融合成可比的排序"，它吃排序位置、不追求精排；
rerank 在融合之后对更大的候选池做一次语义精排——召回（recall）和排序（ranking）
是两个阶段，rerank 只升级排序、不负责把没召回的捞回来（见决策 04）。语料首次用到
时从 PG 拉全量建 BM25 倒排并缓存进程内；205 条规模这是合理取舍，语料变更需重建。
"""
from __future__ import annotations

from app.core.config import Settings
from app.rag.bm25 import Bm25Index
from app.rag.embedder import Embedder, HashEmbedder, SiliconFlowEmbedder
from app.rag.fusion import rrf_fuse
from app.rag.models import ArticleRef, FusedHit, LaneHit, RetrievalResult
from app.rag.reranker import Reranker, SiliconFlowReranker, TermOverlapReranker
from app.rag.vectorstore import PgVectorStore


class HybridRetriever:
    def __init__(
        self,
        store: PgVectorStore,
        embedder: Embedder,
        candidate_k: int = 20,  # 每路取多少候选交给融合，而不是直接把 top-1 当答案
        reranker: Reranker | None = None,
        rerank_pool: int = 30,  # RRF 融合后多少名送 rerank；池越大精排越有得选，代价是成对打分越多
    ):
        self._store = store
        self._embedder = embedder
        self._candidate_k = candidate_k
        self._reranker = reranker
        self._rerank_pool = rerank_pool
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

    def search(self, query: str, top_k: int = 8, use_rerank: bool = True) -> RetrievalResult:
        """两路各取候选 → RRF 融合 → 取前 pool 名送 rerank（可选）→ 取前 top_k。

        use_rerank=False 时退化为纯 RRF 结果——供评测脚本在一套 retriever 实例上
        直接对比"加不加 rerank"的指标差，不用重建两次。"""
        self._load_corpus()
        # 每路拉够候选：至少要覆盖 rerank_pool（精排池可能有重复 key，多拉一点更安全）
        fetch_k = max(self._candidate_k, self._rerank_pool)
        vec_hits = self.search_vector(query, top_k=fetch_k)
        bm_hits = self.search_bm25(query, top_k=fetch_k)
        vec_keys = [(h.law_id, h.article_no) for h in vec_hits]
        bm_keys = [(h.law_id, h.article_no) for h in bm_hits]
        vec_set, bm_set = set(vec_keys), set(bm_keys)

        fused = rrf_fuse([vec_keys, bm_keys])

        # rerank 精排：先取 pool 名给 reranker 成对打分，再按 rerank 分重排，
        # 最后截断回 top_k。pool 比 top_k 大（默认 30 vs 8），给精排留选择余地。
        pool = fused[:self._rerank_pool]
        rerank_scores: dict[tuple[str, int], float] = {}
        if use_rerank and self._reranker is not None and len(pool) > 0:
            texts = [self._by_key[key].text for key, _ in pool]
            scores = self._reranker.score(query, texts)
            for (key, _), s in zip(pool, scores):
                rerank_scores[key] = round(s, 6)
            # 按 rerank 分降序重排 pool，分相同则保留原 RRF 序（稳定排序）
            pool.sort(key=lambda item: rerank_scores.get(item[0], 0.0), reverse=True)

        hits: list[FusedHit] = []
        for key, rrf in pool[:top_k]:
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
                    rerank_score=rerank_scores.get(key),
                )
            )
        return RetrievalResult(query=query, hits=hits)


def build_retriever(settings: Settings) -> HybridRetriever:
    """组装点：按配置选 embedding/rerank 提供方并确保向量库就绪（幂等），供脚本/API 复用。

    siliconflow 模式缺 key 直接报错——与其静默降级到占位，不如让配置错误当场暴露。
    reranker 按 rerank_mode 选：siliconflow=真 bge-reranker、offline=词元重叠占位、
    设为 None=不使用（评测时可传显式参数绕开）。
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

    # reranker 装配：三态——siliconflow / offline / None（不使用）
    reranker = None
    if settings.rerank_mode == "siliconflow":
        reranker = SiliconFlowReranker(
            api_key=settings.siliconflow_api_key or "",
            model=settings.rerank_model,
            base_url=settings.siliconflow_base_url,
        )
    elif settings.rerank_mode == "offline":
        reranker = TermOverlapReranker()

    return HybridRetriever(
        store=store, embedder=embedder,
        reranker=reranker, rerank_pool=settings.rerank_pool,
    )
