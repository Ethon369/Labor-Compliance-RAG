"""混合检索编排：一路 pgvector 余弦近邻 + 一路 BM25，RRF 融合成候选排序，可选 rerank 精排。

检索是两段式：RRF 只负责把两路候选"融合成可比的排序"，它吃排序位置、不追求精排；
rerank 在融合之后对更大的候选池做一次语义精排——召回（recall）和排序（ranking）
是两个阶段，rerank 只升级排序、不负责把没召回的捞回来（见决策 04）。

语料单元是"切片"，检索按知识库作用域（kb_ids）隔离。BM25 倒排是进程内缓存，
按作用域分别缓存并用 DB 指纹失效——上传/删除文档后无需重启即可生效（见决策 12）。
"""
from __future__ import annotations

import threading
from typing import NamedTuple

from app.core.config import Settings
from app.kb.schema import DEFAULT_KB_ID
from app.rag.bm25 import Bm25Index
from app.rag.embedder import Embedder, HashEmbedder, SiliconFlowEmbedder
from app.rag.fusion import rrf_fuse
from app.rag.models import ChunkRef, FusedHit, LaneHit, RetrievalResult
from app.rag.reranker import Reranker, SiliconFlowReranker, TermOverlapReranker
from app.rag.vectorstore import PgVectorStore


class _Bm25Entry(NamedTuple):
    """一个知识库作用域的 BM25 缓存条目：指纹 + 回查表 + 倒排索引。"""

    signature: tuple[int, int, str]
    by_key: dict[int, ChunkRef]
    index: Bm25Index


class HybridRetriever:
    def __init__(
        self,
        store: PgVectorStore,
        embedder: Embedder,
        candidate_k: int = 20,  # 每路取多少候选交给融合，而不是直接把 top-1 当答案
        reranker: Reranker | None = None,
        rerank_pool: int = 30,  # RRF 融合后多少名送 rerank；池越大精排越有得选，代价是成对打分越多
        default_kb_ids: list[int] | None = None,
    ):
        self._store = store
        self._embedder = embedder
        self._candidate_k = candidate_k
        self._reranker = reranker
        self._rerank_pool = rerank_pool
        self._default_kb_ids = list(default_kb_ids) if default_kb_ids else [DEFAULT_KB_ID]
        self._bm25_cache: dict[tuple[int, ...], _Bm25Entry] = {}
        self._cache_lock = threading.Lock()

    # ---- 语料缓存 ----

    def _scope(self, kb_ids: list[int] | None) -> tuple[int, ...]:
        """把 kb 集合归一成可哈希的缓存键；缺省用默认库。"""
        return tuple(sorted(kb_ids)) if kb_ids else tuple(self._default_kb_ids)

    def _bm25_for(self, kb_ids: list[int] | None) -> _Bm25Entry:
        """取作用域内的 BM25 索引；DB 指纹变了就地重建。

        为什么用"指纹探针"而不是显式失效通知：写路径分散在迁移脚本、上传流水线、
        删除接口三处，让它们各自通知缓存容易漏；探针在检索侧统一收口，
        代价是每次检索多一条廉价聚合查询（205 条规模可忽略）。
        """
        scope = self._scope(kb_ids)
        signature = self._store.chunk_signature(list(scope))
        entry = self._bm25_cache.get(scope)
        if entry is not None and entry.signature == signature:
            return entry
        with self._cache_lock:
            # 等锁期间语料可能又变了：进锁后重探一次，避免拿旧指纹建出过期索引
            entry = self._bm25_cache.get(scope)
            if entry is not None and entry.signature == self._store.chunk_signature(list(scope)):
                return entry
            signature = self._store.chunk_signature(list(scope))
            corpus = self._store.all_chunks(list(scope))
            entry = _Bm25Entry(
                signature=signature,
                by_key={c.chunk_id: c for c in corpus},
                index=Bm25Index(corpus),
            )
            self._bm25_cache[scope] = entry
            return entry

    # ---- 两路检索 ----

    def search_vector(self, query: str, top_k: int | None = None,
                      kb_ids: list[int] | None = None) -> list[LaneHit]:
        """稠密路：查询文本 embedding → pgvector 余弦近邻。"""
        qvec = self._embedder.embed_one(query)
        return self._store.vector_topk(qvec, top_k or self._candidate_k, list(self._scope(kb_ids)))

    def search_bm25(self, query: str, top_k: int | None = None,
                    kb_ids: list[int] | None = None) -> list[LaneHit]:
        """稀疏路：BM25 词面匹配。"""
        entry = self._bm25_for(kb_ids)
        k = top_k or self._candidate_k
        hits: list[LaneHit] = []
        for cid, score in entry.index.search(query, k):
            ref = entry.by_key.get(cid)
            if ref is None:  # 与 search() 同款竞态保护：快照里没有就跳过
                continue
            hits.append(LaneHit(chunk_id=cid, doc_id=ref.doc_id, seq=ref.seq, score=score))
        return hits

    def search(self, query: str, top_k: int = 8, use_rerank: bool = True,
               kb_ids: list[int] | None = None) -> RetrievalResult:
        """两路各取候选 → RRF 融合 → 取前 pool 名送 rerank（可选）→ 取前 top_k。

        use_rerank=False 时退化为纯 RRF 结果——供评测脚本在一套 retriever 实例上
        直接对比"加不加 rerank"的指标差，不用重建两次。"""
        entry = self._bm25_for(kb_ids)
        # 每路拉够候选：至少要覆盖 rerank_pool（精排池可能有重复 key，多拉一点更安全）
        fetch_k = max(self._candidate_k, self._rerank_pool)
        vec_hits = self.search_vector(query, top_k=fetch_k, kb_ids=kb_ids)
        bm_hits = self.search_bm25(query, top_k=fetch_k, kb_ids=kb_ids)
        vec_ids = [h.chunk_id for h in vec_hits]
        bm_ids = [h.chunk_id for h in bm_hits]
        vec_set, bm_set = set(vec_ids), set(bm_ids)

        fused = rrf_fuse([vec_ids, bm_ids])

        # 竞态保护：向量路可能召回"语料快照之后才入库"的片，by_key 里查不到就本轮跳过，
        # 下次检索指纹已变会重建索引，届时自然包含它。
        pool = [(cid, s) for cid, s in fused[:self._rerank_pool] if cid in entry.by_key]

        # rerank 精排：先取 pool 名给 reranker 成对打分，再按 rerank 分重排，
        # 最后截断回 top_k。pool 比 top_k 大（默认 30 vs 8），给精排留选择余地。
        rerank_scores: dict[int, float] = {}
        if use_rerank and self._reranker is not None and len(pool) > 0:
            texts = [entry.by_key[cid].text for cid, _ in pool]
            scores = self._reranker.score(query, texts)
            for (cid, _), s in zip(pool, scores):
                rerank_scores[cid] = round(s, 6)
            # 按 rerank 分降序重排 pool，分相同则保留原 RRF 序（稳定排序）
            pool.sort(key=lambda item: rerank_scores.get(item[0], 0.0), reverse=True)

        hits: list[FusedHit] = []
        for cid, rrf in pool[:top_k]:
            ref = entry.by_key[cid]
            lanes = [name for name, ks in (("vector", vec_set), ("bm25", bm_set)) if cid in ks]
            hits.append(
                FusedHit(
                    chunk_id=ref.chunk_id,
                    kb_id=ref.kb_id,
                    doc_id=ref.doc_id,
                    doc_title=ref.doc_title,
                    seq=ref.seq,
                    page=ref.page,
                    heading=ref.heading,
                    text=ref.text,
                    source_law_id=ref.source_law_id,
                    lanes=lanes,
                    rrf_score=round(rrf, 6),
                    rerank_score=rerank_scores.get(cid),
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
        default_kb_ids=[DEFAULT_KB_ID],
    )
