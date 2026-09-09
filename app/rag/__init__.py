"""混合检索（阶段 2 第一块）：pgvector 向量路 + BM25 稀疏路 + RRF 融合 + rerank 精排。

对外入口是 HybridRetriever（含 build_retriever 组装点）；embedder/vectorstore/
bm25/fusion/tokenizer/reranker 是构成它的可单测部件。"""
from __future__ import annotations

from app.rag.embedder import Embedder, HashEmbedder, SiliconFlowEmbedder
from app.rag.models import ChunkRef, FusedHit, LaneHit, RetrievalResult
from app.rag.reranker import Reranker, SiliconFlowReranker, TermOverlapReranker
from app.rag.retriever import HybridRetriever, build_retriever

__all__ = [
    "ChunkRef",
    "Embedder",
    "FusedHit",
    "HashEmbedder",
    "HybridRetriever",
    "LaneHit",
    "Reranker",
    "RetrievalResult",
    "SiliconFlowEmbedder",
    "SiliconFlowReranker",
    "TermOverlapReranker",
    "build_retriever",
]
