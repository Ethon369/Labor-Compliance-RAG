"""混合检索（阶段 2 第一块）：pgvector 向量路 + BM25 稀疏路 + RRF 融合。

对外入口是 HybridRetriever（含 build_retriever 组装点）；embedder/vectorstore/
bm25/fusion/tokenizer 是构成它的可单测部件。"""
from __future__ import annotations

from app.rag.embedder import Embedder, HashEmbedder, SiliconFlowEmbedder
from app.rag.models import ArticleRef, FusedHit, LaneHit, RetrievalResult
from app.rag.retriever import HybridRetriever, build_retriever

__all__ = [
    "ArticleRef",
    "Embedder",
    "FusedHit",
    "HashEmbedder",
    "HybridRetriever",
    "LaneHit",
    "RetrievalResult",
    "SiliconFlowEmbedder",
    "build_retriever",
]
