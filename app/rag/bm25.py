"""BM25 稀疏检索路：把语料建成内存倒排（rank-bm25 的 BM25Okapi），按查询取 top-k。

为什么在 Python 内存做而不是 PG 侧 tsvector/倒排：205 条一次性全量可入内存，
rank-bm25 是锁定清单内的现成实现；将来语料量级上来再评估外置索引（新决策）。
"""
from __future__ import annotations

from rank_bm25 import BM25Okapi

from app.rag.models import ArticleRef
from app.rag.tokenizer import tokenize


class Bm25Index:
    def __init__(self, corpus: list[ArticleRef]):
        # 语料顺序即下标：sorted 过的全量列表，用下标回找 ArticleRef 不丢信息
        self._corpus = corpus
        self._okapi = BM25Okapi([tokenize(a.text) for a in corpus])

    def _key_at(self, i: int) -> tuple[str, int]:
        a = self._corpus[i]
        return (a.law_id, a.article_no)

    def search(self, query: str, top_k: int) -> list[tuple[str, int, float]]:
        """按 BM25 得分降序返回前 top_k 个 (law_id, article_no, score)。"""
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self._okapi.get_scores(q_tokens)
        order = sorted(range(len(self._corpus)), key=lambda i: scores[i], reverse=True)
        return [
            (*self._key_at(i), float(scores[i]))
            for i in order[:top_k]
            if scores[i] > 0.0  # 一句没命中词的文档不占候选位
        ]
