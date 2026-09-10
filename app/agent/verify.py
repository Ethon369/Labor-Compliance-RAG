"""引用核验：纯规则判断检索结果是否支持回答（不依赖 LLM）。

为什么纯规则而不是叫 LLM 判断："要不要拒答"这条边界要确定、可测、离线可复现——
线上与 CI 都要能稳定触发拒答。规则简单但诚实：top-1 条文对查询意图词的覆盖率达标
且确有命中，就说"可答"并给引用；否则走拒答支路。规则的阈值是接口参数，可调。见决策 05。
"""
from __future__ import annotations

from app.rag.models import FusedHit
from app.rag.tokenizer import tokenize


class CitationVerifier:
    def __init__(self, threshold: float = 0.3):
        # 覆盖率门槛：top-1 条文需覆盖至少这么多比例的查询意图词，才认为"能答"
        self._threshold = threshold

    def supported(self, query: str, hits: list[FusedHit]) -> bool:
        """判定是否支持回答：有命中，且 top-1 条文覆盖足够比例的查询意图词。"""
        if not hits:
            return False
        q_terms = set(tokenize(query))
        if not q_terms:
            return False  # 空查询无从谈覆盖，宁拒答
        top_text = hits[0].text
        d_terms = set(tokenize(top_text))
        overlap = len(q_terms & d_terms) / len(q_terms)
        return overlap >= self._threshold
