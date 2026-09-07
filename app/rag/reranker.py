"""对候选文档做语义精排的两个提供方，对外是同一接口（score/rerank），切换不动下游。

为什么存在"离线占位"这条真路径（与 embedder.py 同思路）：本机没有 key 时，要让
rerank 管道离线可跑、可复现、可测试。占位打分基于 query 与 doc 的词元重叠率——
这近似"query 有多少意图词被 doc 覆盖"，恰好能重排掉词面无关的候选；真正的语义
精排（同义改写、语序无关）要等 bge-reranker 上线才有。见决策 04。
"""
from __future__ import annotations

import json
import urllib.request
from typing import Protocol

from app.rag.tokenizer import tokenize


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class TermOverlapReranker:
    """离线确定性占位：score = query 词元在 doc 中出现的比例（q 覆盖率）。

    用"q 覆盖率"而非 Jaccard：法律条文长、查询短，|q∩d|/|q| 直接度量
    "doc 覆盖了多少查询意图"，对短查询更敏锐；Jaccard 会被长条文大量无关词稀释。
    确定性、零依赖——职责是让 rerank 全链路离线成立，语义精度不是它的目标。
    """

    def score(self, query: str, texts: list[str]) -> list[float]:
        q_terms = set(tokenize(query))
        if not q_terms:
            return [0.0] * len(texts)
        out = []
        for text in texts:
            d_terms = set(tokenize(text))
            overlap = len(q_terms & d_terms)
            out.append(overlap / len(q_terms))
        return out


class SiliconFlowReranker:
    """真 bge-reranker：走硅基流动的 /rerank 端点（OpenAI 兼容组织风格）。

    与 SiliconFlowEmbedder 同套路：urllib 零新增依赖、缺 key 当场抛错（与
    embedder 的"配置错误当场暴露"一致）、网络异常原样抛给调用方。
    score() 返回 provider 的原始得分（通常已 sigmoid 到 [0,1]，越高越相关）。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "BAAI/bge-reranker-v2-m3",
        base_url: str = "https://api.siliconflow.cn/v1",
    ):
        if not api_key:
            raise ValueError("siliconflow 模式需要 SILICONFLOW_API_KEY")
        self._api_key = api_key
        self._model = model
        self._endpoint = f"{base_url.rstrip('/')}/rerank"

    def score(self, query: str, texts: list[str]) -> list[float]:
        body = json.dumps({"model": self._model, "query": query, "documents": texts}).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        results = sorted(payload["results"], key=lambda r: r["index"])  # 按 index 对齐输入顺序
        return [float(r["relevance_score"]) for r in results]
