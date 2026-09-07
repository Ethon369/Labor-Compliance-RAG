"""评测框架：50 条劳动争议场景问答集 + top-k 召回率/命中率指标 + 对比脚本。

评测不依赖 API 或特定提供方——只要被评测对象提供 search_vector/search_bm25/search
三条检索管道即可，离线占位向量也可跑，便于"纯向量 vs 混合 vs 混合+rerank"对比。
"""
from __future__ import annotations

from eval.metrics import hit_rate_at_k, mean_recall, recall_at_k
from eval.runner import print_table, run_comparison

__all__ = [
    "hit_rate_at_k",
    "mean_recall",
    "print_table",
    "recall_at_k",
    "run_comparison",
]