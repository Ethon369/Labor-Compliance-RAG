"""评测指标：top-k 召回率（Recall@k）与命中率（Hit Rate@k）。

两个指标的区别（为什么两个都要算而不是二选一）：
- Recall@k：top-k 中命中几条标注 / 标注总数——度量"用户想要的，系统找回了多少比例"。
  适合多标注问题（一条问对应多条应命中法条）——单条命中 ≠ 找全了。
- Hit Rate@k：top-k 中至少命中一条标注的问题占比——度量"用户在 k 条内能不能看到一条
  有用结果"的比例，适合回答"系统对多少比例的用户确实提供了价值"。

用哪个对外展示取决于受众：面试官看重"查全"用 Recall，看重"首条体验"用 Hit Rate；
两者同时列在对比表里，面试官自选角度。
"""
from __future__ import annotations

from collections.abc import Callable

from eval.questions import EvalQuestion

RetrieveFn = Callable[[str, int], list[tuple[str, int]]]
# 检索函数签名：(query, top_k) -> [(law_id, article_no), ...]


def _keys_in_topk(
    query: str,
    gold_keys: set[tuple[str, int]],
    retrieve: RetrieveFn,
    top_k: int,
) -> tuple[list[tuple[str, int]], int]:
    """返回 (命中键列表, 命中数)，不区分 law_id。"""
    hits = retrieve(query, top_k)
    hit_keys = [key for key in hits if key in gold_keys]
    return hit_keys, len(hit_keys)


def recall_at_k(
    query: str,
    gold: list[tuple[str, int]],
    retrieve: RetrieveFn,
    top_k: int,
) -> float:
    """单条问题的 Recall@k = 命中标注数 / 标注总数。

    gold 为空时返回 1.0（没有标注可召回 = 不算失败）。
    """
    gold_set = set(gold)
    if not gold_set:
        return 1.0
    _, hit_count = _keys_in_topk(query, gold_set, retrieve, top_k)
    return hit_count / len(gold_set)


def mean_recall(
    questions: list[EvalQuestion],
    retrieve: RetrieveFn,
    top_k: int,
) -> float:
    """全部问题的平均 Recall@top_k。"""
    scores = [recall_at_k(q, gold, retrieve, top_k) for q, gold in questions]
    return sum(scores) / len(scores) if scores else 0.0


def hit_rate_at_k(
    questions: list[EvalQuestion],
    retrieve: RetrieveFn,
    top_k: int,
) -> float:
    """Hit Rate@k = 至少命中一条标注的问题数 / 总问题数。

    等价于"用户在 k 条内看到有用结果的问题占比"。
    """
    if not questions:
        return 0.0
    hit_count = 0
    for query, gold in questions:
        if not gold:
            hit_count += 1  # 无标注 = 不算失分
            continue
        _, hc = _keys_in_topk(query, set(gold), retrieve, top_k)
        if hc > 0:
            hit_count += 1
    return hit_count / len(questions)