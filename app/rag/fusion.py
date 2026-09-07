"""RRF（Reciprocal Rank Fusion）：把多条检索路的排序位置融合成一条排序。

只吃"排序位置"、不吃各自分数——这是它比加权平均分数稳的关键：BM25 的得分
量纲与 bge-m3 余弦完全不同（一个是对数词频叠加、一个是 [0,1] 相似度），直接加
没有意义；但"它在某一路排第几名"在任何路内部都是可比的。见决策 03 第 4 节。
"""
from __future__ import annotations

from collections import Counter

_Key = tuple[str, int]  # (law_id, article_no)，跨路的稳定文档标识


def rrf_fuse(rank_lists: list[list[_Key]], k: int = 60) -> list[tuple[_Key, float]]:
    """给每路的有序候选列表打分：score(d) = Σ 路 1/(k + rank(d))，k 默认 60（论文值）。

    不在任何一路出现的文档贡献为 0，天然被排到最后；返回按融合分降序。
    """
    scores: Counter[_Key] = Counter()
    for rank_list in rank_lists:
        for pos, key in enumerate(rank_list, start=1):
            scores[key] += 1.0 / (k + pos)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
