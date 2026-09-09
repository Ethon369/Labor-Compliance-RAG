"""运行检索评测对比：纯向量 vs 混合 vs 混合+rerank，输出表格。

用法：
    python eval/runner.py                # 离线模式跑一遍（占位向量 + 占位 rerank）
    python eval/runner.py --top-k 10     # 按指定 top-k 评测

输出的对比表给面试官看——三种方法在 50 条问题上的召回率和命中率一目了然。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 项目根放进 sys.path：`python eval/runner.py` 直接跑时，interpret 只把 eval/ 放
# 进 sys.path[0]，认不到 `eval.metrics`（顶层包）；加 root 让两种跑法（脚本 / -m）都成立
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from dataclasses import dataclass
from typing import Protocol

from eval.metrics import hit_rate_at_k, mean_recall
from eval.questions import QUESTIONS

# 不用 eval.__init__ 的顶层 import（因为 runner 需要可独立运行），
# 指标函数直接按独立模块引用


# ---------- 检索描述：三种方法的 key ----------

class Retriever(Protocol):
    """评测只要三样：纯向量、BM25、混合（可开关 rerank）。HybridRetriever 满足。"""

    def search_vector(self, query: str, top_k: int | None = None) -> list[tuple[int, int]]: ...

    def search_bm25(self, query: str, top_k: int | None = None) -> list[tuple[int, int]]: ...

    def search(self, query: str, top_k: int = 8) -> list[tuple[int, int]]: ...


@dataclass
class Method:
    label: str
    fn: callable  # (query, top_k) -> list[(doc_id, seq)]


def _wrap_vector(rt: Retriever, top_k: int):
    """纯向量：只用 search_vector 这一路的 top-k。"""
    def f(q: str, k: int = top_k) -> list[tuple[int, int]]:
        hits = rt.search_vector(q, top_k=k)
        return [(h.doc_id, h.seq) for h in hits]
    return f


def _wrap_bm25(rt: Retriever, top_k: int):
    def f(q: str, k: int = top_k) -> list[tuple[int, int]]:
        hits = rt.search_bm25(q, top_k=k)
        return [(h.doc_id, h.seq) for h in hits]
    return f


def _wrap_mixed(rt: Retriever, top_k: int, use_rerank: bool):
    def f(q: str, k: int = top_k) -> list[tuple[int, int]]:
        res = rt.search(q, top_k=k, use_rerank=use_rerank)
        return [(h.doc_id, h.seq) for h in res.hits]
    label = "混合+rerank" if use_rerank else "混合(RRF)"
    return label, f


# ---------- 表格输出 ----------

def print_table(methods: list[tuple[str, callable]], questions: list, top_k: int) -> None:
    """跑一遍评测并打印对齐的对比表。"""
    header = f"{'方法':<20s} {'Recall@{:<3d}'.format(top_k)}  {'HitRate@{:<3d}'.format(top_k)}"
    print(header)
    print("-" * len(header))
    for label, fn in methods:
        rec = mean_recall(questions, fn, top_k)
        hr = hit_rate_at_k(questions, fn, top_k)
        print(f"{label:<20s} {rec:>7.4f}      {hr:>7.4f}")
    print()


# ---------- 组装 ----------

def run_comparison(rt: Retriever, questions: list, top_k: int = 8) -> None:
    """拉齐同一 rt 实例，三种方法跑一轮对比并打印表。

    questions 需是已翻译成 (doc_id, seq) 身份的评测集——见 eval/compat.py。"""
    methods: list[tuple[str, callable]] = [
        ("纯向量", _wrap_vector(rt, top_k)),
        ("纯BM25", _wrap_bm25(rt, top_k)),
    ]
    # 混合(RRF)——不用 rerank，直接用 RRF 融合结果
    methods.append(_wrap_mixed(rt, top_k, use_rerank=False))
    # 混合+rerank——开启精排
    methods.append(_wrap_mixed(rt, top_k, use_rerank=True))

    print_table(methods, questions, top_k)


# ---------- CLI ----------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=8,
                    help="评测 cut-off，默认 8")
    args = ap.parse_args(argv)

    from app.core.config import Settings
    from app.rag.retriever import build_retriever
    from eval.compat import law_to_doc, translate_gold

    settings = Settings()
    rt = build_retriever(settings)
    # 评测集 gold 是 (law_id, article_no)；检索返回的是 (doc_id, seq)。入口翻译一次，
    # 指标函数与问题集都不必改（见 eval/compat.py 为何不重标注）
    questions = translate_gold(QUESTIONS, law_to_doc(settings.database_url))

    run_comparison(rt, questions, top_k=args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())