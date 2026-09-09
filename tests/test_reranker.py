"""rerank 单测：TermOverlapReranker 离线行为 + reranker 在 retriever 管道中的效果。

不碰数据库/网络——纯占位 + 假向量库验证 rerank 的全链路逻辑。
"""
from __future__ import annotations

import pytest

from app.rag.embedder import HashEmbedder
from app.rag.reranker import TermOverlapReranker
from app.rag.retriever import HybridRetriever

# 复用 test_rag_offline 的语料与假向量库，避免两处各维护一份（同一目录，pytest 直接可 import）
from test_rag_offline import CORPUS, _FakeVectorStore


# ---------- TermOverlapReranker 纯逻辑 ----------

def test_overlap_full_coverage():
    """q 的全部词元都出现在 doc 中 => score=1.0"""
    r = TermOverlapReranker()
    scores = r.score("延长工作时间", ["安排劳动者延长工作时间的，支付不低于工资的百分之一百五十的工资报酬"])
    assert scores[0] == 1.0


def test_overlap_partial_coverage():
    """q 部分词元命中 => score 介于 0 到 1"""
    r = TermOverlapReranker()
    scores = r.score("克扣拖欠工资", ["工资应当以货币形式按月支付。不得克扣或者无故拖欠劳动者的工资。"])
    assert 0.0 < scores[0] < 1.0


def test_overlap_zero_coverage():
    """q 和 doc 无共享词元 => score=0"""
    r = TermOverlapReranker()
    scores = r.score("ZZZ不存在词元", ["劳动争议发生后，当事人可以向调解委员会申请调解。"])
    assert scores[0] == 0.0


def test_overlap_empty_query():
    """空查询 → 所有 doc score=0（不会除以零）"""
    r = TermOverlapReranker()
    scores = r.score("", ["安排劳动者延长工作时间的"])
    assert scores == [0.0]


def test_overlap_multiple_docs_ordered():
    """多文档打分：覆盖率高的排前面。"""
    r = TermOverlapReranker()
    texts = [
        "劳动合同期满即行终止。",
        "安排劳动者延长工作时间的，支付不低于工资的百分之一百五十的工资报酬。",
        "劳动争议发生后可以申请调解仲裁。",
    ]
    scores = r.score("延长工作时间工资报酬", texts)
    # text[1] 覆盖了"延长""工作时间"，text[0] 和 text[2] 基本不覆盖
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


# ---------- reranker 在 retriever 管道中的效果 ----------

@pytest.fixture()
def retriever_no_rerank():
    """不带 reranker 的 retriever——作为基线。"""
    emb = HashEmbedder(dim=1024)
    return HybridRetriever(_FakeVectorStore(CORPUS, emb), embedder=emb, candidate_k=5)


@pytest.fixture()
def retriever_with_rerank():
    """带占位 reranker 的 retriever。"""
    emb = HashEmbedder(dim=1024)
    return HybridRetriever(
        _FakeVectorStore(CORPUS, emb),
        embedder=emb,
        candidate_k=5,
        reranker=TermOverlapReranker(),
        rerank_pool=5,
    )


def test_rerank_changes_order_when_appropriate():
    """rerank 对排序有影响——验证 rerank_score 字段有值且可能改变 rank-1。

    用"劳动争议"查询：RRF 可能把两路都命中的条排第一，但 rerank 从语义
    （词元覆盖率）角度可能把更贴切的条文排上来。
    """
    emb = HashEmbedder(dim=1024)
    rt = HybridRetriever(
        _FakeVectorStore(CORPUS, emb),
        embedder=emb,
        candidate_k=5,
        reranker=TermOverlapReranker(),
        rerank_pool=5,
    )
    res_no = rt.search("劳动争议调解仲裁", top_k=3, use_rerank=False)
    res_yes = rt.search("劳动争议调解仲裁", top_k=3, use_rerank=True)

    # 验证 rerank_score 字段
    for h in res_yes.hits:
        assert h.rerank_score is not None

    # 基本断言：两种结果都不是空的
    assert len(res_no.hits) > 0
    assert len(res_yes.hits) > 0


def test_use_rerank_false_has_no_rerank_scores(retriever_with_rerank):
    """use_rerank=False 时 rerank_score 全部是 None。"""
    res = retriever_with_rerank.search("延长工作时间工资报酬", top_k=3, use_rerank=False)
    assert all(h.rerank_score is None for h in res.hits)


def test_use_rerank_true_has_rerank_scores(retriever_with_rerank):
    """use_rerank=True 时 rerank_score 有值。"""
    res = retriever_with_rerank.search("延长工作时间工资报酬", top_k=3, use_rerank=True)
    assert all(h.rerank_score is not None for h in res.hits)


def test_no_reranker_and_use_rerank_true_is_harmless(retriever_no_rerank):
    """没有 reranker 的时候 use_rerank=True 也安全——退化为纯 RRF。"""
    res = retriever_no_rerank.search("延长工作时间工资报酬", top_k=3, use_rerank=True)
    assert len(res.hits) > 0
    assert all(h.rerank_score is None for h in res.hits)


def test_rerank_preserves_rrf_score():
    """RRF 分在 rerank 之后仍然保留（用于分析"精排改变了什么"）。"""
    emb = HashEmbedder(dim=1024)
    rt = HybridRetriever(
        _FakeVectorStore(CORPUS, emb),
        embedder=emb,
        candidate_k=5,
        reranker=TermOverlapReranker(),
        rerank_pool=5,
    )
    res = rt.search("克扣拖欠劳动者的工资", top_k=3, use_rerank=True)
    for h in res.hits:
        assert h.rrf_score is not None
        assert h.rerank_score is not None