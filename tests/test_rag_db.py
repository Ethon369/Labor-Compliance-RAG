"""真库集成测试（@pytest.mark.db）：验证 pgvector 向量路在 labor-pg 上真召回。

离线用例（test_rag_offline.py）只验融合/BM25 纯逻辑；pgvector 是 DB 行为，必须连真库
才算数。规则：
- 连不上库 / 无 DATABASE_URL => 整模块 skip，离线环境不影响全绿；
- 若库里有数据但还没向量，就地用离线 embedder 回填后再测（自包含，不依赖先跑脚本）。
验收对照是手挑的 (查询, 应命中条款) 对：占位向量只有词面重叠语义，所以查询刻意
选成与法律原文同词的问法（同义改写要等真 bge-m3，见决策 03 取舍）。
"""
from __future__ import annotations

import psycopg
import pytest

from app.core.config import Settings
from app.rag.embedder import HashEmbedder
from app.rag.retriever import HybridRetriever
from app.rag.vectorstore import PgVectorStore

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
def retriever():
    s = Settings()
    try:
        store = PgVectorStore(s.database_url, s.embedding_dim)
        store.ensure_ready()
    except psycopg.OperationalError as e:
        pytest.skip(f"labor-pg 不可达（docker compose up -d 后重跑）：{e}")

    if store.count_embedded() == 0:  # 就地回填一次，让本模块自包含可跑
        refs = store.all_articles()
        emb = HashEmbedder(s.embedding_dim)
        store.set_embeddings([(r.law_id, r.article_no, emb.embed_one(r.text)) for r in refs])

    embedder = HashEmbedder(s.embedding_dim)
    return HybridRetriever(store=store, embedder=embedder, candidate_k=20)


# (查询, 应命中的 law_id, article_no)——词面与原文强重叠的法律相关问法，见模块 docstring
GOLD_PAIRS = [
    ("用人单位安排劳动者延长工作时间应当按照标准支付高于正常工作时间工资的工资报酬", "labor_law", 44),
    ("劳动者每日工作时间不超过八小时", "labor_law", 36),
    ("用人单位解除劳动合同应当向劳动者支付经济补偿", "labor_contract_law", 46),
    ("用人单位克扣或者无故拖欠劳动者的工资", "labor_law", 50),
]


def test_vector_lane_recalls_gold(retriever):
    for query, law, no in GOLD_PAIRS:
        topk = [(h.law_id, h.article_no) for h in retriever.search_vector(query, top_k=12)]
        assert (law, no) in topk, f"向量路 top-12 未命中 {law}#{no} <- {query}"


def test_bm25_lane_recalls_gold(retriever):
    for query, law, no in GOLD_PAIRS:
        topk = [(h.law_id, h.article_no) for h in retriever.search_bm25(query, top_k=12)]
        assert (law, no) in topk, f"BM25 路 top-12 未命中 {law}#{no} <- {query}"


def test_fused_recalls_gold_with_both_lanes(retriever):
    for query, law, no in GOLD_PAIRS:
        res = retriever.search(query, top_k=12)
        key_to_hit = {(h.law_id, h.article_no): h for h in res.hits}
        assert (law, no) in key_to_hit, f"融合 top-12 未命中 {law}#{no} <- {query}"
        # 关键验收：这一条法律相关条款确实被两路都召回，融合不是单路独裁
        assert set(key_to_hit[(law, no)].lanes) == {"vector", "bm25"}


def test_vector_topk_order_is_cosine_descending(retriever):
    hits = retriever.search_vector("每日工作时间不超过八小时", top_k=10)
    cosines = [h.score for h in hits]
    assert cosines == sorted(cosines, reverse=True)


def test_db_module_import_is_inert_without_settings():
    """模块自身不读 env——Settings 只在 fixture 里读一次，便于测试注入替身。"""
    assert Settings().embedding_mode in {"offline", "siliconflow"}
