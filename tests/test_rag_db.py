"""真库集成测试（@pytest.mark.db）：验证 pgvector 向量路在 labor-pg 上真召回。

离线用例（test_rag_offline.py）只验融合/BM25 纯逻辑；pgvector 是 DB 行为，必须连真库
才算数。规则：
- 连不上库 / 无 DATABASE_URL => 整模块 skip，离线环境不影响全绿；
- 默认库为空 => skip 并提示先跑迁移脚本（本模块不再自己造语料）；
- 若库里有数据但还没向量，就地用离线 embedder 回填后再测（自包含）。

gold 标注仍是法律语义的 (law_id, article_no)，本模块用 eval/compat.law_to_doc 翻译成
切片身份 (doc_id, seq)——不重标注评测集，理由见 eval/compat.py。
验收对照是手挑的 (查询, 应命中条款) 对：占位向量只有词面重叠语义，所以查询刻意
选成与法律原文同词的问法（同义改写要等真 bge-m3，见决策 03 取舍）。
"""
from __future__ import annotations

import psycopg
import pytest

from app.core.config import Settings
from app.kb.schema import DEFAULT_KB_ID
from app.rag.embedder import HashEmbedder
from app.rag.retriever import HybridRetriever
from app.rag.vectorstore import PgVectorStore
from eval.compat import law_to_doc

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
def env():
    """返回 (retriever, law_id→doc_id 映射)；库不可达或默认库未迁移则整模块 skip。"""
    s = Settings()
    try:
        store = PgVectorStore(s.database_url, s.embedding_dim)
        store.ensure_ready()
    except psycopg.OperationalError as e:
        pytest.skip(f"labor-pg 不可达（docker compose up -d 后重跑）：{e}")

    if store.chunk_signature([DEFAULT_KB_ID])[0] == 0:
        pytest.skip("默认库为空，先跑 python scripts/migrate_default_kb.py")

    refs = store.all_chunks([DEFAULT_KB_ID])
    if store.count_embedded([DEFAULT_KB_ID]) == 0:  # 就地回填一次，让本模块自包含可跑
        emb = HashEmbedder(s.embedding_dim)
        store.set_embeddings([(r.chunk_id, emb.embed_one(r.text)) for r in refs])
    else:
        # 模型一致性自检：hash-embed 库内首片文本，查 top-10 应命中自己。
        # 命中 = 库向量是离线占位（hash）；未命中 = 库已被真 bge-m3 重灌——
        # 此时 hash query 对 bge 库做词面断言没有意义（两个语义空间），跳过。
        # 真模型库的检索验收见 eval/runner.py 的真实评测结果。
        probe = refs[0]
        probe_vec = HashEmbedder(s.embedding_dim).embed_one(probe.text)
        probe_hits = {h.chunk_id for h in store.vector_topk(probe_vec, 10, [DEFAULT_KB_ID])}
        if probe.chunk_id not in probe_hits:
            pytest.skip("默认库向量非离线占位（已被真模型重灌），离线检索断言跳过")

    retriever = HybridRetriever(
        store=store, embedder=HashEmbedder(s.embedding_dim), candidate_k=20,
        default_kb_ids=[DEFAULT_KB_ID],
    )
    return retriever, law_to_doc(s.database_url)


# (查询, 应命中的 law_id, article_no)——词面与原文强重叠的法律相关问法，见模块 docstring
GOLD_PAIRS = [
    ("用人单位安排劳动者延长工作时间应当按照标准支付高于正常工作时间工资的工资报酬", "labor_law", 44),
    ("劳动者每日工作时间不超过八小时", "labor_law", 36),
    ("用人单位解除劳动合同应当向劳动者支付经济补偿", "labor_contract_law", 46),
    ("用人单位克扣或者无故拖欠劳动者的工资", "labor_law", 50),
]


def test_vector_lane_recalls_gold(env):
    rt, doc_map = env
    for query, law, no in GOLD_PAIRS:
        topk = [(h.doc_id, h.seq) for h in rt.search_vector(query, top_k=12)]
        assert (doc_map[law], no) in topk, f"向量路 top-12 未命中 {law}#{no} <- {query}"


def test_bm25_lane_recalls_gold(env):
    rt, doc_map = env
    for query, law, no in GOLD_PAIRS:
        topk = [(h.doc_id, h.seq) for h in rt.search_bm25(query, top_k=12)]
        assert (doc_map[law], no) in topk, f"BM25 路 top-12 未命中 {law}#{no} <- {query}"


def test_fused_recalls_gold_with_both_lanes(env):
    rt, doc_map = env
    for query, law, no in GOLD_PAIRS:
        key = (doc_map[law], no)
        res = rt.search(query, top_k=12)
        key_to_hit = {(h.doc_id, h.seq): h for h in res.hits}
        assert key in key_to_hit, f"融合 top-12 未命中 {law}#{no} <- {query}"
        # 关键验收：这一条法律相关条款确实被两路都召回，融合不是单路独裁
        assert set(key_to_hit[key].lanes) == {"vector", "bm25"}


def test_vector_topk_order_is_cosine_descending(env):
    rt, _ = env
    hits = rt.search_vector("每日工作时间不超过八小时", top_k=10)
    cosines = [h.score for h in hits]
    assert cosines == sorted(cosines, reverse=True)


def test_hits_carry_document_identity(env):
    """切片化之后，命中必须能定位到"哪个文档的哪一段"——引用卡片就靠这些字段。"""
    rt, doc_map = env
    hit = rt.search("用人单位克扣或者无故拖欠劳动者的工资", top_k=1).hits[0]
    assert hit.doc_id == doc_map["labor_law"]
    assert hit.doc_title == "劳动法"
    assert hit.seq == 50
    assert hit.source_law_id == "labor_law"


def test_db_module_import_is_inert_without_settings():
    """模块自身不读 env——Settings 只在 fixture 里读一次，便于测试注入替身。"""
    assert Settings().embedding_mode in {"offline", "siliconflow"}
