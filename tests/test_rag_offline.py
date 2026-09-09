"""app/rag 离线单测：切词、embedding、RRF 融合、BM25、混合编排。

不碰数据库/网络。编排测试用一个内存向量库替身顶替 PgVectorStore 的接口
（all_chunks / chunk_signature / vector_topk），真 pgvector 的 SQL 行为由
tests/test_rag_db.py（@pytest.mark.db）验证——离线与真库的分工见决策 03。
"""
from __future__ import annotations

import pytest

from app.rag.bm25 import Bm25Index
from app.rag.embedder import HashEmbedder
from app.rag.fusion import rrf_fuse
from app.rag.models import ChunkRef, LaneHit
from app.rag.retriever import HybridRetriever
from app.rag.tokenizer import tokenize

# ---- 一个小而真实的语料：互有干扰词，检验排序而不只是"能命中" ----
# 检索身份是 chunk_id；doc_id/seq 是"哪个文档的哪一段"（离线测试里 id 只需自洽）
_DOC = {"labor_law": (1, "劳动法"), "labor_contract_law": (2, "劳动合同法")}


def _chunk(cid: int, law_id: str, no: int, text: str) -> ChunkRef:
    doc_id, title = _DOC[law_id]
    return ChunkRef(chunk_id=cid, kb_id=1, doc_id=doc_id, doc_title=title, seq=no,
                    heading="", text=text, source_law_id=law_id)


CORPUS = [
    _chunk(1, "labor_law", 36, "国家实行劳动者每日工作时间不超过八小时、平均每周工作时间不超过四十四小时的工时制度。"),
    _chunk(2, "labor_law", 44, "安排劳动者延长工作时间的，支付不低于工资的百分之一百五十的工资报酬；休息日安排劳动者工作又不能安排补休的，支付不低于工资百分之二百的工资报酬。"),
    _chunk(3, "labor_law", 50, "工资应当以货币形式按月支付给劳动者本人。不得克扣或者无故拖欠劳动者的工资。"),
    _chunk(4, "labor_law", 79, "劳动争议发生后，当事人可以向本单位劳动争议调解委员会申请调解；调解不成，可以向劳动争议仲裁委员会申请仲裁。"),
    _chunk(5, "labor_contract_law", 46, "有下列情形之一的，用人单位应当向劳动者支付经济补偿。"),
]


class _FakeVectorStore:
    """替身：用同一 embedder 在内存算余弦，模拟 PgVectorStore 的契约。"""

    def __init__(self, corpus: list[ChunkRef], embedder: HashEmbedder):
        self._docs = corpus
        self._embs = [embedder.embed_one(d.text) for d in corpus]

    def all_chunks(self, kb_ids: list[int]) -> list[ChunkRef]:
        return self._docs

    def chunk_signature(self, kb_ids: list[int]) -> tuple[int, int, str]:
        return (len(self._docs), max(d.chunk_id for d in self._docs), "")

    def vector_topk(self, qvec: list[float], top_k: int, kb_ids: list[int]) -> list[LaneHit]:
        scored = sorted(
            ((sum(a * b for a, b in zip(qvec, e)), d) for e, d in zip(self._embs, self._docs)),
            key=lambda t: t[0],
            reverse=True,
        )
        return [
            LaneHit(chunk_id=d.chunk_id, doc_id=d.doc_id, seq=d.seq, score=s)
            for s, d in scored[:top_k]
        ]


# ---------- tokenizer ----------

def test_tokenize_han_as_sliding_bigrams():
    toks = tokenize("每日工作时间")
    assert toks == ["每日", "日工", "工作", "作时", "时间"]  # 相邻两字滑窗，非全组合
    assert "时工" not in toks

def test_tokenize_keeps_ascii_word_and_isolated_single_han():
    assert tokenize("法") == ["法"]                    # 单字成段的残段保留
    assert tokenize("abc123") == ["abc123"]            # ASCII 字母数字视为一个词
    assert tokenize("第36条") == ["第", "条", "36"]     # 汉字段在前、ASCII 词在后

def test_tokenize_drops_punctuation_but_keeps_han():
    assert tokenize("（一）延长工作时间；") == ["一", "延长", "长工", "工作", "作时", "时间"]

def test_tokenize_preserves_duplicates_for_tf():
    toks = tokenize("加班加班")
    assert toks.count("加班") == 2


# ---------- embedding（离线占位）----------

def test_hash_embedder_deterministic_unit_and_dim():
    emb = HashEmbedder(dim=64)
    v1, v2 = emb.embed_one("工资报酬"), emb.embed_one("工资报酬")
    assert v1 == v2                      # 同文本必同向量（md5 稳定，非内建 hash）
    assert len(v1) == 64
    assert abs(sum(x * x for x in v1) ** 0.5 - 1.0) < 1e-9   # L2 单位化

def test_hash_embedder_word_overlap_raises_similarity():
    emb = HashEmbedder(dim=1024)
    q = emb.embed_one("延长工作时间")
    same = emb.embed_one("延长工作时间工资报酬")
    other = emb.embed_one("克扣拖欠工资")
    cos = lambda a, b: sum(x * y for x, y in zip(a, b))
    assert cos(q, same) > cos(q, other)


# ---------- RRF 融合（纯函数）----------

def test_rrf_ranks_two_lane_hit_above_one_lane_hit():
    A, B, C = 1, 2, 3                   # 身份键现在是 chunk_id（int）
    ranked = rrf_fuse([[A, B], [B, C]], k=60)
    assert ranked[0][0] == B            # B 在两路都在前列 => 融合第一
    assert ranked[0][1] == 1 / 61 + 1 / 62

def test_rrf_ignores_docs_absent_from_all():
    ranked = rrf_fuse([[1], [2]])
    assert all(key != 9 for key, _ in ranked)

def test_rrf_k_large_flattens_rank_gap():
    A, B = 1, 2
    s_small = dict(rrf_fuse([[A, B]], k=1))
    s_large = dict(rrf_fuse([[A, B]], k=100))
    # k 越大，第 1 名和第 2 名的分差越小（1/(k+1) vs 1/(k+2) 趋近相等）
    assert (s_small[A] - s_small[B]) > (s_large[A] - s_large[B])


# ---------- BM25 稀疏路 ----------

def test_bm25_top_hit_is_query_relevant_doc():
    idx = Bm25Index(CORPUS)
    hits = idx.search("劳动者每日工作时间不超过八小时", top_k=3)
    assert hits[0][0] == 1              # chunk 1 = 劳动法第 36 条（工时）
    hits = idx.search("克扣拖欠劳动者的工资", top_k=3)
    assert hits[0][0] == 3              # chunk 3 = 劳动法第 50 条（克扣拖欠工资）

def test_bm25_no_overlap_returns_empty():
    assert Bm25Index(CORPUS).search("zzzz不存在的词面", top_k=3) == []


# ---------- 混合编排（假向量库替身，真融合/排序逻辑）----------

@pytest.fixture()
def retriever():
    emb = HashEmbedder(dim=1024)
    return HybridRetriever(_FakeVectorStore(CORPUS, emb), embedder=emb, candidate_k=3)


def test_hybrid_recalls_gold_on_both_lanes(retriever):
    q = "用人单位应当向劳动者支付经济补偿"
    gold = (2, 46)                       # (doc_id, seq)：劳动合同法文档的第 46 条
    assert gold in [(h.doc_id, h.seq) for h in retriever.search_vector(q, top_k=3)]
    assert gold in [(h.doc_id, h.seq) for h in retriever.search_bm25(q, top_k=3)]


def test_fused_result_carries_both_lanes_and_text(retriever):
    res = retriever.search("用人单位应当向劳动者支付经济补偿", top_k=3)
    top = res.hits[0]
    assert (top.doc_id, top.seq) == (2, 46)
    assert top.doc_title == "劳动合同法"
    assert set(top.lanes) == {"vector", "bm25"}     # 两路都把它带进了候选
    assert "经济补偿" in top.text                     # 结果自带完整条文（供引用核验）


def test_fused_reorders_beyond_single_lane(retriever):
    """单路 top-1 可能不同，融合后与纯 BM25 排序一致但不等于任一路独裁。"""
    res = retriever.search("克扣拖欠劳动者的工资", top_k=3)
    assert (res.hits[0].doc_id, res.hits[0].seq) == (1, 50)


def test_bm25_cache_rebuilds_when_signature_changes():
    """语料指纹变了就重建索引——上传/删除文档后无需重启进程（决策 12）。"""
    emb = HashEmbedder(dim=1024)
    store = _FakeVectorStore(CORPUS, emb)
    rt = HybridRetriever(store, embedder=emb, candidate_k=3)
    assert rt.search_bm25("经济补偿", top_k=3)          # 先建一次缓存

    # 往"库"里加一片并改变指纹：下一次检索必须能召回它
    store._docs = CORPUS + [_chunk(6, "labor_law", 91, "用人单位招用劳动者，不得扣押劳动者的居民身份证。")]
    store._embs.append(emb.embed_one(store._docs[-1].text))
    hits = rt.search_bm25("扣押居民身份证", top_k=3)
    assert 6 in [h.chunk_id for h in hits]
