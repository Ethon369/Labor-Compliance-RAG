"""评测集与"切片化检索"之间的身份翻译层。

评测集的 gold 标注是人工按法律语义标的 `(law_id, article_no)`——它是标注数据，
不该因为数据层从 articles 换成 chunks 就重标一遍（重标既费人力、又会让历史指标
失去可比性）。这里在评测入口做一次翻译：`(law_id, article_no) → (doc_id, seq)`。

可逆性来自迁移规则：内置法条按条灌成切片，`seq == article_no`（见 migrate_default_kb）。
"""
from __future__ import annotations

import sys

import psycopg

from app.kb.schema import DEFAULT_KB_ID

GoldPair = tuple[str, int]
Question = tuple[str, list[GoldPair]]


def law_to_doc(dsn: str, kb_id: int = DEFAULT_KB_ID) -> dict[str, int]:
    """内置法律的 law_id → 默认库里的文档 id（documents.source_law_id 是权威来源）。"""
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            "SELECT source_law_id, id FROM documents "
            "WHERE kb_id = %s AND source_law_id IS NOT NULL",
            (kb_id,),
        ).fetchall()
    return {law_id: doc_id for law_id, doc_id in rows}


def translate_gold(questions: list[Question], mapping: dict[str, int]) -> list[Question]:
    """把 gold 的 (law_id, no) 翻成 (doc_id, no)。

    翻不到的标注直接丢弃并告警（例如库还没迁移）；指标函数对空 gold 记 1.0，
    所以丢弃不会把结果算成"错"，但告警能让人立刻发现"评测跑在了没数据的库上"。
    """
    out: list[Question] = []
    dropped = 0
    for query, gold in questions:
        pairs = []
        for law_id, no in gold:
            doc_id = mapping.get(law_id)
            if doc_id is None:
                dropped += 1
                continue
            pairs.append((doc_id, no))
        out.append((query, pairs))
    if dropped:
        print(f"[warn] {dropped} 条 gold 标注找不到对应文档，已丢弃"
              f"（默认库是否已迁移？python scripts/migrate_default_kb.py）", file=sys.stderr)
    return out
