"""eval 框架单测：metrics 函数 + 50 条问题集自校验。不碰 DB/网络。"""

from __future__ import annotations

import pytest

from eval.compat import translate_gold
from eval.metrics import hit_rate_at_k, mean_recall, recall_at_k
from eval.questions import QUESTIONS


# ---------- 问题集自校验 ----------

def test_question_count():
    """确保问题集数量在合理范围（至少 50 条），且无重复问题文本。"""
    assert len(QUESTIONS) >= 50, f"期望至少 50 条，实际 {len(QUESTIONS)} 条"
    queries = [q for q, _ in QUESTIONS]
    assert len(queries) == len(set(queries)), "重复的问题文本"


def test_every_question_has_at_least_one_gold():
    """每条问题必须至少标注一条应命中法条——无标注就失去评测含义。"""
    for q, gold in QUESTIONS:
        assert len(gold) >= 1, f"'{q}' 缺少标注"


def test_gold_keys_are_valid():
    """标注的 law_id 只能是 labor_law 或 labor_contract_law，且 article_no 为正数。"""
    for q, gold in QUESTIONS:
        for law_id, no in gold:
            assert law_id in {"labor_law", "labor_contract_law"}, \
                f"'{q}' 的标注 law_id={law_id} 无效"
            assert no >= 1, f"'{q}' 的标注 article_no={no} 无效"


def test_coverage_of_both_laws():
    """验证两部法律都被标注覆盖了（不能只标了一部）。"""
    laws = {law_id for _, golds in QUESTIONS for law_id, _ in golds}
    assert "labor_law" in laws
    assert "labor_contract_law" in laws


# ---------- 身份翻译（eval/compat，纯函数部分） ----------

def test_translate_gold_maps_law_id_to_doc_id():
    """gold 从 (law_id, no) 翻成 (doc_id, no)；seq==article_no 由迁移规则保证。"""
    qs = [("问题一", [("labor_law", 44), ("labor_contract_law", 46)])]
    out = translate_gold(qs, {"labor_law": 2, "labor_contract_law": 1})
    assert out == [("问题一", [(2, 44), (1, 46)])]


def test_translate_gold_drops_unknown_law_without_crashing(capsys):
    """库里没有对应文档时丢弃该标注并告警——不能静默算成"没命中"。"""
    qs = [("问题一", [("labor_law", 44), ("ghost_law", 1)])]
    out = translate_gold(qs, {"labor_law": 2})
    assert out == [("问题一", [(2, 44)])]
    assert "找不到对应文档" in capsys.readouterr().err


def test_multi_gold_questions_exist():
    """验证存在多标注问题——Recall 指标依赖多标注才有意义。"""
    multi = [(q, g) for q, g in QUESTIONS if len(g) > 1]
    assert len(multi) >= 5, f"多标注问题只有 {len(multi)} 条，不够评测 Recall"


# ---------- 指标函数单测（用假检索函数） ----------

def _retrieve_none(_, top_k: int) -> list[tuple[str, int]]:
    """假检索：什么都不返回（完全失分场景）。"""
    return []


def test_recall_at_k_perfect():
    """标注都在 top-k 里 => Recall=1.0"""
    gold = [("labor_law", 36), ("labor_law", 44)]
    r = recall_at_k("测试", gold, lambda q, k: gold[:k], top_k=5)
    assert r == 1.0


def test_recall_at_k_partial():
    """一半命中 => Recall=0.5"""
    def retrieve_half(_, top_k: int) -> list[tuple[str, int]]:
        return [("labor_law", 36)]

    r = recall_at_k("测试", [("labor_law", 36), ("labor_law", 44)], retrieve_half, top_k=5)
    assert r == 0.5


def test_recall_at_k_zero():
    """完全没命中 => Recall=0"""
    r = recall_at_k("测试", [("labor_law", 36)], _retrieve_none, top_k=5)
    assert r == 0.0


def test_recall_empty_gold_is_1():
    """无标注的问题 => Recall=1.0（不算失分）"""
    r = recall_at_k("测试", [], _retrieve_none, top_k=5)
    assert r == 1.0


def test_hit_rate_all_hit():
    """每个问题至少命中一条 => Hit Rate=1.0"""
    questions = [("问题A", [("labor_law", 1)]), ("问题B", [("labor_law", 2)])]
    gold_map = {"问题A": [("labor_law", 1)], "问题B": [("labor_law", 2)]}
    hr = hit_rate_at_k(questions, lambda q, k: gold_map.get(q, [])[:k], top_k=5)
    assert hr == 1.0


def test_hit_rate_half_hit():
    """一半命中 => Hit Rate=0.5"""

    def retrieve_second_only(q: str, top_k: int):   # noqa: ARG001
        if "B" in q:
            return [("labor_law", 2)]
        return []

    questions = [("问题A", [("labor_law", 1)]), ("问题B", [("labor_law", 2)])]
    hr = hit_rate_at_k(questions, retrieve_second_only, top_k=5)
    assert hr == 0.5


def test_hit_rate_none_hit():
    """完全没命中 => Hit Rate=0"""
    questions = [("问题A", [("labor_law", 1)])]
    hr = hit_rate_at_k(questions, _retrieve_none, top_k=5)
    assert hr == 0.0


def test_mean_recall_1_on_perfect():
    gold = [("labor_law", 1), ("labor_law", 2)]
    questions = [("问题A", gold)]
    mr = mean_recall(questions, lambda q, k: gold[:k], top_k=5)
    assert mr == 1.0


def test_mean_recall_0_on_zero():
    questions = [("问题A", [("labor_law", 1)])]
    mr = mean_recall(questions, _retrieve_none, top_k=5)
    assert mr == 0.0