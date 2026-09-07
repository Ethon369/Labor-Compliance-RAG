"""阶段 4 LangGraph Agent 单测：5 节点 + 拒答分支 + 离线占位全链路 + LLM 节点 mock。

面向 RetrievalProtocol 注入内存假检索器——无 PG、无 LLM 也能端到端验证 Graph 拓扑。
覆盖：正常链路（回答+引用）、拒答两条支路（空命中 / top-1 覆盖不足）、改写与阈值可调、
LLM 改写/回答的独立行为（mock HTTP）。
"""
from __future__ import annotations

import json
import io
import urllib.request
from unittest.mock import patch

import pytest

from app.agent.graph import build_agent, build_agent_from_config
from app.agent.protocol import (
    LLMAnswerer,
    LLMRewriter,
    PassThroughRewriter,
    _call_chat,
)
from app.agent.state import AgentAnswer
from app.agent.verify import CitationVerifier
from app.rag.models import FusedHit, RetrievalResult

OVERTIME = FusedHit(
    law_id="labor_law",
    article_no=44,
    text="安排劳动者延长工作时间的，支付不低于工资的百分之一百五十的工资报酬。",
    lanes=["bm25"],
    rrf_score=1.0,
)
UNRELATED = FusedHit(
    law_id="labor_law",
    article_no=36,
    text="工时制度。",
    lanes=["bm25"],
    rrf_score=1.0,
)


class FakeRetriever:
    """可编程内存检索器：无论 query 是什么都返回固定的命中集（测试注入用）。"""

    def __init__(self, hits: list[FusedHit]):
        self.hits = hits

    def search(self, query: str, top_k: int = 8, use_rerank: bool = True) -> RetrievalResult:
        return RetrievalResult(query=query, hits=self.hits)


def _invoke(hits: list[FusedHit], query: str = "公司延长我工作时间，加班费怎么算") -> AgentAnswer:
    app = build_agent(FakeRetriever(hits))
    result = app.invoke({"original": query})
    return result["answer"]


# ---- 正常链路：命中覆盖意图 → 走 answer 支路，带引用 ----


def test_happy_path_answers_with_citation():
    answer = _invoke([OVERTIME])
    assert answer.refuse is False
    assert "延长工作时间" in answer.answer  # 答案引用了 top 条文文本
    assert len(answer.citations) == 1
    assert answer.citations[0].law_id == "labor_law"
    assert answer.citations[0].article_no == 44


def test_rewrite_strips_whitespace():
    app = build_agent(FakeRetriever([OVERTIME]), rewriter=PassThroughRewriter())
    result = app.invoke({"original": "  延长工作时间加班费   "})
    assert result["answer"].rewritten == "延长工作时间加班费"


# ---- 拒答支路一：完全没有命中 ----


def test_refuse_on_empty_hits():
    answer = _invoke([])
    assert answer.refuse is True
    assert answer.citations == []
    assert "无法就该问题给出明确结论" in answer.answer


# ---- 拒答支路二：有命中但 top-1 覆盖不了意图（无关条文）----


def test_refuse_when_top_hit_irrelevant():
    answer = _invoke([UNRELATED])  # 条文只有"工时制度"，覆盖不了"加班费"意图
    assert answer.refuse is True
    assert answer.citations == []


# ---- 核验规则单测 + 阈值可调 ----


def test_verifier_supported_true():
    v = CitationVerifier()
    assert v.supported("延长工作时间工资报酬", [OVERTIME]) is True


def test_verifier_supported_false_empty():
    v = CitationVerifier()
    assert v.supported("延长工作时间", []) is False


def test_verifier_supported_false_irrelevant():
    v = CitationVerifier()
    assert v.supported("加班费怎么计算", [UNRELATED]) is False


def test_threshold_sensitivity():
    v_lo = CitationVerifier(threshold=0.0)  # 只要有命中就支持
    v_hi = CitationVerifier(threshold=1.0)  # 必须全词覆盖才支持
    assert v_lo.supported("ZZZ不相关词", [OVERTIME]) is True
    assert v_hi.supported("延长工作时间工资报酬延长", [OVERTIME]) is False


# ====== LLM 节点单测（mock urllib） ======


def _fake_chat_response(content: str) -> io.BytesIO:
    """构造一个与 OpenAI chat/completions 兼容的假 HTTP 响应体。"""
    body = {"choices": [{"message": {"content": content}}]}
    return io.BytesIO(json.dumps(body).encode("utf-8"))


def test_call_chat_parses_content():
    """_call_chat 应正确从 choices[0].message.content 提取文本。"""
    expected = "加班费 计算依据 延长工作时间 工资报酬"
    with patch("urllib.request.urlopen", return_value=_fake_chat_response(expected)):
        result = _call_chat("http://fake/v1/chat/completions", "sk-fake", "fake-model", [
            {"role": "user", "content": "加班费怎么算"},
        ])
    assert result == expected


def test_llm_rewriter_requires_api_key():
    """空 key 时应立即报错，而不是到请求时才 401。"""
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        LLMRewriter(api_key="")


def test_llm_rewriter_calls_chat_and_strips():
    """LLMRewriter.rewrite 应返回 LLM 返回的去空白结果。"""
    raw = "  加班费 工资报酬 延长工作时间  "
    with patch("urllib.request.urlopen", return_value=_fake_chat_response(raw)):
        rw = LLMRewriter(api_key="sk-test")
        result = rw.rewrite("加班费怎么算")
    assert result == raw.strip()
    assert "加班费" in result


def test_llm_answerer_requires_api_key():
    """空 key 时应立即报错。"""
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        LLMAnswerer(api_key="")


def test_llm_answerer_formats_articles_and_calls_chat():
    """LLMAnswerer.generate 应把条文拼进 user message 发给 LLM。"""
    expected_answer = "根据《劳动法》第44条，加班费应不低于150%。"
    with patch("urllib.request.urlopen", return_value=_fake_chat_response(expected_answer)):
        ans = LLMAnswerer(api_key="sk-test")
        result = ans.generate("加班费怎么算", [OVERTIME])
    assert "加班费" in result
    assert "150" in result


def test_llm_answerer_handles_empty_hits():
    """空命中时也应正常调用 LLM（条文部分给出提示）。"""
    expected = "无法提供明确结论。"
    with patch("urllib.request.urlopen", return_value=_fake_chat_response(expected)):
        ans = LLMAnswerer(api_key="sk-test")
        result = ans.generate("怎么计算加班费", [])
    assert result == expected


# ---- LLM 注入 Graph 端到端（mock） ----


class FakeRewriter:
    """测试用改写器：记录调用并返回固定值。"""

    def __init__(self, return_value: str = "改写后的查询"):
        self.called_with: list[str] = []
        self.return_value = return_value

    def rewrite(self, query: str) -> str:
        self.called_with.append(query)
        return self.return_value


class FakeAnswerer:
    """测试用回答器：记录调用并返回固定值。"""

    def __init__(self, return_value: str = "带引用的回答"):
        self.called_with: list[tuple[str, list[FusedHit]]] = []
        self.return_value = return_value

    def generate(self, query: str, hits: list[FusedHit]) -> str:
        self.called_with.append((query, hits))
        return self.return_value


def test_graph_with_injected_llm_rewriter():
    """注入自定义改写器后 graph 应使用它而非 PassThroughRewriter。"""
    rewriter = FakeRewriter("加班费 延长工作时间")
    app = build_agent(FakeRetriever([OVERTIME]), rewriter=rewriter)
    result = app.invoke({"original": "公司让我加班不给钱怎么办"})
    # 输入应该被改写器处理过
    assert result["rewritten"] == "加班费 延长工作时间"
    assert len(rewriter.called_with) == 1


def test_graph_with_injected_llm_answerer():
    """注入自定义回答器后 graph 应使用它而非 TemplateAnswerer。"""
    answerer = FakeAnswerer("【LLM 生成的回答】根据劳动法第44条...")
    # 使用 FakeRewriter 改写 query，使 top-1 词的覆盖率通过 verify
    # verify 需要"top-1 文本"覆盖足够比例的"改写后查询"词元
    rewriter = FakeRewriter("延长工作时间 工资报酬")
    app = build_agent(FakeRetriever([OVERTIME]), rewriter=rewriter, answerer=answerer)
    result = app.invoke({"original": "加班费怎么算"})
    assert not result["answer"].refuse
    assert "LLM" in result["answer"].answer
    assert len(answerer.called_with) == 1


def test_graph_with_both_llm_components():
    """同时注入改写器和回答器，graph 端到端正常。"""
    rewriter = FakeRewriter("加班费工资报酬")
    answerer = FakeAnswerer("根据第44条，加班费不低于150%")
    app = build_agent(FakeRetriever([OVERTIME]), rewriter=rewriter, answerer=answerer)
    result = app.invoke({"original": "加班没给钱"})
    assert result["answer"].refuse is False
    assert result["rewritten"] == "加班费工资报酬"
    assert "44条" in result["answer"].answer


# ---- build_agent_from_config 测试 ----


class FakeSettings:
    """最小 Settings stub：只模拟 agent 层关心的字段。"""

    def __init__(self, llm_mode: str = "offline", llm_api_key: str = "",
                 llm_model: str = "deepseek-chat", llm_base_url: str = "https://api.deepseek.com/v1",
                 llm_temperature: float = 0.0):
        self.llm_mode = llm_mode
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm_temperature = llm_temperature


def test_build_from_config_offline_uses_placeholder():
    """offline 模式应使用 PassThroughRewriter + TemplateAnswerer。"""
    graph = build_agent_from_config(FakeRetriever([OVERTIME]), FakeSettings("offline"))
    # 需要 query 词元与 OVERTIME 文本有共现，verify 才会走 answer 分支
    result = graph.invoke({"original": "  延长工作时间 工资报酬  "})
    assert result["rewritten"] == "延长工作时间 工资报酬"  # PassThrough 做 strip
    assert "可参考以下条文" in result["answer"].answer  # TemplateAnswerer 模板


def test_build_from_config_llm_mode_requires_key():
    """llm 模式缺 key 应报错。"""
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        build_agent_from_config(FakeRetriever([OVERTIME]), FakeSettings("llm", llm_api_key=""))


def test_build_from_config_llm_mode_with_key():
    """llm 模式有 key 时 graph 应正常编译（mock _call_chat 而非 urlopen——避免 BytesIO 上下文管理问题）。

    注意 graph 走两遍 LLM：rewrite → verify → answer。side_effect 按顺序给返回值。
    """
    with patch("app.agent.protocol._call_chat", side_effect=[
        "延长工作时间 工资报酬",  # 改写结果，与 OVERTIME 文本共词 → verify 通过
        "根据劳动法第44条应当支付150%的加班费",  # 回答结果
    ]):
        graph = build_agent_from_config(
            FakeRetriever([OVERTIME]),
            FakeSettings("llm", llm_api_key="sk-test"),
        )
        result = graph.invoke({"original": "加班费怎么算"})
        assert result["answer"].refuse is False
        assert "150" in result["answer"].answer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])