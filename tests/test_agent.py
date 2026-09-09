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
    _call_chat_stream,
    _call_chat_with_tools_stream,
    answer_sink_get,
    answer_sink_reset,
    answer_sink_set,
    scrub_citation_markers,
)
from app.agent.state import AgentAnswer
from app.agent.verify import CitationVerifier
from app.rag.models import FusedHit, RetrievalResult

def _hit(seq: int, text: str, *, doc_id: int = 1, doc_title: str = "劳动法",
         source_law_id: str | None = "labor_law", lanes: tuple[str, ...] = ("bm25",),
         rrf_score: float = 1.0) -> FusedHit:
    """构造一条切片命中：内置法条默认落在"劳动法"文档上，seq 即条号。"""
    return FusedHit(
        chunk_id=seq, kb_id=1, doc_id=doc_id, doc_title=doc_title, seq=seq,
        heading="", text=text, source_law_id=source_law_id,
        lanes=list(lanes), rrf_score=rrf_score,
    )


OVERTIME = _hit(44, "安排劳动者延长工作时间的，支付不低于工资的百分之一百五十的工资报酬。")
UNRELATED = _hit(36, "工时制度。")


class FakeRetriever:
    """可编程内存检索器：无论 query 是什么都返回固定的命中集（测试注入用）。"""

    def __init__(self, hits: list[FusedHit]):
        self.hits = hits

    def search(self, query: str, top_k: int = 8, use_rerank: bool = True,
               kb_ids: list[int] | None = None) -> RetrievalResult:
        return RetrievalResult(query=query, hits=self.hits)


def _invoke(
    hits: list[FusedHit],
    query: str = "公司延长我工作时间，加班费怎么算",
    verify_mode: str = "rule",
) -> AgentAnswer:
    # 默认 rule 模式：保住"top-1 无关 → 拒答"的既有语义（对应真 LLM 配套的兜底）
    app = build_agent(FakeRetriever(hits), verify_mode=verify_mode)
    result = app.invoke({"original": query})
    return result["answer"]


# ---- 正常链路：命中覆盖意图 → 走 answer 支路，带引用 ----


def test_happy_path_answers_with_citation():
    answer = _invoke([OVERTIME])
    assert answer.refuse is False
    assert "延长工作时间" in answer.answer  # 答案引用了 top 条文文本
    assert len(answer.citations) == 1
    assert answer.citations[0].doc_title == "劳动法"
    assert answer.citations[0].seq == 44
    assert answer.citations[0].source_law_id == "labor_law"
    assert answer.citations[0].score == 1.0     # 无精排分时退融合分


def test_answer_carries_numbered_citation_markers():
    """离线模板也要带 [n] 编号——前端角标与引用卡片靠它对位（决策 13）。"""
    answer = _invoke([OVERTIME], verify_mode="pass")
    assert "[1]" in answer.answer


# ---- 越界角标清理（纯函数）：LLM 偶尔会引用列表外的编号 ----

def test_scrub_removes_out_of_range_markers():
    assert scrub_citation_markers("结论。[1] 另有说法。[9]", 2) == "结论。[1] 另有说法。"

def test_scrub_keeps_valid_markers():
    assert scrub_citation_markers("甲[1]乙[2]丙", 2) == "甲[1]乙[2]丙"

def test_scrub_removes_zero_marker():
    assert scrub_citation_markers("结论[0]。", 3) == "结论。"

def test_scrub_leaves_non_numeric_brackets_alone():
    assert scrub_citation_markers("第[一]条与[abc]", 3) == "第[一]条与[abc]"


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


# 离线模板模式（verify_mode="pass"）：命中即答，不设规则防线。
# 模板只拼条文原文、不产生新结论，无幻觉风险；rule 模式的拒答是给真 LLM 配套的兜底。


def test_pass_mode_answers_when_hits_exist():
    answer = _invoke([UNRELATED], verify_mode="pass")  # top-1 无关也照答
    assert answer.refuse is False
    assert len(answer.citations) == 1
    assert "第36条" in answer.answer  # 模板引用了条文原文


def test_pass_mode_still_refuses_on_empty():
    answer = _invoke([], verify_mode="pass")  # 无命中仍拒答
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


# ---- 流式回答单测（stream:true + SSE 逐 data 行解析）----


def _fake_sse_response(chunks: list[dict]) -> io.BytesIO:
    """构造一个 OpenAI 兼容的流式响应体：若干行 `data: <json>` 之间空行隔开。"""
    body = "".join(
        f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks
    ) + "data: [DONE]\n\n"
    return io.BytesIO(body.encode("utf-8"))


def test_call_chat_stream_parses_deltas_and_callbacks():
    """_call_chat_stream 应逐 delta 调 on_token 并拼回完整文本。"""
    chunks = [
        {"choices": [{"delta": {"content": "根据"}}]},
        {"choices": [{"delta": {"content": "劳动法第44条"}}]},
        {"choices": [{"delta": {}}]},  # 空增量/usage 帧应跳过
        {"choices": [{"delta": {"content": "支付150%"}}]},
    ]
    collected: list[str] = []
    with patch("urllib.request.urlopen", return_value=_fake_sse_response(chunks)):
        text = _call_chat_stream("http://fake/v1/chat/completions", "sk-fake", "m",
                                 [{"role": "user", "content": "q"}],
                                 on_token=collected.append)
    assert text == "根据劳动法第44条支付150%"
    assert collected == ["根据", "劳动法第44条", "支付150%"]


def test_llm_answerer_streams_when_on_token_given():
    """给 on_token 时应走流式请求并回调每个增量，返回值仍是完整答案。"""
    chunks = [{"choices": [{"delta": {"content": p}}]} for p in ["加班费", "不低于", "150%"]]
    collected: list[str] = []
    with patch("urllib.request.urlopen", return_value=_fake_sse_response(chunks)):
        ans = LLMAnswerer(api_key="sk-test")
        result = ans.generate("加班费怎么算", [OVERTIME], on_token=collected.append)
    assert result == "加班费不低于150%"
    assert collected == ["加班费", "不低于", "150%"]


def test_answer_sink_default_none_and_roundtrip():
    """answer_sink 默认 None；注入后能读到、reset 后还原（路由在请求线程里用）。"""
    assert answer_sink_get() is None
    got = []
    tok = answer_sink_set(got.append)
    assert answer_sink_get() == got.append
    answer_sink_reset(tok)
    assert answer_sink_get() is None


class StubToolRegistry:
    """最小工具注册器：record 调用，回一个可序列化 dict。"""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def tool_schemas(self):
        return [{"type": "function", "function": {"name": "calc", "parameters": {}}}]

    def call(self, name: str, args: dict):
        self.calls.append((name, args))
        return {"value": 3000}


def test_tool_calling_stream_runs_tool_then_streams_final_text():
    """流式工具循环：首轮只有 tool_calls（无 content），应执行工具；次轮 content 逐段回调。"""
    # 首轮：tool_call 的 id/name/arguments 拆成多个 chunk 到达（模拟真实网络拆分）
    tool_deltas = [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "type": "function",
             "function": {"name": "calc", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '{"month":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": ' 3}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    # 次轮：纯文本增量（最终回答）
    text_deltas = [{"choices": [{"delta": {"content": p}}]} for p in ["计算结果：", "3000元"]]
    bodies = [_fake_sse_response(tool_deltas), _fake_sse_response(text_deltas)]
    call_no = {"n": 0}

    def fake_urlopen(_req, *_a, **_k):
        body = bodies[min(call_no["n"], len(bodies) - 1)]
        call_no["n"] += 1
        return body

    registry = StubToolRegistry()
    messages = [{"role": "user", "content": "算一下经济补偿"}]
    collected: list[str] = []
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        text = _call_chat_with_tools_stream(
            "http://fake/v1/chat/completions", "sk-fake", "m",
            messages, registry.tool_schemas(), registry,
            on_token=collected.append,
        )
    assert registry.calls == [("calc", {"month": 3})]
    assert collected == ["计算结果：", "3000元"]
    assert text == "计算结果：3000元"
    # assistant(tool_calls) 与 tool 两条消息都进了上下文，闭环成立
    assert any(m["role"] == "tool" and m["tool_call_id"] == "call_1" for m in messages)
    assert any(m["role"] == "assistant" and m.get("tool_calls") for m in messages)


# ---- answer 节点把流式接收器传给 answerer 的接线测试 ----


class StreamingFakeAnswerer:
    """带 on_token 的回答器：把文本拆成若干片段逐段回调（模拟真 LLM 流式）。"""

    def __init__(self):
        self.pieces = ["片段A", "片段B", "片段C"]
        self.on_token_given = None

    def generate(self, query: str, hits: list[FusedHit], history: list[dict] | None = None,
                 on_token=None) -> str:
        self.on_token_given = on_token
        if on_token:
            for p in self.pieces:
                on_token(p)
        return "".join(self.pieces)


def test_graph_streams_answer_tokens_via_sink():
    """路由注入 answer_sink 后，answer 节点应把接收器传给 answerer，token 被逐段收集。"""
    answerer = StreamingFakeAnswerer()
    app = build_agent(FakeRetriever([OVERTIME]), answerer=answerer, verify_mode="pass")
    collected: list[str] = []
    tok = answer_sink_set(collected.append)
    try:
        result = app.invoke({"original": "加班费怎么算"})
    finally:
        answer_sink_reset(tok)
    assert result["answer"].answer == "片段A片段B片段C"
    assert answerer.on_token_given == collected.append
    assert collected == ["片段A", "片段B", "片段C"]


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
        self.called_with: list[tuple[str, list[FusedHit], list[dict]]] = []
        self.return_value = return_value

    def generate(self, query: str, hits: list[FusedHit], history: list[dict] | None = None,
                 on_token=None) -> str:
        # on_token：对齐回答器统一接口（answer 节点恒传此 kwarg），测试用假实现不产生流式文本
        self.called_with.append((query, hits, history or []))
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
    """llm 模式有 key 时 graph 应正常编译。

    rewrite 节点走 _call_chat，answer 节点走 ToolCallingAnswerer → _call_chat_with_tools。
    需同时 mock 两个函数。与阶段 5 之前不同：answer 换成了 ToolCallingAnswerer（带工具注册）。
    """
    with (
        patch("app.agent.protocol._call_chat", return_value="延长工作时间 工资报酬"),
        patch("app.agent.protocol._call_chat_with_tools",
              return_value="根据劳动法第44条应当支付150%的加班费"),
    ):
        graph = build_agent_from_config(
            FakeRetriever([OVERTIME]),
            FakeSettings("llm", llm_api_key="sk-test"),
        )
        result = graph.invoke({"original": "加班费怎么算"})
        assert result["answer"].refuse is False
        assert "150" in result["answer"].answer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])