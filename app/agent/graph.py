"""LangGraph 组装点：把 5 个节点拧成 StateGraph 并编译，对外暴露 build_agent()。

注入方式与检索层 build_retriever 一致：能力对象从参数进，缺省用离线占位——调用方决定
检索器是内存假实现（测试）还是真 HybridRetriever（PG）。Graph 对存储/LLM 无感知，
这正是"进程编排层与外部世界解耦"的落点。见决策 05。

新增 build_agent_from_config() 快捷组装：从 Settings 读 llm_mode，离线=占位，llm=真 LLM。
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    make_answer_node,
    make_refuse_node,
    make_retrieve_node,
    make_rewrite_node,
    make_verify_node,
    route_after_verify,
)
from app.agent.protocol import (
    AnswerGenerator,
    LLMAnswerer,
    LLMRewriter,
    PassThroughRewriter,
    QueryRewriter,
    RetrievalProtocol,
    TemplateAnswerer,
)
from app.agent.state import AgentInput, AgentState
from app.agent.verify import CitationVerifier
from app.core.config import Settings


def build_agent(
    retriever: RetrievalProtocol,
    rewriter: QueryRewriter | None = None,
    answerer: AnswerGenerator | None = None,
    threshold: float = 0.3,
):
    """组装并编译 Graph。retriever 必传；改写/回答缺省用离线占位，核验阈值可调。"""
    rewriter = rewriter or PassThroughRewriter()
    answerer = answerer or TemplateAnswerer()
    verifier = CitationVerifier(threshold)

    g = StateGraph(AgentState, input_schema=AgentInput)
    g.add_node("rewrite", make_rewrite_node(rewriter))
    g.add_node("retrieve", make_retrieve_node(retriever))
    g.add_node("verify", make_verify_node(verifier))
    g.add_node("answer", make_answer_node(answerer))
    g.add_node("refuse", make_refuse_node())

    g.add_edge(START, "rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "verify")
    g.add_conditional_edges("verify", route_after_verify, {"answer": "answer", "refuse": "refuse"})
    g.add_edge("answer", END)
    g.add_edge("refuse", END)

    return g.compile()


def build_agent_from_config(
    retriever: RetrievalProtocol,
    settings: Settings,
) -> StateGraph:
    """快捷组装：从 Settings 读 llm_mode 决定改写/回答用离线占位还是真 LLM。

    离线模式：PassThroughRewriter + TemplateAnswerer（零 key、可测、可演示）
    llm 模式：LLMRewriter + LLMAnswerer（OpenAI 兼容接口，key 从 settings 取）
    检索器由调用方传入——可以是真 PG HybridRetriever，也可以是测试用 FakeRetriever。
    """
    if settings.llm_mode == "llm":
        api_key = settings.llm_api_key or ""
        if not api_key:
            raise ValueError("llm 模式需要 LLM_API_KEY，请填 .env 或改 LLM_MODE=offline")
        rewriter: QueryRewriter | None = LLMRewriter(
            api_key=api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
        )
        answerer: AnswerGenerator | None = LLMAnswerer(
            api_key=api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
        )
    else:
        rewriter = None  # build_agent 缺省用 PassThroughRewriter
        answerer = None  # 缺省用 TemplateAnswerer

    return build_agent(retriever, rewriter=rewriter, answerer=answerer)