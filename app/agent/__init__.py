"""LangGraph 流程编排（阶段 4）：查询改写 → 检索 → 引用核验 → 回答 / 拒答 的 StateGraph。

入口是 build_agent()；节点/协议/状态模型都在此导出，便于测试与 API 层复用。"""
from __future__ import annotations

from app.agent.graph import build_agent, build_agent_from_config
from app.agent.nodes import route_after_verify
from app.agent.protocol import (
    AnswerGenerator,
    LLMAnswerer,
    LLMRewriter,
    PassThroughRewriter,
    QueryRewriter,
    RetrievalProtocol,
    TemplateAnswerer,
)
from app.agent.state import AgentAnswer, AgentInput, AgentState, Citation
from app.agent.verify import CitationVerifier

__all__ = [
    "AgentAnswer",
    "AgentInput",
    "AgentState",
    "AnswerGenerator",
    "Citation",
    "CitationVerifier",
    "LLMAnswerer",
    "LLMRewriter",
    "PassThroughRewriter",
    "QueryRewriter",
    "RetrievalProtocol",
    "TemplateAnswerer",
    "build_agent",
    "build_agent_from_config",
    "route_after_verify",
]
