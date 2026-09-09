"""Agent 的 5 个节点 + 条件边路由。

节点都是"读部分状态 → 返回要更新的字段"的纯函数，自身不碰 DB/网络、只调构造时注入
的能力对象——所以每个节点可单独断言。拓扑（决策 05）：

    START -c-> rewrite -c-> retrieve -c-> verify -?-> answer -c-> END
                                                  └→ refuse -c-> END

verify 之后的条件边是 LangGraph 与 Chain 的本质分界线：它让"下一条走哪"由状态运行时
决定，Chain 只有写死的下一步。这也是面试"为什么用 Graph"的第一句话。
"""
from __future__ import annotations

from typing import Any

from app.agent.protocol import (
    AnswerGenerator,
    QueryRewriter,
    RetrievalProtocol,
    answer_sink_get,
)
from app.agent.state import AgentAnswer, Citation
from app.agent.verify import CitationVerifier

REFUSAL_TEXT = (
    "抱歉，根据现有法律依据无法就该问题给出明确结论。\n"
    "建议咨询专业劳动法律师，或联系当地劳动争议调解仲裁机构获取权威解答。"
)


def make_rewrite_node(rewriter: QueryRewriter):
    """节点工厂：把能力对象封装成 LangGraph 节点函数（闭包注入，见协议 why）。"""

    def node(state: dict[str, Any]) -> dict[str, Any]:
        return {"rewritten": rewriter.rewrite(state["original"])}

    return node


def make_retrieve_node(retriever: RetrievalProtocol):
    def node(state: dict[str, Any]) -> dict[str, Any]:
        # kb_ids 由路由层按可见性解析后放进输入；缺省让检索器用默认库
        result = retriever.search(state["rewritten"], kb_ids=state.get("kb_ids"))
        return {"hits": result.hits}

    return node


def make_verify_node(verifier: CitationVerifier, mode: str = "rule"):
    """核验节点。mode 决定"要不要拒答"这条边界由谁把关：
    - "rule"：CitationVerifier 规则核验（top-1 条文对查询意图词的覆盖率达阈值才可答）
    - "pass"：离线模板模式不设防——有命中即可答（空命中和规则模式一样拒答）

    为什么 offline 默认用 pass：模板回答本身不产生"新结论"（只拼条文原文），
    不存在幻觉风险；设防只会挡住演示和测试。真 LLM 上线后切回 rule，
    让"条文不足以支撑回答"的判断交给规则兜底（LLM 自己也会说"无法提供明确结论"）。
    """

    def node(state: dict[str, Any]) -> dict[str, Any]:
        hits = state.get("hits") or []
        supported = bool(hits) if mode == "pass" else verifier.supported(state["rewritten"], hits)
        return {"supported": supported}

    return node


def route_after_verify(state: dict[str, Any]) -> str:
    """条件边 path：核验支持→回答节点，否则→拒答节点。返回值即 path_map 的键。"""
    return "answer" if state.get("supported") else "refuse"


def make_answer_node(answerer: AnswerGenerator):
    def node(state: dict[str, Any]) -> dict[str, Any]:
        hits = state.get("hits") or []
        # 引用顺序 = 检索排序顺序，与 answerer 拼给 LLM 的 [n] 编号一一对应
        citations = [
            Citation(
                kb_id=h.kb_id, doc_id=h.doc_id, doc_title=h.doc_title, seq=h.seq,
                page=h.page, heading=h.heading, text=h.text,
                # 相关度优先用精排分（更接近"对这个问题有多相关"），无精排时退融合分
                score=h.rerank_score if h.rerank_score is not None else h.rrf_score,
                source_law_id=h.source_law_id,
            )
            for h in hits
        ]
        # 路由层在流式请求时通过 answer_sink 注入 token 接收器（见 protocol.answer_sink_*）。
        # answerer 生成期间逐段回调它；离线/测试没有接收器则为 None，走一次性整段生成。
        return {
            "answer": AgentAnswer(
                refuse=False,
                rewritten=state["rewritten"],
                answer=answerer.generate(
                    state["rewritten"], hits, state.get("history"),
                    on_token=answer_sink_get(),
                ),
                citations=citations,
            )
        }

    return node


def make_refuse_node():
    def node(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "answer": AgentAnswer(
                refuse=True,
                rewritten=state["rewritten"],
                answer=REFUSAL_TEXT,
                citations=[],
            )
        }

    return node
