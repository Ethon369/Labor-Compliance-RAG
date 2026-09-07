"""Agent 的三块可替换能力 + 离线默认实现 + LLM 实现。

与检索层 Reranker/Embedder 同一套路：Graph 面向 Protocol 编程，能力对象由"组装点"
注入。真检索（PG HybridRetriever）、真 LLM（改写/回答）都能在 build_agent 时换上；
离线默认让全链路在"无 key、无 DB"下端到端可跑、可测、可复现——求职演示与 CI 都要它。
见决策 05。

LLM 改写/回答走 urllib（与 SiliconFlowEmbedder 同模式），不引入 requests/httpx——
项目锁定清单里没有它们，urllib 够用且零额外依赖。"""
from __future__ import annotations

import json
import urllib.request
from typing import Protocol

from app.rag.models import FusedHit, RetrievalResult

# ---- 能力协议 ----


class RetrievalProtocol(Protocol):
    """检索能力：给查询，还 RetrievalResult。签名故意对齐 HybridRetriever.search，
    真实现无需适配就是兼容的——这也是"面向接口编程"的落点。"""
    def search(self, query: str, top_k: int = 8, use_rerank: bool = True) -> RetrievalResult: ...


class QueryRewriter(Protocol):
    """查询改写能力：把口语提问润色成检索友好查询。真产品里是 LLM。"""
    def rewrite(self, query: str) -> str: ...


class AnswerGenerator(Protocol):
    """回答生成能力：依据命中的条文生成带引用的答案。真产品里是 LLM。"""
    def generate(self, query: str, hits: list[FusedHit]) -> str: ...


# ---- 离线默认实现（零 key 可跑） ----


class PassThroughRewriter:
    """离线默认改写：去空白恒等变换。让"改写"这一步在无 LLM 时不影响后续检索，
    保证链路离线可复现；换了真 LLM 后这一步才真正润色措辞。"""
    def rewrite(self, query: str) -> str:
        return query.strip()


class TemplateAnswerer:
    """离线默认回答：确定性模板拼条文，不产生引用断言之外的结论（无幻觉）。

    真接 LLM 后这里换成开放式回答；模板句刻意"机械"，只服务两件事——让端到端测试
    能断言"答案引用了 top 条文"，以及演示时结果可预测。"""
    def generate(self, query: str, hits: list[FusedHit]) -> str:
        if not hits:
            return "未检索到可引用的法律条文。"
        lines = [f"《{h.law_id}》第{h.article_no}条：{h.text}" for h in hits]
        return "可参考以下条文：\n" + "\n".join(lines)


# ---- LLM 实现（OpenAI 兼容 chat/completions，urllib 直连） ----


def _call_chat(
    endpoint: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    timeout: int = 60,
) -> str:
    """调用 OpenAI 兼容 /chat/completions，从 choices[0].message.content 取回文本。

    为什么不用 LangChain ChatModel：LangGraph 只需要 node 内部的纯函数，LangChain
    的 ChatModel 包装在这里是多余的一层——本项目"仅用 LangChain 基础组件"，urllib
    直连是更轻、更透明的落点。见决策 05。
    """
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


_REWRITE_SYSTEM = (
    "你是劳动争议智能合规助手的查询理解模块。"
    "将用户输入改写为一个适合法律条文检索的简洁查询。"
    "要求：提取核心法律问题（如工资、加班、解除合同、补偿），"
    "保留关键术语（劳动合同、劳动者、用人单位、工资、补偿、解除等），"
    "去除口语化表达（怎么、怎么办、我/公司等），不添加用户没提到的假设信息。"
    "直接输出改写后的查询，不要加任何解释或附加文字。"
)


class LLMRewriter:
    """LLM 改写：把口语劳动争议问题转成 BM25/语义检索友好的关键词查询。"""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        temperature: float = 0.0,
    ):
        if not api_key:
            raise ValueError("llm 模式需要 LLM_API_KEY")
        self._api_key = api_key
        self._model = model
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._temperature = temperature

    def rewrite(self, query: str) -> str:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _REWRITE_SYSTEM},
            {"role": "user", "content": query},
        ]
        return _call_chat(
            self._endpoint, self._api_key, self._model,
            messages, self._temperature,
        ).strip()


_ANSWER_SYSTEM_HEAD = (
    "你是劳动争议智能合规助手。你的回答必须严格基于以下检索到的法律条文原文。"
    "如果条文能回答用户的问题，请组织一段清晰、易懂的回答，并在每个结论后"
    "注明引用的法条出处（例如: 根据《劳动法》第44条）。"
    "不要编造条文里没有的结论，不要给出法律意见——你只是帮用户理解条文。"
    "如果条文与问题无关或不足以回答，直接说「无法提供明确结论」。"
    "回答字数控制在 300 字以内，通俗易懂。"
)

_ANSWER_USER_TPL = "用户问题：{query}\n\n检索到的法律条文：\n{articles}"


class LLMAnswerer:
    """LLM 回答：基于检索命中的条文组织带引用的自然语言回答。"""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        temperature: float = 0.0,
    ):
        if not api_key:
            raise ValueError("llm 模式需要 LLM_API_KEY")
        self._api_key = api_key
        self._model = model
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._temperature = temperature

    def generate(self, query: str, hits: list[FusedHit]) -> str:
        article_lines: list[str] = []
        for h in hits:
            article_lines.append(f"《{h.law_id}》第{h.article_no}条：{h.text}")
        articles = "\n".join(article_lines) if article_lines else "（未检索到相关条文）"
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _ANSWER_SYSTEM_HEAD},
            {"role": "user", "content": _ANSWER_USER_TPL.format(query=query, articles=articles)},
        ]
        return _call_chat(
            self._endpoint, self._api_key, self._model,
            messages, self._temperature,
        ).strip()