"""Agent 状态与数据模型。跨层数据一律 Pydantic 承载，State 只是 LangGraph 引擎的接线外壳。

为什么状态用 TypedDict、承载用 Pydantic：LangGraph 1.x 的 StateGraph 直接吃 TypedDict
做状态 schema，节点返回"部分更新 dict"由引擎自动合并——这是画图（边）写状态（字段）的
最顺写法；但每次检索/回答产生的"结果对象"要跨层复用、要能被 API 直接序列化，所以用
项目自家 Pydantic 模型（FusedHit / AgentAnswer）兜住。分层：引擎管流程，Pydantic 管数据。
"""
from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel, Field

from app.rag.models import FusedHit


class AgentInput(BaseModel):
    """Graph 入口校验模型。字段名取 original（与状态同名，LangGraph 据此投影进状态），
    API 层若想对外暴露 query 这个名字，可在路由层映射——Graph 内部保持一致。"""
    original: str


class Citation(BaseModel):
    """核验通过、准备写进答案的条文引用：只携带定位键（law_id/article_no）与条文原文。"""
    law_id: str
    article_no: int
    chapter: str | None = None
    text: str


class AgentAnswer(BaseModel):
    """Graph 最终产物：答案文本、是否拒答、命中的条文引用。

    refuse=True 时 citations 应为空——"回答"与"拒答"是 verify 条件边分派的互斥支路。
    """
    refuse: bool = False
    rewritten: str = ""
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    """LangGraph 引擎状态外壳：每个字段对应一个或多个节点要写/读的状态。

    total=False 让中间字段（rewritten/hits/supported）在到达对应节点前可以不存在。
    """
    original: str
    rewritten: str
    hits: list[FusedHit]
    supported: bool
    answer: AgentAnswer
