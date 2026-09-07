# 05 · LangGraph Agent 编排

> 阶段 4 决策记录。对应 `app/agent/`（StateGraph + 5 节点 + 条件路由 + 引用核验 + 拒答分支 + LLM 改写/回答）。
> 格式约定（CLAUDE.md 第 6 条）：选了什么、备选、为什么、已知取舍。

## 1. 一句话结论

**用 LangGraph 的 StateGraph 做流程编排，不用 Chain。** 核心原因：检索结果质量不确定 → 后续路径必须由"状态运行时"的分支逻辑决定，不是写死的线性步骤。条件边（verify → answer/refuse）是 Graph 优于 Chain 的第一句话。

## 2. Graph 拓扑

```
    START -c→ rewrite -c→ retrieve -c→ verify -?→ answer -c→ END
                                                  └→ refuse -c→ END
```

- **rewrite**：口语劳动争议问题 → 检索友好查询（去掉"我怎么/怎么办/公司"、保留法条关键词）
- **retrieve**：检索（面向 RetrievalProtocol，注入真 HybridRetriever 或测试 FakeRetriever）
- **verify**：纯规则核验 top-1 条文能否支持回答（词元覆盖率阈值 ≥ 0.3）
- **answer**（verify 通过）：组织带条文引用的答案
- **refuse**（verify 不通过）：拒答，引导咨询专业人士

## 3. 选型：Graph vs Chain

| 维度 | LangChain Chain | LangGraph StateGraph |
|---|---|---|
| 路径 | 写死（线性或 RunnableBranch） | 运行时根据状态决定下一节点 |
| 条件路由 | 有，但嵌套深了可读性差 | `add_conditional_edges` 是原生能力 |
| 循环/重试 | 需要自己包循环逻辑 | `add_edge` 回到前面节点即可 |
| 状态管理 | 隐式（Chain 内部传递） | TypedDict 显式声明，LangGraph 自动合并部分更新 |
| 面试价值 | "会用框架" | "理解编排的 why：什么时候需要分支、状态怎么设计" |

**为什么不直接用 Chain：** 检索结果质量是不确定的——有可能召回了好条文，也可能召回了完全无关的。后续步骤（回答 vs 拒答）应该由检索结果的质量决定，而不是一条线走到底。Chain 的本质是"下一步总是已知"；Graph 的本质是"下一步取决于状态"。本案的后一个本质才是正确的抽象。

## 4. 核心设计决策

### 4.1 状态用 TypedDict（LangGraph 引擎） + 数据用 Pydantic（跨层复用）

- **AgentState**（TypedDict）：LangGraph 引擎只识 TypedDict 做状态 schema。节点返回"部分更新 dict"，引擎自动合并——这是 LangGraph 1.x 最佳实践，比自定义 Reducer 简单。
- **AgentAnswer / AgentInput / Citation**（Pydantic）：跨层复用对象用 Pydantic 承载校验与序列化。分层：引擎管流程，Pydantic 管数据。
- **total=False**：中间字段（rewritten/hits/supported）在对应节点前可以不存在，LangGraph 在第一次写入时才创建。

### 4.2 能力对象通过 Protocol 注入（与检索层同一套路）

Graph 对存储/LLM/核验规则无感知——这四个是"能力对象"，在 `build_agent()` 组装时注入：

```python
build_agent(
    retriever=...,      # RetrievalProtocol（必传）
    rewriter=...,       # QueryRewriter（缺省 PassThrough）
    answerer=...,       # AnswerGenerator（缺省 TemplateAnswerer）
    threshold=0.3,      # 核验阈值
)
```

为什么这样：每个节点是纯函数 + 注入对象，可单独单测——节点不管对象内部做了什么，只调接口。

### 4.3 引用核验是纯规则，不是 LLM

"要不要拒答"这条边界必须确定、可测、离线可复现。用 LLM 判的坑：同一输入连续两次可能给出不同答案；新模型上线阈值会飘；测试断言不可靠。规则简单但诚实——top-1 条文对查询的意图词覆盖率 ≥ threshold → 可答，否则走拒答。threshold 是接口参数，面试/CI/演示三场景都能稳定触发两条支路。

### 4.4 LLM 调用用 urllib 直连，不引入 LangChain ChatModel

与 `SiliconFlowEmbedder`/`SiliconFlowReranker` 同套路：`urllib.request` + `json` 零新增依赖。LangChain ChatModel 的包装在这里多了一层抽象没有收益——Graph 的 node 只需要"给我一段 LLM 返回的文本"这个纯函数结果，urllib 直连更轻、更透明。（LangChain ChatModel 的流式/重试/token counting 在这个阶段不需要，阶段 6 做 SSE 流式时会在 API 层直接流式转发。）

### 4.5 离线占位（零 key 可跑，保持全链路可复现）

- **PassThroughRewriter**：`query.strip()`——恒等变换，不影响检索
- **TemplateAnswerer**：确定性模板拼条文——答案可预测、测试可断言
- **真 LLM（LLMRewriter / LLMAnswerer）**：填 key 后从 `build_agent_from_config(settings)` 一键换上，接口同 Protocol

## 5. 循环与终止条件

当前 v1 无循环——4 个带条件的线性步骤 + 1 个条件分叉。终止条件：
- answer → END
- refuse → END

如需加入循环（Agentic RAG 常见两个场景）：
1. **检索结果不足 → 改写 → 重新检索**：verify 失败后不走 refuse，而是回到 rewrite 改更宽的关键词（如把"加班费"扩写成"工资报酬 延长工作时间"），再检索一次。**前提**：要限定最大改写次数（如 2 次），否则可能无限循环。
2. **答案质量自检**：在 answer 后加一个 LLM 自检节点，如果发现引用不准确（条文不支持答案里的某个断言），回到 rewrite。

LangGraph 的循环是在 conditional_edges 的 `path_map` 里把某个分支指回上游节点（如 `"rewrite"`），不是另外的 API。这和条件分叉用的是同一个机制——这也是面试"Graph 比 Chain 强在哪"的第二句话。

## 6. 面试高频题：LangGraph

### Q: 为什么用 Graph 不用 Chain？
> 见 §3。核心原因：我的流程有**条件分叉**（检索质量够不够→答还是拒答），不是每一步都确定下一步是什么。Graph 的 conditional_edges 是原生能力；Chain 的 RunnableBranch 嵌套深了是面条代码。

### Q: 状态怎么设计的？
> 见 §4.1。LangGraph 1.x 用 TypedDict 做 State，节点返回部分更新——这和 Redux reducer 是一个思路，区别是 LangGraph 自动合并而不是你写 combine。数据跨层复用用 Pydantic，因为 Pydantic 自带校验+序列化，API 层直接 `.model_dump()` 就输出 JSON。

### Q: 循环怎么控制，怎么防止无限循环？
> 见 §5。v1 没循环——只有一次条件分叉。如果加循环（改写→重新检索），关键是"最大轮数软上限"和一个"放弃后走拒答"的兜底——类似于 HTTP 客户端的 max_retries。LangGraph 没有内置的循环次数限制，要自己在状态里加一个 `retry_count` 字段，每次 rewrite 时 +1，conditional_edges 看到 retry_count > 2 就强制走 refuse。

### Q: LangGraph 和 LangChain 是什么关系？
> LangGraph 的"节点"是纯函数——它不强制你用 LangChain 的 Runnable。我项目里检索节点调的是自家 `HybridRetriever`，LLM 节点用 urllib 直连 OpenAI 兼容接口，都没通过 LangChain。LangGraph 是"编排层"，LangChain 是"组件层"——本项目只用编排层，组件层用项目自己的实现。这恰恰是 LangGraph 作者本意：Graph is a runtime, not a component library。

## 7. 已知取舍

1. **核验是词元覆盖率，不是语义覆盖率**。离线占位下"加班费"和"延长工作时间"算不相关（虽然法律上就是同一件事——加班费是延长工作时间的报酬，见劳动法第 44 条）。真 LLM 上线后可在 rewrite 节点做语义扩展（如"加班费 → 延长工作时间 工资报酬"），或对核验加一层 LLM 语义判断。当前阈值 0.3 是经验值——既不会太严（一句话的 query 只要有一条关键术语命中就过），也不会太松（无关条文几乎过不了）。
2. **v1 无循环**。当前流程是"一次改写→一次检索→一次核验→一次回答/拒答"。不重试的好处是延迟可预测、行为确定，坏处是遇到"改写不到位→召回失败"无法自愈。这是刻意取舍：v1 先验证四步链路正确性，循环在 v2 加。
3. **LLM 调用无重试**。urllib 的网络异常原样抛给调用方——与 `SiliconFlowEmbedder` 同策略：抛出来比静默降级好，让上层（API 层/测试）决定怎么处理。
4. **检索结果全部写入 state.hits**。如果 hits 数量很大，state 会在节点间传一个不小的 list。205 条法条、top-8 结果——这点数据传输量在内存里微不足道，不值得做引用优化。