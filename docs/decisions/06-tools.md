# 06 · Function Calling 工具调用

> 阶段 5 决策记录。对应 `app/tools/compensation.py`（经济补偿金计算器）、`app/tools/registry.py`（ToolRegistry）、`app/agent/protocol.py` 的 `ToolCallingAnswerer`。
> 格式约定（CLAUDE.md 第 6 条）：选了什么、备选、为什么、已知取舍。

## 1. 一句话结论

**工具调用（function calling）的底层是 JSON Schema 约定——LLM 不是"调用函数"，而是输出一个 JSON 对象描述"我想调哪个函数、参数是什么"；宿主程序收到后真正执行，再把结果用 JSON 发回去。** 本项目手写 Schema + urllib 直连，不用 LangChain tool 装饰器——因为 Schema 措辞直接影响 LLM 会不会调、参数怎么填，手写才能精确控制。

## 2. Function Calling 的底层机制（面试必答）

整个过程是 **4 步 JSON 约定循环**，不是魔法：

```
1. 宿主 → LLM（带 tools 列表）
   POST /chat/completions
   {
     "model": "deepseek-chat",
     "messages": [...],
     "tools": [                                    ← 告诉 LLM 有哪些工具
       {
         "type": "function",
         "function": {
           "name": "calculate_compensation",
           "description": "根据《劳动合同法》第47条计算...",
           "parameters": {                         ← JSON Schema 定义入参形态
             "type": "object",
             "properties": {
               "years_of_service": {"type": "number", "description": "工作年限"},
               "monthly_salary": {"type": "number", "description": "月平均工资"}
             },
             "required": ["years_of_service", "monthly_salary"]
           }
         }
       }
     ]
   }

2. LLM → 宿主（返回 tool_call，不是文本）
   {
     "choices": [{
       "message": {
         "role": "assistant",
         "content": null,                          ← 不是文本回答
         "tool_calls": [{                          ← 而是"我要调这个工具"
           "id": "call_abc123",
           "type": "function",
           "function": {
             "name": "calculate_compensation",
             "arguments": "{\"years_of_service\":3,\"monthly_salary\":8000}"
           }
         }]
       }
     }]
   }

3. 宿主执行工具 → 把结果发回 LLM
   messages.append(assistant_msg)                  ← 保留 LLM 的 tool_call 消息
   messages.append({                               ← 追加工具返回
     "role": "tool",
     "tool_call_id": "call_abc123",
     "content": "{\"n_months\":3.0,\"total\":24000.0,...}"
   })

4. LLM → 宿主（这次是文本回答）
   "根据《劳动合同法》第47条，您工作3年、月薪8000元，
    应获得经济补偿金24000元（3个月×8000元）。"
```

**关键理解**：LLM 从头到尾没有"执行"任何代码。它只是按 Schema 的 description 理解什么时候该调、参数该填什么——Schema 措辞质量直接影响调用准确率。

## 3. 为什么手写 Schema，不用 LangChain tool 装饰器

| 维度 | LangChain `@tool` 装饰器 | 本项目手写 Schema |
|---|---|---|
| Schema 来源 | 从函数签名 + docstring 自动生成 | 手写，精确控制每个字段的 description |
| description 质量 | 依赖 docstring 质量，不可控 | 专门为 LLM 阅读优化措辞（见 §4） |
| 依赖 | 引入 langchain 额外依赖 | 零新增依赖（纯 dict） |
| 透明度 | 黑盒——你不知道 LLM 收到的 Schema 长什么样 | 完全透明——Schema 就是代码里的 dict |

**核心原因**：Schema 的 `description` 字段是 LLM 决定"调不调、参数怎么填"的唯一依据。手写让你精确控制这段措辞——比如 `local_avg_salary` 的 description 写了"当劳动者月工资高于此值3倍时，补偿基数按3倍封顶，年限最高12年"，LLM 读到这行就知道"用户没说当地社平工资就不填这个字段"。装饰器从 docstring 自动生成的 description 达不到这个精度。

## 4. 补偿金 Schema 设计要点

经济补偿金计算器（《劳动合同法》第47条 + 第87条违法解除二倍赔偿）：

- **入参设计**：`years_of_service`（支持小数如 3.5）、`monthly_salary`、`local_avg_salary`（可选——LLM 从用户话里没听到就不传）、`reason`（enum 6 个值，含 `illegal` 触发二倍赔偿）
- **为什么 local_avg_salary 是可选的**：用户说"我月薪 8000 被辞退"时没说当地社平工资——LLM 从 Schema 的 description 理解"不填就不设上限"，不会胡乱编一个数字
- **为什么 reason 用 enum 而不是让 LLM 自由填**：enum 约束让 LLM 在有限选项中选——降误判率；如果 LLM 不确定选哪个，6 个选项能覆盖全场景
- **计算结果用 Pydantic 承载**：`CompensationResult` 包含 `n_months`/`base_salary`/`total`/`is_capped`/`is_illegal`/`detail`——前几个是数字（供 LLM 融入回答），`detail` 是中文摘要（LLM 可直接引用或改写）

## 5. ToolRegistry 设计

- **为什么不是全局 dict**：注册中心把"Schema 列表"和"执行分发"绑在一起——给 LLM 的 schemas 和收到 tool_call 后的 dispatch 走同一个对象，新增工具只改注册处
- **ToolEntry.invoke**：支持 input_model 校验（Pydantic），在工具执行前拦截非法参数——LLM 的 JSON 参数即使格式对、值也可能非法（负工资），校验在宿主侧兜底
- **`default_registry()` 工厂**：后续加工具（工伤待遇、生育津贴）只需在此函数里多注册一条

## 6. 工具选择失败怎么兜底（面试高频）

三个层次的兜底，从近到远：

1. **工具未注册（KeyError）**：LLM 可能因为 Schema description 歧义叫了一个不存在的工具名——`_call_chat_with_tools` 捕获 KeyError → 把 `{"error": "工具 'xxx' 未注册"}` 作为 tool 结果发回 LLM → LLM 看到错误消息后重新组织回答（通常会说"抱歉，无法计算"）
2. **工具执行异常（Exception）**：参数格式对但值导致计算异常——同样捕获，error 消息发回 LLM
3. **LLM 不调工具（该调没调）**：Schema description 措辞是主要防线；如果 LLM 跳过工具直接回答，系统不会强制纠正——因为场景是"辅助参考"，漏算一次比误算强。但可通过后续评测统计"该调用但没调"的比例来调优 Schema 措辞
4. **max_rounds 耗尽**：工具调用循环最多 5 轮（model→tool→model→tool→...）——防止 LLM 和工具来回拉扯无限循环（如 LLM 反复微调参数重算）。耗尽后倒序扫 messages 取最后一条 assistant content；如果全程没有文本就返回固定兜底文案

## 7. 为什么 tool_call 循环在 Answerer 内部而不是 Graph 节点

当前设计：`ToolCallingAnswerer.generate()` 内部处理 tool_call 循环，Graph 对此无感知——Graph 只看到"回答器返回了文本"。

备选方案（Graph 节点化）：在 answer 节点后加 `tool_decide` → `tool_execute` → 回到 answer 的条件边。好处是可观测性更强（每个 tool_call 作为 state 中的一个事件），代价是 Graph 拓扑更复杂、节点边界更多。

取舍：**v1 用内部循环**——工具调用循环本质上是 LLM 生成回答过程中的子流程，对 Graph 来说"生成回答"是一步。内部循环让 Graph 保持简洁（5 节点不变），代价是 tool_call 过程不可见。**v2 如果工具变多**（>5个）、需要可观测性，再拆成 Graph 节点。

## 8. 已知取舍

1. **只有一个工具**：当前仅注册经济补偿金计算器。工伤待遇、生育津贴、未签合同双倍工资等后续按同一模式注册——Schema 手写、Pydantic 校验、ToolEntry 包装。
2. **手写 Schema 的人力成本**：每个工具约 20 行 Schema dict。好处是措辞可控，坏处是没自动生成方便。对"展示项目"来说手写是正解——面试官会问"Schema 怎么设计的"，你指着每个字段的 description 讲为什么这么写。
3. **urllib 直连无流式**：function calling 的请求/响应与文本对话走同一 `/chat/completions` 端点，urllib 够用。阶段 6 做 SSE 流式时，流式 tool_calls 的处理会在 API 层实现。
4. **没有并行工具调用**：当前一次 LLM 响应可能返回多个 tool_calls，但 `_call_chat_with_tools` 是串行执行的（逐个调）。对于当前单一工具场景无关紧要；多个独立工具时可以并行执行（用 `concurrent.futures`），这是 v2 优化项。