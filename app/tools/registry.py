"""工具注册中心：把工具函数和 JSON Schema 绑定，统一调度。

为什么要有注册中心：LLM function calling 需要两个配合——(1) 传给 LLM 的
tool schemas 列表（告诉模型"你能调什么"），(2) 收到 LLM tool_call 后的实际
调用分发（exec）。注册中心把两者绑在一个地方，新增工具只需在此注册、不必改其它代码。

为什么不用 LangChain tool 装饰器：本项目的工具是纯 Python 函数 + 手写 JSON
Schema——手写 Schema 让你精确控制传给 LLM 的描述措辞（这直接影响 LLM 会不会调、
参数会怎么填），黑盒装饰器在这方面不可控。见决策 06。
"""

from __future__ import annotations

from typing import Any, Callable

from app.tools.compensation import (
    COMPENSATION_TOOL_SCHEMA,
    CompensationInput,
    CompensationResult,
    calculate_compensation,
)


class ToolEntry:
    """一个已注册工具的元数据：名称、函数、Schema、Pydantic 输入/输出校验模型。"""

    def __init__(
        self,
        name: str,
        func: Callable[..., Any],
        schema: dict[str, Any],
        input_model: type[CompensationInput] | None = None,
    ):
        self.name = name
        self.func = func
        self.schema = schema
        self.input_model = input_model

    def invoke(self, arguments: dict[str, Any]) -> Any:
        """调用已注册工具。input_model 存在时先校验输入。"""
        if self.input_model is not None:
            parsed = self.input_model.model_validate(arguments)
            return self.func(**parsed.model_dump())
        return self.func(**arguments)


class ToolRegistry:
    """工具注册中心：注册工具 → 获取 schemas 列表 → 按名称分发调用。"""

    def __init__(self):
        self._tools: dict[str, ToolEntry] = {}

    def register(self, entry: ToolEntry) -> None:
        if entry.name in self._tools:
            raise ValueError(f"工具 '{entry.name}' 已注册")
        self._tools[entry.name] = entry

    def tool_schemas(self) -> list[dict[str, Any]]:
        """返回 LLM function calling 所需的 tools 列表。"""
        return [entry.schema for entry in self._tools.values()]

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        """按工具名分发调用，未注册的工具抛 KeyError。"""
        entry = self._tools.get(name)
        if entry is None:
            raise KeyError(f"未注册的工具: {name}")
        return entry.invoke(arguments)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def default_registry() -> ToolRegistry:
    """工厂：注册经济补偿金计算器（后续工具继续往里加）。"""
    r = ToolRegistry()
    r.register(
        ToolEntry(
            name="calculate_compensation",
            func=calculate_compensation,
            schema=COMPENSATION_TOOL_SCHEMA,
            input_model=CompensationInput,
        )
    )
    return r