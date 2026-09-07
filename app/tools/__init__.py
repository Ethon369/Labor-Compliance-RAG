"""Function Calling 工具（阶段 5）：经济补偿金计算器 + 工具注册中心。

后续工具（如工伤待遇、生育津贴计算）均在此注册，统一通过 ToolRegistry 调度。"""

from __future__ import annotations

from app.tools.compensation import (
    COMPENSATION_TOOL_SCHEMA,
    CompensationInput,
    CompensationResult,
    calculate_compensation,
)
from app.tools.registry import ToolEntry, ToolRegistry, default_registry

__all__ = [
    "COMPENSATION_TOOL_SCHEMA",
    "CompensationInput",
    "CompensationResult",
    "ToolEntry",
    "ToolRegistry",
    "calculate_compensation",
    "default_registry",
]