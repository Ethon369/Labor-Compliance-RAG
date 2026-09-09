"""阶段 5 工具调用单测：经济补偿金计算器 + 工具注册中心 + ToolCallingAnswerer mock。

覆盖：计算器公式验证（含封顶/违法解除/边界）、ToolRegistry 注册/调用/异常、
ToolCallingAnswerer 的 tool_call 循环（mock HTTP）。全离线可跑，无需 LLM key。
"""
from __future__ import annotations

import json
import io
from unittest.mock import patch

import pytest

from app.tools.compensation import (
    CompensationInput,
    CompensationResult,
    calculate_compensation,
    COMPENSATION_TOOL_SCHEMA,
)
from app.tools.registry import ToolEntry, ToolRegistry, default_registry
from app.agent.protocol import ToolCallingAnswerer, _call_chat_with_tools
from app.rag.models import FusedHit


# ====== 计算器纯函数单测 ======

LABOR_CONTRACT_47 = FusedHit(
    chunk_id=47, kb_id=1, doc_id=1, doc_title="劳动合同法", seq=47,
    heading="第四章 劳动合同的解除和终止",
    text="经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资...",
    source_law_id="labor_contract_law", lanes=["bm25"], rrf_score=1.0,
)


def test_standard_compensation():
    """标准场景：3.5年，月薪8000，无封顶。"""
    r = calculate_compensation(years_of_service=3.5, monthly_salary=8000)
    assert r.n_months == 4.0  # 3年 + 6个月以上=1年
    assert r.base_salary == 8000
    assert r.total == 32000.0
    assert r.is_capped is False
    assert r.is_illegal is False


def test_less_than_six_months():
    """不满6个月：年限0.3 → 0.5个月。"""
    r = calculate_compensation(years_of_service=0.3, monthly_salary=6000)
    assert r.n_months == 0.5
    assert r.total == 3000.0


def test_exactly_six_months():
    """正好6个月：算1年（≥0.5 按1年算）。"""
    r = calculate_compensation(years_of_service=2.5, monthly_salary=5000)
    assert r.n_months == 3.0  # 2 + 1
    assert r.total == 15000.0


def test_more_than_six_months():
    """超过6个月（如0.7年）：算1年。"""
    r = calculate_compensation(years_of_service=1.7, monthly_salary=5000)
    assert r.n_months == 2.0
    assert r.total == 10000.0


def test_full_years():
    """整数年限：5年 → 5个月。"""
    r = calculate_compensation(years_of_service=5.0, monthly_salary=10000)
    assert r.n_months == 5.0
    assert r.total == 50000.0


def test_long_tenure_uncapped():
    """长年限无封顶：20年 → 20个月（未触发3倍封顶所以不限制年限）。"""
    r = calculate_compensation(years_of_service=20.0, monthly_salary=5000)
    assert r.n_months == 20.0
    assert r.total == 100000.0


def test_salary_cap_triggered():
    """月薪30000 > 社平8000×3=24000 → 基数是24000，年限最长12年。"""
    r = calculate_compensation(
        years_of_service=15.0, monthly_salary=30000, local_avg_salary=8000,
    )
    assert r.is_capped is True
    assert r.base_salary == 24000.0  # 三倍封顶
    assert r.n_months == 12.0  # 15年但封顶后限12
    assert r.total == 288000.0


def test_salary_cap_not_triggered_when_just_at_limit():
    """月薪正好3倍：不触发封顶（>才是封顶条件）。"""
    r = calculate_compensation(
        years_of_service=3.0, monthly_salary=24000, local_avg_salary=8000,
    )
    assert r.is_capped is False
    assert r.base_salary == 24000
    assert r.n_months == 3.0


def test_short_tenure_under_cap():
    """封顶触发但年限本来就短：社平3倍封顶 + 年限4年 → 4个月（不截断）。"""
    r = calculate_compensation(
        years_of_service=4.0, monthly_salary=50000, local_avg_salary=6000,
    )
    assert r.is_capped is True
    assert r.n_months == 4.0  # 不到12，不截断
    assert r.base_salary == 18000.0  # 6000×3


def test_illegal_dismissal_double():
    """违法解除：按第87条，赔偿金 = 2x补偿金。"""
    r = calculate_compensation(
        years_of_service=3.0, monthly_salary=8000, reason="illegal",
    )
    assert r.is_illegal is True
    assert r.n_months == 3.0
    assert r.base_salary == 8000
    assert r.total == 48000.0  # 3×8000×2
    assert "赔偿金" in r.detail


def test_zero_years():
    """年限=0。"""
    r = calculate_compensation(years_of_service=0.0, monthly_salary=5000)
    assert r.n_months == 0.0
    assert r.total == 0.0


def test_detail_contains_key_info():
    """计算结果摘要包含必要信息。"""
    r = calculate_compensation(years_of_service=2.0, monthly_salary=7000)
    assert "2.0年" in r.detail
    assert "7000" in r.detail
    assert "14000" in r.detail


# ====== CompensationInput 校验 ======

def test_input_rejects_negative_years():
    with pytest.raises(Exception):
        CompensationInput(years_of_service=-1.0, monthly_salary=5000)


def test_input_rejects_zero_salary():
    with pytest.raises(Exception):
        CompensationInput(years_of_service=3.0, monthly_salary=0)


def test_input_rejects_negative_salary():
    with pytest.raises(Exception):
        CompensationInput(years_of_service=3.0, monthly_salary=-100)


def test_input_accepts_valid():
    i = CompensationInput(years_of_service=3.5, monthly_salary=8000, reason="layoff")
    assert i.years_of_service == 3.5
    assert i.reason == "layoff"


# ====== JSON Schema 验证 ======

def test_schema_has_required_fields():
    s = COMPENSATION_TOOL_SCHEMA
    assert s["type"] == "function"
    fn = s["function"]
    assert fn["name"] == "calculate_compensation"
    assert "parameters" in fn
    assert "required" in fn["parameters"]
    assert "years_of_service" in fn["parameters"]["required"]
    assert "monthly_salary" in fn["parameters"]["required"]


def test_schema_enum_values_complete():
    """Schema 的 reason 枚举应包含全部场景。"""
    reason_enum = COMPENSATION_TOOL_SCHEMA["function"]["parameters"]["properties"]["reason"]["enum"]
    assert "termination" in reason_enum
    assert "expiration" in reason_enum
    assert "illegal" in reason_enum


# ====== ToolRegistry 注册中心 ======

def test_registry_register_and_call():
    r = ToolRegistry()
    r.register(ToolEntry(
        name="calc",
        func=calculate_compensation,
        schema=COMPENSATION_TOOL_SCHEMA,
        input_model=CompensationInput,
    ))
    assert len(r) == 1
    assert "calc" in r
    result = r.call("calc", {"years_of_service": 3.0, "monthly_salary": 8000})
    assert isinstance(result, CompensationResult)
    assert result.total == 24000.0


def test_registry_call_unknown_tool():
    r = ToolRegistry()
    with pytest.raises(KeyError, match="未注册"):
        r.call("nonexistent", {})


def test_registry_duplicate_rejects():
    r = ToolRegistry()
    entry = ToolEntry(name="dup", func=calculate_compensation, schema=COMPENSATION_TOOL_SCHEMA)
    r.register(entry)
    with pytest.raises(ValueError, match="已注册"):
        r.register(entry)


def test_default_registry_has_compensation():
    r = default_registry()
    assert len(r) == 1
    assert "calculate_compensation" in r
    schemas = r.tool_schemas()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "calculate_compensation"


def test_tool_entry_validates_input():
    """ToolEntry 传入 input_model 时应校验输入。"""
    entry = ToolEntry(
        name="calc",
        func=calculate_compensation,
        schema=COMPENSATION_TOOL_SCHEMA,
        input_model=CompensationInput,
    )
    with pytest.raises(Exception):
        entry.invoke({"years_of_service": -5, "monthly_salary": 5000})
    # 合法输入正常
    r = entry.invoke({"years_of_service": 2.0, "monthly_salary": 6000})
    assert r.total == 12000.0


# ====== ToolCallingAnswerer（mock LLM） ======

def _fake_tool_call_response(tool_name: str, args: dict) -> io.BytesIO:
    """构造 LLM 返回 tool_call 的假响应。"""
    body = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_fake_001",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }],
            },
        }],
    }
    return io.BytesIO(json.dumps(body).encode("utf-8"))


def _fake_text_response(text: str) -> io.BytesIO:
    """构造 LLM 返回文本回答的假响应。"""
    body = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": text,
            },
        }],
    }
    return io.BytesIO(json.dumps(body).encode("utf-8"))


def test_call_chat_with_tools_text_only():
    """LLM 直接返回文本（不调工具）。"""
    with patch("urllib.request.urlopen", return_value=_fake_text_response("直接回答")):
        result = _call_chat_with_tools(
            "http://fake/v1/chat/completions", "sk-x", "m",
            [{"role": "user", "content": "问"}],
            [], default_registry(),
        )
    assert result == "直接回答"


def test_call_chat_with_tools_executes_and_returns():
    """LLM 先返回 tool_call → 执行计算 → 下一次返回文本。"""
    # 第一次调返回 tool_call，第二次返回文本
    call_count = [0]

    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return _fake_tool_call_response("calculate_compensation", {
                "years_of_service": 3.0,
                "monthly_salary": 8000,
                "reason": "termination",
            })
        else:
            return _fake_text_response("根据计算，您应获得经济补偿金24000元。")

    with patch("urllib.request.urlopen", side_effect=side_effect):
        result = _call_chat_with_tools(
            "http://fake/v1/chat/completions", "sk-x", "m",
            [{"role": "user", "content": "我工作3年被辞退，月薪8000，能拿多少补偿？"}],
            default_registry().tool_schemas(),
            default_registry(),
        )
    assert "24000" in result
    assert call_count[0] == 2  # 一轮 tool_call + 一轮文本


def test_call_chat_with_tools_fallback_on_unknown_tool():
    """LLM 调了一个未注册的工具 → 兜底返回错误信息，不崩溃。"""
    call_count = [0]

    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return _fake_tool_call_response("nonexistent_tool", {"x": 1})
        else:
            return _fake_text_response("抱歉，无法计算。")

    with patch("urllib.request.urlopen", side_effect=side_effect):
        result = _call_chat_with_tools(
            "http://fake/v1/chat/completions", "sk-x", "m",
            [{"role": "user", "content": "算一下"}],
            default_registry().tool_schemas(),
            default_registry(),
        )
    assert call_count[0] == 2


def test_tool_calling_answerer_requires_key():
    """空 key 应报错。"""
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        ToolCallingAnswerer(api_key="", tools_registry=default_registry())


def test_tool_calling_answerer_generates():
    """端到端：ToolCallingAnswerer.generate 走完整 tool_call 循环。"""
    call_count = [0]

    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return _fake_tool_call_response("calculate_compensation", {
                "years_of_service": 2.0,
                "monthly_salary": 10000,
            })
        else:
            return _fake_text_response("您的经济补偿金为20000元。")

    with patch("urllib.request.urlopen", side_effect=side_effect):
        ans = ToolCallingAnswerer(api_key="sk-test", tools_registry=default_registry())
        result = ans.generate("被公司辞退，工作2年月薪1万，补偿多少？", [LABOR_CONTRACT_47])
    assert "20000" in result
    assert call_count[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])