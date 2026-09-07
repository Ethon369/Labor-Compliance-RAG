"""经济补偿金计算器：基于《劳动合同法》第47条，注册为 LLM function calling tool。

公式（第47条原文）：
- 每满一年支付一个月工资；六个月以上不满一年按一年算；不满六个月支付半个月
- 月工资高于当地社平工资3倍 → 按3倍封顶，且年限最高不超过12年
- 月工资 = 解除/终止前12个月平均工资
- 违法解除（第87条）：按第47条标准的二倍支付赔偿金

为什么纯函数而非类方法：function calling 的 tool 约定是"函数签名→JSON Schema"，
纯函数天然适合这个映射——入参→计算→返回 Pydantic 结果。无副作用，可单测。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CompensationInput(BaseModel):
    """经济补偿金输入。校验：年限≥0，月工资>0。"""
    years_of_service: float = Field(..., ge=0, description="在本单位连续工作年限（年），支持小数")
    monthly_salary: float = Field(..., gt=0, description="解除前12个月平均工资（元/月）")
    local_avg_salary: float | None = Field(None, gt=0, description="当地上年度职工月平均工资（元/月），不填不设上限")
    reason: str = Field("termination", description="解除原因：termination/expiration/mutual/layoff/illegal/other")


class CompensationResult(BaseModel):
    """经济补偿金计算结果。Pydantic 承载，API/LLM 均可直接序列化。"""
    n_months: float          # 折算后的工作年限月数
    base_salary: float       # 月工资基数（可能被3倍封顶）
    total: float             # 经济补偿金总额（元）
    is_capped: bool          # 是否触发三倍社平封顶
    is_illegal: bool         # 是否违法解除（赔偿金=2x）
    detail: str              # 计算过程摘要（给 LLM 理解用）


# OpenAI 兼容 function calling 的 JSON Schema（tool 注册时传给 LLM）
COMPENSATION_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculate_compensation",
        "description": (
            "根据《劳动合同法》第47条计算劳动争议经济补偿金。"
            "输入工作年限、月平均工资和当地社平工资（可选），"
            "返回应得补偿金额及计算明细。违法解除时按第87条支付二倍赔偿金。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "years_of_service": {
                    "type": "number",
                    "description": "劳动者在本单位连续工作的年限，支持小数（如 3.5 表示3年6个月）",
                },
                "monthly_salary": {
                    "type": "number",
                    "description": "解除或终止劳动合同前12个月的平均工资（元/月）",
                },
                "local_avg_salary": {
                    "type": "number",
                    "description": (
                        "当地上年度职工月平均工资（元/月）。不填则不设三倍封顶上限。"
                        "当劳动者月工资高于此值3倍时，补偿基数按3倍封顶，年限最高12年。"
                    ),
                },
                "reason": {
                    "type": "string",
                    "enum": ["termination", "expiration", "mutual", "layoff", "illegal", "other"],
                    "description": (
                        "解除/终止原因：termination=用人单位单方解除, "
                        "expiration=合同期满终止(用人单位不续签), mutual=协商一致解除, "
                        "layoff=经济性裁员, illegal=用人单位违法解除(赔偿金=2倍补偿金), other=其他"
                    ),
                },
            },
            "required": ["years_of_service", "monthly_salary"],
        },
    },
}


def calculate_compensation(
    years_of_service: float,
    monthly_salary: float,
    local_avg_salary: float | None = None,
    reason: str = "termination",
) -> CompensationResult:
    """经济补偿金/赔偿金计算（《劳动合同法》第47条 + 第87条）。

    参数均来自 LLM 从用户问题中抽取——Schema 约束参数形态，
    校验在 Pydantic 层完成，函数只管纯数学逻辑。
    """
    # 1. 折算工作年限 → N
    full_years = int(years_of_service)
    remainder = years_of_service - full_years
    if remainder >= 0.5:
        n_months = float(full_years + 1)
    elif remainder > 0:
        n_months = full_years + 0.5
    else:
        n_months = float(full_years)

    # 2. 月工资基数（三倍社平封顶判断）
    is_capped = False
    base_salary = monthly_salary
    if local_avg_salary is not None and monthly_salary > local_avg_salary * 3:
        base_salary = round(local_avg_salary * 3, 2)
        is_capped = True
        if n_months > 12:
            n_months = 12.0

    # 3. 总额
    total = round(n_months * base_salary, 2)

    # 4. 违法解除 → 二倍赔偿金（第87条）
    is_illegal = reason == "illegal"
    if is_illegal:
        total = round(total * 2, 2)

    # 5. 计算过程摘要
    parts = [
        f"工作年限{years_of_service}年，折算N={n_months}个月",
        f"月工资基数={base_salary:.2f}元" + ("（触发3倍社平封顶）" if is_capped else ""),
    ]
    if is_illegal:
        parts.append(f"违法解除，按第87条支付二倍赔偿金")
    parts.append(
        f"{'赔偿金' if is_illegal else '经济补偿金'} = "
        f"{n_months} × {base_salary:.2f}" +
        (" × 2" if is_illegal else "") +
        f" = {total:.2f}元"
    )

    return CompensationResult(
        n_months=n_months,
        base_salary=base_salary,
        total=total,
        is_capped=is_capped,
        is_illegal=is_illegal,
        detail="；".join(parts),
    )