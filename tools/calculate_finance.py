"""金融方案测算工具"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from hello_agents.tools import Tool, ToolParameter, ToolResponse

try:
    from ._matching import score_match, match_key
except ImportError:  # 直接运行脚本时
    from _matching import score_match, match_key


class CalculateFinanceTool(Tool):
    """金融方案测算工具

    【用途】计算贷款月供、总利息、首付金额等金融方案。
    【输入】车型、首付比例、贷款年限。
    【返回】详细的金融方案（月供、总利息、首付金额等）。

    默认方案（决策点 A 裁定 A1，2026-11-06）：
    - 首付比例/贷款年限为可选参数，缺省时按默认方案计算（首付30%、贷款3年、
      年利率4.5%），返回文本标注"按默认方案计算，可调整"——把确定性留给工具，
      模型无需猜测或追问；
    - 问题 18 同口径：模型显式传默认值（30/3/4.5）与不传等价，属遵守工具契约。

    自动读价（D4 批次，CAPA 问题 10）：
    - 车价由工具自动读取对应车型的指导价，模型无需（也不应）传入 car_price；
    - 返回文本标注"指导价 X 元，另有优惠 Y 未计入"——减价决策权保留在
      Agent 回复层（AI 报价可审计性）；
    - car_model 必填：只传车系（如"E5 Sportback"）会返回多配置清单，
      模型应向用户确认具体配置后重试。

    使用场景：
    - 用户问"首付30%，5年贷款月供多少？"
    - 用户问"计算贷款方案"
    - 用户问"月供是多少？"
    """

    def __init__(self):
        super().__init__(
            name="calculate_finance",
            description="计算贷款月供、总利息、首付金额等金融方案。需要输入车型；首付比例、贷款年限为可选参数（缺省按默认方案：首付30%、3年期、年利率4.5%，返回中标注可调整）；车价由工具自动读取车型指导价，无需传入。注意：这是计算工具，不是查询工具。如需查询历史成交价，请使用query_transaction_price工具。"
        )
        # 加载Mock数据（用于快速返回预设方案）
        self.mock_data = self._load_mock_data()
        # 车型指导价库（text_search mock 复用，保证价格口径一致）
        self._price_data = self._load_price_data()
        self._price_entries = [
            {"title": r.get("title", ""), "price": (r.get("content") or {}).get("price")}
            for data in self._price_data.values()
            for r in data.get("results", [])
        ]
        # 优惠库（返回文本标注"另有优惠 Y 未计入"）
        self._discount_data = self._load_discount_data()

    def _load_mock_data(self) -> Dict[str, Any]:
        """加载Mock数据"""
        mock_file = Path(__file__).parent / "mock_data" / "calculate_finance.json"
        with open(mock_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_price_data(self) -> Dict[str, Any]:
        """加载车型指导价数据（与 text_search 同源）"""
        price_file = Path(__file__).parent / "mock_data" / "text_search.json"
        with open(price_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_discount_data(self) -> Dict[str, Any]:
        """加载优惠数据（与 query_discount_policy 同源）"""
        discount_file = Path(__file__).parent / "mock_data" / "query_discount_policy.json"
        with open(discount_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_parameters(self) -> List[ToolParameter]:
        """定义工具参数"""
        return [
            ToolParameter(
                name="car_model",
                type="string",
                description="车型名称（如'E5 Sportback 2025款先锋型'、'帕萨特2024款280TSI商务版'）。工具自动读取该车型指导价计算。若题面只给车系（如'E5 Sportback'），请向用户确认具体配置。",
                required=True
            ),
            ToolParameter(
                name="down_payment_ratio",
                type="number",
                description="首付比例（0-100）。单位：百分比。可选参数：缺省时按默认方案 30% 计算（返回中标注可调整）。",
                required=False,
                default=30
            ),
            ToolParameter(
                name="loan_years",
                type="integer",
                description="贷款年限（年）。可选参数：缺省时按默认方案 3 年期计算（返回中标注可调整）。必须大于0。",
                required=False,
                default=3
            ),
            ToolParameter(
                name="interest_rate",
                type="number",
                description="年利率（百分比）。默认4.5%。可从优惠政策获取贴息利率。",
                required=False,
                default=4.5
            ),
            ToolParameter(
                name="car_price",
                type="number",
                description="【已废弃】车价由工具自动读取车型指导价，请勿传入。",
                required=False
            )
        ]

    def _resolve_guide_price(self, car_model: str) -> Tuple[Optional[str], Optional[float], Optional[ToolResponse]]:
        """按 car_model 解析具体车型配置与指导价。

        匹配策略（复用 _matching.score_match）：对全部配置条目（trim title）打分，
        - 恰好 1 条强匹配（>=2.5，精确/双向子串）→ 返回该配置；
        - 多条强匹配（车系级名称，如"E5 Sportback"）→ AMBIGUOUS_MODEL partial，
          列出各配置及指导价，引导模型明确配置（向用户确认）；
        - 0 条强匹配 → NOT_FOUND error，列出可用车系。

        Returns:
            (trim_title, guide_price, None) 成功；(None, None, ToolResponse) 失败。
        """
        candidates = [
            r for r in self._price_entries
            if score_match(car_model, r.get("title", "")) >= 2.5
        ]
        if len(candidates) == 1:
            return candidates[0]["title"], candidates[0]["price"], None
        if len(candidates) > 1:
            listing = "；".join(f"{r['title']} {r['price']} 元" for r in candidates)
            return None, None, ToolResponse.partial(
                text=f"车型'{car_model}'包含多个配置，请明确具体配置后重试：{listing}",
                data={
                    "results": [{"title": r["title"], "price": r["price"]} for r in candidates],
                    "total": len(candidates),
                }
            )
        available = "、".join(self._price_data.keys())
        return None, None, ToolResponse.error(
            code="NOT_FOUND",
            message=f"未找到车型'{car_model}'的指导价。知识库可用车系：{available}"
        )

    def _lookup_discount(self, trim_title: str) -> Optional[float]:
        """按车型配置反查车系优惠总额（优先全国键；缺失时回退首个区域键）。"""
        key = match_key(trim_title, self._discount_data.keys())
        if not key:
            return None
        regions = self._discount_data[key]
        region_key = "全国" if "全国" in regions else next(iter(regions))
        return (regions[region_key] or {}).get("total_discount")

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        """执行工具"""
        # 1. 获取参数（决策点 A：首付/年限可选，缺省用默认方案 30%/3年）
        car_model = (parameters.get("car_model") or "").strip()
        down_payment_ratio = parameters.get("down_payment_ratio", 30)
        loan_years = parameters.get("loan_years", 3)
        interest_rate = parameters.get("interest_rate", 4.5)
        # 默认方案标记：模型未显式传首付/年限 → 返回文本标注"按默认方案计算，可调整"
        used_default_plan = "down_payment_ratio" not in parameters or "loan_years" not in parameters

        # 2. 参数验证
        if not car_model:
            return ToolResponse.error(
                code="INVALID_PARAM",
                message="car_model必填：请提供具体车型配置（如'E5 Sportback 2025款先锋型'）"
            )

        if parameters.get("car_price") is not None:
            return ToolResponse.error(
                code="INVALID_PARAM",
                message="car_price不接受显式传入：本工具自动读取车型指导价，请移除car_price后重试"
            )

        if down_payment_ratio < 0 or down_payment_ratio > 100:
            return ToolResponse.error(
                code="INVALID_PARAM",
                message="down_payment_ratio必须在0-100之间"
            )

        if loan_years <= 0:
            return ToolResponse.error(
                code="INVALID_PARAM",
                message="loan_years必须大于0"
            )

        if interest_rate < 0:
            return ToolResponse.error(
                code="INVALID_PARAM",
                message="interest_rate不能为负数"
            )

        # 3. 自动读取车型指导价（查价职责下沉：杜绝模型编造车价）
        trim_title, car_price, error = self._resolve_guide_price(car_model)
        if error is not None:
            return error

        # 4. 反查优惠总额（仅标注，不参与计算——减价决策权在 Agent 回复层）
        discount = self._lookup_discount(trim_title)

        # 5. 尝试从Mock数据中获取预设方案（以自动读取的指导价为键）
        mock_key = f"{int(car_price)}_{int(down_payment_ratio)}_{loan_years}_{interest_rate}"
        if mock_key in self.mock_data:
            result = dict(self.mock_data[mock_key])
            result.update({
                "car_model": trim_title,
                "guide_price": car_price,
                "discount_not_included": discount,
            })
            note = f"，另有优惠 {discount} 元未计入" if discount else ""
            plan_note = "；按默认方案计算（首付30%、3年期），可调整" if used_default_plan else ""
            return ToolResponse.success(
                text=(
                    f"车型 {trim_title} 指导价 {car_price} 元{note}；"
                    f"首付{down_payment_ratio}%，贷款{loan_years}年（年利率{interest_rate}%），"
                    f"月供 {result['monthly_payment']} 元{plan_note}"
                ),
                data=result
            )

        # 6. 实时计算（等额本息）
        try:
            result = self._calculate_finance(car_price, down_payment_ratio, loan_years, interest_rate)
            result.update({
                "car_model": trim_title,
                "guide_price": car_price,
                "discount_not_included": discount,
            })

            note = f"，另有优惠 {discount} 元未计入" if discount else ""
            plan_note = "；按默认方案计算（首付30%、3年期），可调整" if used_default_plan else ""
            return ToolResponse.success(
                text=(
                    f"车型 {trim_title} 指导价 {car_price} 元{note}；"
                    f"首付{down_payment_ratio}%，贷款{loan_years}年（年利率{interest_rate}%），"
                    f"月供 {result['monthly_payment']} 元{plan_note}"
                ),
                data=result
            )
        except Exception as e:
            return ToolResponse.error(
                code="CALCULATION_ERROR",
                message=f"计算失败: {str(e)}"
            )

    def _calculate_finance(
        self,
        car_price: float,
        down_payment_ratio: float,
        loan_years: int,
        interest_rate: float
    ) -> Dict[str, Any]:
        """计算金融方案（等额本息）

        Args:
            car_price: 车价（元）
            down_payment_ratio: 首付比例（百分比）
            loan_years: 贷款年限（年）
            interest_rate: 年利率（百分比）

        Returns:
            金融方案详情
        """
        # 计算首付和贷款金额
        down_payment = car_price * (down_payment_ratio / 100)
        loan_amount = car_price - down_payment

        # 计算月利率和总期数
        monthly_rate = interest_rate / 100 / 12
        total_months = loan_years * 12

        # 计算月供（等额本息公式）
        if monthly_rate == 0:
            # 零利率情况
            monthly_payment = loan_amount / total_months
        else:
            # 等额本息公式：月供 = [贷款本金 × 月利率 × (1+月利率)^还款月数] ÷ [(1+月利率)^还款月数 - 1]
            monthly_payment = loan_amount * monthly_rate * (1 + monthly_rate) ** total_months / \
                            ((1 + monthly_rate) ** total_months - 1)

        # 计算总利息和总费用
        total_interest = monthly_payment * total_months - loan_amount
        total_cost = car_price + total_interest

        # 生成还款计划（前5个月）
        payment_schedule = []
        remaining_principal = loan_amount

        for month in range(1, min(6, total_months + 1)):
            interest = remaining_principal * monthly_rate
            principal = monthly_payment - interest
            remaining_principal -= principal

            payment_schedule.append({
                "month": month,
                "principal": round(principal, 2),
                "interest": round(interest, 2),
                "total": round(monthly_payment, 2)
            })

        return {
            "car_price": round(car_price, 2),
            "down_payment": round(down_payment, 2),
            "loan_amount": round(loan_amount, 2),
            "loan_years": loan_years,
            "interest_rate": interest_rate,
            "monthly_payment": round(monthly_payment, 2),
            "total_interest": round(total_interest, 2),
            "total_cost": round(total_cost, 2),
            "payment_schedule": payment_schedule
        }


# 测试代码
if __name__ == "__main__":
    tool = CalculateFinanceTool()

    # 测试1: 具体配置自动读价（先锋型 235900，全国优惠 30000 未计入）
    result = tool.run({
        "car_model": "E5 Sportback 2025款先锋型",
        "down_payment_ratio": 30,
        "loan_years": 5,
    })
    print(result.to_json())

    # 测试2: 车系级名称 → 多配置清单（引导确认）
    result = tool.run({
        "car_model": "E5 Sportback",
        "down_payment_ratio": 30,
        "loan_years": 5,
    })
    print(result.to_json())

    # 测试3: 不存在的车型 → NOT_FOUND
    result = tool.run({
        "car_model": "星曜X7 2025款",
        "down_payment_ratio": 30,
        "loan_years": 5,
    })
    print(result.to_json())

    # 测试4: 显式传 car_price → 拒绝
    result = tool.run({
        "car_model": "E5 Sportback 2025款先锋型",
        "car_price": 200000,
        "down_payment_ratio": 30,
        "loan_years": 5,
    })
    print(result.to_json())
