"""金融方案测算工具"""

import json
from pathlib import Path
from typing import Dict, Any, List
from hello_agents.tools import Tool, ToolParameter, ToolResponse


class CalculateFinanceTool(Tool):
    """金融方案测算工具
    
    【用途】计算贷款月供、总利息、首付金额等金融方案。
    【输入】车价、首付比例、贷款年限。
    【返回】详细的金融方案（月供、总利息、首付金额等）。
    【注意】这是计算工具，不是查询工具。如果需要查询历史成交价，请使用query_transaction_price工具。
    
    使用场景：
    - 用户问"首付30%，5年贷款月供多少？"
    - 用户问"计算贷款方案"
    - 用户问"月供是多少？"
    
    重要提示：
    - 车价参数（car_price）需要用户提供或从其他工具获取
    - 本工具不会自动查询车价，需要明确传入
    - 如果用户没有提供车价，可以先调用text_search或query_transaction_price查询
    """
    
    def __init__(self):
        super().__init__(
            name="calculate_finance",
            description="计算贷款月供、总利息、首付金额等金融方案。需要输入车价、首付比例、贷款年限。注意：这是计算工具，不是查询工具。如需查询历史成交价，请使用query_transaction_price工具。"
        )
        # 加载Mock数据（用于快速返回预设方案）
        self.mock_data = self._load_mock_data()
    
    def _load_mock_data(self) -> Dict[str, Any]:
        """加载Mock数据"""
        mock_file = Path(__file__).parent / "mock_data" / "calculate_finance.json"
        with open(mock_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def get_parameters(self) -> List[ToolParameter]:
        """定义工具参数"""
        return [
            ToolParameter(
                name="car_price",
                type="number",
                description="车价（元）。可以是指导价或优惠后价格。必须大于0。",
                required=True
            ),
            ToolParameter(
                name="down_payment_ratio",
                type="number",
                description="首付比例（0-100）。单位：百分比。常见值：20、30、50。",
                required=True
            ),
            ToolParameter(
                name="loan_years",
                type="integer",
                description="贷款年限（年）。常见值：1、2、3、5。必须大于0。",
                required=True
            ),
            ToolParameter(
                name="interest_rate",
                type="number",
                description="年利率（百分比）。默认4.5%。可从优惠政策获取贴息利率。",
                required=False,
                default=4.5
            )
        ]
    
    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        """执行工具"""
        # 1. 获取参数
        car_price = parameters.get("car_price", 0)
        down_payment_ratio = parameters.get("down_payment_ratio", 0)
        loan_years = parameters.get("loan_years", 0)
        interest_rate = parameters.get("interest_rate", 4.5)
        
        # 2. 参数验证
        if car_price <= 0:
            return ToolResponse.error(
                code="INVALID_PARAM",
                message="car_price必须大于0"
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
        
        # 3. 尝试从Mock数据中获取预设方案
        mock_key = f"{int(car_price)}_{int(down_payment_ratio)}_{loan_years}_{interest_rate}"
        if mock_key in self.mock_data:
            return ToolResponse.success(
                text=f"车价{car_price}元，首付{down_payment_ratio}%，{loan_years}年，月供{self.mock_data[mock_key]['monthly_payment']}元",
                data=self.mock_data[mock_key]
            )
        
        # 4. 实时计算（等额本息）
        try:
            result = self._calculate_finance(car_price, down_payment_ratio, loan_years, interest_rate)
            
            return ToolResponse.success(
                text=f"车价{car_price}元，首付{down_payment_ratio}%，{loan_years}年，月供{result['monthly_payment']}元",
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
    
    # 测试1: 28.99万，首付30%，5年，利率4.5%（Mock数据中有）
    result = tool.run({
        "car_price": 289900,
        "down_payment_ratio": 30,
        "loan_years": 5,
        "interest_rate": 4.5
    })
    print(result.to_json())
    
    # 测试2: 25万，首付20%，3年，利率4.0%（实时计算）
    result = tool.run({
        "car_price": 250000,
        "down_payment_ratio": 20,
        "loan_years": 3,
        "interest_rate": 4.0
    })
    print(result.to_json())
    
    # 测试3: 参数错误
    result = tool.run({
        "car_price": -100000,
        "down_payment_ratio": 30,
        "loan_years": 5
    })
    print(result.to_json())
