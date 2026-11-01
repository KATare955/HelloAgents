"""历史成交价查询工具"""

import json
from pathlib import Path
from typing import Dict, Any, List
from hello_agents.tools import Tool, ToolParameter, ToolResponse

try:
    from ._matching import match_key
except ImportError:  # 直接运行脚本时
    from _matching import match_key


class QueryTransactionPriceTool(Tool):
    """历史成交价查询工具
    
    【用途】查询车型近期市场实际成交价格，用于判断当前优惠幅度是否合理。
    【返回】价格趋势、历史成交价、购买建议。
    【注意】这是查询工具，不是计算工具。如果需要计算贷款月供，请使用calculate_finance工具。
    
    使用场景：
    - 用户问"这款车最近成交价多少？"
    - 用户问"现在买划算吗？"
    - 用户问"历史价格走势如何？"
    """
    
    def __init__(self):
        super().__init__(
            name="query_transaction_price",
            description="查询车型近期市场实际成交价格（历史成交价），返回价格趋势和购买建议。注意：这是查询工具，不是计算工具。如需计算贷款月供，请使用calculate_finance工具。"
        )
        # 加载Mock数据
        self.mock_data = self._load_mock_data()
    
    def _load_mock_data(self) -> Dict[str, Any]:
        """加载Mock数据"""
        mock_file = Path(__file__).parent / "mock_data" / "query_transaction_price.json"
        with open(mock_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def get_parameters(self) -> List[ToolParameter]:
        """定义工具参数"""
        return [
            ToolParameter(
                name="car_model",
                type="string",
                description="车型名称。需要完整准确，如'E5 Sportback 2025款 旗舰quattro型'。",
                required=True
            ),
            ToolParameter(
                name="days",
                type="integer",
                description="查询天数范围。默认90天（近3个月）。可选值：30、90、180、365。",
                required=False,
                default=90
            ),
            ToolParameter(
                name="region",
                type="string",
                description="地区，用于查询地区成交价格。默认全国范围。",
                required=False,
                default="全国"
            )
        ]
    
    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        """执行工具"""
        # 1. 获取参数
        car_model = parameters.get("car_model", "")
        days = parameters.get("days", 90)
        region = parameters.get("region", "全国")
        
        # 2. 参数验证
        if not car_model:
            return ToolResponse.error(
                code="INVALID_PARAM",
                message="car_model参数不能为空"
            )
        
        if days not in [30, 90, 180, 365]:
            return ToolResponse.error(
                code="INVALID_PARAM",
                message="days参数必须是30、90、180或365"
            )
        
        # 3. 查找匹配的历史成交价数据（健壮匹配：精确/双向子串/token重叠）
        price_data = None
        days_key = str(days)
        
        matched_key = match_key(car_model, self.mock_data.keys())
        if matched_key:
            car_data = self.mock_data[matched_key]
            if days_key in car_data:
                price_data = car_data[days_key]
        
        # 4. 处理未找到的情况
        if not price_data:
            return ToolResponse.success(
                text=f"未找到{car_model}近{days}天的历史成交价数据",
                data={
                    "car_model": car_model,
                    "period": f"近{days}天",
                    "avg_price": 0,
                    "min_price": 0,
                    "max_price": 0,
                    "median_price": 0,
                    "trend": "无数据",
                    "sample_count": 0,
                    "recommendation": "暂无历史成交数据，建议关注官方指导价",
                    "price_history": []
                }
            )
        
        # 5. 返回结果
        return ToolResponse.success(
            text=f"{car_model}近{days}天平均成交价{price_data['avg_price']}元，趋势{price_data['trend']}",
            data=price_data
        )


# 测试代码
if __name__ == "__main__":
    tool = QueryTransactionPriceTool()
    
    # 测试1: 查询E5 Sportback旗舰quattro型近90天成交价
    result = tool.run({"car_model": "E5 Sportback 2025款 旗舰quattro型", "days": 90})
    print(result.to_json())
    
    # 测试2: 查询帕萨特近90天成交价
    result = tool.run({"car_model": "帕萨特2024款 330TSI 精英版", "days": 90})
    print(result.to_json())
    
    # 测试3: 查询不存在的车型
    result = tool.run({"car_model": "不存在的车型", "days": 90})
    print(result.to_json())
