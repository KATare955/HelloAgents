"""优惠政策查询工具"""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from hello_agents.tools import Tool, ToolParameter, ToolResponse


class QueryDiscountPolicyTool(Tool):
    """优惠政策查询工具
    
    查询当前可用的优惠政策，包括厂商补贴、置换补贴、金融贴息、限时活动等。
    返回优惠政策详情和总优惠金额。
    """
    
    def __init__(self):
        super().__init__(
            name="query_discount_policy",
            description="查询当前可用的优惠政策，包括厂商补贴、置换补贴、金融贴息、限时活动等"
        )
        # 加载Mock数据
        self.mock_data = self._load_mock_data()
    
    def _load_mock_data(self) -> Dict[str, Any]:
        """加载Mock数据"""
        mock_file = Path(__file__).parent / "mock_data" / "query_discount_policy.json"
        with open(mock_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def get_parameters(self) -> List[ToolParameter]:
        """定义工具参数"""
        return [
            ToolParameter(
                name="car_model",
                type="string",
                description="车型名称。需要完整准确，如'E5 Sportback'、'E5 Sportback 2025款'。",
                required=True
            ),
            ToolParameter(
                name="region",
                type="string",
                description="地区，用于查询地区限定优惠政策。默认查询全国通用政策。",
                required=False,
                default="全国"
            ),
            ToolParameter(
                name="include_expired",
                type="boolean",
                description="是否包含已过期的政策。默认false（只查询有效政策）。",
                required=False,
                default=False
            )
        ]
    
    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        """执行工具"""
        # 1. 获取参数
        car_model = parameters.get("car_model", "")
        region = parameters.get("region", "全国")
        include_expired = parameters.get("include_expired", False)
        
        # 2. 参数验证
        if not car_model:
            return ToolResponse.error(
                code="INVALID_PARAM",
                message="car_model参数不能为空"
            )
        
        # 3. 查找匹配的优惠政策
        # 先尝试精确匹配，再尝试模糊匹配
        discount_data = None
        
        # 精确匹配
        if car_model in self.mock_data:
            car_data = self.mock_data[car_model]
            if region in car_data:
                discount_data = car_data[region]
            elif "全国" in car_data:
                discount_data = car_data["全国"]
        else:
            # 模糊匹配
            for key in self.mock_data.keys():
                if car_model.lower() in key.lower() or key.lower() in car_model.lower():
                    car_data = self.mock_data[key]
                    if region in car_data:
                        discount_data = car_data[region]
                    elif "全国" in car_data:
                        discount_data = car_data["全国"]
                    break
        
        # 4. 处理未找到的情况
        if not discount_data:
            return ToolResponse.success(
                text=f"未找到{car_model}在{region}的优惠政策",
                data={
                    "car_model": car_model,
                    "region": region,
                    "discounts": [],
                    "total_discount": 0,
                    "query_time": datetime.now().isoformat() + "Z"
                }
            )
        
        # 5. 过滤过期政策（如果需要）
        discounts = discount_data.get("discounts", [])
        if not include_expired:
            current_date = datetime.now().strftime("%Y-%m-%d")
            discounts = [
                d for d in discounts 
                if d.get("valid_until", "9999-12-31") >= current_date
            ]
        
        # 6. 计算总优惠
        total_discount = sum(d.get("amount", 0) for d in discounts)
        
        # 7. 返回结果
        return ToolResponse.success(
            text=f"找到{len(discounts)}条{car_model}在{region}的优惠政策，总优惠{total_discount}元",
            data={
                "car_model": car_model,
                "region": region,
                "discounts": discounts,
                "total_discount": total_discount,
                "query_time": datetime.now().isoformat() + "Z"
            }
        )


# 测试代码
if __name__ == "__main__":
    tool = QueryDiscountPolicyTool()
    
    # 测试1: 查询E5 Sportback在上海的优惠
    result = tool.run({"car_model": "E5 Sportback", "region": "上海"})
    print(result.to_json())
    
    # 测试2: 查询帕萨特的优惠
    result = tool.run({"car_model": "帕萨特2024款", "region": "上海"})
    print(result.to_json())
    
    # 测试3: 查询不存在的车型
    result = tool.run({"car_model": "不存在的车型", "region": "上海"})
    print(result.to_json())
