"""竞品对比工具"""

import json
from pathlib import Path
from typing import Dict, Any, List
from hello_agents.tools import Tool, ToolParameter, ToolResponse

try:
    from ._matching import match_key
except ImportError:  # 直接运行脚本时
    from _matching import match_key


class CompetitorCompareTool(Tool):
    """竞品对比工具
    
    同级别车型参数/配置横向对比，返回对比结果和推荐建议。
    支持多维度对比：价格、配置、空间、动力、电池续航等。
    """
    
    def __init__(self):
        super().__init__(
            name="competitor_compare",
            description="同级别车型参数/配置横向对比，返回对比结果和推荐建议"
        )
        # 加载Mock数据
        self.mock_data = self._load_mock_data()
    
    def _load_mock_data(self) -> Dict[str, Any]:
        """加载Mock数据"""
        mock_file = Path(__file__).parent / "mock_data" / "competitor_compare.json"
        with open(mock_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def get_parameters(self) -> List[ToolParameter]:
        """定义工具参数"""
        return [
            ToolParameter(
                name="car_model_a",
                type="string",
                description="本车车型名称。通常是用户咨询的主车型。",
                required=True
            ),
            ToolParameter(
                name="car_model_b",
                type="string",
                description="竞品车型名称。用户想要对比的车型。",
                required=True
            ),
            ToolParameter(
                name="compare_aspects",
                type="array",
                description="对比维度列表。可选值：价格、配置、空间、动力、电池容量、充电速度、使用成本。默认对比所有可用维度。",
                required=False,
                default=["价格", "配置", "空间", "动力", "电池容量", "充电速度"]
            )
        ]
    
    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        """执行工具"""
        # 1. 获取参数
        car_model_a = parameters.get("car_model_a", "")
        car_model_b = parameters.get("car_model_b", "")
        compare_aspects = parameters.get("compare_aspects", ["价格", "配置", "空间", "动力", "电池容量", "充电速度"])
        
        # 2. 参数验证
        if not car_model_a:
            return ToolResponse.error(
                code="INVALID_PARAM",
                message="car_model_a参数不能为空"
            )
        
        if not car_model_b:
            return ToolResponse.error(
                code="INVALID_PARAM",
                message="car_model_b参数不能为空"
            )
        
        # 3. 查找匹配的对比数据（健壮匹配，两级：车系A -> 竞品B）
        comparison_data = None
        
        # 正向：先匹配一级 key(A)，再匹配二级 key(B)
        key_a = match_key(car_model_a, self.mock_data.keys())
        if key_a:
            key_b = match_key(car_model_b, self.mock_data[key_a].keys())
            if key_b:
                comparison_data = self.mock_data[key_a][key_b]
        
        # 反向：先匹配 B，再在 B 的竞品中匹配 A，最后反转结果
        if not comparison_data:
            key_b = match_key(car_model_b, self.mock_data.keys())
            if key_b:
                key_a = match_key(car_model_a, self.mock_data[key_b].keys())
                if key_a:
                    reverse_data = self.mock_data[key_b][key_a]
                    comparison_data = self._reverse_comparison(reverse_data, car_model_a, car_model_b)
        
        # 4. 处理未找到的情况
        if not comparison_data:
            return ToolResponse.success(
                text=f"未找到{car_model_a}与{car_model_b}的对比数据",
                data={
                    "car_model_a": car_model_a,
                    "car_model_b": car_model_b,
                    "comparison": {},
                    "summary": {
                        "car_a_advantages": 0,
                        "car_b_advantages": 0,
                        "equal_count": 0,
                        "overall_winner": "无数据"
                    },
                    "recommendation": "暂无对比数据，建议分别查询两个车型的详细信息"
                }
            )
        
        # 5. 根据compare_aspects过滤对比维度
        filtered_comparison = {}
        for aspect in compare_aspects:
            if aspect in comparison_data.get("comparison", {}):
                filtered_comparison[aspect] = comparison_data["comparison"][aspect]
        
        # 6. 重新计算summary
        car_a_advantages = sum(1 for v in filtered_comparison.values() if v.get("advantage") == "A")
        car_b_advantages = sum(1 for v in filtered_comparison.values() if v.get("advantage") == "B")
        equal_count = sum(1 for v in filtered_comparison.values() if v.get("advantage") == "相当")
        
        if car_a_advantages > car_b_advantages:
            overall_winner = car_model_a
        elif car_b_advantages > car_a_advantages:
            overall_winner = car_model_b
        else:
            overall_winner = "相当"
        
        # 7. 返回结果
        result_data = {
            "car_model_a": car_model_a,
            "car_model_b": car_model_b,
            "comparison": filtered_comparison,
            "summary": {
                "car_a_advantages": car_a_advantages,
                "car_b_advantages": car_b_advantages,
                "equal_count": equal_count,
                "overall_winner": overall_winner
            },
            "recommendation": comparison_data.get("recommendation", "")
        }
        
        return ToolResponse.success(
            text=f"{car_model_a} vs {car_model_b}：{overall_winner}更胜一筹（{car_a_advantages}:{car_b_advantages}）",
            data=result_data
        )
    
    def _reverse_comparison(self, data: Dict[str, Any], car_model_a: str, car_model_b: str) -> Dict[str, Any]:
        """反转对比结果（A->B 变成 B->A）"""
        reversed_comparison = {}
        
        for aspect, values in data.get("comparison", {}).items():
            # 交换A和B的值
            reversed_comparison[aspect] = {
                "car_a_value": values.get("car_b_value"),
                "car_b_value": values.get("car_a_value"),
                "advantage": self._reverse_advantage(values.get("advantage", "")),
                "note": values.get("note", "")
            }
        
        # 反转summary
        summary = data.get("summary", {})
        reversed_summary = {
            "car_a_advantages": summary.get("car_b_advantages", 0),
            "car_b_advantages": summary.get("car_a_advantages", 0),
            "equal_count": summary.get("equal_count", 0),
            "overall_winner": car_model_a if summary.get("overall_winner") == data.get("car_model_b") else car_model_b
        }
        
        return {
            "car_model_a": car_model_a,
            "car_model_b": car_model_b,
            "comparison": reversed_comparison,
            "summary": reversed_summary,
            "recommendation": data.get("recommendation", "")
        }
    
    def _reverse_advantage(self, advantage: str) -> str:
        """反转优势方"""
        if advantage == "A":
            return "B"
        elif advantage == "B":
            return "A"
        return advantage


# 测试代码
if __name__ == "__main__":
    tool = CompetitorCompareTool()
    
    # 测试1: E5 Sportback vs 小米SU7
    result = tool.run({
        "car_model_a": "E5 Sportback 2025款 旗舰quattro型",
        "car_model_b": "小米SU7 2025款 四驱Max版"
    })
    print(result.to_json())
    
    # 测试2: 小米SU7 vs E5 Sportback（反向对比）
    result = tool.run({
        "car_model_a": "小米SU7 2025款 四驱Max版",
        "car_model_b": "E5 Sportback 2025款 旗舰quattro型"
    })
    print(result.to_json())
    
    # 测试3: 不存在的车型对比
    result = tool.run({
        "car_model_a": "不存在的车型A",
        "car_model_b": "不存在的车型B"
    })
    print(result.to_json())
