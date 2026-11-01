"""销售知识库检索工具"""

import json
from pathlib import Path
from typing import Dict, Any, List
from hello_agents.tools import Tool, ToolParameter, ToolResponse

try:
    from ._matching import is_match
except ImportError:  # 直接运行脚本时
    from _matching import is_match


class TextSearchTool(Tool):
    """销售知识库检索工具
    
    在销售知识库中按关键词检索，返回车型配置、价格、政策等结构化信息。
    """
    
    def __init__(self):
        super().__init__(
            name="text_search",
            description="在销售知识库中按关键词检索，返回车型配置、价格、政策等结构化信息"
        )
        # 加载Mock数据
        self.mock_data = self._load_mock_data()
    
    def _load_mock_data(self) -> Dict[str, Any]:
        """加载Mock数据"""
        mock_file = Path(__file__).parent / "mock_data" / "text_search.json"
        with open(mock_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def get_parameters(self) -> List[ToolParameter]:
        """定义工具参数"""
        return [
            ToolParameter(
                name="query",    # 参数1: query (string, required) - 检索关键词
                type="string",
                description="检索关键词。从用户问题中提取，可以是车型名称、配置项、价格范围等。",
                required=True
            ),
            ToolParameter(
                name="category",  # 参数2: category (string, optional, default="all") - 分类过滤
                type="string",
                description="分类过滤，限定检索范围。可选值：车型配置、价格、政策、all",
                required=False,
                default="all"
            ),
            ToolParameter(
                name="limit",    # 参数3: limit (integer, optional, default=10) - 返回数量限制
                type="integer",
                description="返回结果数量限制。默认10条，最大50条。",
                required=False,
                default=10
            )
        ]
    
    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        """执行工具"""
        # 1. 获取参数
        query = parameters.get("query", "")
        category = parameters.get("category", "all")
        limit = parameters.get("limit", 10)
        
        # 2. 参数验证
        if not query:
            return ToolResponse.error(
                code="INVALID_PARAM",
                message="query参数不能为空"
            )
        
        # 3. 查找匹配的结果（健壮匹配：精确/双向子串/token重叠，兼容带配置后缀或意图词的查询）
        results = []
        for key, data in self.mock_data.items():
            if is_match(query, key):
                results.extend(data.get("results", []))
        
        # 4. 根据category过滤
        if category != "all":
            results = [r for r in results if r.get("category") == category]
        
        # 5. 限制返回数量
        results = results[:limit]
        
        # 6. 返回结果
        if not results:
            return ToolResponse.success(
                text=f"未找到与'{query}'相关的结果",
                data={"results": [], "total": 0, "query": query}
            )
        
        return ToolResponse.success(
            text=f"找到{len(results)}条与'{query}'相关的结果",
            data={
                "results": results,
                "total": len(results),
                "query": query
            }
        )


# 测试代码（可选）
if __name__ == "__main__":
    tool = TextSearchTool()
    
    # 测试1: 查询E5 Sportback
    result = tool.run({"query": "E5 Sportback 2025款"})
    print(result.to_json())
    
    # # 测试2: 查询帕萨特
    # result = tool.run({"query": "帕萨特2024款", "category": "车型配置"})
    # print(result.to_json())

    # 测试3: 不存在的车型返回空结果
    result = tool.run({"query": "A7L"})
    print(result.to_json())