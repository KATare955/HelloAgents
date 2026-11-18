"""网页搜索工具（阶段二新增，23 工具设计之信息检索族）

语义边界（与 text_search 的分工，写进描述防混淆）：
    text_search 查「销售知识库」——车型配置/价格/优惠政策等库内结构化数据；
    web_search 查「公开网页」——召回公告/新车资讯/竞品动态/行业政策等库外时效信息。
    知识库查不到的内容才用它，两工具不可相互替代。

八件套 schema（tools设计.md）：
    名称：web_search
    描述：做什么/何时不用（text_search 能查的不查网页）/参数来源/失败行为
    输入：query 必填；category 可选（召回公告/新车资讯/竞品动态/行业政策/all）
    输出：results[{title, snippet, source, publish_date}] + total
    权限：只读；失败码：INVALID_PARAM / 无结果 partial
    沙盒：无写台账；验证器可读事实：摘要片段全文（GT 数字可核对）
    参数来源：query 来自用户问题文本，无工具间 L1 依赖 → 不进 dep_edges

无状态性（Reward 规格书 §7 沙盒规则 3）：run() 只读 mock 数据，跨样本调用无残留。
"""

import json
from pathlib import Path
from typing import Dict, Any, List

from hello_agents.tools import Tool, ToolParameter, ToolResponse

try:
    from ._matching import score_match, has_substantive_overlap
except ImportError:  # 直接运行脚本时
    from _matching import score_match, has_substantive_overlap


class WebSearchTool(Tool):
    """网页搜索工具：对销售知识库未覆盖的内容做补充 web 搜索，返回摘要片段"""

    def __init__(self):
        super().__init__(
            name="web_search",
            description=(
                "对销售知识库未覆盖的内容进行补充网页搜索，返回召回公告、新车资讯、"
                "竞品动态、行业政策等公开网页摘要片段。注意：车型配置、指导价、优惠政策"
                "等知识库内结构化信息请用 text_search 查询，不要用本工具。"
            ),
        )
        self.mock_data = self._load_mock_data()

    def _load_mock_data(self) -> Dict[str, Any]:
        mock_file = Path(__file__).parent / "mock_data" / "web_search.json"
        with open(mock_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description=(
                    "检索关键词，从用户问题中提取（如'召回公告'、'购置税政策'、"
                    "'Model 3 价格'、车型名+召回/资讯等）。"
                ),
                required=True,
            ),
            ToolParameter(
                name="category",
                type="string",
                description=(
                    "分类过滤，限定检索范围。可选值：召回公告、新车资讯、竞品动态、"
                    "行业政策、all"
                ),
                required=False,
                default="all",
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="返回结果数量限制。默认5条，最大10条。",
                required=False,
                default=5,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        query = (parameters.get("query") or "").strip()
        category = parameters.get("category", "all")
        limit = parameters.get("limit", 5)

        if not query:
            return ToolResponse.error(
                code="INVALID_PARAM", message="query参数不能为空"
            )

        # 标题 + 摘要联合打分（资讯类标题不含的实体词常在摘要中，如"Model 3"）
        scored = []
        for cat, data in self.mock_data.items():
            for result in data.get("results", []):
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                s = max(
                    score_match(query, title),
                    score_match(query, snippet),
                )
                scored.append((s, result))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [r for s, r in scored if s >= 2.5]
        if not results:
            # 兜底：实质 token 重叠（与 text_search 问题 8 修复同口径）
            results = [
                r for s, r in scored
                if s > 0
                and has_substantive_overlap(query, r.get("title", "") + " " + r.get("snippet", ""))
            ]

        if category != "all":
            results = [r for r in results if r.get("category") == category]

        results = results[:limit]

        if not results:
            return ToolResponse.partial(
                text=(
                    f"网页搜索未找到与'{query}'相关的结果。"
                    f"可检索分类：召回公告、新车资讯、竞品动态、行业政策"
                ),
                data={
                    "results": [],
                    "total": 0,
                    "query": query,
                    "available_categories": list(self.mock_data.keys()),
                },
            )

        return ToolResponse.success(
            text=f"网页搜索找到{len(results)}条与'{query}'相关的信息",
            data={"results": results, "total": len(results), "query": query},
        )


if __name__ == "__main__":
    tool = WebSearchTool()
    print(tool.run({"query": "E5 Sportback 召回"}).to_json())
    print(tool.run({"query": "Model 3 价格"}).to_json())
    print(tool.run({"query": "购置税"}).to_json())
    print(tool.run({"query": "不存在的新闻"}).to_json())
