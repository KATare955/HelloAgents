"""跟进任务查询工具（阶段二新增，23 工具设计之客户关系管理族）

CRM 场景：查询销售顾问名下指定客户的未完成跟进任务列表。

八件套 schema（tools设计.md）：
    名称：follow_up_query
    描述：做什么/参数来源/失败行为
    输入：customer_name 必填（客户姓名）
    输出：customer{name, phone, level} + tasks[{task_type, deadline, priority,
          status, last_followup_summary}] + total
    权限：只读；失败码：INVALID_PARAM / NOT_FOUND
    沙盒：无写台账；验证器可读事实：任务列表全文（GT 可核对任务数/截止日/优先级）
    参数来源：customer_name 来自用户问题文本，无工具间 L1 依赖 → 不进 dep_edges

隐私设计：未知名客户只报 NOT_FOUND，不返回全部客户清单（CRM 数据不可枚举）。

无状态性（Reward 规格书 §7 沙盒规则 3）：run() 只读 mock 数据，跨样本调用无残留。
"""

import json
from pathlib import Path
from typing import Dict, Any, List

from hello_agents.tools import Tool, ToolParameter, ToolResponse

try:
    from ._matching import score_match
except ImportError:  # 直接运行脚本时
    from _matching import score_match


class FollowUpQueryTool(Tool):
    """跟进任务查询工具：查询指定客户的未完成跟进任务列表"""

    def __init__(self):
        super().__init__(
            name="follow_up_query",
            description=(
                "查询 CRM 中指定客户的未完成跟进任务列表，返回任务类型、截止日期、"
                "优先级和上次跟进摘要。需要客户姓名作为查询条件。"
            ),
        )
        self.mock_data = self._load_mock_data()

    def _load_mock_data(self) -> Dict[str, Any]:
        mock_file = Path(__file__).parent / "mock_data" / "follow_up_query.json"
        with open(mock_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="customer_name",
                type="string",
                description=(
                    "客户姓名（如'王建国'）。从用户问题中提取；若用户未提供姓名，"
                    "请先向用户询问，不要编造。"
                ),
                required=True,
            )
        ]

    def _match_customer(self, customer_name: str):
        """按姓名匹配客户（精确/子串 ≥2.5 强匹配；未命中返回 None）。"""
        candidates = [
            key for key in self.mock_data
            if score_match(customer_name, key) >= 2.5
        ]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        customer_name = (parameters.get("customer_name") or "").strip()

        if not customer_name:
            return ToolResponse.error(
                code="INVALID_PARAM", message="customer_name参数不能为空"
            )

        key = self._match_customer(customer_name)
        if key is None:
            return ToolResponse.error(
                code="NOT_FOUND",
                message=(
                    f"未找到客户'{customer_name}'的跟进记录，"
                    f"请与用户确认客户姓名是否准确。"
                ),
            )

        customer = dict(self.mock_data[key])
        tasks = list(customer.get("tasks") or [])

        if not tasks:
            return ToolResponse.success(
                text=f"客户{customer_name}当前没有未完成的跟进任务",
                data={"customer": customer, "tasks": [], "total": 0},
            )

        return ToolResponse.success(
            text=(
                f"客户{customer_name}有{len(tasks)}个未完成的跟进任务："
                + "；".join(
                    f"{t['task_type']}({t['deadline']}, {t['priority']}优先级)"
                    for t in tasks
                )
            ),
            data={"customer": customer, "tasks": tasks, "total": len(tasks)},
        )


if __name__ == "__main__":
    tool = FollowUpQueryTool()
    print(tool.run({"customer_name": "王建国"}).to_json())
    print(tool.run({"customer_name": "周伟"}).to_json())
    print(tool.run({"customer_name": "不存在的人"}).to_json())
    print(tool.run({"customer_name": ""}).to_json())
