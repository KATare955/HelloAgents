"""阶段二新增工具测试（web_search / follow_up_query）

覆盖：基本行为（命中/未命中/空任务/参数校验）+ 无状态性（跨样本调用无残留，
Reward 规格书 §7 沙盒规则 3）+ 与既有 6 工具注册无冲突。
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hello_agents import ToolRegistry
from tools import (
    TextSearchTool,
    QueryDiscountPolicyTool,
    QueryTransactionPriceTool,
    CompetitorCompareTool,
    CalculateFinanceTool,
    BookTestDriveTool,
)
from tools.web_search import WebSearchTool
from tools.follow_up_query import FollowUpQueryTool


def _check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    return cond


def main():
    print("=" * 72)
    print("阶段二新增工具测试（web_search / follow_up_query）")
    print("=" * 72)
    all_pass = True

    # ---- web_search ----
    ws = WebSearchTool()

    r = ws.run({"query": "E5 Sportback 召回"}).to_dict()
    hit = r["data"].get("total", 0) >= 1
    all_pass &= _check("web_search 召回查询命中", hit, f"total={r['data'].get('total')}")

    r = ws.run({"query": "Model 3 价格"}).to_dict()
    hit = r["data"].get("total", 0) >= 1
    all_pass &= _check("web_search 摘要词命中（Model 3）", hit)

    r = ws.run({"query": "新能源购置税"}).to_dict()
    hit = r["data"].get("total", 0) >= 1
    all_pass &= _check("web_search 政策查询命中", hit)

    r = ws.run({"query": "量子纠缠"}).to_dict()
    miss = r["data"].get("total", 0) == 0
    all_pass &= _check("web_search 无结果返回空", miss)

    r1 = ws.run({"query": "召回"})
    r2 = ws.run({"query": "召回"})
    same = r1.to_json() == r2.to_json()
    all_pass &= _check("web_search 无状态（同参两次一致）", same)

    r = ws.run({"query": ""}).to_dict()
    err = r["status"] == "error"
    all_pass &= _check("web_search 空 query 报 INVALID_PARAM", err)

    # ---- follow_up_query ----
    fu = FollowUpQueryTool()

    r = fu.run({"customer_name": "王建国"}).to_dict()
    hit = r["data"].get("total", 0) == 2
    all_pass &= _check("跟进查询 王建国 2 个任务", hit, f"total={r['data'].get('total')}")

    r = fu.run({"customer_name": "周伟"}).to_dict()
    empty = r["data"].get("total", 0) == 0
    all_pass &= _check("跟进查询 周伟 无未完成任务返回空", empty)

    r = fu.run({"customer_name": "不存在的人"}).to_dict()
    err = r["status"] == "error"
    all_pass &= _check("跟进查询 未知客户报 NOT_FOUND", err)

    r = fu.run({"customer_name": ""}).to_dict()
    err = r["status"] == "error"
    all_pass &= _check("跟进查询 空姓名报 INVALID_PARAM", err)

    r1 = fu.run({"customer_name": "孙丽"})
    r2 = fu.run({"customer_name": "孙丽"})
    same = r1.to_json() == r2.to_json()
    all_pass &= _check("跟进查询 无状态（同参两次一致）", same)

    # ---- 与既有 6 工具注册无冲突 ----
    registry = ToolRegistry()
    for cls in (TextSearchTool, QueryDiscountPolicyTool, QueryTransactionPriceTool,
                CompetitorCompareTool, CalculateFinanceTool, BookTestDriveTool,
                WebSearchTool, FollowUpQueryTool):
        registry.register_tool(cls())
    names = registry.list_tools()
    no_conflict = len(names) == len(set(names)) == 8
    all_pass &= _check("8 工具注册无重名", no_conflict, f"names={names}")

    print("=" * 72)
    print(f"总体：{'ALL PASS' if all_pass else 'HAS FAILURE'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
