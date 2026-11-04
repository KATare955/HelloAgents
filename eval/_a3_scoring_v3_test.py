"""A3 S4 单元测试（临时）：验证 scoring.py v3 精确豁免口径。

验收基准（打样 D4）：rescoreB 的 4 个 accept_clarification 翻案 case 行为不变。
另验证 v3 新行为：双跳题漏第一跳不被澄清豁免掩盖（v2 会误判对）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scoring import judge_tool_correct, JUDGE_VERSION  # noqa: E402

RESCOREB = (
    Path(__file__).resolve().parent
    / "eval_data/results/baseline/qwen3-14b/v1.1"
    / "baseline_qwen3-14b_v1.1_20261101_140513_rescoreB.json"
)

passed = failed = 0
def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✅ {name} {detail}")
    else:
        failed += 1
        print(f"  ❌ {name} {detail}")


print(f"== T0 版本号 ==")
check("JUDGE_VERSION = v3", JUDGE_VERSION == "v3", JUDGE_VERSION)

print("== T1 基础子集：expected ⊆ actual 即正确 ==")
ok, missing, extra = judge_tool_correct(["text_search"], ["text_search", "query_discount_policy"])
check("判正确", ok)
check("missing 空", missing == [], str(missing))

print("== T2 纯反问题（test_drive_002 形态）：expected 仅 target，反问即豁免 ==")
ok, missing, extra = judge_tool_correct(
    ["book_test_drive"], [],
    response="请问您的姓名和联系方式是？我好为您预约试驾",
    accept_clarification={"target_tool": "book_test_drive", "reason": "故意缺姓名/电话"},
)
check("判正确（remaining 空集）", ok, f"missing={missing}")

print("== T3 multi_hop 完整链（multi_hop_005 形态）：缺 target 但剩余链路全命中 ==")
ok, missing, extra = judge_tool_correct(
    ["text_search", "query_discount_policy", "calculate_finance", "book_test_drive"],
    ["text_search", "query_discount_policy", "calculate_finance"],
    response="以上信息已为您汇总，如需预约试驾请提供姓名和电话",
    accept_clarification={"target_tool": "book_test_drive", "reason": "缺必填信息"},
)
check("判正确（剩余链路命中）", ok, f"missing={missing}")

print("== T4 一揽子免单修复（test_drive_010 形态）：漏第一跳不被豁免掩盖 ==")
ok, missing, extra = judge_tool_correct(
    ["query_discount_policy", "calculate_finance", "book_test_drive"],
    ["query_discount_policy"],
    response="优惠已查询，如需预约试驾请提供姓名电话",
    accept_clarification={"target_tool": "book_test_drive", "reason": "题面未提供预约必填信息"},
)
check("判错误（精确豁免不掩盖第一跳缺失）", not ok)
check("missing 报剩余缺失 calculate_finance", missing == ["calculate_finance"], str(missing))

print("== T5 无 accept 普通缺失：target 照报 ==")
ok, missing, extra = judge_tool_correct(["book_test_drive"], [])
check("判错误", not ok)
check("missing 含 book_test_drive", missing == ["book_test_drive"], str(missing))

print("== T6 无回应不豁免：target 未调但 response 为空 ==")
ok, missing, extra = judge_tool_correct(
    ["book_test_drive"], [],
    response="",
    accept_clarification={"target_tool": "book_test_drive", "reason": "故意缺参数"},
)
check("判错误（空回复不豁免）", not ok, f"missing={missing}")

print("== T9 澄清意图验证：幻觉完成不豁免（8B test_drive_002 语料） ==")
ok, missing, extra = judge_tool_correct(
    ["book_test_drive"], [],
    response="您的试驾预约已确认！预订编号：TD20261104881，预约成功通知已发送至您的手机",
    accept_clarification={"target_tool": "book_test_drive", "reason": "故意缺姓名/电话"},
)
check("判错误（非空但无澄清意图）", not ok, f"missing={missing}")
check("missing 含 target（无豁免）", missing == ["book_test_drive"], str(missing))

print("== T10 澄清意图验证：手算回复不豁免 ==")
ok, missing, extra = judge_tool_correct(
    ["book_test_drive"], [],
    response="您的月供约为 8000 元",
    accept_clarification={"target_tool": "book_test_drive", "reason": "故意缺参数"},
)
check("判错误（事实陈述非澄清）", not ok, f"missing={missing}")

print("== T11 澄清意图验证：不可达+替代形态豁免（test_drive_007 语料） ==")
ok, missing, extra = judge_tool_correct(
    ["book_test_drive"], [],
    response="经核实，您咨询的车型不存在。但为您找到以下可预约试驾的车型：E5 Sportback 先锋型 235,900元",
    accept_clarification={"target_tool": "book_test_drive", "reason": "车型不存在（目标不可达）"},
)
check("判正确（不存在+为您找到）", ok, f"missing={missing}")

print("== T12 澄清意图验证：需要您提供形态豁免（test_drive_006 语料） ==")
ok, missing, extra = judge_tool_correct(
    ["book_test_drive"], [],
    response="需要您提供以下信息才能预约试驾：1. 姓名 2. 手机号 3. 试驾时间。请告知这些信息",
    accept_clarification={"target_tool": "book_test_drive", "reason": "题面未提供必填信息"},
)
check("判正确（需要您提供+请告知）", ok, f"missing={missing}")

print("== T7 回归：rescoreB 4 翻案 case 行为不变 ==")
flip_ids = ["test_drive_002", "test_drive_003", "test_drive_007", "multi_hop_005"]
b = json.load(open(RESCOREB, encoding="utf-8"))
questions = {q["id"]: q for q in json.load(
    open(Path(__file__).resolve().parent / "eval_data/questions.json", encoding="utf-8")
)["questions"]}
for r in b["results"]:
    if r["id"] in flip_ids:
        ok, missing, extra = judge_tool_correct(
            r.get("expected_tools", []), r.get("actual_tools", []),
            response=r.get("response", ""),
            accept_clarification=questions[r["id"]].get("accept_clarification"),
        )
        check(f"{r['id']} 保持判对（rescoreB={r.get('tool_correct')}）",
              ok == r.get("tool_correct", False), f"v3={ok} missing={missing}")

print("== T8 全量回归：rescoreB 所有 case 判定不变 ==")
diffs = []
for r in b["results"]:
    ok, missing, extra = judge_tool_correct(
        r.get("expected_tools", []), r.get("actual_tools", []),
        response=r.get("response", ""),
        accept_clarification=questions.get(r["id"], {}).get("accept_clarification"),
    )
    if ok != r.get("tool_correct", False):
        diffs.append((r["id"], r.get("tool_correct"), ok, missing))
check("全量 diff = 0", not diffs, str(diffs[:5]))

print(f"\n=== 结果：{passed} 通过 / {failed} 失败 ===")
sys.exit(1 if failed else 0)
