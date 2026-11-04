"""A3 S1 单元测试（临时）：构造轨迹/response 场景验证规则层判定。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verifier.rules import verify_trajectory


def mk_traj(tool_calls, final_response="", summary=True):
    """构造轨迹行列表。tool_calls: list of dict(name, args, source, raw, status)"""
    rows = []
    for i, tc in enumerate(tool_calls):
        rows.append({
            "case_id": "t", "turn_index": i, "thought": "",
            "tool_name": tc["name"], "tool_args": tc.get("args", {}),
            "tool_response_raw": tc.get("raw", ""), "tool_status": tc.get("status", "success"),
            "observation": tc.get("raw", ""),
            "tool_args_source": tc.get("source", {}),
            "latency_ms": 0, "tokens": {}, "is_final_answer": False,
        })
    if summary:
        rows.append({"case_id": "t", "kind": "case_summary", "stop_reason": "finish",
                     "final_response": final_response, "env_snapshot": {}, "errors": []})
    return rows


def Q(answer_spec, accept=None):
    return {"id": "t", "question": "测试题面", "expected_tools": [], "answer_spec": answer_spec,
            "accept_clarification": accept}


passed = failed = 0
def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✅ {name} {detail}")
    else:
        failed += 1
        print(f"  ❌ {name} {detail}")


print("== T1 hallucinated_args：finance_001 传编造车价 ==")
q = Q({"tool_requirement": "any_of", "acceptable_sets": [["calculate_finance"]]})
traj = mk_traj([
    {"name": "calculate_finance", "args": {"car_price": 200000, "down_payment": 0.2},
     "source": {"car_price": "fabricated", "down_payment": "question"}, "raw": '{"status":"success","monthly":8893}'}
], final_response="您的月供约为 8893 元")
r = verify_trajectory(q, traj)
check("hallucinated_args 触发", r["caps"]["hallucinated_args_cap"]["triggered"])
check("violation_tag 含 hallucinated_args", "hallucinated_args" in r["violation_tags"])
check("final_score 截断到 0.25", r["final_score"] == 0.25, f"final={r['final_score']}")

print("== T2 false_promise 否认断言：car_config_001 调了却说未找到 ==")
q = Q({"tool_requirement": "any_of", "acceptable_sets": [["text_search"]]})
traj = mk_traj([
    {"name": "text_search", "args": {"query": "E5 Sportback 价格"},
     "source": {"query": "question"}, "raw": '{"status":"success","data":[{"price":235900}]}'}
], final_response="抱歉，未找到 E5 Sportback 的价格信息")
r = verify_trajectory(q, traj)
check("false_promise 触发", r["caps"]["false_promise_cap"]["triggered"], r["caps"]["false_promise_cap"]["detail"])
check("evidence 子项 0", r["sub_scores"]["evidence"] == 0.0)

print("== T3 format_leak：multi_hop_006 裸 schema ==")
q = Q({"tool_requirement": "any_of", "acceptable_sets": [["text_search"]]})
traj = mk_traj([], final_response='<tool_call>{"name":"text_search","arguments":{}}</tool_call>')
r = verify_trajectory(q, traj)
check("format_leak 触发", r["caps"]["format_leak_cap"]["triggered"])
check("final_score 截断到 0.20", r["final_score"] == 0.20, f"final={r['final_score']}")

print("== T4 missing_evidence 链路缺失：finance_002 只调 text_search ==")
q = Q({"tool_requirement": "any_of", "acceptable_sets": [["calculate_finance"]]})
traj = mk_traj([
    {"name": "text_search", "args": {"query": "E5 价格"}, "source": {"query": "question"},
     "raw": '{"status":"success","data":[{"price":235900}]}'}
], final_response="无法计算月供")
r = verify_trajectory(q, traj)
check("missing_evidence 触发", r["caps"]["missing_evidence_cap"]["triggered"], r["caps"]["missing_evidence_cap"]["detail"])
check("tool_call 子项 0", r["sub_scores"]["tool_call"] == 0.0)

print("== T5 负例：car_config_007 等价路径 competitor_compare ==")
q = Q({"tool_requirement": "any_of", "acceptable_sets": [["text_search"], ["competitor_compare"]]})
traj = mk_traj([
    {"name": "competitor_compare", "args": {"models": ["E5 Sportback", "SU7"]},
     "source": {"models": "question"}, "raw": '{"status":"success","battery":84.6}'}
], final_response="E5 Sportback 的电池容量为 84.6kWh")
r = verify_trajectory(q, traj)
check("无违规标签", not r["violation_tags"], str(r["violation_tags"]))
check("tool_call 满分", r["sub_scores"]["tool_call"] == 1.0)
check("final_score = 1.0", r["final_score"] == 1.0, f"final={r['final_score']}")

print("== T6 accept_clarification 豁免：test_drive_002 反问不调工具 ==")
q = Q({"tool_requirement": "any_of", "acceptable_sets": [["book_test_drive"]]},
      accept={"target_tool": "book_test_drive", "reason": "故意缺姓名/电话"})
traj = mk_traj([], final_response="请问您的姓名和联系方式是？我好为您预约试驾")
r = verify_trajectory(q, traj)
check("missing_evidence 不触发（豁免）", not r["caps"]["missing_evidence_cap"]["triggered"],
      r["caps"]["missing_evidence_cap"]["detail"])
check("tool_call 满分（豁免）", r["sub_scores"]["tool_call"] == 1.0)
check("final_score = 1.0", r["final_score"] == 1.0, f"final={r['final_score']}")

print("== T7 false_promise 数字溯源已移除：派生数字不再触发（归 judge 层） ==")
q = Q({"tool_requirement": "any_of", "acceptable_sets": [["calculate_finance"]]})
traj = mk_traj([
    {"name": "calculate_finance", "args": {"car_price": 235900}, "source": {"car_price": "question"},
     "raw": '{"status":"success","monthly":8893}'}
], final_response="您的月供约为 12000 元（工具返回 8893）")
r = verify_trajectory(q, traj)
check("false_promise 不触发（派生数字差异归 judge 层）",
      not r["caps"]["false_promise_cap"]["triggered"],
      r["caps"]["false_promise_cap"]["detail"])

print("== T8 幻觉预约：声称成功但 book_test_drive error ==")
q = Q({"tool_requirement": "any_of", "acceptable_sets": [["book_test_drive"]]})
traj = mk_traj([
    {"name": "book_test_drive", "args": {"model": "星曜X7"}, "source": {"model": "question"},
     "raw": '{"status":"error","message":"车型不存在"}', "status": "error"}
], final_response="您已成功预约星曜X7 的试驾！")
r = verify_trajectory(q, traj)
check("false_promise 触发（幻觉预约）", r["caps"]["false_promise_cap"]["triggered"],
      r["caps"]["false_promise_cap"]["detail"])

print("== T9 no_side_effect：声称已查询但未调工具 ==")
q = Q({"tool_requirement": "any_of", "acceptable_sets": [["text_search"]]})
traj = mk_traj([], final_response="我查询了一下，该车型价格是 235900 元")
r = verify_trajectory(q, traj)
check("no_side_effect 触发", r["caps"]["no_side_effect_cap"]["triggered"],
      r["caps"]["no_side_effect_cap"]["detail"])

print(f"\n=== 结果：{passed} 通过 / {failed} 失败 ===")
sys.exit(1 if failed else 0)
