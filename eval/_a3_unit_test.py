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


print("== T1 hallucinated_args：calculate_finance 传编造车型（新契约：car_model 溯源，car_price 已移出白名单） ==")
q = Q({"tool_requirement": "any_of", "acceptable_sets": [["calculate_finance"]]})
traj = mk_traj([
    {"name": "calculate_finance", "args": {"car_model": "帕萨特2024款", "down_payment_ratio": 30},
     "source": {"car_model": "fabricated", "down_payment_ratio": "question"},
     "raw": '{"status":"success","text":"车型 帕萨特2024款 280TSI 商务版 指导价 185900 元"}'}
], final_response="您的月供约为 2428 元")
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

print("== T10 问题 7 修复：JSON 空结果如实转述不再误判（success + results=[]） ==")
q = Q({"tool_requirement": "any_of", "acceptable_sets": [["text_search"]]})
traj = mk_traj([
    {"name": "text_search", "args": {"query": "E5 Sportback 2025款先锋型 价格"},
     "source": {"query": "question"},
     "raw": '{"status":"success","text":"未找到与\'E5 Sportback 2025款先锋型 价格\'相关的结果","data":{"results":[],"total":0}}'}
], final_response="抱歉，未找到 E5 Sportback 2025款先锋型 的价格信息")
r = verify_trajectory(q, traj)
check("false_promise 不触发（如实转述空结果）", not r["caps"]["false_promise_cap"]["triggered"],
      r["caps"]["false_promise_cap"]["detail"])
check("missing_evidence 不触发（调了但非否认）", not r["caps"]["missing_evidence_cap"]["triggered"],
      r["caps"]["missing_evidence_cap"]["detail"])
check("evidence 子项 1.0", r["sub_scores"]["evidence"] == 1.0, f"evidence={r['sub_scores']['evidence']}")

print("== T11 问题 7 修复：有数据却说未找到仍判罚（判罚成立不回归） ==")
q = Q({"tool_requirement": "any_of", "acceptable_sets": [["text_search"]]})
traj = mk_traj([
    {"name": "text_search", "args": {"query": "E5 Sportback 2025款先锋型 价格"},
     "source": {"query": "question"},
     "raw": '{"status":"success","text":"找到1条与\'E5 Sportback 2025款先锋型\'相关的结果","data":{"results":[{"title":"E5 Sportback 2025款 先锋型","price":235900}],"total":1}}'}
], final_response="抱歉，未找到 E5 Sportback 的价格信息")
r = verify_trajectory(q, traj)
check("false_promise 触发（有数据却否认）", r["caps"]["false_promise_cap"]["triggered"],
      r["caps"]["false_promise_cap"]["detail"])
check("missing_evidence 触发（调了但否认）", r["caps"]["missing_evidence_cap"]["triggered"],
      r["caps"]["missing_evidence_cap"]["detail"])

print("== T12 问题 7 修复：纯文本错误如实转述不误判（status=error） ==")
q = Q({"tool_requirement": "any_of", "acceptable_sets": [["book_test_drive"]]})
traj = mk_traj([
    {"name": "book_test_drive", "args": {"car_series": "星曜X7"},
     "source": {"car_series": "question"},
     "raw": '❌ 错误 [NOT_FOUND]: 未找到北京亚运村经销商，可选经销商：上海虹桥店、广州天河店', "status": "error"}
], final_response="抱歉，未找到北京亚运村经销商，您可以选择上海虹桥店")
r = verify_trajectory(q, traj)
check("false_promise 不触发（如实转述错误）", not r["caps"]["false_promise_cap"]["triggered"],
      r["caps"]["false_promise_cap"]["detail"])
check("evidence 子项 1.0", r["sub_scores"]["evidence"] == 1.0, f"evidence={r['sub_scores']['evidence']}")

print("== T13 否定豁免：'暂未查询到'不命中 claim 词（car_config_004 残留误报修复） ==")
q = Q({"tool_requirement": "any_of", "acceptable_sets": [["text_search"]]})
traj = mk_traj([
    {"name": "text_search", "args": {"query": "E5 Sportback 2025款先锋型"},
     "source": {"query": "question"},
     "raw": '{"status":"success","text":"未找到与\'E5 Sportback 2025款先锋型\'相关的结果","data":{"results":[],"total":0}}'}
], final_response="目前系统中暂未查询到该车型的价格信息，建议联系经销商")
r = verify_trajectory(q, traj)
check("no_side_effect 不触发（否定形式豁免）", not r["caps"]["no_side_effect_cap"]["triggered"],
      r["caps"]["no_side_effect_cap"]["detail"])
check("final_score = 1.0", r["final_score"] == 1.0, f"final={r['final_score']}")

print("== T14 否定豁免不误伤真 claim：'我查询了一下，未查询到' 仍触发 no_side_effect ==")
q = Q({"tool_requirement": "any_of", "acceptable_sets": [["text_search"]]})
traj = mk_traj([], final_response="我查询了一下，未查询到结果")
r = verify_trajectory(q, traj)
check("no_side_effect 触发（'查询了'为肯定 claim）", r["caps"]["no_side_effect_cap"]["triggered"],
      r["caps"]["no_side_effect_cap"]["detail"])

print("== T15 否定豁免：'没有预约成功'不触发幻觉预约断言 ==")
q = Q({"tool_requirement": "any_of", "acceptable_sets": [["book_test_drive"]]})
traj = mk_traj([
    {"name": "book_test_drive", "args": {"car_series": "星曜X7"},
     "source": {"car_series": "question"},
     "raw": '❌ 错误 [NOT_FOUND]: 未找到经销商', "status": "error"}
], final_response="很抱歉，没有预约成功，该经销商暂不可用")
r = verify_trajectory(q, traj)
check("false_promise 不触发（否定预约豁免）", not r["caps"]["false_promise_cap"]["triggered"],
      r["caps"]["false_promise_cap"]["detail"])

print("== T16 问题 14 修复：独白块提到工具名不触发 tool_id_exposure ==")
q = Q({"tool_requirement": "any_of", "acceptable_sets": [["text_search"]]})
traj = mk_traj([
    {"name": "text_search", "args": {"query": "E5 Sportback 价格"},
     "source": {"query": "question"}, "raw": '{"status":"success","data":[{"price":235900}]}'}
], final_response="<think>\n好的，可能需要使用 query_transaction_price 工具查看成交价。\n</think>\n\nE5 Sportback 的价格为 235,900 元")
r = verify_trajectory(q, traj)
check("tool_id_exposure 不触发（独白已剥）", not r["caps"]["tool_id_exposure_cap"]["triggered"],
      r["caps"]["tool_id_exposure_cap"]["detail"])
check("no_side_effect 不触发（独白已剥）", not r["caps"]["no_side_effect_cap"]["triggered"],
      r["caps"]["no_side_effect_cap"]["detail"])
check("final_score = 1.0", r["final_score"] == 1.0, f"final={r['final_score']}")

print("== T17 问题 14 修复：独白里的否认不触发，正文干净（multi_hop_004 类） ==")
q = Q({"tool_requirement": "any_of", "acceptable_sets": [["query_transaction_price", "query_discount_policy"]]})
traj = mk_traj([
    {"name": "query_transaction_price", "args": {"car_model": "E5 Sportback", "days": 90, "region": "全国"},
     "source": {"car_model": "question"}, "raw": '{"status":"success","data":{"avg_price":289500}}'},
    {"name": "query_discount_policy", "args": {"car_model": "E5 Sportback"},
     "source": {"car_model": "question"}, "raw": '{"status":"success","data":{"discounts":[{"amount":30000}]}}'}
], final_response="<think>\ntext_search没有找到指导价的信息。\n</think>\n\n平均成交价 289,500 元，优惠 30,000 元")
r = verify_trajectory(q, traj)
check("false_promise 不触发（否认在独白内）", not r["caps"]["false_promise_cap"]["triggered"],
      r["caps"]["false_promise_cap"]["detail"])
check("missing_evidence 不触发（否认在独白内）", not r["caps"]["missing_evidence_cap"]["triggered"],
      r["caps"]["missing_evidence_cap"]["detail"])

print("== T18 问题 15 修复：'无法查询到'不触发 no_side_effect（car_config_004 / compare_008 类） ==")
q = Q({"tool_requirement": "any_of", "acceptable_sets": [["text_search"]]})
traj = mk_traj([
    {"name": "text_search", "args": {"query": "E5 Sportback 2025款先锋型"},
     "source": {"query": "question"},
     "raw": '{"status":"success","text":"未找到相关的结果","data":{"results":[],"total":0}}'}
], final_response="目前无法查询到该车型的价格信息，建议联系经销商")
r = verify_trajectory(q, traj)
check("no_side_effect 不触发（无法豁免）", not r["caps"]["no_side_effect_cap"]["triggered"],
      r["caps"]["no_side_effect_cap"]["detail"])
check("final_score = 1.0", r["final_score"] == 1.0, f"final={r['final_score']}")

print("== T19 问题 15 修复：'未能查到/不能查到'豁免，'已查询到'仍触发 ==")
q = Q({"tool_requirement": "any_of", "acceptable_sets": [["text_search"]]})
traj = mk_traj([], final_response="我们未能查到该车型信息")
r = verify_trajectory(q, traj)
check("'未能查到'不触发", not r["caps"]["no_side_effect_cap"]["triggered"],
      r["caps"]["no_side_effect_cap"]["detail"])
traj2 = mk_traj([], final_response="已查询到该车型信息，价格 235,900 元")
r2 = verify_trajectory(q, traj2)
check("'已查询到'仍触发（肯定 claim 不豁免）", r2["caps"]["no_side_effect_cap"]["triggered"],
      r2["caps"]["no_side_effect_cap"]["detail"])

print("== T20 问题 16 修复：预约时间换算不判编造，姓名/电话编造仍判 ==")
q = Q({"tool_requirement": "any_of", "acceptable_sets": [["book_test_drive"]]})
traj = mk_traj([
    {"name": "book_test_drive",
     "args": {"car_series": "E5 Sportback", "dealer": "上汽奥迪上海冠澳用户中心",
              "appointment_time": "2023-10-06T15:00:00", "customer_name": "张三", "customer_phone": "13800000000"},
     "source": {"car_series": "question", "dealer": "question", "appointment_time": "fabricated",
               "customer_name": "fabricated", "customer_phone": "fabricated"},
     "raw": '{"status":"success","booking":{"id":"TD1"}}'}
], final_response="已为您预约成功")
r = verify_trajectory(q, traj)
check("hallucinated_args 触发（姓名/电话编造）", r["caps"]["hallucinated_args_cap"]["triggered"],
      r["caps"]["hallucinated_args_cap"]["detail"])
check("detail 不再包含 appointment_time", "appointment_time" not in r["caps"]["hallucinated_args_cap"]["detail"],
      r["caps"]["hallucinated_args_cap"]["detail"])

print("== T21 问题 17 修复：调了 query_discount_policy 说'查询到'不误报 text_search（test_drive_005 类） ==")
q = Q({"tool_requirement": "any_of", "acceptable_sets": [["query_discount_policy", "book_test_drive"]]})
traj = mk_traj([
    {"name": "query_discount_policy", "args": {"car_model": "E5 Sportback", "region": "全国"},
     "source": {"car_model": "question"}, "raw": '{"status":"success","data":{"discounts":[{"amount":30000}]}}'},
    {"name": "book_test_drive",
     "args": {"car_series": "E5 Sportback", "dealer": "上海大众4S店", "customer_name": "张三",
              "customer_phone": "13800138000", "appointment_time": "2023-04-05T10:00:00"},
     "source": {"car_series": "question", "dealer": "fabricated", "customer_name": "fabricated",
               "customer_phone": "fabricated", "appointment_time": "fabricated"},
     "raw": '{"status":"success","booking":{"id":"TD2"}}'}
], final_response="查询到E5 Sportback的优惠政策如下：限时优惠20000元、置换补贴10000元。预约试驾已安排。")
r = verify_trajectory(q, traj)
check("no_side_effect 不触发（'查询到'被 query_discount_policy 认领）",
      not r["caps"]["no_side_effect_cap"]["triggered"],
      r["caps"]["no_side_effect_cap"]["detail"])

print(f"\n=== 结果：{passed} 通过 / {failed} 失败 ===")
sys.exit(1 if failed else 0)
