"""GRPO 训练循环假 rollout 单测（无 torch / 无 GPU / 无真实 LLM）

覆盖：
  1. compute_advantages：组内均值方差归一 + 单侧 clip + 零方差归零
  2. sequence_loss：手算值校验 + assistant mask 对齐
  3. tokenize_with_mask：diff 法逐消息定界（假 tokenizer 验证 mask 正确性）
  4. FailInjectTool / build_rl_registry：前 N 次注入失败 + 实例级计数隔离
  5. rollout_from_events：事件流重建 steps/messages；连续失败 K=3 截断；
     失败后恢复不截断；Finish 收尾
  6. compute_reward 离线可复算：同轨迹两次判分逐字段一致
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.book_test_drive import BookTestDriveTool
from tools.text_search import TextSearchTool
from tools.calculate_finance import CalculateFinanceTool

from rl.grpo_train import (
    build_rl_registry,
    compute_advantages,
    rollout_from_events,
    sequence_loss,
    tokenize_with_mask,
    FailInjectTool,
)
from rl.reward import compute_reward
from rl.pass_at_k_filter import select_frontier
from rl.eval_rl_subset import summarize_subset


def _check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    return cond


# ---------------------------------------------------------------------------
# 假 tokenizer：逐消息定长渲染（前缀闭包），不依赖任何真实词表
# ---------------------------------------------------------------------------

class FakeTokenizer:
    """apply_chat_template(msgs, tokenize=True, tools=...)

    渲染规则：system→[100]；user→[200, 201]；assistant→[300] + 每个 tool_call
    追加 [301, 302]；tool→[400, 401]。前缀闭包成立（逐消息拼接）。
    """

    def apply_chat_template(self, msgs, tokenize=True, **kwargs):
        ids = []
        for m in msgs:
            role = m.get("role")
            if role == "system":
                ids += [100]
            elif role == "user":
                ids += [200, 201]
            elif role == "assistant":
                ids += [300]
                for _ in m.get("tool_calls") or []:
                    ids += [301, 302]
            elif role == "tool":
                ids += [400, 401]
        return ids if tokenize else ""


# ---------------------------------------------------------------------------
# 假事件流构造
# ---------------------------------------------------------------------------

def _ev(kind, **kw):
    return {"kind": kind, **kw}


def _success_raw(tool, args):
    r = tool.run(args)
    return r.to_json()


def test_compute_advantages():
    ok = True
    # 组 [0.8,0,0.8,0]：mean=0.4 std=0.4 → 正侧 1.0 被 clip 到 1.0，负侧 -1.0 不 clip
    advs = compute_advantages([0.8, 0.0, 0.8, 0.0], group_size=4)
    ok &= _check("advantage 正侧 clip / 负侧保留", advs == [1.0, -1.0, 1.0, -1.0], f"{advs}")
    # 零方差 → 全零
    advs = compute_advantages([0.5, 0.5], group_size=2)
    ok &= _check("零方差组 advantage 全零", advs == [0.0, 0.0], f"{advs}")
    # 两组拼接
    advs = compute_advantages([1.0, 0.0, 0.0, 0.0], group_size=2)
    ok &= _check("多组拼接顺序", advs == [1.0, -1.0, 0.0, 0.0], f"{advs}")
    return ok


def test_sequence_loss():
    ok = True
    mask = [0, 1, 1, 0]           # 位置 1、2 生效
    lp_t = [2.0, 3.0, 4.0]
    lp_r = [1.0, 1.0, 1.0]
    adv, beta = 1.0, 0.04
    loss, n, kl = sequence_loss(lp_t, lp_r, adv, beta, mask)
    expect = (-adv * 2.0 - beta * (2.0 - 1.0)) + (-adv * 3.0 - beta * (3.0 - 1.0))
    ok &= _check("sequence_loss 数值", abs(loss - expect) < 1e-9, f"{loss} vs {expect}")
    ok &= _check("生效 token 计数", n == 2, f"{n}")
    ok &= _check("KL 均值", abs(kl - ((2.0 - 1.0) + (3.0 - 1.0)) / 2) < 1e-9, f"{kl}")
    # 全 mask=0 → 零损失零 token
    loss, n, kl = sequence_loss(lp_t, lp_r, adv, beta, [0, 0, 0, 0])
    ok &= _check("全 mask=0 零贡献", loss == 0.0 and n == 0)
    return ok


def test_tokenize_with_mask():
    ok = True
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "text_search", "arguments": "{}"}},
            {"id": "b", "type": "function", "function": {"name": "Finish", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "a", "content": "obs1"},
        {"role": "tool", "tool_call_id": "b", "content": "obs2"},
        {"role": "assistant", "content": "最终回复"},
    ]
    tok = FakeTokenizer()
    ids, mask = tokenize_with_mask(tok, msgs)
    expect_ids = [100, 200, 201, 300, 301, 302, 301, 302, 400, 401, 400, 401, 300]
    expect_mask = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1]
    ok &= _check("ids 拼接", ids == expect_ids, f"{ids}")
    ok &= _check("assistant-token mask（含工具调用 JSON）", mask == expect_mask, f"{mask}")
    ok &= _check("mask 与 ids 等长", len(ids) == len(mask))
    return ok


def test_fail_inject():
    ok = True
    inner = BookTestDriveTool()
    args = {
        "car_series": "E5 Sportback",
        "dealer": "上汽奥迪上海郑隆奥翟",
        "appointment_time": "2026-11-22T15:00:00",
        "customer_name": "张伟",
        "customer_phone": "13012340000",
    }
    wrapped = FailInjectTool(inner, fail_times=1, error_code="TIMEOUT")
    r1 = wrapped.run(args).to_dict()
    ok &= _check("第 1 次调用注入 TIMEOUT",
                 r1.get("status") == "error"
                 and r1.get("error", {}).get("code") == "TIMEOUT",
                 f"code={r1.get('error', {}).get('code')}")
    r2 = wrapped.run(args).to_dict()
    ok &= _check("第 2 次调用恢复正常（委托 inner）", r2.get("status") == "success")

    # 实例级计数隔离：新包装器不受上一个影响
    wrapped2 = FailInjectTool(BookTestDriveTool(), fail_times=1, error_code="TIMEOUT")
    r3 = wrapped2.run(args).to_dict()
    ok &= _check("per-sample 计数隔离（新实例仍注入）", r3.get("status") == "error")

    # fail_times=2：前两次都失败，第三次成功
    wrapped3 = FailInjectTool(BookTestDriveTool(), fail_times=2, error_code="NOT_FOUND")
    ok &= _check("fail_times=2 前两次注入",
                 wrapped3.run(args).to_dict()["status"] == "error"
                 and wrapped3.run(args).to_dict()["status"] == "error"
                 and wrapped3.run(args).to_dict()["status"] == "success")
    return ok


def test_build_rl_registry():
    ok = True
    reg = build_rl_registry()
    names = sorted(t.name for t in reg.get_all_tools())
    expect = sorted([
        "text_search", "query_discount_policy", "query_transaction_price",
        "competitor_compare", "calculate_finance", "book_test_drive",
        "web_search", "follow_up_query",
    ])
    ok &= _check("8 工具注册", names == expect, f"{names}")

    reg2 = build_rl_registry({"tool": "book_test_drive", "fail_times": 1,
                              "error_code": "TIMEOUT"})
    bk = reg2.get_tool("book_test_drive")
    ok &= _check("fail_inject 包装目标工具", isinstance(bk, FailInjectTool)
                 and bk.name == "book_test_drive")

    reg3 = build_rl_registry({"tool": "web_search", "fail_times": 2,
                              "error_code": "TIMEOUT"})
    ws = reg3.get_tool("web_search")
    ok &= _check("web_search 亦可包装", isinstance(ws, FailInjectTool))
    return ok


def _scripted_ok_flow(question, tools_map):
    """构造一条含 text_search→calculate_finance→Finish 的合法事件流。"""
    ts = tools_map["text_search"]
    cf = tools_map["calculate_finance"]
    ts_raw = _success_raw(ts, {"query": "E5 Sportback 配置 价格"})
    cf_raw = _success_raw(cf, {
        "car_model": "E5 Sportback 2025款 旗舰quattro型",
        "down_payment_ratio": 30, "loan_years": 5,
    })
    answer = "顶配是旗舰quattro型319900元，首付30%贷5年月供约4174.73元。"
    return [
        _ev("model", content="", usage={}),
        _ev("tool_start", id="c1", name="text_search",
            arguments=json.dumps({"query": "E5 Sportback 配置 价格"})),
        _ev("tool_finish", id="c1", result=ts_raw),
        _ev("model", content="", usage={}),
        _ev("tool_start", id="c2", name="calculate_finance",
            arguments=json.dumps({
                "car_model": "E5 Sportback 2025款 旗舰quattro型",
                "down_payment_ratio": 30, "loan_years": 5,
            })),
        _ev("tool_finish", id="c2", result=cf_raw),
        _ev("model", content="", usage={}),
        _ev("tool_start", id="c3", name="Finish",
            arguments=json.dumps({"answer": answer})),
        _ev("tool_finish", id="c3", result="{}"),
    ]


def test_rollout_from_events_ok():
    ok = True
    question = "帮我查下E5 Sportback的配置和价格，算下顶配首付30%贷5年的月供"
    system = "系统提示"
    tools_map = {"text_search": TextSearchTool(),
                 "calculate_finance": CalculateFinanceTool()}
    out = rollout_from_events(_scripted_ok_flow(question, tools_map), question, system)
    steps, resp, msgs, meta = out["steps"], out["response"], out["messages"], out["meta"]
    ok &= _check("步骤数=3（ts/cf/Finish）", [s["tool_name"] for s in steps]
                 == ["text_search", "calculate_finance", "Finish"])
    ok &= _check("response 取自 Finish", "4174" in resp, resp[:40])
    ok &= _check("messages 结构：3 assistant + 2 tool + system/user",
                 [m["role"] for m in msgs] ==
                 ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"])
    ok &= _check("tool_calls 挂接正确",
                 [tc["function"]["name"] for tc in msgs[2]["tool_calls"]] == ["text_search"])
    ok &= _check("meta 计数", meta["n_calls"] == 2 and meta["n_failed"] == 0
                 and not meta["truncated"])
    return ok


def test_rollout_truncation():
    ok = True
    question = "帮我约试驾"
    system = "S"
    fail_raw = json.dumps({"status": "error", "code": "TIMEOUT",
                           "message": "模拟故障注入"})
    events = []
    for i in range(4):  # 4 次失败调用：第 3 次连续失败即截断
        events.append(_ev("model", content="", usage={}))
        events.append(_ev("tool_start", id=f"f{i}", name="web_search",
                          arguments=json.dumps({"query": "召回"})))
        events.append(_ev("tool_finish", id=f"f{i}", result=fail_raw))
    events.append(_ev("model", content="对不起，服务不可用", usage={}))  # 截断后的事件不应被消费
    out = rollout_from_events(events, question, system)
    ok &= _check("连续 3 次失败截断（第 4 次未消费）",
                 len(out["steps"]) == 3 and out["meta"]["truncated"]
                 and out["meta"]["n_failed"] == 3, f"{out['meta']}")

    # 失败后恢复：fail, fail, success → 不截断
    ok_raw = _success_raw(WebSearchToolShim(), {"query": "召回"})
    events = []
    for i in range(2):
        events += [_ev("model", content="", usage={}),
                   _ev("tool_start", id=f"g{i}", name="web_search",
                       arguments=json.dumps({"query": "召回"})),
                   _ev("tool_finish", id=f"g{i}", result=fail_raw)]
    events += [_ev("model", content="", usage={}),
               _ev("tool_start", id="g2", name="web_search",
                   arguments=json.dumps({"query": "召回"})),
               _ev("tool_finish", id="g2", result=ok_raw)]
    out = rollout_from_events(events, question, system)
    ok &= _check("失败后恢复不截断",
                 len(out["steps"]) == 3 and not out["meta"]["truncated"]
                 and out["meta"]["n_failed"] == 2, f"{out['meta']}")
    return ok


class WebSearchToolShim:
    """截断测试用：直接产出一条成功 JSON（避免依赖 web_search 内部实现）。"""

    def run(self, args):
        class R:
            def to_json(self):
                return json.dumps({"status": "success",
                                   "data": {"total": 1, "items": []}})
        return R()


def test_reward_recomputable():
    ok = True
    question = ("帮我查下E5 Sportback的配置和价格，算下顶配首付30%贷5年的月供，"
                "合适的话帮我约11月22日下午3点上汽奥迪上海郑隆奥翟的试驾，"
                "我叫张伟，电话13012340000")
    label = {
        "question": question,
        "expected_tools": ["text_search", "calculate_finance", "book_test_drive"],
        "gt_facts": {"指导价": 319900, "月供": 4174.73, "试驾预约": "预约成功"},
        "accept_clarification": False, "fail_inject": None, "dep_edges": [],
        "difficulty": "multi_hop_3",
    }
    # 用真实工具输出拼一条轨迹：text_search 查到顶配 → 金融 → 预约
    ts = TextSearchTool()
    cf = CalculateFinanceTool()
    bk = BookTestDriveTool()
    events = [
        _ev("model", content="", usage={}),
        _ev("tool_start", id="c1", name="text_search",
            arguments=json.dumps({"query": "E5 Sportback 配置"})),
        _ev("tool_finish", id="c1", result=_success_raw(ts, {"query": "E5 Sportback 配置"})),
        _ev("model", content="", usage={}),
        _ev("tool_start", id="c2", name="calculate_finance",
            arguments=json.dumps({"car_model": "E5 Sportback 2025款 旗舰quattro型",
                                  "down_payment_ratio": 30, "loan_years": 5})),
        _ev("tool_finish", id="c2", result=_success_raw(cf, {
            "car_model": "E5 Sportback 2025款 旗舰quattro型",
            "down_payment_ratio": 30, "loan_years": 5})),
        _ev("model", content="", usage={}),
        _ev("tool_start", id="c3", name="book_test_drive",
            arguments=json.dumps({
                "car_series": "E5 Sportback", "dealer": "上汽奥迪上海郑隆奥翟",
                "appointment_time": "2026-11-22T15:00:00",
                "customer_name": "张伟", "customer_phone": "13012340000"})),
        _ev("tool_finish", id="c3", result=_success_raw(bk, {
            "car_series": "E5 Sportback", "dealer": "上汽奥迪上海郑隆奥翟",
            "appointment_time": "2026-11-22T15:00:00",
            "customer_name": "张伟", "customer_phone": "13012340000"})),
        _ev("model", content="", usage={}),
        _ev("tool_start", id="c4", name="Finish",
            arguments=json.dumps({"answer": (
                "顶配E5 Sportback 2025款 旗舰quattro型指导价319900元，"
                "首付30%贷5年月供4174.73元，已为您预约11月22日试驾，预约成功。")})),
        _ev("tool_finish", id="c4", result="{}"),
    ]
    out = rollout_from_events(events, question, "S")
    r1 = compute_reward(label, out["steps"], out["response"], eval_year="2026")
    r2 = compute_reward(label, out["steps"], out["response"], eval_year="2026")
    ok &= _check("同轨迹两次判分逐字段一致", r1 == r2)
    ok &= _check("golden 轨迹 r>0", r1["r"] > 0, f"r={r1['r']}")
    return ok


def test_select_frontier():
    ok = True
    rows = [{"id": f"p{i}", "rate": r} for i, r in
            enumerate([0.0, 0.125, 0.2, 0.5, 0.8, 0.875, 1.0])]
    s = select_frontier(rows, min_rate=0.2, max_rate=0.8, min_keep=2)
    ok &= _check("20~80% 闭区间保留 3 条", s["n_keep"] == 3 and s["ok"],
                 f"kept={s['n_keep']} ids={s['kept_ids']}")
    ok &= _check("全对(1.0)/全错(0.0)丢弃", "p0" in s["dropped_ids"] and "p6" in s["dropped_ids"])
    s2 = select_frontier(rows, min_rate=0.2, max_rate=0.8, min_keep=10)
    ok &= _check("不足 min_keep 时 ok=False 并给提示", not s2["ok"] and "hint" in s2)
    return ok


def test_summarize_subset():
    ok = True
    rows = [
        {"pass": True, "reward": {"r": 0.8, "r_acc": 1.0, "r_gate": 1},
         "meta": {"truncated": False}},
        {"pass": False, "reward": {"r": 0.0, "r_acc": 0.5, "r_gate": 0},
         "meta": {"truncated": False}},
        {"pass": False, "reward": {"r": 0.4, "r_acc": 0.5, "r_gate": 1},
         "meta": {"truncated": True}},
        {"pass": True, "reward": {"r": 0.8, "r_acc": 1.0, "r_gate": 1},
         "meta": {"truncated": False}},
    ]
    s = summarize_subset(rows)
    ok &= _check("pass_rate=0.5", s["pass_rate"] == 0.5, str(s["pass_rate"]))
    ok &= _check("mean_r=0.5", s["mean_r"] == 0.5, str(s["mean_r"]))
    ok &= _check("gate_zero_rate=0.25", s["gate_zero_rate"] == 0.25, str(s["gate_zero_rate"]))
    ok &= _check("truncated_rate=0.25", s["truncated_rate"] == 0.25, str(s["truncated_rate"]))
    return ok


def main():
    print("=" * 72)
    print("GRPO 训练循环假 rollout 单测")
    print("=" * 72)
    all_pass = True
    all_pass &= test_compute_advantages()
    all_pass &= test_sequence_loss()
    all_pass &= test_tokenize_with_mask()
    all_pass &= test_fail_inject()
    all_pass &= test_build_rl_registry()
    all_pass &= test_rollout_from_events_ok()
    all_pass &= test_rollout_truncation()
    all_pass &= test_reward_recomputable()
    all_pass &= test_select_frontier()
    all_pass &= test_summarize_subset()
    print("=" * 72)
    print("总体：", "ALL PASS" if all_pass else "HAS FAILURES")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
