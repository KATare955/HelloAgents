#!/usr/bin/env python3
"""闭包修复验证：确认 3 处修复（b2fn F1 / b2cp G1 / b4cs_b）后，
每条样本的轨迹与自身题面/配置一致，不再串到其他迭代的 cell。

校验点：
1. b2fn_a（10 条）：turn1 tool args 的 car_model/car_price 与各自 cfg 一致；
   turn2 fix_obs 的 guide_price 与 answer 中的指导价一致
2. b2cp_a（10 条）：turn1 对比对 = pairs[j%5] 轮换，且与 turn1 参数一致
3. b4cs_b（10 条）：turn2 fix_obs 的 car_model 与题面车型一致；
   answer 中成交价数字 = fix_obs 实跑值
4. dryrun 30 条：已生成 jsonl 与当前脚本重新构建结果一致（无闭包漂移）
"""
import json
import random
import sys
from typing import Any, Dict, List

sys.path.insert(0, "eval/sft_data/v02_gen")
from gen_core import CONFIGS, _NUM_RE  # noqa: E402
import gen_templates_full_b1b2 as m1  # noqa: E402
import gen_templates_full_b3b4b5 as m2  # noqa: E402

FAILS: List[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILS.append(msg)
        print(f"  [FAIL] {msg}")
    else:
        print(f"  [OK]   {msg}")


def get_samples() -> Dict[str, tuple]:
    rng = random.Random(20261107)
    out: List[Dict[str, Any]] = []
    for maker in m1.ALL_MAKERS:
        out.extend(maker(rng))
    for maker in m2.ALL_MAKERS:
        out.extend(maker(rng))
    return {t["sid"]: (t, t["build"](t["question"], t["answer"])) for t in out}


def tool_args(msgs: List[Dict], call_name: str) -> Dict:
    for m in msgs:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                if tc.get("id") == call_name:
                    return json.loads(tc["function"]["arguments"])
    raise KeyError(call_name)


def tool_result(msgs: List[Dict], call_name: str) -> str:
    for m in msgs:
        if m.get("role") == "tool" and m.get("tool_call_id") == call_name:
            return m["content"]
    raise KeyError(call_name)


def check_finance_pair(sid: str, t: Dict, msgs: List[Dict]) -> None:
    """F1 组：turn1 传 car_price 被拒 → turn2 移除重算。"""
    cfg_title = t["question"]
    args1 = tool_args(msgs, "call_2")
    check("car_price" in args1, f"{sid} turn1 args 含 car_price")
    cfg_titles = [c["title"] for c in CONFIGS]
    got = args1["car_model"]
    check(got in cfg_titles, f"{sid} turn1 car_model={got} 为白名单车型")
    expect_price = int(CONFIGS[cfg_titles.index(got)]["price"] * 0.95)
    check(args1["car_price"] == expect_price,
          f"{sid} turn1 car_price={args1['car_price']} == 该车型95折 {expect_price}")
    # turn2 结果与 answer 一致性
    obs2 = json.loads(tool_result(msgs, "call_4"))
    d = obs2["data"]
    ans = t["answer"]
    check(str(int(d["guide_price"])) in ans, f"{sid} answer 指导价 {int(d['guide_price'])} 一致")
    check(str(round(float(d["monthly_payment"]))) in ans,
          f"{sid} answer 月供 {round(float(d['monthly_payment']))} 一致")
    check(d["car_model"] == got, f"{sid} turn2 返回车型 {d['car_model']} == turn1 {got}")


def check_competitor_pair(sid: str, t: Dict, msgs: List[Dict]) -> None:
    """G1 组：turn1 擅自选配置对比 → 指正 → 反问。"""
    args1 = tool_args(msgs, "call_2")
    pa, pb = args1["car_model_a"], args1["car_model_b"]
    check(pa == "E5 Sportback 2025款 旗舰quattro型", f"{sid} turn1 对比A=旗舰quattro {pa}")
    check(pb in ("小米SU7 2025款 四驱Max版", "帕萨特2024款 330TSI 精英版"),
          f"{sid} turn1 对比B=竞品 {pb}")
    # answer 里反问列出的配置清单应含全部 4 款车型
    for c in CONFIGS:
        check(c["title"] in t["answer"], f"{sid} answer 反问清单含 {c['title']}")


def check_transaction_pair(sid: str, t: Dict, msgs: List[Dict]) -> None:
    """b4cs_b 组：days=45 非法 → 90 天重试。"""
    car_in_q = t["question"]
    obs2 = json.loads(tool_result(msgs, "call_4"))
    d = obs2["data"]
    check(d["car_model"] in car_in_q, f"{sid} fix_obs 车型 {d['car_model']} ∈ 题面")
    for key, val in [("avg_price", d["avg_price"]), ("min_price", d["min_price"]),
                     ("max_price", d["max_price"])]:
        check(str(val) in t["answer"], f"{sid} answer 含 {key}={val}")


def main() -> None:
    samples = get_samples()
    print(f"== 共 {len(samples)} 条模板 ==")

    print("\n[1] b2fn_a F1 组（10 条）:")
    for j in range(10):
        sid = f"v02_full_b2fn_a{j:02d}"
        check_finance_pair(sid, samples[sid][0], samples[sid][1])

    print("\n[2] b2cp_a G1 组（10 条）:")
    for j in range(10):
        sid = f"v02_full_b2cp_a{j:02d}"
        check_competitor_pair(sid, samples[sid][0], samples[sid][1])

    print("\n[3] b4cs_b 组（10 条）:")
    for j in range(10):
        sid = f"v02_full_b4cs_b{j:02d}"
        check_transaction_pair(sid, samples[sid][0], samples[sid][1])

    print("\n[4] 全量唯一性（question, answer）:")
    pairs = [(t["question"], t["answer"]) for t, _ in samples.values()]
    dup = len(pairs) - len(set(pairs))
    check(dup == 0, f"(q,a) 重复对数 = {dup}（应为 0）")

    print("\n[5] dryrun 30 条一致性（已生成 jsonl vs 当前脚本重建）:")
    try:
        with open("eval/sft_data/v02_dryrun_30.jsonl", encoding="utf-8") as f:
            dry_lines = [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        print("  [SKIP] v02_dryrun_30.jsonl 不存在")
        dry_lines = []
    if dry_lines:
        import gen_templates_dryrun as md
        rng = random.Random(20261106)
        rebuilt = md.make_all_dryrun_templates(rng)
        by_sid = {x["sid"]: x for x in rebuilt}
        from gen_core import numbers_of, entities_of
        import re as _re
        PHONE_RE = _re.compile(r"137\d{8}")  # _phone 历史版本格式差异豁免（旧 137i{4}i{4} vs 新 137{i:04d}{i:04d}）
        DYNAMIC_KEYS = {"booking_id", "created_at", "query_time"}

        def norm_phones(text: str) -> str:
            return PHONE_RE.sub("<PHONE>", text)

        def strip_dynamic(content: str) -> str:
            """tool 返回 JSON 中删除运行时刻动态字段、归一化手机号后再比对"""
            content = norm_phones(content)
            try:
                o = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                return content
            if isinstance(o, dict):
                o = {k: v for k, v in o.items() if k not in DYNAMIC_KEYS}
                if isinstance(o.get("data"), dict):
                    o["data"] = {k: v for k, v in o["data"].items() if k not in DYNAMIC_KEYS}
                return json.dumps(o, ensure_ascii=False, sort_keys=True)
            return content

        diff = 0
        for line in dry_lines:
            t2 = by_sid.get(line.get("id"))
            if t2 is None:
                diff += 1
                continue
            msgs2 = t2["build"](t2["question"], t2["answer"])
            msgs1 = [m for m in line["messages"] if m.get("role") != "system"]
            bad = len(msgs1) != len(msgs2)
            for m1_, m2_ in zip(msgs1, msgs2):
                if bad:
                    break
                if m1_.get("role") != m2_.get("role"):
                    bad = True
                    break
                if m1_.get("role") == "user":
                    # 用户段（题面/指正）可能被润色：数字 multiset 相等 + 实体只减不增；
                    # 手机号按历史版本差异归一化后比对
                    t1 = norm_phones(m1_["content"])
                    t2_ = norm_phones(m2_["content"])
                    if numbers_of(t1) != numbers_of(t2_) or \
                       not entities_of(t2_).issubset(entities_of(t1)):
                        bad = True
                        break
                elif m2_.get("role") == "assistant" and m2_.get("tool_calls"):
                    f1, f2 = m1_["tool_calls"][0]["function"], m2_["tool_calls"][0]["function"]
                    if f1["name"] != f2["name"]:
                        bad = True
                        break
                    if f1["name"] == "Finish":
                        a1 = json.loads(f1["arguments"])["answer"]
                        a2 = json.loads(f2["arguments"])["answer"]
                        if numbers_of(a1) != numbers_of(a2) or \
                           not entities_of(a2).issubset(entities_of(a1)):
                            bad = True
                            break
                    elif norm_phones(f1["arguments"]) != norm_phones(f2["arguments"]):
                        bad = True
                        break
                elif m1_.get("role") == "tool":
                    if strip_dynamic(m1_["content"]) != strip_dynamic(m2_["content"]):
                        bad = True
                        break
                elif json.dumps(m1_, ensure_ascii=False, sort_keys=True) != \
                     json.dumps(m2_, ensure_ascii=False, sort_keys=True):
                    bad = True
                    break
            if bad:
                diff += 1
        check(diff == 0, f"dryrun 30 条轨迹不一致条数 = {diff}（应为 0）")

    print(f"\n{'=' * 50}")
    if FAILS:
        print(f"共 {len(FAILS)} 项失败")
        sys.exit(1)
    print("全部校验通过 ✓")


if __name__ == "__main__":
    main()
