#!/usr/bin/env python3
"""全量轨迹自洽验证（T9）：对 v02_full_1000.jsonl 每条新生成样本，
以最终题面/答复重放模板 build，逐段比对 jsonl 轨迹。

覆盖的缺陷类型：循环闭包串位（Thought 错位/工具参数串位）、
build 与模板声明不一致、工具返回 mock 漂移等。
dryrun 30 条为已审核版本，由 verify_closure_fix.py 覆盖，此处跳过。

比对口径：
- system 段跳过
- user 段完全相等（题面直接用最终 q_final 重放；多轮指正段为程序文本不润色）
- assistant 段：tool_calls 的 (name, arguments) 逐对相等；loss_mask 归一化比对
- tool 返回段：剔除动态字段（booking_id/created_at）+ 手机号归一化后相等
"""
import json
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, "eval/sft_data/v02_gen")
import gen_templates_full_b1b2 as m1  # noqa: E402
import gen_templates_full_b3b4b5 as m2  # noqa: E402

JSONL = "eval/sft_data/v02_full_1000.jsonl"

PHONE_RE = re.compile(r"137\d{8}")
# booking_id/created_at: 试驾预约生成; query_time: 优惠查询生成（每次调用时间戳不同）
DYNAMIC_KEYS = {"booking_id", "created_at", "query_time"}


def norm_phones(text: str) -> str:
    return PHONE_RE.sub("<PHONE>", text)


def strip_dynamic(content: str) -> str:
    """tool 返回 JSON 中删除运行时刻动态字段、归一化手机号后再比对。"""
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


def _mask_norm(m: Dict) -> str:
    """loss_mask 归一化：None/True → 'T'，False → 'F'。"""
    return "F" if m.get("loss_mask") is False else "T"


def seg_signature(m: Dict) -> Tuple:
    if m.get("role") == "user":
        return ("user", m["content"])
    if m.get("role") == "assistant":
        calls = tuple(
            (tc.get("function", {}).get("name"),
             norm_phones(str(tc.get("function", {}).get("arguments", ""))))
            for tc in (m.get("tool_calls") or [])
        )
        return ("assistant", calls, _mask_norm(m))
    if m.get("role") == "tool":
        return ("tool", strip_dynamic(m.get("content", "")))
    return ("other", json.dumps(m, ensure_ascii=False, sort_keys=True))


def finish_answer(msgs: List[Dict]) -> Optional[str]:
    for m in msgs:
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {})
                if fn.get("name") == "Finish":
                    return json.loads(fn["arguments"]).get("answer", "")
    return None


def build_templates() -> Dict[str, Dict[str, Any]]:
    rng = random.Random(20261107)
    out: Dict[str, Dict[str, Any]] = {}
    for maker in m1.ALL_MAKERS + m2.ALL_MAKERS:
        for t in maker(rng):
            out[t["sid"]] = t
    return out


def compare(sid: str, msgs_jsonl: List[Dict], msgs_replay: List[Dict]) -> Optional[str]:
    a = [m for m in msgs_jsonl if m.get("role") != "system"]
    b = [m for m in msgs_replay if m.get("role") != "system"]
    if len(a) != len(b):
        return f"段数不一致: jsonl={len(a)} vs replay={len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        sx, sy = seg_signature(x), seg_signature(y)
        if sx != sy:
            return (f"段[{i}]不一致:\n"
                    f"    jsonl : {str(sx)[:160]}\n"
                    f"    replay: {str(sy)[:160]}")
    return None


def main() -> None:
    tpls = build_templates()
    with open(JSONL, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    print(f"模板 {len(tpls)} 条 | jsonl {len(lines)} 条")

    fails: List[Tuple[str, str]] = []
    checked = 0
    for o in lines:
        sid = o["id"]
        if sid.startswith("v02_dry"):
            continue  # dryrun 已审核版本，由 verify_closure_fix 覆盖
        t = tpls.get(sid)
        if t is None:
            fails.append((sid, "模板缺失")); continue
        a_final = finish_answer(o["messages"])
        if a_final is None:
            fails.append((sid, "Finish 段缺失")); continue
        replay = t["build"](o["question"], a_final)
        err = compare(sid, o["messages"], replay)
        if err:
            fails.append((sid, err))
        checked += 1

    print(f"校验新生成样本 {checked} 条 | 不一致 {len(fails)} 条")
    for sid, err in fails[:20]:
        print(f"\n[{sid}] {err}")

    print("\n" + "=" * 56)
    if fails:
        print(f"FAIL: {len(fails)} 条轨迹不自洽")
        sys.exit(1)
    print("PASS: 全部轨迹与模板重放一致 ✓")


if __name__ == "__main__":
    main()
