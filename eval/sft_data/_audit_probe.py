#!/usr/bin/env python3
"""_audit_probe.py — SFT 训练数据(sft_trajectories.jsonl)审核探查脚本
只读，不改数据。用于人工审核前的「脚本统计」层：盘点分布 / 结构合法性 / 参数完整性 / 接地防编造。
用法: python3 eval/sft_data/_audit_probe.py [jsonl路径]
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "sft_trajectories.jsonl"

REQUIRED = {
    "text_search": ["query"],
    "query_discount_policy": ["car_model"],
    "query_transaction_price": ["car_model"],
    "competitor_compare": ["car_model_a", "car_model_b"],
    "calculate_finance": ["car_price", "down_payment_ratio", "loan_years"],
    "book_test_drive": ["car_series", "dealer", "appointment_time", "customer_name", "customer_phone"],
}
BIZ_TOOLS = set(REQUIRED)
VALID_DEALERS = {"上汽奥迪上海郑隆奥翟", "上汽奥迪上海大宁音乐广场", "上汽奥迪西上海用户中心", "上汽奥迪上海东昌用户中心", "上汽奥迪上海冠澳用户中心", "上汽奥迪上海冠松用户中心", "上汽奥迪奥迪之城旗舰店"}
VALID_PAIRS = {
    frozenset({"E5 Sportback 2025款 旗舰quattro型", "小米SU7 2025款 四驱Max版"}),
    frozenset({"E5 Sportback 2025款 旗舰quattro型", "帕萨特2024款 330TSI 精英版"}),
}
DISCOUNT_MODELS = {"E5 Sportback", "帕萨特2024款", "小米SU7 2025款"}

rows = [json.loads(l) for l in open(PATH, encoding="utf-8")]
N = len(rows)
print("=" * 64)
print(f"总条数: {N}    文件: {PATH.name}")
print(f"顶层keys: {list(rows[0].keys())}")
print(f"meta keys: {list(rows[0]['meta'].keys())}")

# ---------- 1) 分布盘点 ----------
print("\n" + "=" * 64 + "\n[1] 桶/场景分布")
bk = Counter(r["bucket"] for r in rows)
for k, c in bk.most_common():
    print(f"  {k:18s} {c:3d}  ({c/N*100:.0f}%)")
print("  -- scenario 分布 --")
for k, c in Counter(r["scenario"] for r in rows).most_common():
    print(f"  {k:14s} {c:3d}")
print("  -- is_clarify / is_degrade --")
print(f"  is_clarify=True : {sum(1 for r in rows if r.get('is_clarify'))}")
print(f"  is_degrade=True : {sum(1 for r in rows if r.get('is_degrade'))}")
print("  -- bucket × scenario 交叉表 --")
cross = defaultdict(Counter)
for r in rows:
    cross[r["bucket"]][r["scenario"]] += 1
for b, sc in cross.items():
    print(f"  {b:18s} -> {dict(sc)}")

# ---------- 2) 工具频次 ----------
print("\n" + "=" * 64 + "\n[2] 工具调用频次(assistant.tool_calls.function.name)")
tool_freq = Counter()
thought_freq = finish_freq = 0
for r in rows:
    for m in r["messages"]:
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                nm = tc["function"]["name"]
                if nm == "Thought":
                    thought_freq += 1
                elif nm == "Finish":
                    finish_freq += 1
                else:
                    tool_freq[nm] += 1
for k, c in tool_freq.most_common():
    print(f"  {k:22s} {c:3d}")
print(f"  {'Thought':22s} {thought_freq:3d}")
print(f"  {'Finish':22s} {finish_freq:3d}")

# ---------- 3) book_test_drive 落点 ----------
print("\n" + "=" * 64 + "\n[3] book_test_drive 落点(核对 48:11 失衡)")
book_by_bucket = Counter()
book_ids = []
for r in rows:
    names = [tc["function"]["name"] for m in r["messages"] if m.get("role") == "assistant"
             for tc in (m.get("tool_calls") or [])]
    if "book_test_drive" in names:
        book_by_bucket[r["bucket"]] += 1
        book_ids.append(r["id"])
print(f"  真调 book_test_drive 的条数: {sum(book_by_bucket.values())}")
print(f"  按桶: {dict(book_by_bucket)}")
print(f"  clarify 桶调 book 数(应=0): {book_by_bucket.get('clarify', 0)}")
print(f"  direct_single 调 book 数: {book_by_bucket.get('direct_single', 0)}")
# 试驾场景下 clarify vs book
clarify_cnt = sum(1 for r in rows if r.get("is_clarify"))
print(f"  is_clarify 条数 : book 条数 = {clarify_cnt} : {sum(book_by_bucket.values())}")

# ---------- 4) 结构合法性 ----------
print("\n" + "=" * 64 + "\n[4] 消息结构合法性(逐条校验)")
err = defaultdict(list)
for r in rows:
    rid, msgs = r["id"], r["messages"]
    roles = [m.get("role") for m in msgs]
    if roles[0] != "system":
        err["首条非system"].append(rid)
    if len(roles) < 2 or roles[1] != "user":
        err["次条非user"].append(rid)
    # tool_call_id 配对
    open_ids = []
    for m in msgs:
        if m.get("role") == "assistant":
            if m.get("content") is not None and m.get("tool_calls"):
                err["assistant同时有content和tool_calls"].append(rid)
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {})
                if not fn.get("name"):
                    err["tool_call缺name"].append(rid)
                try:
                    json.loads(fn.get("arguments", ""))
                except Exception:
                    err["arguments非合法JSON"].append(rid)
                open_ids.append(tc.get("id"))
        elif m.get("role") == "tool":
            tid = m.get("tool_call_id")
            if tid not in open_ids:
                err["tool消息tool_call_id无匹配"].append(rid)
    # 末条必须是 Finish
    last_asst = [m for m in msgs if m.get("role") == "assistant"]
    if not last_asst:
        err["无assistant消息"].append(rid)
    else:
        last_calls = last_asst[-1].get("tool_calls") or []
        if not last_calls or last_calls[-1]["function"]["name"] != "Finish":
            err["末条非Finish"].append(rid)
if not err:
    print("  ✅ 全部通过结构校验")
for k, v in err.items():
    print(f"  ❌ {k}: {len(v)} 条  例:{v[:5]}")

# ---------- 5) 必填参数完整性 ----------
print("\n" + "=" * 64 + "\n[5] 必填参数完整性(逐 tool_call)")
miss = Counter()
miss_ids = defaultdict(list)
for r in rows:
    for m in r["messages"]:
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                nm = tc["function"]["name"]
                if nm not in BIZ_TOOLS:
                    continue
                args = json.loads(tc["function"]["arguments"])
                for rq in REQUIRED[nm]:
                    if args.get(rq) in (None, ""):
                        miss[f"{nm}.{rq}"] += 1
                        miss_ids[f"{nm}.{rq}"].append(r["id"])
if not miss:
    print("  ✅ 所有业务工具调用必填参数齐全")
for k, c in miss.most_common():
    print(f"  ❌ 缺 {k}: {c} 次  例:{miss_ids[k][:3]}")

# ---------- 6) 接地/防编造 抽检 ----------
print("\n" + "=" * 64 + "\n[6] 接地防编造抽检")
fab = defaultdict(list)
for r in rows:
    rid = r["id"]
    biz = []
    for m in r["messages"]:
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                nm = tc["function"]["name"]
                if nm in BIZ_TOOLS:
                    biz.append((nm, json.loads(tc["function"]["arguments"])))
    names = [n for n, _ in biz]
    for nm, a in biz:
        if nm == "book_test_drive":
            if a.get("dealer") not in VALID_DEALERS:
                fab["book.dealer非法"].append((rid, a.get("dealer")))
        if nm == "competitor_compare" and not r.get("is_degrade"):
            pair = frozenset({a.get("car_model_a"), a.get("car_model_b")})
            if pair not in VALID_PAIRS:
                fab["compare.车型对非法"].append((rid, tuple(pair)))
        if nm == "text_search":
            if a.get("category") not in (None, "", "all"):
                fab["text_search.误传category"].append((rid, a.get("category")))
        if nm == "calculate_finance":
            cp = a.get("car_price")
            if not isinstance(cp, (int, float)) or cp <= 0:
                fab["finance.car_price非法"].append((rid, cp))
    # 金融连贯: 同轨迹有 text_search 时 car_price 应对得上(粗检:有查价才算)
    if "calculate_finance" in names and "text_search" in names:
        pass  # 需读 observation, 留给人工/深度脚本
for k, v in fab.items():
    print(f"  ⚠️ {k}: {len(v)}  例:{v[:4]}")
if not fab:
    print("  ✅ 抽检未发现明显编造(dealer/车型对/category/car_price 均合法)")

# 6b) degrade 桶「推荐接地表外车型」扫描(违反 generator hint: 应推荐接地表真实替代车型)
print("\n" + "-" * 40 + "\n[6b] degrade 桶非接地推荐扫描")
NON_GROUNDED = ["奥迪A6L", "奥迪A8L", "奥迪A4L", "奥迪A7L", "特斯拉", "Model S", "Model 3",
                "小鹏", "比亚迪", "海豹", "唐EV", "秦PLUS", "宝马i4", "宝马5系", "e-tron"]
dg_total = sum(1 for r in rows if r.get("is_degrade"))
dg_bad = []
for r in rows:
    if not r.get("is_degrade"):
        continue
    ans = r["meta"].get("final_answer", "")
    hit = [t for t in NON_GROUNDED if t in ans]
    if hit:
        dg_bad.append((r["id"], hit))
print(f"  degrade 共 {dg_total} 条 | 推荐了接地表外车型: {len(dg_bad)} 条 ({len(dg_bad)/max(dg_total,1)*100:.0f}%)")
for rid, hit in dg_bad[:8]:
    print(f"    {rid}: {hit}")

# ---------- 7) clarify / degrade 桶深检 ----------
print("\n" + "=" * 64 + "\n[7] clarify / degrade 桶专项检查")
cl = [r for r in rows if r.get("is_clarify")]
cl_nobook = sum(1 for r in cl if "book_test_drive" not in
                [tc["function"]["name"] for m in r["messages"] if m.get("role") == "assistant"
                 for tc in (m.get("tool_calls") or [])])
cl_missing = sum(1 for r in cl if r["meta"].get("missing_params"))
print(f"  clarify 桶: {len(cl)} 条 | 未调book: {cl_nobook} | 有missing_params: {cl_missing}")
print(f"  clarify scenario 分布: {dict(Counter(r['scenario'] for r in cl))}")
dg = [r for r in rows if r.get("is_degrade")]
dg_zero = sum(1 for r in dg if not [tc for m in r["messages"] if m.get("role") == "assistant"
                                    for tc in (m.get("tool_calls") or [])
                                    if tc["function"]["name"] in BIZ_TOOLS])
print(f"  degrade 桶: {len(dg)} 条 | 0业务工具调用(直接拒,应警惕): {dg_zero}")
print(f"  degrade scenario 分布: {dict(Counter(r['scenario'] for r in dg))}")

# ---------- 8) 步数分布 ----------
print("\n" + "=" * 64 + "\n[8] 步数分布(business tool 调用数, 应≤5含Thought/Finish)")
steps = Counter()
for r in rows:
    n = sum(1 for m in r["messages"] if m.get("role") == "assistant"
            for tc in (m.get("tool_calls") or []))
    steps[n] += 1
for k in sorted(steps):
    print(f"  {k} 次assistant调用: {steps[k]} 条")

print("\n" + "=" * 64 + "\n探查完成。")
