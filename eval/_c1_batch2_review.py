# -*- coding: utf-8 -*-
"""C1 第二批审核清单生成：hallucinated_args 7 题参数溯源版（think/nothink 双模式）。

审核核心：verify 规则层把某些参数判为 fabricated（值 ∉ 题面 ∪ 前序工具返回）。
本清单把每步调用的参数值 + 溯源标注 + 工具返回 + 完整回复全部铺开，
供人工裁定：(a) 编造坐实 (b) 规则误伤 (c) 合理默认值口径。
"""
import json
import re
from pathlib import Path

BASE = Path(__file__).parent
RES = BASE / "eval_data" / "results" / "baseline"
QFILE = BASE / "eval_data" / "questions_v1.2.json"
OUT = Path("/Users/kay/Desktop/hello-agents-family/实验记录/BadCase分析记录")
TARGETS = ["finance_001", "finance_003", "finance_004",
           "multi_hop_001", "multi_hop_003", "multi_hop_006", "multi_hop_007"]

TAG = {"question": "题面✓", "prior_obs": "前序✓", "fabricated": "⚠️编造"}

# 与 verifier/rules.py TRACE_ARGS 保持一致：只有白名单内参数的 fabricated 才触发判罚
TRACE_ARGS = {
    "calculate_finance": {"car_price", "down_payment_ratio", "loan_years", "interest_rate"},
    "book_test_drive": {"car_series", "dealer", "appointment_time", "customer_name", "customer_phone"},
}


def fold_think(text: str) -> str:
    """think 块折叠为占位（保留块外正文），返回 (折叠后文本, think字数)。"""
    think_words = 0
    def repl(m):
        nonlocal think_words
        think_words += len(m.group(1))
        return f"〔think 块 {len(m.group(1))} 字已折叠〕"
    folded = re.sub(r"<think>(.*?)</think>", repl, text, flags=re.S)
    return folded, think_words


def load_rows(mode: str, cid: str):
    f = RES / f"qwen3-14b-{mode}" / "v1.2" / "trajectories" / f"{cid}.jsonl"
    return [json.loads(l) for l in open(f)] if f.exists() else []


def load_verify(mode: str, cid: str):
    d = json.load(open(RES / f"qwen3-14b-{mode}" / "v1.2" /
                       f"baseline_qwen3-14b-{mode}_v1.2_20261105_rescoreB.json"))
    rs = d["results"] if "results" in d else d
    for r in rs:
        if r["id"] == cid:
            return r.get("verify", {})
    return {}


def fmt_args(args, sources, tool_name):
    parts = []
    whitelist = TRACE_ARGS.get(tool_name, set())
    for k, v in (args or {}).items():
        if k not in whitelist:
            parts.append(f"{k}={v}（自由）")
            continue
        src = (sources or {}).get(k, "?")
        mark = TAG.get(src, src)
        parts.append(f"{k}={v}（{mark}）")
    return "、".join(parts) if parts else "（无参数）"


def fmt_traj(mode: str, cid: str):
    rows = load_rows(mode, cid)
    table_lines = []
    # 工具调用序列表
    tool_rows = [r for r in rows if r.get("tool_name")]
    for i, r in enumerate(tool_rows, 1):
        ret = (r.get("tool_response_raw") or "").strip().replace("\n", " ")
        table_lines.append(f"| {i} | {r['tool_name']} | {fmt_args(r.get('tool_args'), r.get('tool_args_source'), r['tool_name'])} | {ret[:120]} |")
    # 中间纯文本回合（model_output 且 tool_count==0，最终回复之前）
    mo = [r for r in rows if r.get("kind") == "model_output"]
    final_rows = [r for r in rows if r.get("is_final_answer") or r.get("kind") == "case_summary"]
    if final_rows:
        last_turn = max((r.get("turn_index", 0) for r in rows), default=0)
        mo_inter = [r for r in mo if r["turn_index"] < last_turn and r.get("tool_count", 0) == 0]
    else:
        mo_inter = []
    # 最终回复
    cs = [r for r in rows if r.get("kind") == "case_summary"]
    final_text = cs[0].get("final_response", "") if cs else ""
    folded, tw = fold_think(final_text)
    return table_lines, mo_inter, final_text, folded, tw


def main():
    d = json.load(open(QFILE))
    qs = d if isinstance(d, list) else d["questions"]
    qmap = {q["id"]: q for q in qs}
    lines = []
    A = lines.append
    A("# C1 第二批审核清单：hallucinated_args 7 题 · 参数溯源版")
    A("")
    A("> 生成时间：2026-11-05 · 数据源：20261105 重跑轨迹（含 model_output 行）+ rescoreB 判罚")
    A("> 溯源标注含义：**题面✓**=值出自题面；**前序✓**=值出自前面工具返回；**⚠️编造**=规则判定值 ∉ 题面 ∪ 前序返回")
    A("> 本次只需复核：**① 每个 ⚠️编造 参数是否真的无来源；② 若真编造，属「工具合理默认值口径」还是「高危编造」**")
    A("")
    A("## 裁定口径备忘（供勾选时参照）")
    A("")
    A("- **口径 A（高危编造）**：值凭空产生且直接影响答案正确性（如车价编造→月供算错）→ 判罚合理")
    A("- **口径 B（合理默认值）**：金融通用默认在题面未给时采用行业惯例 → 需讨论是否豁免。**已知铁证**：`calculate_finance` 工具 schema 显式声明 `interest_rate` 参数 `default=4.5`（description 写「默认4.5%」），模型传 4.5 与不传等价、属遵守工具契约 → 建议豁免 4.5")
    A("- **口径 C（规则误伤）**：值实际来自前序工具返回/题面暗示，规则没识别 → 判罚撤销、修规则")
    A("")
    A("## 已知规则疑点（审核时顺带裁定）")
    A("")
    A("- **疑点 1 · 子串匹配**：溯源是纯子串匹配（`text in question`）。题面含「90天」时 `car_price=0` 被标「题面✓」——0 是模型试探性调用（先触发参数校验错），不是题面给的。子串匹配对数字/短词误标率高，属规则层待修项")
    A("- **疑点 2 · 白名单外自由参数**：`text_search.query/category/limit`、`query_discount_policy.region/include_expired`、`compare_aspects`、`notes` 等标「自由」，规则不判罚（A3 校准实证：自由组织参数误伤率高）。若审核发现某自由参数值危害大（如 region 编造导致查错地区），请在备注标注")
    A("- **疑点 3 · mock 数据盲区**：`calculate_finance` 的 mock 预设键只有 235900/289900/319900 等组合；`car_price=200000` 不在 mock 里会走实时计算路径，仍能算出月供——所以模型编个 20 万也能「成功」出结果，不会被工具报错拦截")
    A("")
    for cid in TARGETS:
        q = qmap[cid]
        A("---")
        A("")
        A(f"## {cid} · {q['scenario']}（{q['type']} / {q['difficulty']}）")
        A("")
        A(f"**题面**：{q['question']}")
        A("")
        A(f"**expected_tools**：{', '.join(q['expected_tools'])}")
        note = q.get("note")
        if note:
            A(f"**note**：{note}")
        A("")
        for mode in ["nothink", "think"]:
            table_lines, mo_inter, final_text, folded, tw = fmt_traj(mode, cid)
            v = load_verify(mode, cid)
            tags = v.get("violation_tags", [])
            A(f"### {mode} 轨迹")
            A("")
            A("| # | 工具 | 参数（溯源标注） | 返回摘要 |")
            A("| - | ---- | ---------------- | -------- |")
            if table_lines:
                for row in table_lines:
                    A(row)
            else:
                A("| — | （无工具调用） | — | — |")
            A("")
            if mo_inter:
                A(f"**中间纯文本回合（{len(mo_inter)} 个）**：")
                for r in mo_inter:
                    t, _ = fold_think(r["content"])
                    A(f"> [turn {r['turn_index']}] {t[:200]}")
                A("")
            A(f"**最终回复**（{len(final_text)} 字{'，含 think ' + str(tw) + ' 字' if tw else ''}）：")
            A("")
            A(f"> {folded}")
            A("")
            if "hallucinated_args" in tags:
                fab = "fabricated 参数见上表 ⚠️编造 标注"
                A(f"**verify 判罚**：❌ hallucinated_args（{fab}）→ 总分 {v.get('final_score')}")
            else:
                A(f"**verify 判罚**：✅ 无 hallucinated_args（tags={tags or '无'}）")
            A("")
        A("### 审核结论（待填）")
        A("")
        A("- nothink：□ 编造坐实（口径A） □ 合理默认值（口径B） □ 规则误伤（口径C） □ 其他：__")
        A("- think：□ 编造坐实（口径A） □ 合理默认值（口径B） □ 规则误伤（口径C） □ 其他：__")
        A("- 备注：__")
        A("")
    out = OUT / "v1.2基线_C1第二批_参数溯源审核清单_2026-11-05.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ 已生成：{out}")


if __name__ == "__main__":
    main()
