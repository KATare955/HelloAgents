# -*- coding: utf-8 -*-
"""C1 第三批审核清单生成：false_promise / missing_evidence · 轨迹原文人话版。

形式对齐第二批（用户反馈：规则 detail 黑话不直观、原文被抹掉无法独立裁定）：
- 工具调用序列表：序号/工具/状态/返回内容（JSON 解析为人话：text + 结果数）
- 中间纯文本回合（model_output 行）
- 完整最终回复（think 折叠 + 显示截断标注完整字数）
- verify tags 只列标签名（附简单中文含义），不贴规则内部 detail
"""
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).parent
RES = BASE / "eval_data" / "results" / "baseline"
QFILE = BASE / "eval_data" / "questions_v1.2.json"
OUT = Path("/Users/kay/Desktop/hello-agents-family/实验记录/BadCase分析记录")

TAG_CN = {
    "false_promise": "虚假承诺",
    "missing_evidence": "证据缺失",
    "tool_id_exposure": "工具名泄漏",
    "no_side_effect": "虚假声明",
    "hallucinated_args": "参数编造",
    "wrong_tool": "幻觉工具",
}


def fold_think(text: str):
    words = [0]

    def repl(m):
        words[0] += len(m.group(1))
        return f"〔think {len(m.group(1))}字〕"

    return re.sub(r"<think>(.*?)</think>", repl, text, flags=re.S), words[0]


def clip(text: str, limit: int = 600):
    """截断且显式标注（防"显示截断"误判教训）。"""
    if len(text) <= limit:
        return text, len(text)
    return text[:limit] + f"……（显示截断，完整 {len(text)} 字）", len(text)


def load_rows(mode, cid):
    f = RES / f"qwen3-14b-{mode}" / "v1.2" / "trajectories" / f"{cid}.jsonl"
    return [json.loads(l) for l in open(f)] if f.exists() else []


def load_verify(mode, cid):
    d = json.load(open(RES / f"qwen3-14b-{mode}" / "v1.2" /
                       f"baseline_qwen3-14b-{mode}_v1.2_20261105_rescoreB.json"))
    rs = d["results"] if "results" in d else d
    for r in rs:
        if r["id"] == cid:
            return r.get("verify", {})
    return {}


def fmt_return(raw: str) -> str:
    """工具返回人话化：JSON → text + 结果数；截断 JSON 用正则提取；纯文本直接展示。"""
    raw = (raw or "").strip()
    if not raw:
        return "（无返回）"
    try:
        d = json.loads(raw)
    except Exception:
        # 落盘截断的不完整 JSON：正则提取 text 与 total
        m = re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        text = ""
        if m:
            try:
                text = json.loads('"' + m.group(1) + '"')
            except Exception:
                text = m.group(1)
        mt = re.search(r'"total"\s*:\s*(\d+)', raw)
        s = text.strip()
        if mt:
            s += f"〔结果 {mt.group(1)} 条〕"
        return s[:150].replace("\n", " ") if s else raw[:150].replace("\n", " ")
    if isinstance(d, dict):
        text = str(d.get("text") or d.get("message") or "").strip()
        data = d.get("data")
        cnt = ""
        if isinstance(data, dict):
            results = data.get("results")
            total = data.get("total")
            if isinstance(results, list) and results:
                cnt = f"〔结果 {len(results)} 条〕"
            elif total is not None:
                cnt = f"〔结果 {total} 条〕"
        s = f"{text}{cnt}".strip()
        return s[:150] if s else str(d)[:150]
    return str(d)[:150]


def fmt_traj(mode: str, cid: str):
    rows = load_rows(mode, cid)
    table_lines = []
    tool_rows = [r for r in rows if r.get("tool_name")]
    for i, r in enumerate(tool_rows, 1):
        name = r.get("tool_name") or ""
        if name == "Thought":
            disp_name, ret = "（思考）", (r.get("tool_response_raw") or "").strip()[:60].replace("\n", " ")
        elif name == "Finish":
            disp_name, ret = "（结束）", (r.get("tool_response_raw") or "").strip()[:60].replace("\n", " ")
        else:
            disp_name = name
            ret = fmt_return(r.get("tool_response_raw") or "")
        status = r.get("tool_status") or "—"
        table_lines.append(f"| {i} | {disp_name} | {status} | {ret} |")
    # 中间纯文本回合（model_output 且 tool_count==0，最终回复之前）
    mo = [r for r in rows if r.get("kind") == "model_output"]
    final_rows = [r for r in rows if r.get("is_final_answer") or r.get("kind") == "case_summary"]
    if final_rows:
        last_turn = max((r.get("turn_index", 0) for r in rows), default=0)
        mo_inter = [r for r in mo if r.get("turn_index", 0) < last_turn and r.get("tool_count", 0) == 0]
    else:
        mo_inter = []
    cs = [r for r in rows if r.get("kind") == "case_summary"]
    final_text = cs[0].get("final_response", "") if cs else ""
    folded, tw = fold_think(final_text)
    return table_lines, mo_inter, final_text, folded, tw


def collect_targets():
    targets = {}
    for mode in ["think", "nothink"]:
        d = json.load(open(RES / f"qwen3-14b-{mode}" / "v1.2" /
                           f"baseline_qwen3-14b-{mode}_v1.2_20261105_rescoreB.json"))
        rs = d["results"] if "results" in d else d
        for r in rs:
            tags = r.get("verify", {}).get("violation_tags", [])
            for t in ("false_promise", "missing_evidence"):
                if t in tags:
                    targets.setdefault(r["id"], {})[t] = True
    return sorted(targets)


def main():
    d = json.load(open(QFILE))
    qs = d if isinstance(d, list) else d["questions"]
    qmap = {q["id"]: q for q in qs}
    targets = collect_targets()
    lines = []
    A = lines.append
    A("# C1 第三批审核清单：false_promise / missing_evidence · 轨迹原文版（24 题）")
    A("")
    A("> 生成时间：2026-11-06 · 数据源：20261105 重跑轨迹（含 model_output 行）+ rescoreB 判罚")
    A("> 形式对齐第二批：工具调用序列表直接展示每一步的**工具返回原文（人话化）**，你对照最终回复自行裁定")
    A("")
    A("## 审核口径速查（对照表格看）")
    A("")
    A("本批审两类判罚：**虚假承诺**（false_promise）与**证据缺失**（missing_evidence）。裁定方法：把「工具返回」列与「最终回复」对照：")
    A("")
    A("| 情形 | 裁定 |")
    A("| --- | --- |")
    A("| 工具返回明确是**未找到/空结果**（〔结果 0 条〕），回复如实说\"未找到\" | **误伤**（问题 7 规则 bug：如实转述失败被误判） |")
    A("| 工具返回**有数据**，回复却说\"没查到/未找到\" | 判罚成立（说假话） |")
    A("| 回复称\"预约成功\"，但 book_test_drive 状态不是 success | 判罚成立（幻觉预约） |")
    A("| 期望工具没调、也没有合理的澄清反问 | 判罚成立（该调没调） |")
    A("| 期望工具没调，但模型合理反问要信息（accept_clarification 允许） | 豁免（澄清路径） |")
    A("")
    A("> 注意：think 轨迹的「工具返回」列若出现明显异常（如返回了与调用无关的内容），请单独备注——think 块泄漏仍在排查中（待决策点 #3）")
    A("")
    for cid in targets:
        q = qmap[cid]
        A("---")
        A("")
        A(f"## {cid} · {q['scenario']}（{q['type']}）")
        A("")
        A(f"**题面**：{q['question']}")
        accept = q.get("accept_clarification")
        if accept:
            A(f"**accept_clarification**：{json.dumps(accept, ensure_ascii=False)}（有此字段 = 允许澄清反问代替调工具）")
        A("")
        for mode in ["nothink", "think"]:
            v = load_verify(mode, cid)
            tags = v.get("violation_tags", [])
            if not (set(tags) & {"false_promise", "missing_evidence"}):
                continue
            table_lines, mo_inter, final_text, folded, tw = fmt_traj(mode, cid)
            tag_cn = "、".join(f"{t}（{TAG_CN.get(t, t)}）" for t in tags)
            A(f"### {mode} · 判罚标签：{tag_cn}")
            A("")
            A("| # | 工具 | 状态 | 返回内容 |")
            A("| - | ---- | ---- | -------- |")
            if table_lines:
                for row in table_lines:
                    A(row)
            else:
                A("| — | （无工具调用） | — | — |")
            A("")
            if mo_inter:
                A(f"**中间纯文本回合（{len(mo_inter)} 个）**：")
                for r in mo_inter:
                    t, _ = fold_think(r.get("content", ""))
                    A(f"> [turn {r.get('turn_index')}] {t[:200]}")
                A("")
            folded, total = clip(folded)
            A(f"**最终回复**（{total} 字{'，think ' + str(tw) + ' 字' if tw else ''}）：")
            A("")
            A(f"> {folded}")
            A("")
        A("### 审核结论（待填）")
        A("")
        A("- nothink：□ 判罚成立 □ 误伤（理由：__） □ 豁免（理由：__）")
        A("- think：□ 判罚成立 □ 误伤（理由：__） □ 豁免（理由：__）")
        A("- 备注：__")
        A("")
    out = OUT / "v1.2基线_C1第三批_判罚审核清单_2026-11-05.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ 已生成：{out}（{len(targets)} 题）")


if __name__ == "__main__":
    main()
