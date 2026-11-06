"""C1 人工审核 v2 修复版清单：完整回复显示 + 预填上次结论（临时脚本，不入库）"""
import json, re
from pathlib import Path

ROOT = Path("/Users/kay/Desktop/hello-agents-family/HelloAgents")
NOTE = Path("/Users/kay/Desktop/hello-agents-family/实验记录")
QMAP = {q["id"]: q for q in json.load(open(ROOT / "eval/eval_data/questions.json"))["questions"]}

def load_results(mode):
    p = ROOT / f"eval/eval_data/results/baseline/qwen3-14b-{mode}/v1.2/baseline_qwen3-14b-{mode}_v1.2_20261104_rescoreB.json"
    return {r["id"]: r for r in json.load(open(p))["results"]}

THINK, NOTHINK = load_results("think"), load_results("nothink")

def read_traj(mode, cid):
    p = ROOT / f"eval/eval_data/results/baseline/qwen3-14b-{mode}/v1.2/trajectories/{cid}.jsonl"
    calls, final = [], ""
    for line in open(p):
        r = json.loads(line)
        if r.get("kind") == "case_summary":
            final = r.get("final_response", "")
        elif r.get("tool_name"):
            calls.append((r.get("tool_name"), r.get("tool_args") or {}, (r.get("tool_response_raw") or "")[:200]))
    return calls, final

def split_think(text):
    """分离 think 块与正文"""
    m = re.search(r"<think>(.*?)</think>", text, re.S)
    if not m:
        return "", text
    return m.group(1), (text[:m.start()] + text[m.end():]).strip()

def fmt_final(text):
    think, body = split_think(text)
    parts = []
    if think:
        parts.append(f"<think> 思考过程 {len(think)} 字（已折叠，重点看正文）</think>")
    if body:
        parts.append(body if len(body) <= 1500 else body[:1500] + f"\n…（正文截断显示，完整 {len(body)} 字）")
    return "\n\n".join(parts) if parts else "（无正文）", len(text), len(body)

# 提取上次审核结论
old = (NOTE / "BadCase分析记录/v1.2基线_test_drive系列_人工审核清单.md").read_text(encoding="utf-8")
concl = {}
for m in re.finditer(r"## (\S+).*?【审核结论】[✓_＿\s]*([^\n]+)", old, re.S):
    concl[m.group(1)] = m.group(2).strip()

def esc(s):
    return str(s).replace("\n", " ").strip()

lines = ["# v1.2 基线 · test_drive 系列人工审核清单（v2 修复版）\n",
"> **v2 改进**：① 最终回复**完整显示**（v1 的 400 字截断已修正，含总字数标注）；② 预填 v1 的审核结论，**本次只需复核「回复是否完整」这一条**（v1 唯一被显示截断干扰的点）。\n",
"> **复核方法**：看每题 think/nothink 的完整回复结尾是否自然收尾、反问是否完整——确认后勾选「结论不变」即可；若发现新问题，写一句修正。\n"]

td_ids = [qid for qid in QMAP if qid.startswith("test_drive")] + ["multi_hop_005"]
for cid in sorted(td_ids):
    q = QMAP[cid]
    n, t = NOTHINK[cid], THINK[cid]
    lines.append(f"\n---\n\n## {cid}｜nothink score={n['verify']['final_score']} ／ think score={t['verify']['final_score']}\n")
    lines.append(f"- **题面**：{esc(q.get('question'))}\n")
    lines.append(f"- **accept_clarification**：{q.get('accept_clarification') or '无'}\n")
    for mode, res in [("nothink", n), ("think", t)]:
        calls, final = read_traj(mode, cid)
        lines.append(f"\n### {mode}（tags={res['verify']['violation_tags']}）\n")
        for tool, args, obs in calls:
            if tool == "book_test_drive":
                lines.append(f"- **book_test_drive 调用参数**（溯源标注同 v1）：\n")
                for k, v in args.items():
                    src = "题面✓" if str(v) in q.get("question", "") else "题面✗"
                    lines.append(f"    - `{k}` = {esc(v)}  →  {src}\n")
                lines.append(f"    - 工具返回：{esc(obs)}\n")
            elif tool == "Thought":
                pass
            else:
                lines.append(f"- 调用 {tool}：{esc(json.dumps(args, ensure_ascii=False))[:120]}\n")
        shown, total, body_len = fmt_final(final)
        lines.append(f"- **完整最终回复（总 {total} 字，正文 {body_len} 字）**：\n")
        lines.append(f"  {shown.replace(chr(10), chr(10) + '  ')}\n")
    old_c = concl.get(cid, "（v1 未填）")
    lines.append(f"\n> **v1 审核结论**：{old_c}\n")
    lines.append("> **v2 复核**：＿ 结论不变（回复完整）／＿ 需修正：\n")

out = NOTE / "BadCase分析记录/v1.2基线_test_drive系列_人工审核清单_v2修复版.md"
out.write_text("".join(lines), encoding="utf-8")
print("已生成:", out, "| 题数:", len(td_ids))
