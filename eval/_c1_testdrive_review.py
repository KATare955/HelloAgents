"""C1 人工审核辅助：生成 test_drive 系列 11 题审核清单（临时脚本，不入库）"""
import json
from pathlib import Path

ROOT = Path("/Users/kay/Desktop/hello-agents-family/HelloAgents")
QMAP = {q["id"]: q for q in json.load(open(ROOT / "eval/eval_data/questions.json"))["questions"]}

def load_results(mode):
    p = ROOT / f"eval/eval_data/results/baseline/qwen3-14b-{mode}/v1.2/baseline_qwen3-14b-{mode}_v1.2_20261104_rescoreB.json"
    return {r["id"]: r for r in json.load(open(p))["results"]}

THINK = load_results("think")
NOTHINK = load_results("nothink")

def read_traj(mode, cid):
    p = ROOT / f"eval/eval_data/results/baseline/qwen3-14b-{mode}/v1.2/trajectories/{cid}.jsonl"
    calls, final = [], ""
    for line in open(p):
        r = json.loads(line)
        if r.get("kind") == "case_summary":
            final = r.get("final_response", "")
        elif r.get("tool_name"):
            calls.append((r.get("tool_name"), r.get("tool_args") or {}, (r.get("tool_response_raw") or "")[:250]))
    return calls, final

def trace(value, question):
    """字面溯源：值是否出现在题面"""
    v = str(value)
    return "题面✓" if v in question else "题面✗"

def esc(s):
    return str(s).replace("\n", " ").strip()

lines = []
lines.append("# v1.2 基线 · test_drive 系列人工审核清单\n")
lines.append("> **审核目标**：确认 `hallucinated_args` 在试驾预约 10/11 全线失守中的**误伤面**——哪些参数是模型纯编造（真违规），哪些是题面相对时间/隐式信息的合理推导（疑似误伤）。\n")
lines.append("> **审核方法**：每题看三处——题面给了什么、模型传了什么、工具回了什么。重点看 `book_test_drive` 的 5 个参数（车型/门店/时间/姓名/电话）各自是「题面✓」还是「题面✗」，题面✗ 的参数再判断是**推导**（如\"明天\"→日期）还是**编造**（题面完全没提）。\n")

td_ids = [qid for qid in QMAP if qid.startswith("test_drive")] + ["multi_hop_005"]
for cid in sorted(td_ids):
    q = QMAP[cid]
    qt = esc(q.get("question"))
    n, t = NOTHINK[cid], THINK[cid]
    nv, tv = n["verify"], t["verify"]
    lines.append(f"\n---\n\n## {cid}｜nothink score={nv['final_score']} ／ think score={tv['final_score']}\n")
    lines.append(f"- **题面**：{qt}\n")
    lines.append(f"- **accept_clarification**：{q.get('accept_clarification') or '无'}｜expected_tools={q.get('expected_tools')}\n")
    for mode, res in [("nothink", n), ("think", t)]:
        calls, final = read_traj(mode, cid)
        lines.append(f"\n### {mode} 轨迹（verify tags={res['verify']['violation_tags']} caps={res['verify']['caps_triggered']}）\n")
        for tool, args, obs in calls:
            if tool == "book_test_drive":
                lines.append(f"- **调用 {tool}**：\n")
                for k, v in args.items():
                    lines.append(f"    - `{k}` = {esc(v)}  →  {trace(v, qt)}\n")
                lines.append(f"    - 工具返回：{esc(obs)}\n")
            else:
                lines.append(f"- 调用 {tool}：{esc(json.dumps(args, ensure_ascii=False))[:150]}\n")
        lines.append(f"- 最终回复：{esc(final)[:400]}\n")
    lines.append("\n> 【审核结论】＿ 真违规／＿ 疑似误伤（相对时间推导）／＿ 混合 —— 一句话理由：\n")

out = Path("/Users/kay/Desktop/hello-agents-family/实验记录/BadCase分析记录/v1.2基线_test_drive系列_人工审核清单.md")
out.write_text("".join(lines), encoding="utf-8")
print("已生成:", out)
print("题数:", len(td_ids))
