"""Reward 离线回归三断言（Reward 规格书 §8；上机前必过，本地 0 元）

数据源：
    G1 SFT golden 1000 条（eval/sft_data/v02_full_1000.jsonl，messages 格式）
    G2 SFT-v02 评测轨迹 68 条（A4 格式，63/68 正确口径，附加证据组）
    F  Baseline think + nothink 评测轨迹 136 条（A4 格式，含失败样本）

三断言：
    ① golden 得分显著高于失败轨迹（reward 区分度，diff > 0.5）
    ② 编造轨迹（TRACE_ARGS fabricated / 预约年份幻觉）gate=0，且样本存在
    ③ golden gate 误杀率 < 5%（规格书：失败恢复题单独统计允许更宽——本数据源
       无 fail_inject 注入题，该项在 RL prompt 生成后复跑）

附加证据（不阻断）：r_acc 与 verify final_score 的 Pearson 相关性（训练尺与
评测尺对齐度，简历素材）。

用法：
    cd HelloAgents && uv run python rl/offline_regression.py
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.verifier.rules import verify_trajectory  # noqa: E402
from rl.reward import compute_reward, from_a4_rows, from_sft_messages  # noqa: E402

TODAY = datetime.now().strftime("%Y-%m-%d")
REPORT_PATH = project_root / "rl" / f"regression_report_{TODAY}.json"
SAMPLE_PATH = project_root / "rl" / f"regression_per_sample_{TODAY}.jsonl"

G1_PATH = project_root / "eval/sft_data/v02_full_1000.jsonl"
G2_DIR = (
    project_root
    / "eval/eval_data/results/sft/qwen3-14b-sales-sft_v02/v1.2/trajectories"
)
F_DIRS = [
    ("F_think",
     project_root / "eval/eval_data/results/baseline/qwen3-14b-think/v1.2/trajectories"),
    ("F_nothink",
     project_root / "eval/eval_data/results/baseline/qwen3-14b-nothink/v1.2/trajectories"),
]
QUESTIONS_PATH = project_root / "eval/eval_data/questions.json"

DIFF_THRESHOLD = 0.5    # 断言①：golden 均值 − 失败均值
MISKILL_THRESHOLD = 0.05  # 断言③：golden gate 误杀率


def load_questions(path: Path) -> Dict[str, Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {q["id"]: q for q in data["questions"]}


def iter_a4_cases(traj_dir: Path):
    """逐 case 产出 (case_id, rows)；截断行跳过并计数（生产轨迹稳健解析）。"""
    broken = 0
    for p in sorted(traj_dir.glob("*.jsonl")):
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                broken += 1
        yield p.stem, rows
    if broken:
        print(f"[warn] {traj_dir.name}: {broken} 行 JSON 截断跳过")


def pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx * vy else float("nan")


def group_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    if not n:
        return {"n": 0}
    rs = [r["r"] for r in records]
    return {
        "n": n,
        "mean_r": round(sum(rs) / n, 4),
        "mean_r_acc": round(sum(r["r_acc"] for r in records) / n, 4),
        "gate0_rate": round(sum(1 for r in records if r["r_gate"] == 0) / n, 4),
        "tool_correct_rate": round(
            sum(1 for r in records if r["tool_correct"]) / n, 4
        ),
    }


def main() -> None:
    questions = load_questions(QUESTIONS_PATH)
    records: List[Dict[str, Any]] = []
    sample_out = open(SAMPLE_PATH, "w", encoding="utf-8")

    def add_record(group, cid, label, steps, resp, meta, extra=None):
        year = None
        snap = meta.get("env_snapshot") or {}
        if snap.get("eval_today"):
            year = str(snap["eval_today"])[:4]
        out = compute_reward(
            label, steps, resp,
            eval_year=year,
            require_finish=(group == "G1"),
        )
        rec = {
            "group": group, "id": cid,
            "question": str(label.get("question") or "")[:40],
            "r": out["r"], "r_gate": out["r_gate"], "gate_hits": out["gate_hits"],
            "r_acc": out["r_acc"], "tool_correct": out["acc"]["tool_correct"],
            "fabrication_hit": out["fabrication"]["hit"],
        }
        if extra:
            rec.update(extra)
        records.append(rec)
        sample_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return out

    # ---- G1: SFT golden 1000 ----
    g1_gate_reasons: Dict[str, int] = {}
    with open(G1_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            steps, resp, meta = from_sft_messages(d["messages"])
            label = {
                "question": meta.get("question_text") or d["question"],
                "expected_tools": d.get("expected_tools") or [],
            }
            out = add_record("G1", d["id"], label, steps, resp, meta,
                             extra={"bucket": d.get("bucket", "?")})
            for h in out["gate_hits"]:
                g1_gate_reasons[h] = g1_gate_reasons.get(h, 0) + 1

    # ---- G2 + F: A4 评测轨迹 ----
    verify_align: List[Tuple[float, float]] = []
    fab_cases: List[Tuple[str, str, str]] = []
    for group, d in [("G2", G2_DIR), *F_DIRS]:
        for cid, rows in iter_a4_cases(d):
            q = questions.get(cid)
            if q is None:
                print(f"[warn] {cid} 不在 questions.json，跳过")
                continue
            steps, resp, meta = from_a4_rows(rows)
            label = {
                "question": q["question"],
                "expected_tools": q.get("expected_tools") or [],
                "accept_clarification": q.get("accept_clarification"),
            }
            out = add_record(group, cid, label, steps, resp, meta)
            if out["fabrication"]["hit"]:
                fab_cases.append((group, cid, out["fabrication"]["hit"]))
            v = verify_trajectory(q, rows, resp)
            verify_align.append((out["r_acc"], v["final_score"]))

    sample_out.close()

    # ---- 断言计算 ----
    g1 = [r for r in records if r["group"] == "G1"]
    g2 = [r for r in records if r["group"] == "G2"]
    f_records = [r for r in records if r["group"].startswith("F")]
    f_fail = [r for r in f_records if not r["tool_correct"]]
    f_correct = [r for r in f_records if r["tool_correct"]]

    # 纠错桶轨迹有意嵌入负面轮（首轮编造→用户批评→正确澄清），gate 命中该轮属
    # 真阳性而非误杀——按规格书"失败恢复题单独统计，允许更宽"口径拆开统计。
    g1_corr = [r for r in g1 if r.get("bucket") == "bucket2_correction"]
    g1_noncorr = [r for r in g1 if r.get("bucket") != "bucket2_correction"]

    golden_mean = sum(r["r"] for r in g1) / len(g1)
    fail_mean = sum(r["r"] for r in f_fail) / len(f_fail) if f_fail else float("nan")
    diff = golden_mean - fail_mean

    n_gate0_noncorr = sum(1 for r in g1_noncorr if r["r_gate"] == 0)
    miskill_rate = n_gate0_noncorr / len(g1_noncorr)
    corr_gate_rate = (
        sum(1 for r in g1_corr if r["r_gate"] == 0) / len(g1_corr)
        if g1_corr else 0.0
    )

    rec_by_key = {(r["group"], r["id"]): r for r in records}
    fab_gated = all(
        rec_by_key.get((g, c)) is not None and rec_by_key[(g, c)]["r_gate"] == 0
        for g, c, _ in fab_cases
    )

    a1 = {"golden_mean_r": round(golden_mean, 4),
          "failure_mean_r": round(fail_mean, 4),
          "diff": round(diff, 4), "threshold": DIFF_THRESHOLD,
          "passed": bool(f_fail) and diff > DIFF_THRESHOLD}
    a2 = {"n_fabricated_cases": len(fab_cases),
          "cases": [{"group": g, "id": c, "reason": h} for g, c, h in fab_cases],
          "all_gated": fab_gated,
          "passed": bool(fab_cases) and fab_gated}
    a3 = {"n_gate0": n_gate0_noncorr, "total": len(g1_noncorr),
          "rate": round(miskill_rate, 4), "threshold": MISKILL_THRESHOLD,
          "gate_reasons": g1_gate_reasons,
          "note": "非纠错桶口径；纠错桶嵌入负面轮为真阳性，单独统计",
          "correction_bucket_gate_rate": round(corr_gate_rate, 4),
          "passed": miskill_rate < MISKILL_THRESHOLD}

    corr = pearson([x for x, _ in verify_align], [y for _, y in verify_align])

    report = {
        "date": TODAY,
        "config": {"w_acc": 0.8, "w_proc": 0.0, "recover_bonus": 0.15},
        "groups": {
            "G1_noncorr_750": group_stats(g1_noncorr),
            "G1_corr_250": group_stats(g1_corr),
            "G2_sft_v02_eval_68": group_stats(g2),
            "F_baseline_136": group_stats(f_records),
            "F_fail_subset": group_stats(f_fail),
            "F_correct_subset": group_stats(f_correct),
        },
        "verify_alignment": {
            "n": len(verify_align),
            "pearson_r(r_acc, verify_final_score)": round(corr, 4),
        },
        "assertions": {"1_distinguish": a1, "2_fabrication_gated": a2,
                       "3_gate_miskill": a3},
        "per_sample_file": str(SAMPLE_PATH),
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- 输出 ----
    print("=" * 72)
    print("Reward 离线回归报告（Reward 规格书 §8 三断言）")
    print("=" * 72)
    for name, g in report["groups"].items():
        if g.get("n"):
            print(f"{name:<22} n={g['n']:<5} mean_r={g['mean_r']:<7} "
                  f"mean_r_acc={g['mean_r_acc']:<7} gate0={g['gate0_rate']:<7} "
                  f"tool_correct={g['tool_correct_rate']}")
    print("-" * 72)
    print(f"断言① 区分度：golden {a1['golden_mean_r']} vs 失败 {a1['failure_mean_r']} "
          f"(diff={a1['diff']}, 阈值 {DIFF_THRESHOLD}) -> "
          f"{'PASS' if a1['passed'] else 'FAIL'}")
    print(f"断言② 编造清零：发现 {a2['n_fabricated_cases']} 条编造轨迹，全部 gate=0？"
          f" {a2['all_gated']} -> {'PASS' if a2['passed'] else 'FAIL'}")
    for c in a2["cases"]:
        print(f"    - {c['id']}: {c['reason']}")
    print(f"断言③ gate 误杀（非纠错桶 {a3['total']} 条）：{a3['n_gate0']}/{a3['total']} "
          f"({a3['rate']:.2%}, 阈值 {MISKILL_THRESHOLD}) -> "
          f"{'PASS' if a3['passed'] else 'FAIL'}")
    print(f"    纠错桶（嵌入负面轮，真阳性口径）：gate 命中率 "
          f"{a3['correction_bucket_gate_rate']:.2%}")
    if g1_gate_reasons:
        print(f"    golden gate 触发原因分布：{g1_gate_reasons}")
    print(f"对齐度：r_acc 与 verify final_score Pearson = "
          f"{report['verify_alignment']['pearson_r(r_acc, verify_final_score)']} "
          f"(n={report['verify_alignment']['n']})")
    print("-" * 72)
    all_pass = a1["passed"] and a2["passed"] and a3["passed"]
    print(f"总体：{'ALL PASS' if all_pass else 'HAS FAILURE'}")
    print(f"报告已存：{REPORT_PATH}")
    print(f"逐样本已存：{SAMPLE_PATH}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
