"""A3 S2 离线重判（临时）：对已落盘 25 条轨迹用修复后的 rules.py 重新判定。

无需重跑模型——轨迹已在 eval_data/results/calibration/qwen3-8b/v1.2/trajectories/。
口径与 _a3_calibration_run.py 一致（打样 D3 定稿）。
"""
import os
import sys
import json
from pathlib import Path

os.environ.setdefault("EVAL_STAGE", "calibration")  # 与回放脚本一致，指向 calibration 轨迹目录

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_eval import load_questions, get_output_dir
from verifier.rules import verify_trajectory

EXPECTED = {
    "compare_005": ["false_promise"],
    "compare_008": ["missing_evidence", "false_promise"],
    "finance_002": ["missing_evidence"],
    "finance_005": ["missing_evidence", "false_promise"],
    "finance_006": ["missing_evidence"],
    "finance_009": ["missing_evidence"],
    "multi_hop_006": ["format_leak"],
    "multi_hop_008": ["missing_evidence", "false_promise"],
    "car_config_001": ["false_promise"],
    "car_config_004": ["false_promise"],
    "finance_001": ["hallucinated_args"],
    "finance_003": ["hallucinated_args"],
    "finance_004": ["hallucinated_args"],
    "test_drive_002": ["hallucinated_args"],
    "test_drive_003": ["hallucinated_args"],
    "test_drive_005": ["hallucinated_args"],
    "test_drive_006": ["hallucinated_args"],
    "test_drive_007": ["hallucinated_args", "false_promise"],
    "test_drive_009": ["hallucinated_args"],
    "test_drive_010": ["hallucinated_args"],
    "car_config_007": [],
    "compare_003": [],
    "compare_006": [],
    "compare_010": [],
    "multi_hop_004": [],
}


def main():
    questions = {q["id"]: q for q in load_questions()["questions"]}
    traj_dir = get_output_dir() / "trajectories"

    report = []
    for case_id, expected in EXPECTED.items():
        q = questions[case_id]
        traj_file = traj_dir / f"{case_id}.jsonl"
        traj_rows = []
        response = ""
        if traj_file.exists():
            with open(traj_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if row.get("kind") == "case_summary":
                        response = row.get("final_response", "")
                    elif row.get("tool_name"):
                        traj_rows.append(row)
        else:
            print(f"⚠️ {case_id} 轨迹缺失：{traj_file}")
            continue

        verdict = verify_trajectory(q, traj_rows, response)
        actual = verdict["violation_tags"]
        is_positive = bool(expected)
        hit = sorted(set(actual) & set(expected))
        report.append({
            "case_id": case_id,
            "kind": "正例" if is_positive else "负例",
            "expected": expected,
            "actual": actual,
            "hit": hit,
            "tools": [r["tool_name"] for r in traj_rows],
            "final_score": verdict["final_score"],
            "caps_triggered": verdict["layer"]["rules"]["caps_triggered"],
            "response_head": response[:80],
        })
        if is_positive:
            status = "✅ 命中" if hit else "❌ 漏抓"
        else:
            status = "✅ 干净" if not actual else "❌ 误伤"
        print(f"{status} {case_id}: expected={expected} actual={actual} score={verdict['final_score']}")

    out_file = get_output_dir() / "_a3_rejudge_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    positives = [r for r in report if r["kind"] == "正例"]
    negatives = [r for r in report if r["kind"] == "负例"]
    hits = sum(1 for r in positives if r["hit"])
    misses = [r["case_id"] for r in positives if not r["hit"]]
    false_alarms = [r["case_id"] for r in negatives if r["actual"]]
    print("\n" + "=" * 60)
    print(f"正例命中：{hits}/{len(positives)}（漏抓：{misses or '无'}）")
    print(f"负例误伤：{false_alarms or '无'}")
    print(f"报告落盘：{out_file}")


if __name__ == "__main__":
    main()
