"""A3 S2 校准回放（临时）：本地重跑 25 校准 case + 规则层回放判定。

口径（打样 D3 定稿）：
    - 正例 20 个：规则层命中 ≥1 个预期标签 → 捕获；一个都没命中 → 漏抓；
    - 负例 5 个：规则层输出任何标签 → 误伤（一票否决）；
    - "未再现"（v1.2 新体系下模型不再犯该违规）的 case 由人工判读轨迹后排除分母。
"""
import os
import sys
import json
from pathlib import Path

os.environ.setdefault("EVAL_STAGE", "calibration")  # 独立结果目录，不污染正式基线

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_eval import create_sales_agent, evaluate_single_question, load_questions, get_output_dir
from verifier.rules import verify_trajectory

# 25 校准 case 预期标签（来源：badcase_review_nothink 第四节 4.1/4.2）
EXPECTED = {
    # 正例 20
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
    # 负例 5
    "car_config_007": [],
    "compare_003": [],
    "compare_006": [],
    "compare_010": [],
    "multi_hop_004": [],
}


def main():
    questions = {q["id"]: q for q in load_questions()["questions"]}
    out_dir = get_output_dir()
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    agent = create_sales_agent()
    print(f"Agent 就绪：{agent.name}，工具数 {len(agent.tool_registry.list_tools())}")

    env_snapshot = {"eval_version": "v1.2", "judge_version": "v3", "note": "A3 校准回放"}

    report = []
    for i, case_id in enumerate(EXPECTED, 1):
        q = questions[case_id]
        print(f"\n[{i}/25] {case_id}（{q['type']}）{q['question'][:40]}")
        result = evaluate_single_question(agent, q, traj_dir, env_snapshot)
        response = result.get("response", "")

        # 读轨迹回放判定
        traj_file = traj_dir / f"{case_id}.jsonl"
        traj_rows = []
        if traj_file.exists():
            with open(traj_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        traj_rows.append(json.loads(line))
        verdict = verify_trajectory(q, traj_rows, response)

        expected = EXPECTED[case_id]
        actual = verdict["violation_tags"]
        is_positive = bool(expected)
        hit = sorted(set(actual) & set(expected))
        report.append({
            "case_id": case_id,
            "kind": "正例" if is_positive else "负例",
            "expected": expected,
            "actual": actual,
            "hit": hit,
            "tools": [r["tool_name"] for r in traj_rows if r.get("tool_name")],
            "final_score": verdict["final_score"],
            "caps_triggered": verdict["layer"]["rules"]["caps_triggered"],
            "response_head": response[:100],
        })
        status = "✅ 命中" if is_positive and hit else ("✅ 干净" if not is_positive and not actual else "❌ 漏抓" if is_positive else "❌ 误伤")
        print(f"   {status} expected={expected} actual={actual} score={verdict['final_score']}")

    # 汇总
    out_file = out_dir / "_a3_calibration_report.json"
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
    print("注：漏抓 case 需人工判读轨迹区分「未再现」与「真漏抓」")


if __name__ == "__main__":
    main()
