"""四组 4B 模型 verify 规则层离线重判对比（阶段二 RL 叙事补充，2026-11-19）

用法：
    cd HelloAgents
    python eval/_a4_verify_4b_compare.py

说明：
    - 不碰 GPU：读已落盘轨迹（trajectories/{id}.jsonl）调用 verify_trajectory 离线重判
    - 口径与 rescore.py 一致（case_summary 行取 final_response，tool_name 行入 traj_rows）
    - 输出：总表 / 三维子项 / 分场景 / cap 频率 + 明细 JSONL
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rescore import load_traj_rows  # noqa: E402
from verifier.rules import verify_trajectory  # noqa: E402

BASE = Path(__file__).parent / "eval_data" / "results" / "baseline"
QUESTIONS = Path(__file__).parent / "eval_data" / "questions.json"
MODELS = ["qwen3-4b-base", "qwen3-4b-sft", "qwen3-4b-rl", "qwen3-4b-rl80"]
OUT = Path(__file__).parent / "eval_data" / "verify_4b_compare_2026-11-19.jsonl"


def load_q_map():
    with open(QUESTIONS, encoding="utf-8") as f:
        data = json.load(f)
    return {q["id"]: q for q in data["questions"]}


def rescore_one_model(model, q_map):
    main_json = BASE / model / "v1.2" / f"baseline_{model}_v1.2_20261119.json"
    with open(main_json, encoding="utf-8") as f:
        payload = json.load(f)
    per_case = {}
    for r in payload["results"]:
        cid = r["id"]
        traj_file = BASE / model / "v1.2" / "trajectories" / f"{cid}.jsonl"
        if not traj_file.exists():
            per_case[cid] = None
            continue
        traj_rows, response = load_traj_rows(traj_file)
        verdict = verify_trajectory(q_map.get(cid, {}), traj_rows, response)
        per_case[cid] = {
            "scenario": r.get("scenario", "?"),
            "type": r.get("type", "?"),
            "final_score": verdict["final_score"],
            "weighted": verdict["weighted_score"],
            "tool_call": verdict["sub_scores"]["tool_call"],
            "evidence": verdict["sub_scores"]["evidence"],
            "caps_triggered": [n for n in verdict["caps"] if verdict["caps"][n]["triggered"]],
            "violation_tags": verdict["violation_tags"],
        }
    return per_case


def main():
    q_map = load_q_map()
    all_cases = defaultdict(dict)
    for m in MODELS:
        per = rescore_one_model(m, q_map)
        for cid, v in per.items():
            all_cases[cid][m] = v

    n = len(q_map)
    print("=" * 78)
    print(f"四组 4B 模型 verify 规则层离线重判（{n} 题，questions v1.2.1）")
    print("=" * 78)

    # 总表
    print("\n【总表】")
    print(f"{'模型':<18}{'mean_final':>12}{'满分(1.0)':>10}{'>=0.8':>8}{'<0.5':>8}{'mean_tool_call':>16}{'mean_evidence':>15}")
    stats = {}
    for m in MODELS:
        vals = [all_cases[c][m]["final_score"] for c in q_map if all_cases[c].get(m)]
        tc = [all_cases[c][m]["tool_call"] for c in q_map if all_cases[c].get(m)]
        ev = [all_cases[c][m]["evidence"] for c in q_map if all_cases[c].get(m)]
        stats[m] = {
            "mean_final": sum(vals) / len(vals),
            "full": sum(1 for v in vals if v >= 0.999),
            "ge08": sum(1 for v in vals if v >= 0.8),
            "lt05": sum(1 for v in vals if v < 0.5),
            "mean_tc": sum(tc) / len(tc),
            "mean_ev": sum(ev) / len(ev),
        }
        print(f"{m:<18}{stats[m]['mean_final']:>12.4f}{stats[m]['full']:>8}/{n}{stats[m]['ge08']:>6}/{n}{stats[m]['lt05']:>6}{stats[m]['mean_tc']:>16.4f}{stats[m]['mean_ev']:>15.4f}")

    # 分场景
    print("\n【分场景 mean_final】")
    scenarios = sorted({all_cases[c][m]["scenario"] for c in q_map for m in MODELS if all_cases[c].get(m)})
    hdr = f"{'场景':<14}" + "".join(f"{m.replace('qwen3-4b-', ''):>10}" for m in MODELS)
    print(hdr)
    for sc in scenarios:
        row = f"{sc:<14}"
        for m in MODELS:
            vals = [all_cases[c][m]["final_score"] for c in q_map
                    if all_cases[c].get(m) and all_cases[c][m]["scenario"] == sc]
            row += f"{sum(vals) / len(vals):>10.3f}" if vals else f"{'-':>10}"
        print(row)

    # cap 触发频率
    print("\n【7 个 hard cap 触发题数】")
    cap_names = ["hallucinated_args_cap", "false_promise_cap", "missing_evidence_cap",
                 "wrong_tool_cap", "no_side_effect_cap", "format_leak_cap", "tool_id_exposure_cap"]
    hdr = f"{'cap':<24}" + "".join(f"{m.replace('qwen3-4b-', ''):>8}" for m in MODELS)
    print(hdr)
    for cap in cap_names:
        row = f"{cap:<24}"
        for m in MODELS:
            cnt = sum(1 for c in q_map if all_cases[c].get(m) and cap in all_cases[c][m]["caps_triggered"])
            row += f"{cnt:>8}"
        print(row)

    # 明细落盘
    with open(OUT, "w", encoding="utf-8") as f:
        for cid in sorted(all_cases):
            row = {"case_id": cid}
            for m in MODELS:
                v = all_cases[cid].get(m)
                row[m] = None if v is None else {
                    "final_score": v["final_score"], "scenario": v["scenario"],
                    "violation_tags": v["violation_tags"],
                }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n明细已落盘: {OUT}")


if __name__ == "__main__":
    main()
