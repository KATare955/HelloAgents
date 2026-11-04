"""离线重判分脚本（不碰 GPU、不重新推理）

用途：
    1) 判分逻辑变更（如 accept_clarification / v3）时，直接读已保存结果 JSON
       （含 actual_tools / expected_tools / response）重算 tool_correct（judge_tool_correct 路径）
    2) verify 规则层离线重判（T4 新增）：读 trajectory jsonl 调 verify_trajectory，
       输出三维子项 + 7 cap + final_score + violation_tags
    3) 无 trajectory 的旧结果（如 v1.1）明确报"不可重判"而非误判

为什么不需要重跑推理：判分逻辑只影响"如何评判已产出的输出"，
不改变模型输入，故模型输出不变，可离线重算。

轨迹发现顺序：result["trajectory_ref"]（相对主 JSON 目录，A4 落盘）→
同目录 trajectories/{id}.jsonl → --traj-dir 显式覆盖。

用法：
    cd HelloAgents
    python eval/rescore.py eval/eval_data/results/baseline/qwen3-14b/v1.1/baseline_qwen3-14b_v1.1_20261101_140513.json

可选：--save 将重判分结果另存为 *_rescoreB.json（不覆盖原文件）；
      --traj-dir 目录  显式指定轨迹目录（冒烟验证用）。
"""

import json
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scoring import judge_tool_correct, JUDGE_VERSION  # noqa: E402
from verifier.rules import verify_trajectory  # noqa: E402

QUESTIONS_FILE = Path(__file__).parent / "eval_data" / "questions.json"


def load_clarification_map():
    """从评测集读取 {question_id: accept_clarification}"""
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return {q["id"]: q.get("accept_clarification") for q in data["questions"]}


def load_questions_map():
    """从评测集读取 {question_id: question}（verify_trajectory 需要 answer_spec 等完整字段）"""
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return {q["id"]: q for q in data["questions"]}


def load_traj_rows(traj_file: Path):
    """读 trajectory jsonl（口径与 _a3_rejudge.py 一致）：
    case_summary 行取 final_response，tool_name 行入 traj_rows。"""
    traj_rows = []
    response = ""
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
    return traj_rows, response


def find_traj_file(result_dir: Path, result: dict, traj_dir_override=None):
    """返回 (traj_file, note)。None + note 表示不可重判。"""
    case_id = result.get("id", "?")
    if traj_dir_override:
        f = Path(traj_dir_override) / f"{case_id}.jsonl"
        return (f, "") if f.exists() else (None, f"指定轨迹目录缺失 {f}")
    ref = result.get("trajectory_ref")
    if ref:
        f = result_dir / ref
        return (f, "") if f.exists() else (None, f"trajectory_ref 指向缺失 {ref}")
    f = result_dir / "trajectories" / f"{case_id}.jsonl"
    return (f, "") if f.exists() else (None, "无 trajectory（旧结果不可重判）")


def rescore(result_file: str, save: bool = False, traj_dir: str = None):
    with open(result_file, encoding="utf-8") as f:
        payload = json.load(f, object_pairs_hook=OrderedDict)

    clar_map = load_clarification_map()
    q_map = load_questions_map()
    result_dir = Path(result_file).parent
    results = payload["results"]

    old_correct = sum(1 for r in results if r.get("tool_correct"))
    flipped = []

    for r in results:
        # v1 路径：tool_correct 重算（judge_tool_correct，历史能力保留）
        old = r.get("tool_correct", False)
        new, missing, extra = judge_tool_correct(
            r.get("expected_tools", []),
            r.get("actual_tools", []),
            response=r.get("response", ""),
            accept_clarification=clar_map.get(r["id"]),
        )
        r["tool_correct"] = new
        if not new:
            r["missing_tools"] = missing
            r["extra_tools"] = extra
        else:
            r.pop("missing_tools", None)
            r.pop("extra_tools", None)
        if new != old:
            flipped.append((r["id"], old, new))

        # T4 路径：verify 规则层重判（读 trajectory；无轨迹 → 明确报不可重判）
        traj_file, note = find_traj_file(result_dir, r, traj_dir)
        if traj_file:
            traj_rows, response = load_traj_rows(traj_file)
            verdict = verify_trajectory(q_map.get(r["id"], {}), traj_rows, response)
            r["verify"] = {
                "rejudgeable": True,
                "final_score": verdict["final_score"],
                "violation_tags": verdict["violation_tags"],
                "sub_scores": verdict["sub_scores"],
                "caps_triggered": [k for k, v in verdict["caps"].items() if v["triggered"]],
            }
        else:
            r["verify"] = {"rejudgeable": False, "reason": note}

    total = len(results)
    new_correct = sum(1 for r in results if r.get("tool_correct"))

    # 重算场景 / 类型统计
    scenario_stats = {}
    type_stats = {}
    for r in results:
        sc = scenario_stats.setdefault(
            r["scenario"], {"total": 0, "success": 0, "tool_correct": 0, "errors": []}
        )
        sc["total"] += 1
        if r.get("status") == "success":
            sc["success"] += 1
        if r.get("tool_correct"):
            sc["tool_correct"] += 1
        else:
            sc["errors"].append(r["id"])

        ty = type_stats.setdefault(r["type"], {"total": 0, "tool_correct": 0})
        ty["total"] += 1
        if r.get("tool_correct"):
            ty["tool_correct"] += 1

    print("=" * 70)
    print(f"离线重判分（judge {JUDGE_VERSION} + verify 规则层）: {Path(result_file).name}")
    print("=" * 70)
    print(f"旧判分正确: {old_correct}/{total} = {old_correct/total*100:.1f}%")
    print(f"新判分正确: {new_correct}/{total} = {new_correct/total*100:.1f}%")
    print(f"翻转题数: {len(flipped)}")
    for qid, old, new in flipped:
        print(f"  {'✅' if new else '❌'} {qid}: {old} -> {new}")
    n_ok = sum(1 for r in results if r.get("verify", {}).get("rejudgeable"))
    n_bad = total - n_ok
    print(f"verify 规则层重判：可重判 {n_ok}/{total}；不可重判 {n_bad}（无 trajectory 旧结果）")
    print("-" * 70)
    print("场景准确率（新）:")
    for sc, st in scenario_stats.items():
        print(f"  {sc:<12} {st['tool_correct']}/{st['total']} = {st['tool_correct']/st['total']*100:.1f}%  errors={st['errors']}")
    print("类型准确率（新）:")
    for ty, st in type_stats.items():
        print(f"  {ty:<8} {st['tool_correct']}/{st['total']} = {st['tool_correct']/st['total']*100:.1f}%")

    # 更新 metadata 并重存
    payload["metadata"]["overall_accuracy"] = new_correct / total * 100
    payload["metadata"]["bad_cases_count"] = total - new_correct
    payload["metadata"]["rescore_note"] = f"判分逻辑 {JUDGE_VERSION} + verify 规则层离线重判分"
    payload["scenario_stats"] = scenario_stats
    payload["type_stats"] = type_stats

    if save:
        out = Path(result_file).with_name(
            Path(result_file).stem.replace(".json", "") + "_rescoreB.json"
        )
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n💾 已另存: {out}")

    # 重存 bad cases（新判分下的错误集）
    bad = [r for r in results if not r.get("tool_correct")]
    bad_file = Path(result_file).with_name(
        Path(result_file).stem.replace(".json", "").replace("_bad_cases", "") + "_bad_cases_rescoreB.json"
    )
    with open(bad_file, "w", encoding="utf-8") as f:
        json.dump(bad, f, ensure_ascii=False, indent=2)
    print(f"💾 新 bad cases（{len(bad)}个）: {bad_file}")

    return new_correct, total


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    save = "--save" in sys.argv
    traj_dir = None
    if "--traj-dir" in sys.argv:
        traj_dir = sys.argv[sys.argv.index("--traj-dir") + 1]
    path = [a for a in sys.argv[1:] if not a.startswith("--") and a != traj_dir][0]
    rescore(path, save=save, traj_dir=traj_dir)
