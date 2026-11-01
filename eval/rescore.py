"""离线重判分脚本（不碰 GPU、不重新推理）

用途：当**判分逻辑**变更（如 B 方案引入 accept_clarification）时，
直接读取已保存的评测结果 JSON（含 actual_tools / expected_tools / response），
用新判分逻辑重算 tool_correct 与各项统计，得到新基线。

为什么不需要重跑推理：判分逻辑只影响"如何评判已产出的输出"，
不改变模型输入，故模型输出不变，可离线重算。

用法：
    cd HelloAgents
    python eval/rescore.py eval/eval_data/results/baseline/qwen3-14b/v1.1/baseline_qwen3-14b_v1.1_20261101_140513.json

可选：加 --save 将重判分结果另存为 *_rescoreB.json（不覆盖原文件）。
"""

import json
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scoring import judge_tool_correct  # noqa: E402

QUESTIONS_FILE = Path(__file__).parent / "eval_data" / "questions.json"


def load_clarification_map():
    """从评测集读取 {question_id: accept_clarification}"""
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return {q["id"]: q.get("accept_clarification") for q in data["questions"]}


def rescore(result_file: str, save: bool = False):
    with open(result_file, encoding="utf-8") as f:
        payload = json.load(f, object_pairs_hook=OrderedDict)

    clar_map = load_clarification_map()
    results = payload["results"]

    old_correct = sum(1 for r in results if r.get("tool_correct"))
    flipped = []

    for r in results:
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

    # 输出对比
    print("=" * 70)
    print(f"离线重判分（B 方案 accept_clarification）: {Path(result_file).name}")
    print("=" * 70)
    print(f"旧判分正确: {old_correct}/{total} = {old_correct/total*100:.1f}%")
    print(f"新判分正确: {new_correct}/{total} = {new_correct/total*100:.1f}%")
    print(f"翻转题数: {len(flipped)}")
    for qid, old, new in flipped:
        print(f"  {'✅' if new else '❌'} {qid}: {old} -> {new}")
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
    payload["metadata"]["rescore_note"] = "判分逻辑 v2（accept_clarification）离线重判分"
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
    path = [a for a in sys.argv[1:] if not a.startswith("--")][0]
    rescore(path, save=save)
