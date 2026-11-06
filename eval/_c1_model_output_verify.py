"""A4.1 落盘优化本地验证：单题跑出轨迹，检查 model_output 行（临时脚本，不入库）"""
import json, sys
from pathlib import Path

ROOT = Path("/Users/kay/Desktop/hello-agents-family/HelloAgents")
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT))
from baseline_eval import create_sales_agent, evaluate_single_question, _get_git_hash

qs = json.load(open(ROOT / "eval/eval_data/questions.json"))["questions"]
qmap = {q["id"]: q for q in qs}

traj_dir = ROOT / "eval/eval_data/_tmp_traj_a41"
traj_dir.mkdir(parents=True, exist_ok=True)

env_snapshot = {
    "tools_git_hash": _get_git_hash(),
    "judge_version": "v3",
    "model": "base-nothink-local",
    "llm_model_id": "qwen3:8b",
    "thinking_observed": "unknown",
    "temperature": 0,
    "eval_version": "v1.2",
}

agent = create_sales_agent()
for cid in ["test_drive_001", "test_drive_002"]:
    print(f"\n===== {cid} =====")
    result = evaluate_single_question(agent, qmap[cid], traj_dir, env_snapshot)
    print(f"status={result['status']} tool_correct={result.get('tool_correct')}")
    lines = [json.loads(l) for l in open(traj_dir / f"{cid}.jsonl")]
    kinds = [r.get("kind", "tool_row") for r in lines]
    print("行结构:", [(r.get("kind", "tool_row") if r.get("kind") else r.get("tool_name", "?")) for r in lines])
    mo = [r for r in lines if r.get("kind") == "model_output"]
    print(f"model_output 行数: {len(mo)}")
    for r in mo:
        print(f"  - turn={r['turn_index']} tool_count={r['tool_count']} 文本前 60 字: {r['content'][:60]!r}")
