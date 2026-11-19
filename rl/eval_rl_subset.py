"""RL 评测子集（15 题）评测 runner（三组对照共用，temp=0）

questions_v1.3_rl.json 是 RL prompt schema（expected_tools/gt_facts/fail_inject/
dep_edges/accept_clarification），不是 68 题的 answer_spec schema，不能走
baseline_eval——用训练同款 run_rollout + compute_reward 判分（同一把尺子：
r = r_gate × (0.8·r_acc + 0·r_proc)，fail_inject 按题面注入）。

用法（GPU 日，某组模型已 serve）：
    python rl/eval_rl_subset.py \
        --data eval/eval_data/questions_v1.3_rl.json \
        --llm-base-url http://127.0.0.1:8000/v1 --llm-model base4b --llm-api-key EMPTY \
        --eval-today 2026-11-19 --temperature 0 \
        --out rl/data/eval_subset_base4b_2026-11-19.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rl.grpo_train import run_rollout
from rl.pass_at_k_filter import is_pass


def summarize_subset(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """纯函数：子集汇总口径（pass = gate==1 ∧ r_acc==1）。"""
    n = len(rows)
    if n == 0:
        return {"n": 0, "pass_rate": 0.0, "mean_r": 0.0, "mean_r_acc": 0.0,
                "gate_zero_rate": 0.0, "truncated_rate": 0.0}
    n_pass = sum(1 for r in rows if r["pass"])
    return {
        "n": n,
        "n_pass": n_pass,
        "pass_rate": round(n_pass / n, 4),
        "mean_r": round(sum(r["reward"]["r"] for r in rows) / n, 4),
        "mean_r_acc": round(sum(r["reward"]["r_acc"] for r in rows) / n, 4),
        "gate_zero_rate": round(sum(1 for r in rows if r["reward"]["r_gate"] == 0) / n, 4),
        "truncated_rate": round(sum(1 for r in rows if r["meta"]["truncated"]) / n, 4),
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RL 子集 15 题评测（reward 口径）")
    ap.add_argument("--data", default="eval/eval_data/questions_v1.3_rl.json")
    ap.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--llm-model", default=os.getenv("LLM_MODEL_ID", "base4b"))
    ap.add_argument("--llm-api-key", default=os.getenv("LLM_API_KEY", "EMPTY"))
    ap.add_argument("--eval-today", default=os.getenv("EVAL_TODAY", datetime.now().strftime("%Y-%m-%d")))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    questions = data["questions"] if isinstance(data, dict) and "questions" in data else data
    weekday_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][
        datetime.strptime(args.eval_today, "%Y-%m-%d").weekday()
    ]
    llm_kwargs = dict(
        model=args.llm_model, api_key=args.llm_api_key,
        base_url=args.llm_base_url, temperature=args.temperature,
    )
    print(f"[subset] {len(questions)} 题 temp={args.temperature} → {args.llm_base_url} ({args.llm_model})")

    rows: List[Dict[str, Any]] = []
    for i, q in enumerate(questions, 1):
        rec = run_rollout(q, llm_kwargs, args.eval_today, weekday_cn)
        rw = rec["reward"]
        rows.append({
            "id": q.get("id", f"#{i}"),
            "scenario": q.get("scenario"),
            "difficulty": q.get("difficulty"),
            "reward": rw,
            "pass": is_pass(rw),
            "meta": rec["meta"],
        })
        print(f"[subset] {i}/{len(questions)} {rows[-1]['id']} "
              f"r={rw['r']} gate={rw['r_gate']} r_acc={rw['r_acc']} pass={rows[-1]['pass']}")

    summary = summarize_subset(rows)
    with Path(args.out).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path(args.out + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[subset] 汇总: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
