"""pass@8 前沿过滤（RL 数据规格书 §4 / DoD：通过率 20~80% 保留，≥150 条）

用即将进 RL 的起点模型（SFT4B，vLLM serve）对每条 RL prompt 采样 N 条、
按 reward 打分（训练同口径 w_acc=0.8 / w_proc=0），只保留通过率落在
[min_rate, max_rate] 的 prompt——全对/全错组组内零方差 → 零 advantage → 纯烧机时。
副产品：起点模型的 pass@k 基线（简历对比表一列）。

通过定义：r_gate == 1 且 r_acc == 1（首跑 λ=0 口径下等价于 r == 0.8）。

用法（GPU 日，SFT4B 已 serve）：
    python rl/pass_at_k_filter.py \
        --data rl/data/rl_prompts_v1_2026-11-21.jsonl \
        --n-samples 8 --temperature 0.7 \
        --llm-base-url http://127.0.0.1:8000/v1 --llm-model sft4b --llm-api-key EMPTY \
        --eval-today 2026-11-21 \
        --min-rate 0.2 --max-rate 0.8 --min-keep 150 \
        --out-filtered rl/data/rl_prompts_filtered.jsonl \
        --out-report rl/data/pass_at_k_sft4b.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rl.grpo_train import run_rollout

FULL_R = 0.8  # w_acc=0.8, w_proc=0, gate=1, r_acc=1 时的满分


def is_pass(reward: Dict[str, Any]) -> bool:
    return reward.get("r_gate") == 1 and reward.get("r_acc", 0) >= 0.9999


def select_frontier(
    rows: List[Dict[str, Any]],
    min_rate: float = 0.2,
    max_rate: float = 0.8,
    min_keep: int = 150,
) -> Dict[str, Any]:
    """纯函数：按通过率区间筛选。rows 每项需含 id/rate。"""
    kept, dropped = [], []
    for row in rows:
        rate = row["rate"]
        row["keep"] = min_rate <= rate <= max_rate
        (kept if row["keep"] else dropped).append(row)
    summary = {
        "n_total": len(rows),
        "n_keep": len(kept),
        "n_drop": len(dropped),
        "min_rate": min_rate,
        "max_rate": max_rate,
        "min_keep": min_keep,
        "ok": len(kept) >= min_keep,
        "kept_ids": [r["id"] for r in kept],
        "dropped_ids": [r["id"] for r in dropped],
    }
    if not summary["ok"]:
        summary["hint"] = (
            f"保留 {len(kept)} 条 < {min_keep}：先核 SFT4B 训练质量，"
            "仍不足则报告后放宽 max_rate（DoD 要求 ≥150）"
        )
    return summary


def _pass_at_k(flags: List[bool], k: int) -> bool:
    return any(flags[:k])


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="pass@N 前沿过滤 + 起点模型 pass@k 基线")
    ap.add_argument("--data", required=True, help="RL prompt jsonl（250 条候选）")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--llm-model", default=os.getenv("LLM_MODEL_ID", "sft4b"))
    ap.add_argument("--llm-api-key", default=os.getenv("LLM_API_KEY", "EMPTY"))
    ap.add_argument("--eval-today", default=os.getenv("EVAL_TODAY", datetime.now().strftime("%Y-%m-%d")))
    ap.add_argument("--min-rate", type=float, default=0.2)
    ap.add_argument("--max-rate", type=float, default=0.8)
    ap.add_argument("--min-keep", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8, help="每 prompt 采样并发数（run_rollout 每样本独立实例无共享状态，线程安全）")
    ap.add_argument("--save-traces", default=None, help="轨迹明细落盘目录（可选）：每 prompt 一个 {id}.jsonl，含 steps/reward 分解，供失败模式拆解")
    ap.add_argument("--out-filtered", required=True)
    ap.add_argument("--out-report", required=True)
    args = ap.parse_args(argv)

    traces_dir = Path(args.save_traces) if args.save_traces else None
    if traces_dir:
        traces_dir.mkdir(parents=True, exist_ok=True)

    prompts = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    weekday_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][
        datetime.strptime(args.eval_today, "%Y-%m-%d").weekday()
    ]
    llm_kwargs = dict(
        model=args.llm_model, api_key=args.llm_api_key,
        base_url=args.llm_base_url, temperature=args.temperature,
    )
    print(f"[filter] {len(prompts)} 条 × {args.n_samples} 采样 "
          f"temp={args.temperature} → {args.llm_base_url} ({args.llm_model})")

    rows: List[Dict[str, Any]] = []
    for i, p in enumerate(prompts, 1):
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            recs = list(ex.map(
                lambda s: run_rollout(p, llm_kwargs, args.eval_today, weekday_cn),
                range(args.n_samples),
            ))
        flags = [is_pass(rec["reward"]) for rec in recs]
        rewards = [rec["reward"]["r"] for rec in recs]
        n_pass = sum(flags)
        if traces_dir:
            # 轨迹明细落盘（§5.4 schema）：每行一条采样轨迹，供失败模式拆解
            with (traces_dir / f"{p.get('id', f'#{i}')}.jsonl").open("w", encoding="utf-8") as tf:
                for si, rec in enumerate(recs):
                    tf.write(json.dumps({
                        "prompt_id": p.get("id", f"#{i}"),
                        "bucket": p.get("bucket"),
                        "difficulty": p.get("difficulty"),
                        "sample_idx": si,
                        "temperature": args.temperature,
                        "reward": rec.get("reward"),
                        "steps": rec.get("steps", []),
                        "response": rec.get("response", ""),
                        "meta": rec.get("meta", {}),
                    }, ensure_ascii=False) + "\n")
        rows.append({
            "id": p.get("id", f"#{i}"),
            "difficulty": p.get("difficulty"),
            "n": args.n_samples,
            "n_pass": n_pass,
            "rate": round(n_pass / args.n_samples, 4),
            "rewards": rewards,
            "pass_at_k": {str(k): any(flags[:k]) for k in range(1, args.n_samples + 1)},
        })
        print(f"[filter] {i}/{len(prompts)} {rows[-1]['id']} "
              f"pass={n_pass}/{args.n_samples} rate={rows[-1]['rate']}")

    summary = select_frontier(rows, args.min_rate, args.max_rate, args.min_keep)
    kk = args.n_samples
    summary["pass_at_k_mean"] = {
        str(k): round(sum(r["pass_at_k"][str(k)] for r in rows) / len(rows), 4)
        for k in range(1, kk + 1)
    }
    summary["rate_distribution"] = {
        b: sum(1 for r in rows if r["rate"] == b) for b in sorted({r["rate"] for r in rows})
    }

    with Path(args.out_report).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path(args.out_report + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    kept_prompts = [p for r, p in zip(rows, prompts) if r["keep"]]
    with Path(args.out_filtered).open("w", encoding="utf-8") as f:
        for p in kept_prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"[filter] 保留 {summary['n_keep']}/{summary['n_total']} 条 → {args.out_filtered}")
    print(f"[filter] pass@1..{kk} 均值: {summary['pass_at_k_mean']}")
    if not summary["ok"]:
        print(f"[filter] FAIL: {summary['hint']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
