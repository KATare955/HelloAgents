"""probe 轨迹失败模式拆解（RL规划/分层训练与工具覆盖设计讨论_2026-11-21_v1.0.md §5.5）

输入：pass_at_k_filter.py --save-traces 落盘的轨迹明细目录 + 候选 prompt 集。
输出三个矩阵（控制台 + 可选 --out json）：
  A. 桶 × 失败模式：gate_hits 分布 / r_acc 失分子项 / 断链（调用数 < expected_tools 数）
  B. 工具 × 失败模式：args_source fabricated 命中率 / 失败后下一步行为（重试/换路/停下）
  C. 多跳专用：断链位置直方图 / 调用序列与 expected_tools 顺序比对的首处失配位置

用法：
    python rl/analyze_probe_traces.py \
        --traces rl/data/probe_traces \
        --data rl/data/rl_prompts_v1_2026-11-18.jsonl \
        [--out rl/data/probe_failure_matrix.json]

纯 stdlib，不依赖 hello_agents，可在无 GPU 环境离线跑。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 8 个业务工具白名单：steps 里混有模型输出/思考事件（Thought/Answer/Question/Finish 等），
# 统计前先过滤，否则失败模式矩阵被噪声行污染。
BUSINESS_TOOLS = {
    "text_search", "competitor_compare", "query_discount_policy",
    "book_test_drive", "query_transaction_price", "calculate_finance",
    "web_search", "follow_up_query",
}


def _clean_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [s for s in steps if (s.get("tool_name") or "") in BUSINESS_TOOLS]


def load_traces(traces_dir: Path) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for f in sorted(traces_dir.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                if "prompt_id" in d:  # 跳过同目录下混入的其他 jsonl（如 prompts 集）
                    d["steps"] = _clean_steps(d.get("steps", []))
                    recs.append(d)
    return recs


def load_labels(data_path: Path) -> Dict[str, Dict[str, Any]]:
    return {p["id"]: p for p in
            (json.loads(l) for l in data_path.read_text(encoding="utf-8").splitlines() if l.strip())}


# ---------------------------------------------------------------------------
# A. 桶 × 失败模式
# ---------------------------------------------------------------------------

def matrix_bucket(recs: List[Dict[str, Any]], labels: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in recs:
        by_bucket[r.get("bucket") or labels.get(r["prompt_id"], {}).get("bucket", "?" )].append(r)

    out: Dict[str, Any] = {}
    for bucket, rs in sorted(by_bucket.items()):
        n = len(rs)
        gate = Counter()
        tool_fail = Counter()      # missing_tools 计数
        facts_miss = 0             # 有 gt_facts 但 facts_matched < 1 的轨迹数
        facts_have = 0
        broken = 0                 # 调用数 < expected_tools 数（断链）
        calls_total = 0
        for r in rs:
            rw = r.get("reward") or {}
            for h in rw.get("gate_hits") or []:
                gate[h] += 1
            if not (rw.get("gate_hits")):
                gate["(无拦截)"] += 1
            acc = rw.get("acc") or {}
            if not acc.get("tool_correct"):
                for t in acc.get("missing_tools") or []:
                    tool_fail[t] += 1
                if not acc.get("missing_tools"):
                    tool_fail["(extra_only)"] += 1
            if "facts_matched" in acc:
                facts_have += 1
                if acc.get("facts_matched", 1.0) < 1.0:
                    facts_miss += 1
            exp = labels.get(r["prompt_id"], {}).get("expected_tools") or []
            n_calls = len(r.get("steps") or [])  # 已过滤噪声事件
            calls_total += n_calls
            if n_calls < len(exp):
                broken += 1
        out[bucket] = {
            "n_traj": n,
            "pass_rate": round(sum(1 for r in rs if (r.get("reward") or {}).get("r", 0) >= 0.7999) / n, 3),
            "gate_hits": dict(gate),
            "tool_miss": dict(tool_fail),
            "facts_miss_rate": round(facts_miss / facts_have, 3) if facts_have else None,
            "broken_rate": round(broken / n, 3),
            "avg_calls": round(calls_total / n, 2),
        }
    return out


# ---------------------------------------------------------------------------
# B. 工具 × 失败模式
# ---------------------------------------------------------------------------

def matrix_tool(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "n_calls": 0, "n_fab_params": 0, "fab_traj": set(),
        "after_fail": Counter(),
    })
    for r in recs:
        steps = r.get("steps") or []
        for i, s in enumerate(steps):
            t = s.get("tool_name") or "?"
            st = stats[t]
            st["n_calls"] += 1
            src = s.get("args_source") or {}
            for v in src.values():
                if v == "fabricated":
                    st["n_fab_params"] += 1
                    st["fab_traj"].add(r["prompt_id"] + "#" + str(r.get("sample_idx", "")))
            if (s.get("tool_status") or "") != "success":
                later = [x.get("tool_name") for x in steps[i + 1:]]
                if later and later[0] == t:
                    st["after_fail"]["retry"] += 1
                elif later:
                    st["after_fail"]["switch"] += 1
                else:
                    st["after_fail"]["stop"] += 1
    traj_with_tool: Dict[str, set] = defaultdict(set)
    for r in recs:
        for s in (r.get("steps") or []):
            traj_with_tool[s.get("tool_name") or "?"].add(r["prompt_id"] + "#" + str(r.get("sample_idx", "")))
    for t, st in stats.items():
        st["n_traj"] = len(traj_with_tool.get(t, set()))
        st["fab_traj_ratio"] = round(len(st["fab_traj"]) / st["n_traj"], 3) if st["n_traj"] else 0.0
        st["after_fail"] = dict(st["after_fail"])
        st.pop("fab_traj", None)
    return dict(stats)


# ---------------------------------------------------------------------------
# C. 多跳专用：断链位置 + 首处失配
# ---------------------------------------------------------------------------

def matrix_multihop(recs: List[Dict[str, Any]], labels: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    mh = [r for r in recs
          if (r.get("difficulty") or labels.get(r["prompt_id"], {}).get("difficulty", "")) and
          str(r.get("difficulty") or labels.get(r["prompt_id"], {}).get("difficulty", "")).startswith("multi_hop")]
    if not mh:
        return {"note": "无多跳轨迹"}
    n = len(mh)
    calls_hist = Counter()
    first_mismatch = Counter()   # 调用序列与 expected 顺序比对的首处失配位置（1-based）
    mismatch_total = 0
    for r in mh:
        exp = labels.get(r["prompt_id"], {}).get("expected_tools") or []
        calls = [s.get("tool_name") for s in (r.get("steps") or [])]
        calls_hist[len(calls)] += 1
        # 顺序比对：在 expected 序列里找第一个"调用序列与剩余期望不匹配"的位置
        mm = 0
        remaining = list(exp)
        for c in calls:
            if c in remaining:
                # 允许跳过前面的工具（宽松比对），但记录当前位置
                remaining = [x for x in remaining if x != c]
            else:
                mm = calls.index(c) + 1 if mm == 0 else mm
                mismatch_total += 1
        first_mismatch[mm] += 1
    return {
        "n_traj": n,
        "calls_hist": {str(k): v for k, v in sorted(calls_hist.items())},
        "first_mismatch": {str(k): v for k, v in sorted(first_mismatch.items())},
        "mismatch_total": mismatch_total,
        "expected_len_hist": Counter(str(len(labels[r["prompt_id"]].get("expected_tools") or []))
                                     for r in mh),
    }


# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="probe 轨迹失败模式拆解（三维矩阵）")
    ap.add_argument("--traces", required=True, help="--save-traces 落盘目录")
    ap.add_argument("--data", required=True, help="候选 prompt 集 jsonl（取 bucket/expected_tools 标签）")
    ap.add_argument("--out", default=None, help="可选：结果落盘 json")
    args = ap.parse_args(argv)

    recs = load_traces(Path(args.traces))
    labels = load_labels(Path(args.data))
    print(f"[analyze] {len(recs)} 条轨迹 / {len(labels)} 个 prompt 标签")

    a = matrix_bucket(recs, labels)
    b = matrix_tool(recs)
    c = matrix_multihop(recs, labels)

    print("\n=== A. 桶 × 失败模式 ===")
    for bucket, d in a.items():
        print(f"\n[{bucket}] n={d['n_traj']} pass={d['pass_rate']} "
              f"断链={d['broken_rate']} 均调用={d['avg_calls']}")
        print(f"  gate_hits: {d['gate_hits']}")
        print(f"  missing_tools: {d['tool_miss']}")
        print(f"  facts_miss_rate: {d['facts_miss_rate']}")

    print("\n=== B. 工具 × 失败模式 ===")
    for t, d in sorted(b.items()):
        print(f"[{t}] 调用={d['n_calls']} 轨迹={d['n_traj']} "
              f"fab轨迹比={d['fab_traj_ratio']} fab参数={d['n_fab_params']} "
              f"失败后={d['after_fail']}")

    print("\n=== C. 多跳专用 ===")
    for k, v in c.items():
        print(f"{k}: {v}")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"bucket": a, "tool": b, "multihop": c}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n[analyze] 结果已落盘 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
