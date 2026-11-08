#!/usr/bin/env python3
"""split_train_val.py — v02 1000 条按【桶×场景】格分层抽样划分 960/40

规则（《D1训练任务清单 3.2》，2026-11-08 用户已确认）：
  1. 每格保底抽 1 条（22 个非空格全覆盖）
  2. 剩余配额按"格样本量 × 4%"取整分配，不足 40 时从最大格依次补足
  3. 固定 seed=42，可复现
输出：
  train_ids.jsonl / val_ids.jsonl（每行 {id, bucket, scenario}，供审计与转换脚本消费）
  控制台审计报告（每格 train/val 数量 + 覆盖断言）
"""

import json
import random
from collections import defaultdict
from pathlib import Path

SFT_DIR = Path(__file__).resolve().parent
SRC = SFT_DIR.parent / "v02_full_1000.jsonl"
OUT_TRAIN = SFT_DIR / "train_ids.jsonl"
OUT_VAL = SFT_DIR / "val_ids.jsonl"
VAL_TOTAL = 40
SEED = 42


def main():
    records = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(records) == 1000, f"期望 1000 条，实读 {len(records)}"

    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        by_cell[(r["bucket"], r["scenario"])].append(r)
    cells = sorted(by_cell.keys())
    print(f"[in] {len(records)} 条 / {len(cells)} 个非空格")

    # 配额：保底 1 + 4% 取整，不足从大格补
    quota: dict[tuple[str, str], int] = {}
    for c in cells:
        n = len(by_cell[c])
        quota[c] = max(1, int(n * 0.04))
    deficit = VAL_TOTAL - sum(quota.values())
    assert deficit >= 0, f"保底已超 {VAL_TOTAL}"
    for c in sorted(cells, key=lambda c: -len(by_cell[c])):
        if deficit == 0:
            break
        if quota[c] < len(by_cell[c]):
            quota[c] += 1
            deficit -= 1
    assert sum(quota.values()) == VAL_TOTAL, f"配额合计 {sum(quota.values())} != {VAL_TOTAL}"

    rng = random.Random(SEED)
    train_rows, val_rows = [], []
    report = []
    for c in cells:
        recs = by_cell[c][:]
        rng.shuffle(recs)
        n_val = quota[c]
        val_rows += [{"id": r["id"], "bucket": r["bucket"], "scenario": r["scenario"]} for r in recs[:n_val]]
        train_rows += [{"id": r["id"], "bucket": r["bucket"], "scenario": r["scenario"]} for r in recs[n_val:]]
        report.append(f"  {c[0]} | {c[1]} | 格={len(recs):>3} | val={n_val:>2} | train={len(recs)-n_val:>3}")

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    OUT_TRAIN.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train_rows) + "\n", encoding="utf-8")
    OUT_VAL.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in val_rows) + "\n", encoding="utf-8")

    # 审计报告
    print(f"[out] train={len(train_rows)} → {OUT_TRAIN.name}")
    print(f"[out] val  ={len(val_rows)} → {OUT_VAL.name}")
    print("[dist] 每格分配：")
    print("\n".join(report))
    # 覆盖断言：22 格全部 ≥1
    val_cells = {(r["bucket"], r["scenario"]) for r in val_rows}
    missing = set(cells) - val_cells
    assert not missing, f"val 未覆盖格：{missing}"
    print(f"[assert] val 覆盖 {len(val_cells)}/{len(cells)} 格 ✓（每格 ≥1）")
    # id 无重复且无交叉
    t_ids = {r["id"] for r in train_rows}
    v_ids = {r["id"] for r in val_rows}
    assert len(t_ids) == 960 and len(v_ids) == 40
    assert not (t_ids & v_ids), "train/val id 交叉"
    print("[assert] id 无重复、train/val 无交叉 ✓")


if __name__ == "__main__":
    main()
