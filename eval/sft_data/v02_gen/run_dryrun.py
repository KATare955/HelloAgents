# -*- coding: utf-8 -*-
"""v02 dry-run 入口：桶1×试驾 20 条 + 桶2×试驾 10 条

流程：构建骨架 → Claude 润色题面/答复（校验失败回退程序模板）→ 组装消息（工具实跑）
→ 零重叠校验 → 输出 v02_dryrun_30.jsonl + 树状可读视图 + 统计报告

用法（在 HelloAgents 目录下）：
    .venv/bin/python eval/sft_data/v02_gen/run_dryrun.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_core import (
    SFT_DATA_DIR, make_sample, polish_qa, check_overlap, load_eval_questions, render_tree,
)
from gen_templates_dryrun import make_all_dryrun_templates

OUT_JSONL = SFT_DATA_DIR / "v02_dryrun_30.jsonl"
OUT_VIEW = SFT_DATA_DIR / "v02_dryrun_30_可读视图.md"


def main() -> None:
    rng = random.Random(20261107)
    templates = make_all_dryrun_templates(rng)
    eval_questions = load_eval_questions()

    samples = []
    polish_ok = 0
    polish_fallback = 0
    overlap_hits = 0
    for tpl in templates:
        q, a = tpl["question"], tpl["answer"]
        polished = polish_qa(q, a)
        if polished is not None:
            q_final, a_final = polished
            polish_ok += 1
        else:
            q_final, a_final = q, a
            polish_fallback += 1
        messages = tpl["build"](q_final, a_final)
        if check_overlap(q_final, eval_questions):
            overlap_hits += 1
            print(f"[WARN] 与评测集重叠: {tpl['sid']} {q_final}")
        samples.append(make_sample(
            sid=tpl["sid"], bucket=tpl["bucket"], scenario=tpl["scenario"],
            question=q_final, targets=tpl["targets"], is_clarify=tpl["is_clarify"],
            is_degrade=tpl["is_degrade"], messages=messages,
            design_intent=tpl["design_intent"], review_focus=tpl["review_focus"],
            stage="v0.2_dryrun_30条"))

    # ---- 校验报告 ----
    mask_stats = {"bucket1": 0, "bucket2": 0, "bucket2_masked_segments": 0}
    for s in samples:
        for m in s["messages"]:
            if m.get("loss_mask") is False:
                mask_stats["bucket2_masked_segments"] += 1
    b1 = sum(1 for s in samples if s["bucket"] == "bucket1_clarify")
    b2 = sum(1 for s in samples if s["bucket"] == "bucket2_correction")
    mask_stats["bucket1"] = b1
    mask_stats["bucket2"] = b2

    # 实跑一致性抽样断言：每条样本的 tool content 必须是合法 JSON 且 status 合法
    bad_tool = 0
    for s in samples:
        for m in s["messages"]:
            if m["role"] == "tool" and m["content"].startswith("{"):
                try:
                    obj = json.loads(m["content"])
                    assert obj.get("status") in ("success", "partial", "error")
                except (json.JSONDecodeError, AssertionError):
                    bad_tool += 1
                    print(f"[WARN] tool content 异常: {s['id']} {m['content'][:80]}")

    # ---- 输出 ----
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    OUT_VIEW.write_text(render_tree(samples), encoding="utf-8")

    print("=" * 60)
    print(f"总样本: {len(samples)}（桶1 {b1} / 桶2 {b2}）")
    print(f"润色成功: {polish_ok} | 回退程序模板: {polish_fallback}")
    print(f"评测集重叠: {overlap_hits} | tool content 异常: {bad_tool}")
    print(f"loss_mask=false 段数（应=桶2条数×2=20）: {mask_stats['bucket2_masked_segments']}")
    print(f"输出: {OUT_JSONL}")
    print(f"视图: {OUT_VIEW}")


if __name__ == "__main__":
    main()
