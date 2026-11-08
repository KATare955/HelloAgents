# -*- coding: utf-8 -*-
"""v02 全量 1000 条批量生成入口（T6）

组成：dryrun 30 条（已审核，原样并入）+ b1b2 补充 470 条 + b3b4b5 500 条 = 1000 条
矩阵核对：
    桶1 250 = dryrun 20 + 试驾 extra 70 + 金融 80 + 跨场景 80
    桶2 250 = dryrun 10 + 试驾 extra 190 + 金融 30 + 竞品 20
    桶3 150（试驾 40/金融 45/竞品 35/跨场景 30）
    桶4 100（试驾 40/金融 20/车型参数 20/跨场景 20）
    桶5 250（试驾 80/车型参数 40/优惠 40/成交价 40/竞品 25/金融 25）

流程：构建骨架 → Claude 润色（缓存 key=sid 持久化 + 并发）→ build
→ 校验（零重叠 / loss_mask 覆盖率 / tool content JSON / 矩阵条数）→ 输出报告

注意：润色缓存按 sid 存。若模板题面/答案有改动，须删除 polish_cache.json 后重跑，
否则会命中旧模板的润色结果。

用法（在 HelloAgents 目录下）：
    .venv/bin/python eval/sft_data/v02_gen/run_batch.py [--workers 6] [--refresh-polish]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_core import (
    SFT_DATA_DIR, make_sample, polish_qa, check_overlap,
    load_eval_questions, render_tree,
)
from gen_templates_full_b1b2 import ALL_MAKERS as B1B2_MAKERS
from gen_templates_full_b3b4b5 import ALL_MAKERS as B3B4B5_MAKERS

OUT_JSONL = SFT_DATA_DIR / "v02_full_1000.jsonl"
OUT_VIEW = SFT_DATA_DIR / "v02_full_1000_可读视图.md"
DRYRUN_JSONL = SFT_DATA_DIR / "v02_dryrun_30.jsonl"
CACHE_PATH = Path(__file__).resolve().parent / "polish_cache.json"

EXPECTED_BUCKETS = {
    "bucket1_clarify": 250,
    "bucket2_correction": 250,
    "bucket3_multihop": 150,
    "bucket4_degrade": 100,
    "bucket5_coverage": 250,
}


def _cache_key(sid: str) -> str:
    """润色缓存按样本 sid 独立存：即使两条模板题面相同也各自润色，保证文字多样。"""
    return sid


def load_cache() -> Dict[str, Optional[List[str]]]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: Dict[str, Optional[List[str]]]) -> None:
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def polish_all(templates: List[Dict[str, Any]], cache: Dict[str, Optional[List[str]]],
               workers: int, refresh: bool) -> Tuple[int, int, int]:
    """并发润色全部模板；结果写入 cache（每 50 条落盘一次，中途崩溃可断点续跑）。"""
    to_polish = [
        (i, t) for i, t in enumerate(templates)
        if refresh or _cache_key(t["sid"]) not in cache
    ]
    cached = len(templates) - len(to_polish)
    print(f"润色任务: 总 {len(templates)} | 缓存命中 {cached} | 需调用 API {len(to_polish)}",
          flush=True)

    done = ok = 0
    if to_polish:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(polish_qa, t["question"], t["answer"]): i for i, t in to_polish}
            for fut in as_completed(futs):
                i = futs[fut]
                t = templates[i]
                try:
                    res = fut.result()
                except Exception:
                    res = None
                cache[_cache_key(t["sid"])] = list(res) if res else None
                done += 1
                ok += 1 if res else 0
                if done % 50 == 0:
                    save_cache(cache)
                    print(f"  ... {done}/{len(to_polish)}（成功 {ok}）", flush=True)
        save_cache(cache)
    return len(to_polish), ok, cached


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--refresh-polish", action="store_true")
    ap.add_argument("--no-polish", action="store_true",
                    help="跳过润色直接用程序模板（调试管线用）")
    args = ap.parse_args()

    rng = random.Random(20261107)

    # 1) 组装全部模板（970 条新生成 + 30 条已审核 dryrun 原样并入）
    templates: List[Dict[str, Any]] = []
    for maker in B1B2_MAKERS:
        templates.extend(maker(rng))
    for maker in B3B4B5_MAKERS:
        templates.extend(maker(rng))
    assert len(templates) == 970, f"模板数异常: {len(templates)}"

    dryrun_samples = [
        json.loads(line) for line in DRYRUN_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(dryrun_samples) == 30, f"dryrun 样本数异常: {len(dryrun_samples)}"

    # 模板题面唯一性（润色前保证）：模板间不重复 + 不与 dryrun 已审核题面撞
    _tpl_qs = [t["question"] for t in templates]
    _dup_tpl = sorted({q for q in _tpl_qs if _tpl_qs.count(q) > 1})
    _dry_qs = {s["question"] for s in dryrun_samples}
    _hit_dry = [t["sid"] for t in templates if t["question"] in _dry_qs]
    assert not _dup_tpl and not _hit_dry, \
        f"模板题面重复: {_dup_tpl[:3]}; 撞 dryrun 题面: {_hit_dry[:3]}"

    # 2) 润色（缓存 + 并发）
    cache = load_cache()
    if args.no_polish:
        n_api, n_ok, n_cached = 0, 0, 0
    else:
        n_api, n_ok, n_cached = polish_all(templates, cache, args.workers, args.refresh_polish)

    # 3) 组装样本（build 主线程执行，工具实跑；dryrun 30 条保持已审核版本原样）
    samples: List[Dict[str, Any]] = []
    fallback = 0
    # 全局题面唯一：以 dryrun 30 条已审核题面为种子；润色后题面撞车 → 回退程序模板原文
    seen_qs = {s["question"] for s in dryrun_samples}
    dup_q_fallback = 0
    dup_q_conflict: List[str] = []
    for t in templates:
        polished = cache.get(_cache_key(t["sid"]))
        if polished:
            q_final, a_final = polished[0], polished[1]
        else:
            q_final, a_final = t["question"], t["answer"]
            fallback += 1
        if q_final in seen_qs:
            q_final, a_final = t["question"], t["answer"]
            fallback += 1
            dup_q_fallback += 1
            if q_final in seen_qs:
                dup_q_conflict.append(t["sid"])
        seen_qs.add(q_final)
        messages = t["build"](q_final, a_final)
        samples.append(make_sample(
            sid=t["sid"], bucket=t["bucket"], scenario=t["scenario"],
            question=q_final, targets=t["targets"], is_clarify=t["is_clarify"],
            is_degrade=t["is_degrade"], messages=messages,
            design_intent=t["design_intent"], review_focus=t["review_focus"],
            stage="v0.2_批量1000条"))
    samples.extend(dryrun_samples)

    # 4) 校验
    eval_questions = load_eval_questions()
    overlap_hits = [s["id"] for s in samples if check_overlap(s["question"], eval_questions)]

    bucket_counts: Dict[str, int] = {}
    mask_segments = 0
    mask_samples = set()
    bad_tool = []
    ids: set = set()
    dup_ids = set()
    for s in samples:
        bucket_counts[s["bucket"]] = bucket_counts.get(s["bucket"], 0) + 1
        if s["id"] in ids:
            dup_ids.add(s["id"])
        ids.add(s["id"])
        for m in s["messages"]:
            if m.get("loss_mask") is False:
                mask_segments += 1
                mask_samples.add(s["id"])
            if m["role"] == "tool" and isinstance(m.get("content"), str) \
                    and m["content"].startswith("{"):
                try:
                    obj = json.loads(m["content"])
                    if obj.get("status") not in ("success", "partial", "error"):
                        bad_tool.append(s["id"])
                except json.JSONDecodeError:
                    bad_tool.append(s["id"])

    b2_count = bucket_counts.get("bucket2_correction", 0)
    matrix_ok = all(bucket_counts.get(k, 0) == v for k, v in EXPECTED_BUCKETS.items())
    mask_ok = (mask_segments == b2_count * 2
               and mask_samples == {s["id"] for s in samples if s["bucket"] == "bucket2_correction"})

    # 5) 输出
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    OUT_VIEW.write_text(render_tree(samples), encoding="utf-8")

    print("=" * 62)
    print(f"总样本: {len(samples)}（新生成 {len(samples) - 30} + dryrun 已审核 30）")
    for k, v in EXPECTED_BUCKETS.items():
        got = bucket_counts.get(k, 0)
        print(f"  {k}: {got}/{v} {'OK' if got == v else '!! MISMATCH'}")
    print(f"矩阵条数: {'OK' if matrix_ok else '!! MISMATCH'}")
    print(f"润色: API 调用 {n_api}（成功 {n_ok}）| 缓存命中 {n_cached} | 程序模板回退 {fallback}"
          f"（其中题面撞车回退 {dup_q_fallback}）")
    print(f"题面唯一: 冲突 {len(dup_q_conflict)} "
          f"{'OK' if not dup_q_conflict else dup_q_conflict[:5]}")
    print(f"评测集重叠: {len(overlap_hits)} {'OK' if not overlap_hits else overlap_hits[:5]}")
    print(f"loss_mask=false: 段数 {mask_segments}（期望 {b2_count * 2}）"
          f"| 覆盖样本 {len(mask_samples)}（期望 {b2_count}）| {'OK' if mask_ok else '!! MISMATCH'}")
    print(f"tool content 异常: {len(bad_tool)} {'OK' if not bad_tool else bad_tool[:5]}")
    print(f"sid 重复: {len(dup_ids)} {'OK' if not dup_ids else list(dup_ids)[:5]}")
    print(f"输出: {OUT_JSONL}")
    print(f"视图: {OUT_VIEW}")


if __name__ == "__main__":
    main()
