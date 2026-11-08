# -*- coding: utf-8 -*-
"""T7 全量清洗验收（03 v0.2 的 2.4 红线 + 05 文档第五节 mask 覆盖率）

验收项：
A1. 润色零注入（主校验，管线硬性红线）：最终 question/Finish.answer vs 模板原文
    数字 multiset 相等 + 实体只减不增 + 长度≥50%
A2. 可溯源率（参考指标）：计入 loss 教师段数字能否从 system/user/tool 段扣除
B. 与评测集零重叠（0 题面重合）+ 相似度初筛 top20 供人工复核
C. clarify 桶配平 ≥1:1（2.5 节场景级计数：试驾 1.05:1、金融执行主导）
D. 正例工具链完整 100%（执行类正例 tool_call 序列 ⊇ targets）
E. final_answer 接地 100%（推荐替代车型/门店 ∈ 接地表白名单）
F. mask 覆盖率（05 文档第五节：桶2 每条恰 2 段、其余桶 0 段、false 段只出现在 assistant）
G. 结构卫生（sid 唯一 / 消息角色序列 / tool content JSON 合法）

用法（在 HelloAgents 目录下）：
    .venv/bin/python eval/sft_data/v02_gen/verify_t7.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_core import (  # noqa: E402
    BASE_DATE_POOL, CONFIGS, CUSTOMER_NAMES, KNOWN_ENTITIES, SFT_DATA_DIR,
    check_overlap, entities_of, load_eval_questions, numbers_of, validate_polish,
)
import gen_templates_full_b1b2  # noqa: E402
import gen_templates_full_b3b4b5  # noqa: E402

DATA = SFT_DATA_DIR / "v02_full_1000.jsonl"
REPORT = SFT_DATA_DIR / "v02_T7_验收报告.md"

# ---------- A 项：参数数字检测（带业务单位的数字才视为"参数数字"） ----------
_UNITS = "元|%|年|天|点|月|日|度|km|公里|米|升|L|万|期|kw|kwh|kg|秒"
_UNIT_RE = re.compile(r"(\d{1,3}(?:,\d{3})*|\d+(?:\.\d+)?)\s*(" + _UNITS + r")", re.I)
_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})")  # 只验收小时，分钟由程序保证整点/半点
_PHONE_RE = re.compile(r"1[3-9]\d{9}")
# 话术常数豁免：营业时间整段（系统级事实，非业务参数）
_BUSINESS_HOURS_RE = re.compile(r"营业时间是?\s*9:00-18:00")
# Thought 段日期换算豁免（基准日由程序注入，模板级已验证自洽）
_DATE_TALK_RE = re.compile(r"\d{4}年\d{1,2}月\d{1,2}日")


def _msg_texts(m: Dict[str, Any]) -> List[str]:
    """一条消息的全部文本：tool_calls 的 name+arguments，或 content。"""
    if m.get("tool_calls"):
        out = []
        for c in m["tool_calls"]:
            fn = c["function"]
            out.append(fn["name"])
            out.append(fn.get("arguments", "") or "")
        return out
    return [m.get("content") or ""]


def _tool_name(m: Dict[str, Any]) -> str:
    if m.get("tool_calls"):
        return m["tool_calls"][0]["function"]["name"]
    return ""


def _load() -> List[Dict[str, Any]]:
    return [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]


# ============================================================
# A1. 润色零注入（主校验，管线硬性红线）
# 最终 question / Finish.answer 与模板原文比对：数字 multiset 相等 + 实体只减不增
# + 长度 ≥50%（validate_polish 口径）。回退程序模板的样本（q/a 全等）必然通过。
# dryrun 30 条为已审核历史版本（旧模板+润色），不适用此校验。
# ============================================================
def check_polish_no_inject(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    import random
    rng = random.Random(20261107)  # 与 run_batch.py 同 seed
    tpls: Dict[str, Tuple[str, str]] = {}
    for maker in gen_templates_full_b1b2.ALL_MAKERS:
        for t in maker(rng):
            tpls[t["sid"]] = (t["question"], t["answer"])
    for maker in gen_templates_full_b3b4b5.ALL_MAKERS:
        for t in maker(rng):
            tpls[t["sid"]] = (t["question"], t["answer"])

    bad: List[Tuple[str, str]] = []
    checked = 0
    for s in samples:
        t0 = tpls.get(s["id"])
        if t0 is None:
            continue  # dryrun 30 条历史版本跳过
        checked += 1
        q0, a0 = t0
        # 最终 answer 在 Finish 段
        last = s["messages"][-1]
        a_fin = ""
        for c in last.get("tool_calls", []):
            if c["function"]["name"] == "Finish":
                a_fin = json.loads(c["function"]["arguments"]).get("answer", "")
        for tag, fin, orig in (("question", s["question"], q0), ("answer", a_fin, a0)):
            if numbers_of(fin) != numbers_of(orig):
                bad.append((s["id"], f"{tag} 数字不一致: {numbers_of(fin)} vs {numbers_of(orig)}"))
            extra = entities_of(fin) - entities_of(orig)
            if extra:
                bad.append((s["id"], f"{tag} 实体新增 {sorted(extra)}"))
            if len(fin) < len(orig) * 0.5:
                bad.append((s["id"], f"{tag} 长度坍缩 {len(fin)} < {len(orig) * 0.5:.0f}"))
    return {"checked": checked, "bad": bad, "pass": not bad}


# ============================================================
# A2. 可溯源率（参考指标，不参与总判定）
# 计入 loss 教师段中数字能否从 system/user/tool 段扣除；存在合法程序换算
# （24小时制、万→元、round/int 取整）与白名单数据，仅供人工参考。
# ============================================================
def _time_exists(hh: int, blob: str) -> bool:
    """时间存在性判定：24 小时制 ↔ 12 小时制（下午2点）互认。"""
    cands = {f"{hh:02d}:", f"{hh}点"}
    if hh > 12:
        cands |= {f"下午{hh - 12}点", f"{hh - 12}点"}
    elif hh == 0:
        cands |= {"晚上12点", "0点"}
    return any(c in blob for c in cands)


def check_traceability(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A2 参考指标：计入 loss 教师段中数字能否从 src（system/user/tool + CONFIGS
    白名单数据）中溯源。存在性判定（同一数字被多个教师段转述不重复扣分）。
    返回统计与违规明细（不参与总判定）。"""
    total = 0
    unfounded = 0
    bad_entities: List[Tuple[str, str]] = []     # (sid, 实体)
    bad_nums: List[Tuple[str, str]] = []         # (sid, 数字)
    for s in samples:
        src_texts = [m.get("content", "") for m in s["messages"] if m["role"] in ("system", "user", "tool")]
        src_blob = "".join(src_texts)
        # 存在性池：src 数字 + CONFIGS 白名单数据（price/range/motor/battery 等合法引用）
        pool: Set[float] = {float(x) for x in numbers_of(src_blob)}
        import json as _json
        pool |= {float(x) for c in CONFIGS for x in numbers_of(_json.dumps(c, ensure_ascii=False))}
        src_ents: Set[str] = set(KNOWN_ENTITIES) | set(CUSTOMER_NAMES) | entities_of(src_blob)

        def exists(num: str) -> bool:
            key = float(num.replace(",", ""))
            if key in pool:
                return True
            if key == int(key):  # 整数 key 走 round/int 取整容差（如月供 5620 vs 5620.09）
                return any(k != key and (round(k) == key or int(k) == key) for k in pool)
            return False

        for m in s["messages"]:
            if m["role"] != "assistant" or m.get("loss_mask") is False:
                continue
            seg = "".join(_msg_texts(m))
            # 实体：硬性要求全部可溯源
            for e in entities_of(seg):
                if e not in src_ents:
                    bad_entities.append((s["id"], e))
            # 日期话术豁免（程序换算：基准日+"明天"→具体日期，模板级已验证自洽）
            seg = _DATE_TALK_RE.sub("", seg)
            seg = re.sub(r"\d{1,2}月\d{1,2}日", "", seg)
            # 数字：Finish/工具参数/Thought 全查
            name = _tool_name(m)
            if name != "Thought" and name != "Finish":
                # 工具调用参数（JSON）中的数字也查
                for c in m.get("tool_calls", []):
                    seg += c["function"].get("arguments", "") or ""
            seg = _BUSINESS_HOURS_RE.sub("", seg)  # 剔除营业时间话术整段
            for m_num, unit in _UNIT_RE.findall(seg):
                total += 1
                # 程序换算豁免："最近一个月"→"30天"
                if m_num == "30" and unit == "天" and "一个月" in src_blob:
                    continue
                # 工具契约枚举话术豁免：成交价查询 days 仅支持 30/90/180/365
                if unit == "天" and m_num in ("30", "90", "180", "365") \
                        and ("成交价" in src_blob or "成交价" in seg):
                    continue
                if not exists(m_num):
                    unfounded += 1
                    bad_nums.append((s["id"], f"{m_num}{unit}"))
            for hh, _mm in _TIME_RE.findall(seg):
                total += 1
                if not _time_exists(int(hh), src_blob):
                    unfounded += 1
                    bad_nums.append((s["id"], f"{hh}:xx"))
            for ph in _PHONE_RE.findall(seg):
                total += 1
                if ph not in src_blob:
                    unfounded += 1
                    bad_nums.append((s["id"], ph))
    rate = (unfounded / total) if total else 0.0
    num_categories = Counter(v for _, v in bad_nums).most_common(15)
    return {
        "total": total, "unfounded": unfounded, "rate": rate,
        "pass": unfounded <= total * 0.02,
        "bad_entities": bad_entities[:20], "bad_nums": bad_nums[:20],
        "num_categories": num_categories,
        "n_bad_entities": len(bad_entities), "n_bad_nums": len(bad_nums),
    }


# ============================================================
# B. 零重叠 + 相似度初筛
# ============================================================
def check_overlaps(samples: List[Dict[str, Any]], eval_qs: List[str]) -> Dict[str, Any]:
    hits = [s["id"] for s in samples if check_overlap(s["question"], eval_qs)]
    sim_pairs: List[Tuple[float, str, str, str, str]] = []
    q_by_id = {s["id"]: s["question"] for s in samples}
    for s in samples:
        q = s["question"].replace(" ", "")
        for i, eq in enumerate(eval_qs):
            e = eq.replace(" ", "")
            r = SequenceMatcher(None, q, e).ratio()
            if r >= 0.65:
                sim_pairs.append((r, s["id"], f"eval_{i:02d}", s["question"][:48], eval_qs[i][:48]))
    sim_pairs.sort(reverse=True)
    return {"hits": hits, "pass": not hits, "top_sim": sim_pairs[:20], "q_by_id": q_by_id}


# ============================================================
# C. clarify 配平（场景级）
# ============================================================
def check_balance(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows: Dict[str, Dict[str, int]] = {}
    for s in samples:
        row = rows.setdefault(s["scenario"], {"clarify": 0, "execute": 0, "degrade": 0})
        if s["is_clarify"]:
            row["clarify"] += 1
        elif s["is_degrade"]:
            row["degrade"] += 1
        else:
            row["execute"] += 1
    td = rows.get("试驾预约", {"clarify": 0, "execute": 0, "degrade": 0})
    fin = rows.get("金融方案测算", {"clarify": 0, "execute": 0, "degrade": 0})
    td_pass = td["clarify"] >= td["execute"] and td["clarify"] == 210 and td["execute"] == 200
    fin_pass = fin["execute"] >= fin["clarify"]
    return {"rows": rows, "td_pass": td_pass, "fin_pass": fin_pass}


# ============================================================
# D. 正例工具链完整
# ============================================================
def check_toolchains(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    missing: List[Tuple[str, str]] = []
    unpaired = 0
    checked = 0
    for s in samples:
        if not s["targets"] or s["is_clarify"]:
            continue
        checked += 1
        called: Set[str] = set()
        n_calls = 0
        for m in s["messages"]:
            if m["role"] == "assistant":
                for c in m.get("tool_calls", []):
                    nm = c["function"]["name"]
                    if nm not in ("Thought", "Finish"):
                        called.add(nm)
                        n_calls += 1
            elif m["role"] == "tool" and isinstance(m.get("content"), str) \
                    and m["content"].startswith("{"):
                n_calls -= 1  # 仅 JSON 工具返回配对（Thought 的"推理:..."返回不配对）
        if n_calls != 0:
            unpaired += 1
        for t in s["targets"]:
            if t not in called:
                missing.append((s["id"], t))
    return {"checked": checked, "missing": missing, "unpaired": unpaired,
            "pass": not missing and unpaired == 0}


# ============================================================
# E. final_answer 接地
# ============================================================
def check_grounding(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    bad: List[Tuple[str, str]] = []
    for s in samples:
        last = s["messages"][-1]
        seg = "".join(_msg_texts(last))
        src_ents = set(KNOWN_ENTITIES) | set(CUSTOMER_NAMES) | entities_of(
            "".join(m.get("content", "") for m in s["messages"] if m["role"] in ("user", "tool")))
        for e in entities_of(seg):
            if e not in src_ents:
                bad.append((s["id"], e))
    return {"bad": bad, "pass": not bad}


# ============================================================
# F. mask 覆盖率（05 文档第五节）
# ============================================================
def check_masks(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_bucket: Dict[str, int] = {}
    non_assistant = []
    per_sample: Dict[str, int] = {}
    for s in samples:
        n = 0
        for m in s["messages"]:
            if m.get("loss_mask") is False:
                n += 1
                if m["role"] != "assistant":
                    non_assistant.append(s["id"])
        per_bucket[s["bucket"]] = per_bucket.get(s["bucket"], 0) + n
        per_sample[s["id"]] = n
    b2 = [s["id"] for s in samples if s["bucket"] == "bucket2_correction"]
    others = [s["id"] for s in samples if s["bucket"] != "bucket2_correction"]
    b2_bad = [sid for sid in b2 if per_sample[sid] != 2]
    other_bad = [sid for sid in others if per_sample[sid] != 0]
    pass_ = not non_assistant and not b2_bad and not other_bad
    return {"per_bucket": per_bucket, "non_assistant": non_assistant,
            "b2_bad": b2_bad[:10], "other_bad": other_bad[:10], "pass": pass_}


# ============================================================
# G. 结构卫生
# ============================================================
def check_structure(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    dup_ids: Set[str] = set()
    seen: Set[str] = set()
    seq_bad: List[str] = []
    tool_json_bad: List[str] = []
    for s in samples:
        if s["id"] in seen:
            dup_ids.add(s["id"])
        seen.add(s["id"])
        msgs = s["messages"]
        if not msgs or msgs[0]["role"] != "system":
            seq_bad.append(s["id"] + ":no_system")
            continue
        for i, m in enumerate(msgs):
            if m["role"] == "tool":
                if i == 0 or msgs[i - 1]["role"] != "assistant":
                    seq_bad.append(s["id"] + ":tool_orphan")
                content = m.get("content")
                if isinstance(content, str) and content.startswith("{"):
                    try:
                        obj = json.loads(content)
                        if obj.get("status") not in ("success", "partial", "error"):
                            tool_json_bad.append(s["id"])
                    except json.JSONDecodeError:
                        tool_json_bad.append(s["id"])
    return {"dup_ids": dup_ids, "seq_bad": seq_bad[:10], "tool_json_bad": tool_json_bad[:10],
            "pass": not dup_ids and not seq_bad and not tool_json_bad}


# ============================================================
def main() -> None:
    samples = _load()
    eval_qs = load_eval_questions()
    print(f"样本 {len(samples)} 条 | 评测集 {len(eval_qs)} 题")

    a1 = check_polish_no_inject(samples)
    a2 = check_traceability(samples)
    b = check_overlaps(samples, eval_qs)
    c = check_balance(samples)
    d = check_toolchains(samples)
    e = check_grounding(samples)
    f = check_masks(samples)
    g = check_structure(samples)

    lines: List[str] = []
    def out(t: str = "") -> None:
        print(t)
        lines.append(t)

    out("=" * 64)
    out("T7 全量清洗验收报告")
    out("=" * 64)
    out(f"\n数据: {DATA}（{len(samples)} 条）")

    out("\n[A1] 润色零注入（主校验，管线硬性红线：数字 multiset 相等 + 实体只减不增 + 长度≥50%）")
    out(f"  校验 {a1['checked']} 条（dryrun 30 条历史版本除外）| 违规 {len(a1['bad'])} 处"
        f" | {'PASS' if a1['pass'] else '!! FAIL'}")
    for sid, msg in a1["bad"][:10]:
        out(f"    {sid}: {msg}")

    out("\n[A2] 可溯源率（参考指标，不参与总判定；口径 04-3.3 节）")
    out(f"  计入 loss 教师段参数检查: {a2['total']} 处 | 不可溯源 {a2['unfounded']} 处"
        f" | 占比 {a2['rate']:.2%}")
    out(f"  实体硬性检查违规: {a2['n_bad_entities']} 处 {a2['bad_entities'][:10]}")
    out(f"  不可溯源数字 top 分类: {a2['num_categories']}")
    out(f"  不可溯源明细（前 10）: {a2['bad_nums'][:10]}")

    out("\n[B] 与评测集零重叠（阈值 0 题面重合）")
    out(f"  重叠命中: {len(b['hits'])} | {'PASS' if b['pass'] else '!! FAIL ' + str(b['hits'][:5])}")
    out("  相似度初筛 top20（供人工复核，确认非直接复用题面）:")
    for r, sid, eq, sq, eqt in b["top_sim"]:
        out(f"    ratio={r:.3f}  {sid}  vs  {eq}")
        out(f"      SFT: {sq}")
        out(f"      EVL: {eqt}")

    out("\n[C] clarify 配平（≥1:1，2.5 节场景级）")
    for sc, row in c["rows"].items():
        out(f"  {sc}: 澄清 {row['clarify']} | 执行 {row['execute']} | 降级 {row['degrade']}")
    out(f"  试驾 210:200（1.05:1）: {'PASS' if c['td_pass'] else '!! FAIL'}")
    out(f"  金融执行主导: {'PASS' if c['fin_pass'] else '!! FAIL'}")

    out("\n[D] 正例工具链完整（100%）")
    out(f"  执行类正例检查 {d['checked']} 条 | targets 缺调用 {len(d['missing'])}"
        f" | 调用/返回不配对 {d['unpaired']} | {'PASS' if d['pass'] else '!! FAIL ' + str(d['missing'][:5])}")

    out("\n[E] final_answer 接地（100%，白名单实体）")
    out(f"  表外实体 {len(e['bad'])} 处 | {'PASS' if e['pass'] else '!! FAIL ' + str(e['bad'][:5])}")

    out("\n[F] mask 覆盖率（05 文档第五节）")
    for bk, n in sorted(f["per_bucket"].items()):
        out(f"  {bk}: loss_mask=false 段数 {n}")
    out(f"  false 段非 assistant: {f['non_assistant']}")
    out(f"  桶2 未达 2 段: {f['b2_bad']} | 其余桶非 0 段: {f['other_bad']}")
    out(f"  {'PASS' if f['pass'] else '!! FAIL'}（训练脚本读取 loss_mask 字段：待训练脚本实现时补验）")

    out("\n[G] 结构卫生")
    out(f"  sid 重复: {len(g['dup_ids'])} | 角色序列违规: {g['seq_bad']}"
        f" | tool JSON 异常: {g['tool_json_bad']} | {'PASS' if g['pass'] else '!! FAIL'}")

    all_pass = all(x["pass"] for x in (a1, b, d, e, f, g)) and c["td_pass"] and c["fin_pass"]
    out("\n" + ("=" * 64))
    out(f"总判定: {'PASS — 全部红线达标' if all_pass else '!! FAIL — 存在未达标项'}")
    out("=" * 64)

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告: {REPORT}")


if __name__ == "__main__":
    main()
