#!/usr/bin/env python3
"""A3 三重终验：转换后训练数据的红线复验

① mask 覆盖率校验：v02 桶 2 编造段（loss_mask=false）共 500 段，
   转换后 train+val 的 loss=false 段数必须 1:1 全覆盖（05 文档标准）；
② JSON 合法 + 字段完整：train/val JSON 可解析，每条 conversations/system/tools
   字段齐备、conversations 非空、末段 gpt；
③ 与评测集 68 题零重叠复验：train+val 的 human 段（用户问题）与
   questions.json 生效版 68 题的 question 文本，精确匹配 + 归一化匹配均为 0。

用法：python3 eval/sft_data/v02_gen/a3_triple_check.py
"""

import json
import re
import sys
from pathlib import Path

SFT_DIR = Path(__file__).resolve().parent.parent
GEN_DIR = Path(__file__).resolve().parent
V02_SRC = SFT_DIR / "v02_full_1000.jsonl"
TRAIN_JSON = GEN_DIR / "llamafactory_v02" / "sft_train.json"
VAL_JSON = GEN_DIR / "llamafactory_v02" / "sft_val.json"
QUESTIONS = SFT_DIR.parent / "eval_data" / "questions.json"

sys.stdout.reconfigure(encoding="utf-8")


def norm(text: str) -> str:
    """归一化：去空白与标点，用于宽松匹配。"""
    return re.sub(r"[\s，。！？、,.!?：:；;\"'“”‘’（）()\[\]【】-]", "", text or "")


def main():
    all_ok = True

    # ---- ① mask 覆盖率 ----
    print("=== ① mask 覆盖率校验（桶2 编造段 500 段全覆盖） ===")
    v02 = [json.loads(l) for l in V02_SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    src_fake = sum(1 for r in v02 for m in r["messages"]
                   if m.get("loss_mask") is False and m["role"] == "assistant")
    src_fake_ids = {r["id"] for r in v02 for m in r["messages"]
                    if m.get("loss_mask") is False and m["role"] == "assistant"}
    print(f"  v02 源数据 loss_mask=false 段总数: {src_fake}（预期 500），涉及 {len(src_fake_ids)} 条样本")

    train = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
    val = json.loads(VAL_JSON.read_text(encoding="utf-8"))
    out_fake = sum(1 for d in train + val for c in d["conversations"] if c.get("loss") is False)
    print(f"  转换后 loss=false 段总数: {out_fake}（train+val）")
    ok1 = src_fake == 500 and out_fake == src_fake
    print(f"  [{'OK' if ok1 else 'FAIL'}] 500 段全覆盖且 1:1 透传")
    all_ok &= ok1

    # ---- ② JSON 合法 + 字段完整 ----
    print("\n=== ② JSON 合法 + 字段完整 ===")
    n_bad = 0
    for name, data in (("train", train), ("val", val)):
        for i, d in enumerate(data):
            if not isinstance(d.get("conversations"), list) or not d["conversations"]:
                print(f"  [FAIL] {name}[{i}] conversations 缺失/空"); n_bad += 1; continue
            if not isinstance(d.get("system"), str):
                print(f"  [FAIL] {name}[{i}] system 缺失"); n_bad += 1; continue
            if not isinstance(d.get("tools"), str) or not d["tools"]:
                print(f"  [FAIL] {name}[{i}] tools 缺失"); n_bad += 1; continue
            if d["conversations"][-1]["from"] != "gpt":
                print(f"  [FAIL] {name}[{i}] 末段非 gpt"); n_bad += 1; continue
            for c in d["conversations"]:
                if not isinstance(c.get("value"), str):
                    print(f"  [FAIL] {name}[{i}] value 缺失"); n_bad += 1; break
    ok2 = n_bad == 0 and len(train) == 960 and len(val) == 40
    print(f"  train={len(train)} val={len(val)}（预期 960/40），字段异常 {n_bad} 条")
    print(f"  [{'OK' if ok2 else 'FAIL'}]")
    all_ok &= ok2

    # ---- ③ 与评测集 68 题零重叠 ----
    print("\n=== ③ 与评测集 68 题零重叠复验 ===")
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    q_items = questions if isinstance(questions, list) else questions.get("questions", [])
    q_texts = [q.get("question", "") for q in q_items]
    print(f"  评测集题目数: {len(q_texts)}（预期 68）")
    human_texts = [c["value"] for d in train + val for c in d["conversations"] if c["from"] == "human"]

    exact_hits, norm_hits = [], []
    for t in human_texts:
        if t in q_texts:
            exact_hits.append(t[:50])
        nt = norm(t)
        if nt and any(nt == norm(q) for q in q_texts):
            norm_hits.append(t[:50])
    ok3 = len(q_texts) == 68 and not exact_hits and not norm_hits
    print(f"  human 段总数: {len(human_texts)}; 精确匹配重叠 {len(exact_hits)}; 归一化重叠 {len(norm_hits)}")
    print(f"  [{'OK' if ok3 else 'FAIL'}]")
    all_ok &= ok3

    print(f"\n=== A3 三重终验: {'ALL PASS' if all_ok else 'FAIL'} ===")


if __name__ == "__main__":
    main()
