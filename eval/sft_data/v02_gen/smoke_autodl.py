#!/usr/bin/env python3
"""AutoDL 上机 smoke test：llamafactory 0.9.6.dev0 下验证 v02 扩展点（C1 前置）。

验证点：
  1. TEMPLATES["qwen3_semantic_pair"] / DATASET_CONVERTERS["v02_react"] 注册成功
  2. 桶 2 样本语义配对 label 断言：编造段(target0/1)不学、纠错段(target2)学、Finish(target3)学
  3. encode_multiturn 兼容 processor 的第 5 位置参数（discarding_history_cot）
  4. 渲染长度 sanity：cutoff 8192 下无截断
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/LLaMA-Factory/src")
IGNORE = -100

from transformers import AutoTokenizer
from llamafactory.data import TEMPLATES, DATASET_CONVERTERS  # noqa: F401  (触发扩展点注册)

TOK_PATH = "/root/autodl-tmp/models/Qwen3-14B"
DATA = Path("/root/autodl-tmp/HelloAgents/eval/sft_data/v02_gen/llamafactory_v02/sft_train.json")

print("=== 1. 扩展点注册 ===")
assert "qwen3_semantic_pair" in TEMPLATES, "template 未注册"
assert "v02_react" in DATASET_CONVERTERS, "converter 未注册"
print(f"  [OK] qwen3_semantic_pair + v02_react 已注册")

tokenizer = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
tpl = TEMPLATES["qwen3_semantic_pair"]
tpl.enable_thinking = False

print("=== 2. 桶 2 语义配对 label 断言 ===")
recs = json.loads(DATA.read_text(encoding="utf-8"))
bucket2 = [r for r in recs if any(("loss" in c) for c in r["conversations"])]
assert bucket2, "训练集中无 loss 字段样本"
rec = bucket2[0]
print(f"  样本: {rec['conversations'][0]['value'][:30]}... (loss 段数 {sum(1 for c in rec['conversations'] if 'loss' in c)})")

system = rec.get("system", "")
msgs = []
for c in rec["conversations"]:
    role_map = {"human": "user", "gpt": "assistant", "function_call": "function", "observation": "observation"}
    m = {"role": role_map[c["from"]], "content": c.get("value") or ""}
    if "loss" in c:
        m["loss"] = bool(c["loss"])
    msgs.append(m)

pairs = tpl.encode_multiturn(tokenizer, msgs, system, None, discarding_history_cot=False)
print(f"  语义配对对数: {len(pairs)}")
for i, (src, tgt) in enumerate(pairs):
    print(f"  pair{i}: src={len(src)}tok tgt={len(tgt)}tok")

# 断言：loss=false 的段 tgt 为空（并入 source），loss=true 的段 tgt 非空
target_msgs = [m for m in msgs if m["role"] in ("assistant", "function")]
assert len(pairs) == len(target_msgs), f"配对数量不符 {len(pairs)} vs {len(target_msgs)}"
for i, (m, (src, tgt)) in enumerate(zip(target_msgs, pairs)):
    if m.get("loss") is False:
        assert len(tgt) == 0, f"pair{i} 编造段 tgt 应为空"
    else:
        assert len(tgt) > 0, f"pair{i} 学习段 tgt 不应为空"
print("  [OK] 编造段 tgt 空 / 学习段 tgt 非空")

print("=== 3. 第 5 位置参数兼容 ===")
pairs2 = tpl.encode_multiturn(tokenizer, msgs, system, None, False)
assert len(pairs2) == len(pairs), "discarding_history_cot 参数破坏配对"
print("  [OK] discarding_history_cot=False 正常")

print("=== 4. 长度 sanity（全量 train 渲染） ===")
max_len, n = 0, 0
for r in recs:
    sys2 = r.get("system", "")
    mm = []
    for c in r["conversations"]:
        role_map = {"human": "user", "gpt": "assistant", "function_call": "function", "observation": "observation"}
        x = {"role": role_map[c["from"]], "content": c.get("value") or ""}
        if "loss" in c:
            x["loss"] = bool(c["loss"])
        mm.append(x)
    pp = tpl.encode_multiturn(tokenizer, mm, sys2, None)
    total = sum(len(s) + len(t) for s, t in pp)
    max_len = max(max_len, total)
    n += 1
print(f"  train n={n} 渲染 max={max_len} (cutoff 8192 {'OK' if max_len <= 8192 else 'FAIL'})")
assert max_len <= 8192

print()
print("=== SMOKE TEST: ALL PASS ===")
