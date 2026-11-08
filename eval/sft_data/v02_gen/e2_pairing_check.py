#!/usr/bin/env python3
"""E2：v02 桶2 形态 × llamafactory 配对逻辑 label 位置验证（真实 tokenizer，最小原型）

验证目标（评估报告 8.3 不确定项 1/2）：
  1. 官方两两配对（encode_multiturn 语义）对 v02 桶2 9 段形态会错位到什么程度；
  2. 自定义"语义配对"（assistant/function_call 段为 target，其前积累的
     human/observation 段为 source）的 label 位置是否完全正确：
     - 编造段（loss_mask=false）target_label 全 IGNORE
     - 纠错段 target_label = ids
     - Finish gpt 段 target_label = ids
     - 所有 source（user/obs）label 全 IGNORE
  3. 语义配对下 input_ids 与整段渲染（verl 无 think 拼接）逐 token 一致。
"""

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
SFT_DIR = HERE.parent
TOK_DIR = HERE / ".tokenizer_qwen3"
IGNORE = -100

sys.stdout.reconfigure(encoding="utf-8")


def render_each_message(tokenizer, msgs, tools_flat, ctkw):
    """逐消息渲染（verl 逻辑，无 think），返回每消息 token 列表。"""
    return [
        tokenizer.apply_chat_template([m], tools=tools_flat if i == 0 else None,
                                      add_generation_prompt=False, tokenize=True, **ctkw)
        for i, m in enumerate(msgs)
    ]


def official_pairs(per_msg_tokens):
    """复刻 llamafactory 官方配对：两两 (m[2k] 为 source, m[2k+1] 为 target)。"""
    pairs = []
    for i in range(0, len(per_msg_tokens) - 1, 2):
        pairs.append((per_msg_tokens[i], per_msg_tokens[i + 1]))
    if len(per_msg_tokens) % 2 == 1:  # 官方假设偶数，奇数时末段悬空
        pairs.append((per_msg_tokens[-1], []))
    return pairs


def semantic_pairs(msgs, per_msg_tokens):
    """自定义语义配对：fc/gpt 段为 target，其前积累的 human/obs 段为 source。
    返回 [(source_ids, target_ids, loss_flag)]，loss_flag=False 表示 target 不学。"""
    pairs, buf = [], []
    for m, toks in zip(msgs, per_msg_tokens):
        if m["role"] == "assistant":  # v02 中 assistant 段即工具调用（Thought/业务/Finish）
            loss = m.get("loss_mask", True)
            pairs.append((sum(buf, []), toks, bool(loss)))
            buf = []
        else:
            buf.append(toks)
    return pairs


def build_input_labels(pairs, source_label=IGNORE):
    """复刻 processor 的 input_ids/labels 组装：source 全 IGNORE，target 按 loss_flag。"""
    input_ids, labels = [], []
    for src, tgt, learn in pairs:
        input_ids += src + tgt
        labels += [source_label] * len(src)
        labels += (tgt if learn else [IGNORE] * len(tgt))
    return input_ids, labels


def main():
    tokenizer = AutoTokenizer.from_pretrained(str(TOK_DIR), trust_remote_code=True)
    ctkw = {"chat_template_kwargs": {"enable_thinking": False}}
    tools_flat = [t["function"] for t in json.loads((SFT_DIR / "tool_schemas_react.json").read_text(encoding="utf-8"))]

    rec = None
    for line in (SFT_DIR / "v02_full_1000.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d["bucket"] == "bucket2_correction":
            rec = d
            break
    msgs = [m for m in rec["messages"] if m["role"] != "system"]
    per_msg = render_each_message(tokenizer, msgs, tools_flat, ctkw)

    # ---- 1. 官方配对错位演示（忠实复刻：target 一律 label=ids，不区分角色）----
    print("=== 1. 官方两两配对（9 段形态）错位演示 ===")
    off_pairs = official_pairs(per_msg)
    n_learn_bad = 0
    for k, (src, tgt) in enumerate(off_pairs):
        s_role = msgs[2 * k]["role"] if 2 * k < len(msgs) else "?"
        t_role = msgs[2 * k + 1]["role"] if 2 * k + 1 < len(msgs) else "(悬空)"
        if tgt:
            print(f"  对{k}: source={s_role}({len(src)}tok, label=IGNORE) "
                  f"-> target={t_role}({len(tgt)}tok, label=ids 学习!)")
            if t_role != "assistant":
                n_learn_bad += 1
        else:
            print(f"  对{k}: source={s_role}({len(src)}tok, label=IGNORE) -> target={t_role}(悬空)")
    # 编造段在官方配对下的命运
    b2_fake_learned = (msgs[3]["role"] == "assistant" and msgs[3].get("loss_mask") is False)
    print(f"  结论: {n_learn_bad} 个 user/tool 段被当作 target 学习（错位）;")
    print(f"        编造段(应 IGNORE)在官方配对下被学习: {b2_fake_learned}; "
          f"Finish 段(应学习)悬空丢失: {len(per_msg) % 2 == 1}")

    # ---- 2. 语义配对断言 ----
    print("\n=== 2. 语义配对 label 位置断言 ===")
    pairs = semantic_pairs(msgs, per_msg)
    for k, (src, tgt, learn) in enumerate(pairs):
        print(f"  target{k}: src={len(src)}tok, tgt={len(tgt)}tok, learn={learn}")

    input_ids, labels = build_input_labels(pairs)

    # 断言：input_ids 与整段拼接一致
    full_ids = sum(per_msg, [])
    assert input_ids == full_ids, "input_ids 与整段拼接不一致"
    print(f"  [OK] input_ids 与整段拼接一致 ({len(input_ids)} tokens)")

    # 逐段校验 label
    pos = 0
    all_ok = True
    for i, m in enumerate(msgs):
        seg = per_msg[i]
        lab = labels[pos:pos + len(seg)]
        pos += len(seg)
        if m["role"] == "assistant":
            expected = IGNORE if m.get("loss_mask", True) is False else "ids"
            ok = (expected == IGNORE and all(x == IGNORE for x in lab)) or \
                 (expected == "ids" and lab == seg)
            print(f"  [{ 'OK' if ok else 'FAIL'}] {m['role']}(loss_mask={m.get('loss_mask', True)}): "
                  f"label={'IGNORE' if expected == IGNORE else 'ids'} -> {'OK' if ok else 'FAIL'}")
            all_ok &= ok
        else:
            ok = all(x == IGNORE for x in lab)
            print(f"  [{'OK' if ok else 'FAIL'}] {m['role']}: label 全 IGNORE -> {'OK' if ok else 'FAIL'}")
            all_ok &= ok

    # 统计
    n_learn = sum(1 for x in labels if x != IGNORE)
    n_total = len(labels)
    print(f"\n  学习 token {n_learn}/{n_total} ({n_learn / n_total:.1%})，其余 IGNORE")
    print(f"  编造段(target0/target1)应全 IGNORE、纠错段(target2)+Finish(target3)应为 ids")
    print(f"\n=== 断言结果: {'ALL PASS' if all_ok else 'FAIL'} ===")


if __name__ == "__main__":
    main()
