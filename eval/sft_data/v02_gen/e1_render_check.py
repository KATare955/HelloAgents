#!/usr/bin/env python3
"""E1 v2：v02 桶2样本 × Qwen3 tokenizer 渲染一致性验证（精确复刻 verl 0.9.1 逻辑）

精确复刻 verl/utils/tokenizer/chat_template.py 的 extract_system_prompt_and_generation
（无 tools 空 user 消息探测前缀）与 MultiTurnSFTDataset._process_single_message
（单消息渲染 + index!=0 切 system 前缀 + tools 仅第一条传）。
另验证 enable_thinking 两种传参方式（直接 kwarg vs chat_template_kwargs）在本地
transformers 4.57 上的生效性。
"""

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
SFT_DIR = HERE.parent
TOK_DIR = HERE / ".tokenizer_qwen3"

sys.stdout.reconfigure(encoding="utf-8")


def verl_extract_system_and_generation(tokenizer, **kwargs):
    """复刻 verl 0.9.1 extract_system_prompt_and_generation（无 tools 探测）。"""
    token1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True, **kwargs)
    token2 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True, **kwargs)
    system_prompt = token1[: -(len(token2) - len(token1))]
    token3 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True, **kwargs)
    generate_prompt = token3[len(token1):]
    return system_prompt, generate_prompt


def main():
    tokenizer = AutoTokenizer.from_pretrained(str(TOK_DIR), trust_remote_code=True)
    rec = None
    for line in (SFT_DIR / "v02_full_1000.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d["bucket"] == "bucket2_correction":
            rec = d
            break
    system = next(m["content"] for m in rec["messages"] if m["role"] == "system")
    msgs = [m for m in rec["messages"] if m["role"] != "system"]
    tools_flat = [t["function"] for t in json.loads((SFT_DIR / "tool_schemas_react.json").read_text(encoding="utf-8"))]

    # ---- 0. enable_thinking 传参方式对照（nothink 目标）----
    print("=== 0. enable_thinking 传参方式对照 ===")
    for label, kwargs in [
        ("直接 kwarg", {"enable_thinking": False}),
        ("chat_template_kwargs", {"chat_template_kwargs": {"enable_thinking": False}}),
    ]:
        ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}],
            tokenize=True, add_generation_prompt=False, **kwargs)
        txt = tokenizer.decode(ids)
        print(f"  {label}: 含<think>={('<think>' in txt or '</think>' in txt)}, "
              f"含/no_think={('/no_think' in txt)}, 含think字样={('think' in txt)}")

    # ---- 1. verl 精确复刻：逐消息渲染拼接 vs 整段渲染 ----
    print("\n=== 1. verl 逐消息拼接 vs 整段渲染 ===")
    ctkw = {"chat_template_kwargs": {"enable_thinking": False}}
    sys_ids, gen_ids = verl_extract_system_and_generation(tokenizer, **ctkw)
    print(f"  verl 探测 system_prompt={len(sys_ids)} tokens, generation_prompt={len(gen_ids)} tokens")

    verl_ids = []
    for i, m in enumerate(msgs):
        ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}] + [m],
            tools=tools_flat if i == 0 else None,
            add_generation_prompt=False, tokenize=True, **ctkw)
        if i != 0:
            ids = ids[len(sys_ids):]
        verl_ids += ids

    full_ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}] + msgs,
        tools=tools_flat, add_generation_prompt=False, tokenize=True, **ctkw)

    print(f"  整段渲染 {len(full_ids)} tokens vs verl 拼接 {len(verl_ids)} tokens")
    print(f"  token 序列一致: {full_ids == verl_ids}")
    if full_ids != verl_ids:
        n = min(len(full_ids), len(verl_ids))
        fd = next((k for k in range(n) if full_ids[k] != verl_ids[k]), n)
        print(f"  首个分歧位置 {fd}")
        print(f"   整段: ...{tokenizer.decode(full_ids[max(0,fd-30):fd+60])!r}")
        print(f"   拼接: ...{tokenizer.decode(verl_ids[max(0,fd-30):fd+60])!r}")

    # ---- 2. 整段渲染文本关键检查 ----
    full_text = tokenizer.decode(full_ids)
    print("\n=== 2. 整段渲染文本检查 ===")
    print(f"  含 <tool_call> 块数: {full_text.count('<tool_call>')}")
    print(f"  含 <tool_response> 块数: {full_text.count('<tool_response>')}")
    print(f"  含 <think> 标签: {('<think>' in full_text or '</think>' in full_text)}")
    print(f"  含编造段(book_test_drive): {'book_test_drive' in full_text}")
    print(f"  含纠错 user 段: {'我没有告诉过你' in full_text}")
    print(f"  含 7 家门店反问: {'7 家可选' in full_text or '7家可选' in full_text}")

    # ---- 3. tool 消息渲染角色检查 ----
    print("\n=== 3. tool 消息渲染角色 ===")
    m_tool = msgs[2]
    ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, m_tool],
        tools=None, add_generation_prompt=False, tokenize=True, **ctkw)
    tool_part = tokenizer.decode(ids[len(sys_ids):])
    print(f"  tool 消息渲染: {tool_part[:200]!r}")

    # ---- 4. 整段渲染中编造段与纠错段的角色分布（供 loss mask 参考）----
    print("\n=== 4. 渲染文本角色分布（前 6 个 <|im_start|>） ===")
    import re
    starts = re.findall(r"<\|im_start\|>\w+", full_text)
    print("  " + " -> ".join(starts[:10]))


if __name__ == "__main__":
    main()
