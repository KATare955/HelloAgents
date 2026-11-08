#!/usr/bin/env python3
"""E2 真框架版：llamafactory 0.9.3 自定义 Template 语义配对验证

验证（评估报告不确定项 1/2 的框架侧确认）：
  1. 官方 qwen3 模板渲染（ReasoningTemplate, enable_thinking=False）的 input_ids
     与 transformers 整段渲染的 token 差异（预期：仅空 think 块位置差异）；
  2. 官方 encode_multiturn 两两配对对 v02 桶2 9 段形态的错位（框架真实行为）；
  3. 自定义 SemanticPairTemplate 子类重写 encode_multiturn：
     - 语义配对（assistant/function 段为 target 边界）
     - loss=false 的编造段并入 source（(src+tgt, [])，无需 patch processor）
     - 断言 label 位置：编造段全 IGNORE、纠错段/Finish=ids、source 全 IGNORE。
"""

import json
import sys
from copy import deepcopy
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 允许 import llamafactory
from llamafactory.data.template import TEMPLATES, ReasoningTemplate, Role

HERE = Path(__file__).resolve().parent
SFT_DIR = HERE.parent
TOK_DIR = HERE / ".tokenizer_qwen3"
IGNORE = -100

sys.stdout.reconfigure(encoding="utf-8")


def v02_to_lf_messages(msgs):
    """v02 messages → llamafactory sharegpt 角色映射（复用 R1 规则 + 修正）。
    返回 (system, lf_messages, loss_flags)。loss 字段直接写入消息供 encode_multiturn 读取；
    function 段 arguments 解析为 JSON 对象（Qwen3 模板要求 arguments 为对象而非字符串）。"""
    system = next(m["content"] for m in msgs if m["role"] == "system")
    out, flags = [], []
    for m in msgs:
        if m["role"] == "system":
            continue
        if m["role"] == "user":
            out.append({"role": Role.USER.value, "content": m["content"]})
        elif m["role"] == "tool":
            out.append({"role": Role.OBSERVATION.value, "content": m["content"]})
        elif m["role"] == "assistant":
            if m.get("tool_calls"):  # Thought/业务工具 → function
                fn = m["tool_calls"][0]["function"]
                args = fn["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)  # 字符串 → 对象（Qwen3 模板要求）
                out.append({"role": Role.FUNCTION.value,
                            "content": json.dumps({"name": fn["name"], "arguments": args},
                                                  ensure_ascii=False)})
            else:  # Finish → gpt 收尾
                out.append({"role": Role.ASSISTANT.value, "content": m["content"]})
            out[-1]["loss"] = m.get("loss_mask", True)
        else:
            continue
        flags.append(out[-1].get("loss", True))
    return system, out, flags


class SemanticPairTemplate(ReasoningTemplate):
    """语义配对模板：assistant/function 段为 target 边界，loss=false 的 target 并入 source。
    继承 ReasoningTemplate 以复用 remove_thought/get_thought_word_ids/thought_words。"""

    def encode_multiturn(self, tokenizer, messages, system=None, tools=None):
        messages = deepcopy(messages)
        if self.enable_thinking is False:  # 保留官方 nothink 行为
            for i in range(1, len(messages), 2):
                messages[i]["content"] = self.remove_thought(messages[i]["content"])

        encoded = self._encode(tokenizer, messages, system, tools)

        # 语义配对：target 边界 = assistant/function 角色
        pairs, buf = [], []
        for m, toks in zip(messages, encoded):
            if m["role"] in (Role.ASSISTANT.value, Role.FUNCTION.value):
                pairs.append({"src": buf, "tgt": toks, "learn": m.get("loss", True)})
                buf = []
            else:
                buf.append(toks)
        # 尾部残留（不应有，但保险并入最后一对 src）
        if buf and pairs:
            pairs[-1]["src"] += buf

        # 官方空 cot 逻辑（对真实 target 无 think 词时，空 cot 追加到其 source 末尾）
        for p, m in zip(pairs, [m for m in messages if m["role"] in (Role.ASSISTANT.value, Role.FUNCTION.value)]):
            if (self.thought_words[0] not in (m["content"] or "")
                    and self.thought_words[1] not in (m["content"] or "")):
                if not self.enable_thinking:
                    if p["src"]:
                        p["src"][-1] = p["src"][-1] + self.get_thought_word_ids(tokenizer)
                    else:
                        p["src"].append(self.get_thought_word_ids(tokenizer))
                else:
                    p["tgt"] = self.get_thought_word_ids(tokenizer) + p["tgt"]

        # loss=false → target 并入 source（返回 (src+tgt, [])，processor 无需改动）
        result = []
        for p in pairs:
            src_ids = [t for seg in p["src"] for t in seg]
            if p["learn"]:
                result.append((src_ids, p["tgt"]))
            else:
                result.append((src_ids + p["tgt"], []))
        return result


def processor_assemble(encoded_pairs, eos_token_id, efficient_eos=False):
    """复刻 processor._encode_data_example 的组装循环（train_on_prompt=False）。"""
    input_ids, labels = [], []
    for source_ids, target_ids in encoded_pairs:
        source_len, target_len = len(source_ids), len(target_ids)
        if efficient_eos:
            source_label = [eos_token_id] + [IGNORE] * max(0, source_len - 1)
        else:
            source_label = [IGNORE] * source_len
        target_label = target_ids
        input_ids += source_ids + target_ids
        labels += source_label + target_label
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
    system, lf_msgs, flags = v02_to_lf_messages(rec["messages"])

    # ---- 1. 官方 qwen3 模板（ReasoningTemplate） ----
    print("=== 1. 官方 qwen3 模板渲染 vs transformers 整段渲染 ===")
    off_tpl = TEMPLATES["qwen3"]
    off_tpl.enable_thinking = False
    encoded = off_tpl._encode(tokenizer, lf_msgs, system, None)
    lf_ids = [t for seg in encoded for t in seg]
    # transformers 整段渲染（verl 无 think 拼接；tools 不传以对齐 llamafactory tools=None——
    # v02 system 文本自带工具描述，与 R1 训练一致）
    verl_ids = []
    for i, m in enumerate(rec["messages"]):
        ids = tokenizer.apply_chat_template([m], tools=None,
                                            add_generation_prompt=False, tokenize=True, **ctkw)
        verl_ids += ids
    print(f"  llamafactory _encode: {len(lf_ids)} tokens; transformers 拼接: {len(verl_ids)} tokens; 一致: {lf_ids == verl_ids}")
    if lf_ids != verl_ids:
        n = min(len(lf_ids), len(verl_ids))
        fd = next((k for k in range(n) if lf_ids[k] != verl_ids[k]), n)
        print(f"  首个分歧 {fd}")
        print(f"   LF: ...{tokenizer.decode(lf_ids[max(0,fd-30):fd+60])!r}")
        print(f"   TF: ...{tokenizer.decode(verl_ids[max(0,fd-30):fd+60])!r}")

    # ---- 2. 官方 encode_multiturn（两两配对）对 9 段形态 ----
    print("\n=== 2. 官方 encode_multiturn（两两配对）对 9 段形态 ===")
    try:
        off_pairs = off_tpl.encode_multiturn(tokenizer, lf_msgs, system, None)
    except IndexError as e:
        print(f"  官方 encode_multiturn 对 9 段（奇数）消息直接崩溃: IndexError({e})")
        print("  -> 官方两两配对假设偶数段（converter 层位置校验亦拦截），桶 2 形态无法进入官方管线")
        off_pairs = None
    if off_pairs:
        off_input_ids, off_labels = processor_assemble(off_pairs, tokenizer.eos_token_id, off_tpl.efficient_eos)
        n_learn_bad = 0
        for k, (s, t) in enumerate(off_pairs):
            s_role = lf_msgs[2 * k]["role"] if 2 * k < len(lf_msgs) else "?"
            t_role = lf_msgs[2 * k + 1]["role"] if 2 * k + 1 < len(lf_msgs) else "(悬空)"
            print(f"  对{k}: source={s_role}({len(s)}tok) -> target={t_role}({len(t)}tok)")
            if t and t_role not in ("assistant",):
                n_learn_bad += 1
        print(f"  错位: {n_learn_bad} 个 user/observation 段被当作 target 学习")

    # ---- 3. 语义配对子类 ----
    print("\n=== 3. SemanticPairTemplate 语义配对 ===")
    from dataclasses import fields
    sem_tpl = SemanticPairTemplate(**{f.name: getattr(off_tpl, f.name) for f in fields(off_tpl)})
    sem_tpl.enable_thinking = False

    pairs = sem_tpl.encode_multiturn(tokenizer, lf_msgs, system, None)
    input_ids, labels = processor_assemble(pairs, tokenizer.eos_token_id, sem_tpl.efficient_eos)
    print(f"  配对结果: {len(pairs)} 对; input_ids {len(input_ids)} tokens")
    for k, (s, t) in enumerate(pairs):
        print(f"  对{k}: src={len(s)}tok, tgt={len(t)}tok {'(并入source,不学)' if not t else '(学习)'}")

    # 断言：input_ids 与官方 _encode 全序列 + 空 cot 追加一致（空 cot 追加到每对 source 末尾）
    def strip_cot(ids):
        cot = sem_tpl.get_thought_word_ids(tokenizer)
        out, i = [], 0
        while i < len(ids):
            if ids[i:i + len(cot)] == cot:
                i += len(cot)
            else:
                out.append(ids[i]); i += 1
        return out
    assert strip_cot(input_ids) == lf_ids, "语义配对 input_ids（移除空 cot 后）与官方 _encode 全序列不一致"
    print("  [OK] input_ids 与官方渲染全序列一致（空 cot 按官方规则追加到每对 source 末尾）")

    # 断言 label 位置：逐对检查（src 全 IGNORE 含空 cot；tgt 按 loss 标志）
    pos = 0
    all_ok = True
    fake_msgs = [m for m in lf_msgs if m["role"] in ("assistant", "function")]
    for k, (s, t) in enumerate(pairs):
        src_lab = labels[pos:pos + len(s)]
        pos += len(s)
        tgt_lab = labels[pos:pos + len(t)]
        pos += len(t)
        ok_src = all(x == IGNORE for x in src_lab)
        learn = fake_msgs[k].get("loss", True)
        if learn:
            ok_tgt = tgt_lab == t and len(t) > 0
            tag = f"learn target"
        else:
            ok_tgt = len(t) == 0 and len(tgt_lab) == 0
            tag = "loss=false 编造段(并入source)"
        print(f"  [{'OK' if (ok_src and ok_tgt) else 'FAIL'}] 对{k} {tag}: "
              f"src全IGNORE={ok_src}, tgt={'=ids' if learn else '空(不学)'}={ok_tgt}")
        all_ok &= ok_src and ok_tgt
    assert pos == len(labels), f"label 总长度不匹配 {pos} != {len(labels)}"

    n_learn = sum(1 for x in labels if x != IGNORE)
    print(f"\n  学习 token {n_learn}/{len(labels)} ({n_learn/len(labels):.1%})")
    print(f"=== 断言结果: {'ALL PASS' if all_ok else 'FAIL'} ===")


if __name__ == "__main__":
    main()
