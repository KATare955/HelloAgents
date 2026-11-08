#!/usr/bin/env python3
"""远程诊断 v2：手动走 converter → encode_multiturn 链，定位 labels 全 -100 根因。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/LLaMA-Factory/src")

from llamafactory.hparams import get_train_args, read_args
from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.converter import DATASET_CONVERTERS
from llamafactory.data.template import TEMPLATES

model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(read_args())

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
template = get_template_and_fix_tokenizer(tokenizer, data_args)
print("template:", type(template).__name__, "| enable_thinking:", template.enable_thinking)

recs = json.loads(
    Path("/root/autodl-tmp/HelloAgents/eval/sft_data/v02_gen/llamafactory_v02/sft_train.json").read_text()
)
rec = recs[0]
print("原始 from 序列:", [c["from"] for c in rec["conversations"]])

Conv = DATASET_CONVERTERS["v02_react"]
conv = Conv(dataset_attr=None, data_args=data_args)
out = conv(rec)
print("converter 输出 role 序列:", [m["role"] for m in out["_prompt"] + out["_response"]])
print("converter loss 键:", {i: m.get("loss") for i, m in enumerate(out["_prompt"] + out["_response"]) if "loss" in m})

# 模拟 _encode_data_example 的调用
messages = template.mm_plugin.process_messages(out["_prompt"] + out["_response"], [], [], [], None)
print("process_messages 后 role 序列:", [m["role"] for m in messages])

encoded_pairs = template.encode_multiturn(tokenizer, messages, out["_system"], out["_tools"])
print(f"encode_multiturn pairs: {len(encoded_pairs)}")
for i, (src, tgt) in enumerate(encoded_pairs):
    print(f"  pair{i}: src={len(src)} tgt={len(tgt)}")

IGNORE = -100
input_ids, labels = [], []
for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
    source_label = [IGNORE] * len(source_ids)
    target_label = target_ids
    input_ids += source_ids + target_ids
    labels += source_label + target_label
valid = [x for x in labels if x != IGNORE]
print(f"模拟组装: input_ids={len(input_ids)} valid={len(valid)}")
