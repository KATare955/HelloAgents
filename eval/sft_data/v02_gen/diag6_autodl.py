#!/usr/bin/env python3
"""远程诊断 v6：二分对比 aligned 数据 vs 手动 json.loads 数据，定位 valid=0 差异点。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/LLaMA-Factory/src")

from llamafactory.hparams import get_train_args, read_args
from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.parser import get_dataset_list
from llamafactory.data.converter import align_dataset, DATASET_CONVERTERS
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor, IGNORE_INDEX

model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(read_args())

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
template = get_template_and_fix_tokenizer(tokenizer, data_args)

proc = SupervisedDatasetProcessor(template=template, tokenizer=tokenizer, processor=None, data_args=data_args)

# --- A 路：aligned（diag5 环节3 的输出）---
attr = get_dataset_list(data_args.dataset, data_args.dataset_dir)[0]
from datasets import load_dataset

dataset = load_dataset(path="json", data_files=[str(Path(data_args.dataset_dir) / attr.dataset_name)], split="train", num_proc=1)
aligned = align_dataset(dataset, attr, data_args, training_args)

# --- B 路：json.loads 手动 converter（diag4 复刻）---
recs = json.loads((Path(data_args.dataset_dir) / attr.dataset_name).read_text())
Conv = DATASET_CONVERTERS["v02_react"]
conv_b = Conv(dataset_attr=None, data_args=data_args)
out_b = conv_b(recs[0])

# --- 对比打印 ---
a0 = aligned[0]
print("A _prompt roles:", [m["role"] for m in a0["_prompt"]])
print("B _prompt roles:", [m["role"] for m in out_b["_prompt"]])
print("A _prompt[0] content[:80]:", repr(a0["_prompt"][0]["content"][:80]))
print("B _prompt[0] content[:80]:", repr(out_b["_prompt"][0]["content"][:80]))
print("A _response[0] content[:80]:", repr(a0["_response"][0]["content"][:80]))
print("B _response[0] content[:80]:", repr(out_b["_response"][0]["content"][:80]))
print("A _system[:80]:", repr(a0["_system"][:80]))
print("B _system[:80]:", repr(out_b["_system"][:80]))
print("A _tools[:80]:", repr(a0["_tools"][:80]))
print("B _tools[:80]:", repr(out_b["_tools"][:80]))
print("A _prompt keys:", [list(m.keys()) for m in a0["_prompt"]])
print("B _prompt keys:", [list(m.keys()) for m in out_b["_prompt"]])

# --- 单条直调 preprocess：A 路 ---
def preprocess_one(prompt, response, system, tools):
    batched = {"_prompt": [prompt], "_response": [response], "_system": [system], "_tools": [tools],
               "_images": [None], "_videos": [None], "_audios": [None]}
    out = proc.preprocess_dataset(batched)
    labels = out["labels"][0]
    valid = [x for x in labels if x != IGNORE_INDEX]
    return len(out["input_ids"][0]), len(valid)

n_ids, n_valid = preprocess_one(a0["_prompt"], a0["_response"], a0["_system"], a0["_tools"])
print(f"A 单条 preprocess: input_ids={n_ids} valid={n_valid}")
n_ids, n_valid = preprocess_one(out_b["_prompt"], out_b["_response"], out_b["_system"], out_b["_tools"])
print(f"B 单条 preprocess: input_ids={n_ids} valid={n_valid}")

# --- 单条直调 encode_multiturn 对比 ---
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor as _S
import inspect

for tag, out in [("A", a0), ("B", out_b)]:
    messages = proc.template.mm_plugin.process_messages(out["_prompt"] + out["_response"], [], [], [], proc.processor)
    pairs = proc.template.encode_multiturn(proc.tokenizer, messages, out["_system"], out["_tools"], False)
    print(f"{tag} encode_multiturn pairs: {[(len(s), len(t)) for s, t in pairs]}")
