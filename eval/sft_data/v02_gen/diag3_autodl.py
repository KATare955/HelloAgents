#!/usr/bin/env python3
"""远程诊断 v3：直接实例化 SupervisedDatasetProcessor，复刻 preprocess 内部调用。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/LLaMA-Factory/src")

from llamafactory.hparams import get_train_args, read_args
from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.converter import DATASET_CONVERTERS
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor, IGNORE_INDEX

model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(read_args())

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
template = get_template_and_fix_tokenizer(tokenizer, data_args)

proc = SupervisedDatasetProcessor(template=template, tokenizer=tokenizer, processor=None, data_args=data_args)
print("processor.template:", type(proc.template).__name__)
print("processor.template.enable_thinking:", proc.template.enable_thinking)

recs = json.loads(
    Path("/root/autodl-tmp/HelloAgents/eval/sft_data/v02_gen/llamafactory_v02/sft_train.json").read_text()
)
Conv = DATASET_CONVERTERS["v02_react"]
conv = Conv(dataset_attr=None, data_args=data_args)
out = conv(recs[0])

# 复刻 _encode_data_example 的调用
prompt, response = out["_prompt"], out["_response"]
messages = proc.template.mm_plugin.process_messages(prompt + response, [], [], [], proc.processor)
discarding_history_cot = proc.data_args.mask_history and not proc.template.preserve_thinking
print("discarding_history_cot:", discarding_history_cot, "| preserve_thinking:", proc.template.preserve_thinking)
encoded_pairs = proc.template.encode_multiturn(proc.tokenizer, messages, out["_system"], out["_tools"], discarding_history_cot)
print(f"encode_multiturn pairs: {len(encoded_pairs)}")
for i, (src, tgt) in enumerate(encoded_pairs):
    print(f"  pair{i}: src={len(src)} tgt={len(tgt)}")

# 完整走 processor 的 _encode_data_example
input_ids, labels = proc._encode_data_example(
    prompt=out["_prompt"], response=out["_response"], system=out["_system"], tools=out["_tools"],
    images=[], videos=[], audios=[],
)
valid = [x for x in labels if x != IGNORE_INDEX]
print(f"_encode_data_example: input_ids={len(input_ids)} labels={len(labels)} valid={len(valid)}")
