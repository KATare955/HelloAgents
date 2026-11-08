#!/usr/bin/env python3
"""远程诊断 v4：直接调 preprocess_dataset（等价 dataset.map 单进程），定位差异。"""
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

recs = json.loads(
    Path("/root/autodl-tmp/HelloAgents/eval/sft_data/v02_gen/llamafactory_v02/sft_train.json").read_text()
)
Conv = DATASET_CONVERTERS["v02_react"]

# 构造 batched examples（等价 loader 里 converter map 后的输入）
batched = {"_prompt": [], "_response": [], "_system": [], "_tools": [], "_images": [], "_videos": [], "_audios": []}
for rec in recs[:3]:
    conv = Conv(dataset_attr=None, data_args=data_args)
    out = conv(rec)
    for k in batched:
        batched[k].append(out[k])

out2 = proc.preprocess_dataset(batched)
for i in range(3):
    labels = out2["labels"][i]
    valid = [x for x in labels if x != IGNORE_INDEX]
    print(f"preprocess sample[{i}]: input_ids={len(out2['input_ids'][i])} valid={len(valid)}")
