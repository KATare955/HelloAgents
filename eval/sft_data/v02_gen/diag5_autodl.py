#!/usr/bin/env python3
"""远程诊断 v5：逐步复刻 loader 链路，打印每环节中间状态，精确定位。"""
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/LLaMA-Factory/src")

from llamafactory.hparams import get_train_args, read_args
from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.parser import get_dataset_list
from llamafactory.data.converter import align_dataset
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor, IGNORE_INDEX

model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(read_args())

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
template = get_template_and_fix_tokenizer(tokenizer, data_args)

# 环节1：get_dataset_list（dataset_info 解析）
ds_list = get_dataset_list(data_args.dataset, data_args.dataset_dir)
attr = ds_list[0]
print("环节1 formatting:", attr.formatting, "| dataset_name:", attr.dataset_name, "| split:", attr.split)

# 环节2：load_dataset（原始 JSON，复刻 _load_single_dataset 的 file 分支）
from datasets import load_dataset

local_path = str(Path(data_args.dataset_dir) / attr.dataset_name)
data_files = [local_path]
data_path = "json"  # FILEEXT2TYPE["json"]
dataset = load_dataset(
    path=data_path, data_files=data_files, split=attr.split, num_proc=1
)
print("环节2 原始 dataset:", len(dataset), "条 | 列:", dataset.column_names)
print("环节2 首条 from:", [c["from"] for c in dataset[0]["conversations"]])

# 环节3：align_dataset（converter map）
aligned = align_dataset(dataset, attr, data_args, training_args)
print("环节3 align 后列:", aligned.column_names)
print("环节3 _prompt role:", [m["role"] for m in aligned[0]["_prompt"]])
print("环节3 _response role:", [m["role"] for m in aligned[0]["_response"]])

# 环节4：preprocess（num_proc=1）
proc = SupervisedDatasetProcessor(template=template, tokenizer=tokenizer, processor=None, data_args=data_args)
kwargs = dict(num_proc=1, load_from_cache_file=False, desc="diag")
encoded = aligned.map(
    proc.preprocess_dataset, batched=True, batch_size=data_args.preprocessing_batch_size,
    remove_columns=aligned.column_names, **kwargs,
)
labels = encoded[0]["labels"]
valid = [x for x in labels if x != IGNORE_INDEX]
print(f"环节4 preprocess 后 sample[0]: input_ids={len(encoded[0]['input_ids'])} valid={len(valid)}")
