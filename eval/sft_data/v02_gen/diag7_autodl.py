#!/usr/bin/env python3
"""远程诊断 v7：验证 cache 假说——load_dataset 原始输出 vs 磁盘文件 vs aligned 输出。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/LLaMA-Factory/src")

from llamafactory.hparams import get_train_args, read_args
from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.parser import get_dataset_list
from llamafactory.data.converter import align_dataset

model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(read_args())

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
template = get_template_and_fix_tokenizer(tokenizer, data_args)

attr = get_dataset_list(data_args.dataset, data_args.dataset_dir)[0]
file_path = Path(data_args.dataset_dir) / attr.dataset_name

# 1) 磁盘文件 json.loads
recs = json.loads(file_path.read_text())
print("磁盘 recs[0] conversations keys:", [sorted(c.keys()) for c in recs[0]["conversations"]])
print("磁盘 recs[0] loss 值:", [c.get("loss", "<无键>") for c in recs[0]["conversations"]])

# 2) load_dataset 原始输出（注意 cache）
from datasets import load_dataset

dataset = load_dataset(path="json", data_files=[str(file_path)], split="train", num_proc=1)
d0 = dataset[0]["conversations"]
print("load_dataset[0] keys:", [sorted(c.keys()) for c in d0])
print("load_dataset[0] loss 值:", [c.get("loss", "<无键>") for c in d0])

# 3) 全数据集 loss 键统计（load_dataset 侧）
all_keys = {}
for ex in dataset:
    for c in ex["conversations"]:
        k = "has_loss" if "loss" in c else "no_loss"
        all_keys[k] = all_keys.get(k, 0) + 1
print("load_dataset 全量段统计:", all_keys)

# 4) 磁盘文件全量 loss 键统计
disk_keys = {}
for ex in recs:
    for c in ex["conversations"]:
        k = "has_loss" if "loss" in c else "no_loss"
        disk_keys[k] = disk_keys.get(k, 0) + 1
print("磁盘全量段统计:", disk_keys)

# 5) 文件 mtime 与 cache 目录
import os

print("磁盘文件 mtime:", os.path.getmtime(file_path))
cache_dir = Path(os.path.expanduser("~/.cache/huggingface/datasets"))
if cache_dir.exists():
    jsons = sorted(cache_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    print(f"cache 最近 json 文件（前5）:")
    for p in jsons[:5]:
        import datetime

        t = datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime("%m-%d %H:%M")
        print(f"  [{t}] {p}")
