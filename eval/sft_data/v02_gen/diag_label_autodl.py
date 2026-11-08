#!/usr/bin/env python3
"""远程诊断：走训练同款完整 loader 链，检查 1 条样本的 labels 有效 token 数。"""
import sys

sys.path.insert(0, "/root/autodl-tmp/LLaMA-Factory/src")

from llamafactory.hparams import get_train_args, read_args
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer

YAML = "/root/autodl-tmp/HelloAgents/eval/sft_data/autodl/qwen3_14b_lora_sft.yaml"

model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(
    read_args()
)
print("dataset:", data_args.dataset, "| eval:", data_args.eval_dataset)
print("template:", data_args.template, "| mask_history:", data_args.mask_history)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
template = get_template_and_fix_tokenizer(tokenizer, data_args)
print("template class:", type(template).__name__)
print("enable_thinking:", template.enable_thinking)

dataset_module = get_dataset(template=template, model_args=model_args, data_args=data_args, training_args=training_args, stage="sft", tokenizer=tokenizer)
train_ds = dataset_module["train_dataset"]

print("train size:", len(train_ds))
IGNORE = -100
for idx in [0, 1, 2]:
    ex = train_ds[idx]
    labels = ex["labels"]
    valid = [x for x in labels if x != IGNORE]
    print(f"sample[{idx}]: input_ids={len(ex['input_ids'])} labels={len(labels)} valid={len(valid)}")

# 找一条桶2样本（多 loss 段）
for idx in range(len(train_ds)):
    ex = train_ds[idx]
    labels = ex["labels"]
    valid = [x for x in labels if x != IGNORE]
    if valid:
        print(f"first-valid sample[{idx}]: input_ids={len(ex['input_ids'])} valid={len(valid)}")
        break
