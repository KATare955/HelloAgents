#!/usr/bin/env python3
"""convert_v02_to_llamafactory.py — v02 轨迹 → llamafactory sharegpt 兼容 JSON（语义配对扩展点配套）

与 R1 版 convert_to_llamafactory.py 的差异：
  1. 输入 v02_full_1000.jsonl，按 split_train_val.py 的划分清单分流 train/val
  2. loss_mask 透传：assistant 段 loss_mask=false → 该条 conversation 加 "loss": false
     （由 V02ReactConverter 透传给 SemanticPairTemplate 消费，无需源码 patch loss 逻辑）
  3. 输出 dataset_info.json 注册名 sales_agent_v02_train / val，formatting=v02_react
  4. 位置校验替换为语义配对前提校验（官方奇偶交替约束对桶 2 形态不成立——
     官方 converter/encode_multiturn 均无法表达，由 lf_extensions.py 扩展点承接）
映射规则同 R1：system→顶层；user→human；assistant 调 Thought/业务工具→function_call
（arguments json.loads 展开为对象）；Finish→gpt 收尾；tool→observation。

用法：
  python3 eval/sft_data/v02_gen/convert_v02_to_llamafactory.py
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

SFT_DIR = Path(__file__).resolve().parent
SRC = SFT_DIR.parent / "v02_full_1000.jsonl"
OUTDIR = SFT_DIR / "llamafactory_v02"
TRAIN_IDS = SFT_DIR / "train_ids.jsonl"
VAL_IDS = SFT_DIR / "val_ids.jsonl"


def load_tools_str() -> str:
    schemas = json.loads((SFT_DIR.parent / "tool_schemas_react.json").read_text(encoding="utf-8"))
    return json.dumps(schemas, ensure_ascii=False)


def convert_record(rec: Dict[str, Any], tools_str: str) -> Optional[Dict[str, Any]]:
    convs: List[Dict[str, str]] = []
    system = ""
    n_masked = 0
    for m in rec.get("messages", []):
        role = m.get("role")
        if role == "system":
            system = m.get("content") or ""
        elif role == "user":
            convs.append({"from": "human", "value": m.get("content") or ""})
        elif role == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                fn = tcs[0].get("function", {})
                name = fn.get("name")
                raw_args = fn.get("arguments", {})
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                if name == "Finish":
                    convs.append({"from": "gpt", "value": args.get("answer", "")})
                else:
                    convs.append({
                        "from": "function_call",
                        "value": json.dumps({"name": name, "arguments": args}, ensure_ascii=False),
                    })
            elif m.get("content"):
                convs.append({"from": "gpt", "value": m["content"]})
            else:
                continue  # assistant 段既无 tool_calls 也无 content，跳过
            # loss_mask 透传：编造段不参与训练 loss
            if m.get("loss_mask", True) is False:
                convs[-1]["loss"] = False
                n_masked += 1
        elif role == "tool":
            convs.append({"from": "observation", "value": m.get("content") or ""})
    if not convs:
        return None
    return {"conversations": convs, "system": system, "tools": tools_str, "_n_masked": n_masked}


def validate_semantic(convs: List[Dict[str, str]]) -> Optional[str]:
    """语义配对前提校验（替换官方奇偶位置校验）：
    - 首段 human、末段 gpt（末轮 gpt 收尾）
    - loss=false 仅允许出现在 function_call 段（编造轮）
    - target 段（gpt/function_call）不相邻（每段 target 前必有 source）
    """
    if not convs:
        return "空 conversations"
    if convs[0]["from"] != "human":
        return f"首段应为human，实为{convs[0]['from']}"
    if convs[-1]["from"] != "gpt":
        return f"末段应为gpt，实为{convs[-1]['from']}"
    TARGET = ("gpt", "function_call")
    for i, c in enumerate(convs):
        if c.get("loss") is False and c["from"] != "function_call":
            return f"loss=false 仅允许 function_call 段，位置{i+1}实为{c['from']}"
        if i > 0 and c["from"] in TARGET and convs[i - 1]["from"] in TARGET:
            return f"target 段相邻：位置{i}与{i+1}"
    return None


def main():
    records = {r["id"]: r for r in
               (json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip())}
    train_ids = [json.loads(l)["id"] for l in TRAIN_IDS.read_text(encoding="utf-8").splitlines() if l.strip()]
    val_ids = [json.loads(l)["id"] for l in VAL_IDS.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(train_ids) == 960 and len(val_ids) == 40

    OUTDIR.mkdir(parents=True, exist_ok=True)
    tools_str = load_tools_str()

    def build(id_list: List[str]):
        out, bad, masked_by_bucket = [], [], defaultdict(int)
        for sid in id_list:
            rec = records[sid]
            conv = convert_record(rec, tools_str)
            if conv is None:
                bad.append({"id": sid, "reason": "空 conversations"})
                continue
            n_masked = conv.pop("_n_masked")
            verr = validate_semantic(conv["conversations"])
            if verr:
                bad.append({"id": sid, "reason": verr})
                continue
            if n_masked:
                masked_by_bucket[rec["bucket"]] += 1
            out.append(conv)
        return out, bad, masked_by_bucket

    train, bad_t, masked_t = build(train_ids)
    val, bad_v, masked_v = build(val_ids)
    assert not bad_t and not bad_v, f"位置校验失败：{bad_t[:3]} {bad_v[:3]}"

    (OUTDIR / "sft_train.json").write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTDIR / "sft_val.json").write_text(json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8")

    dataset_info = {
        "sales_agent_v02_train": {
            "file_name": "sft_train.json",
            "formatting": "v02_react",
        },
        "sales_agent_v02_val": {
            "file_name": "sft_val.json",
            "formatting": "v02_react",
        },
    }
    (OUTDIR / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[out] train={len(train)} → {OUTDIR/'sft_train.json'}")
    print(f"[out] val  ={len(val)} → {OUTDIR/'sft_val.json'}")
    print(f"[mask] train 含 loss=false 段的样本按桶分布：{dict(sorted(masked_t.items()))}")
    print(f"[mask] val   含 loss=false 段的样本按桶分布：{dict(sorted(masked_v.items()))}")
    # 抽样展示一条桶2样本的角色序列 + loss 标记
    for conv in train:
        if any(c.get("loss") is False for c in conv["conversations"]):
            seq = [c["from"] + ("[loss=false]" if c.get("loss") is False else "") for c in conv["conversations"]]
            print("\n[样例] 桶2样本 conversations：")
            print("   " + " → ".join(seq))
            break
    print("\n[完成] 上传 llamafactory_v02/ 三文件至 AutoDL LLaMA-Factory/data/ 并合并 dataset_info.json；")
    print("[完成] 同时将 lf_extensions.py 放至 src/llamafactory/data/ 并按模块 docstring 应用 tuner import + processor patch")


if __name__ == "__main__":
    main()
