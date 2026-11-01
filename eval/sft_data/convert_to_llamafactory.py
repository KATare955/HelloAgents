#!/usr/bin/env python3
"""convert_to_llamafactory.py — rich 轨迹 JSONL → LLaMA-Factory sharegpt+tools 格式

映射规则（对齐官方 glaive_toolcall 格式 + ReActAgent 推理形态）：
  system 消息               → 顶层 "system" 字段（字符串）
  8 工具 schema(Thought/Finish/6业务) → 顶层 "tools" 字段（JSON 数组的字符串化）
  user 消息                 → {"from":"human","value":...}                    (奇数位)
  assistant 调 Thought/业务工具 → {"from":"function_call","value":'{"name":..,"arguments":{..}}'} (偶数位)
  tool observation          → {"from":"observation","value":<原始 to_model_text>} (奇数位)
  assistant 调 Finish(终止)  → {"from":"gpt","value":<answer>}                 (偶数位, 收尾)
      —— ReActAgent 也接受纯文本作答(runtime.py:235-241)，故 Finish 转标准 gpt 收尾最稳妥

官方约束：human/observation 必须奇数位、gpt/function_call 偶数位；损失只在 gpt/function_call 上算。
本映射天然满足交替；转换器会逐条校验位置，违规则丢弃并记录。

用法：
  .venv/bin/python eval/sft_data/convert_to_llamafactory.py \
      --in eval/sft_data/sft_trajectories.jsonl \
      --outdir eval/sft_data/llamafactory --val-ratio 0.1 --seed 42
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SFT_DIR = Path(__file__).resolve().parent


def load_tools_str() -> str:
    """8 工具 schema（Thought+Finish+6业务）字符串化，作为顶层 tools 字段。"""
    p = SFT_DIR / "tool_schemas_react.json"
    schemas = json.loads(p.read_text(encoding="utf-8"))
    return json.dumps(schemas, ensure_ascii=False)


# 数据质量闸门：已知 mock 工具间价格不一致
#   competitor_compare 对 E5旗舰quattro 返 289900，而 text_search/calculate_finance 用指导价 319900。
#   若同一条答案同时引用两价，会造成“同一车双报价”的内部矛盾（虽各自接地于工具）。
#   此类记录仅 ~1.2%，且不影响工具调用评测，但为训练集干净故过滤。
CONFLICT_PRICE_PAIRS = [("28.99万", "31.99万"), ("289900", "319900")]


def has_price_conflict(rec: Dict[str, Any]) -> bool:
    """检测 final_answer 是否对同一车同时报了两个不一致的价格。"""
    fa = (rec.get("meta", {}) or {}).get("final_answer", "") or ""
    return any(a in fa and b in fa for a, b in CONFLICT_PRICE_PAIRS)


def convert_record(rec: Dict[str, Any], tools_str: str) -> Optional[Dict[str, Any]]:
    """rich messages → LLaMA-Factory sharegpt 单条。返回 None 表示无法转换。"""
    convs: List[Dict[str, str]] = []
    system = ""
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
                    # 终止：转标准 gpt 文本收尾（harness 接受纯文本作答）
                    convs.append({"from": "gpt", "value": args.get("answer", "")})
                else:
                    # Thought 或业务工具：function_call
                    convs.append({
                        "from": "function_call",
                        "value": json.dumps({"name": name, "arguments": args}, ensure_ascii=False),
                    })
            elif m.get("content"):
                # 兜底：纯文本 assistant（正常轨迹不出现）
                convs.append({"from": "gpt", "value": m["content"]})
        elif role == "tool":
            convs.append({"from": "observation", "value": m.get("content") or ""})
    if not convs:
        return None
    return {"conversations": convs, "system": system, "tools": tools_str}


def validate_positions(convs: List[Dict[str, str]]) -> Optional[str]:
    """校验官方位置约束：奇数位 human/observation，偶数位 gpt/function_call；末轮须 gpt。"""
    for i, c in enumerate(convs):
        pos = i + 1
        f = c["from"]
        if pos % 2 == 1:
            if f not in ("human", "observation"):
                return f"位置{pos}应为human/observation，实为{f}"
        else:
            if f not in ("gpt", "function_call"):
                return f"位置{pos}应为gpt/function_call，实为{f}"
    if convs[-1]["from"] != "gpt":
        return f"末轮应为gpt，实为{convs[-1]['from']}"
    if convs[0]["from"] != "human":
        return f"首轮应为human，实为{convs[0]['from']}"
    return None


def stratified_split(records: List[Dict[str, Any]], val_ratio: float, seed: int
                     ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """按 bucket 分层 9:1 划分，保证各桶在 train/val 均有代表。"""
    rng = random.Random(seed)
    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_bucket[r.get("bucket", "_")].append(r)
    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    for b in sorted(by_bucket):
        recs = by_bucket[b][:]
        rng.shuffle(recs)
        n = len(recs)
        n_val = int(round(n * val_ratio))
        if n >= 2:
            n_val = max(1, min(n_val, n - 1))  # 至少1条val、至少1条train
        else:
            n_val = 0
        val.extend(recs[:n_val])
        train.extend(recs[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def main():
    ap = argparse.ArgumentParser(description="rich 轨迹 → LLaMA-Factory sharegpt+tools")
    ap.add_argument("--in", dest="inp", default=str(SFT_DIR / "sft_trajectories.jsonl"))
    ap.add_argument("--outdir", default=str(SFT_DIR / "llamafactory"))
    ap.add_argument("--val-ratio", dest="val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_path = Path(args.inp)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tools_str = load_tools_str()

    raw = [json.loads(l) for l in in_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[in] 读入 {len(raw)} 条 rich 轨迹 ← {in_path}")

    converted: List[Dict[str, Any]] = []
    meta_index: List[Dict[str, Any]] = []  # 保留 bucket 用于分层
    bad: List[Dict[str, Any]] = []
    for rec in raw:
        if has_price_conflict(rec):
            bad.append({"id": rec.get("id"), "reason": "同车双报价(mock价格不一致)"})
            continue
        conv = convert_record(rec, tools_str)
        if conv is None:
            bad.append({"id": rec.get("id"), "reason": "空 conversations"})
            continue
        verr = validate_positions(conv["conversations"])
        if verr:
            bad.append({"id": rec.get("id"), "reason": verr})
            continue
        converted.append(conv)
        meta_index.append({"bucket": rec.get("bucket", "_"), "id": rec.get("id")})

    # 分层划分（用带 bucket 的包装，划分后再剥离）
    paired = list(zip(converted, meta_index))
    rng = random.Random(args.seed)
    by_bucket: Dict[str, List[Tuple]] = defaultdict(list)
    for pr in paired:
        by_bucket[pr[1]["bucket"]].append(pr)
    train_pr: List[Tuple] = []
    val_pr: List[Tuple] = []
    for b in sorted(by_bucket):
        items = by_bucket[b][:]
        rng.shuffle(items)
        n = len(items)
        n_val = int(round(n * args.val_ratio))
        if n >= 2:
            n_val = max(1, min(n_val, n - 1))
        else:
            n_val = 0
        val_pr.extend(items[:n_val])
        train_pr.extend(items[n_val:])
    rng.shuffle(train_pr)
    rng.shuffle(val_pr)

    train = [p[0] for p in train_pr]
    val = [p[0] for p in val_pr]

    train_path = outdir / "sft_train.json"
    val_path = outdir / "sft_val.json"
    train_path.write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    val_path.write_text(json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8")

    # dataset_info.json 注册片段（拷贝到 LLaMA-Factory/data/ 后合并）
    dataset_info = {
        "sales_agent_sft_train": {
            "file_name": "sft_train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
        },
        "sales_agent_sft_val": {
            "file_name": "sft_val.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
        },
    }
    (outdir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")

    # 报告
    def dist(prs):
        d = defaultdict(int)
        for _, m in prs:
            d[m["bucket"]] += 1
        return dict(sorted(d.items()))

    print(f"[out] train={len(train)} → {train_path}")
    print(f"[out] val  ={len(val)} → {val_path}")
    print(f"[dist] train {dist(train_pr)}")
    print(f"[dist] val   {dist(val_pr)}")
    print(f"[tools] 顶层 tools 字段含 {len(json.loads(tools_str))} 工具schema")
    if bad:
        print(f"[warn] 质量闸门丢弃 {len(bad)} 条(位置校验/同车双报价)：{bad[:5]}")
    # 抽样展示一条
    if train:
        print("\n[样例] train[0] conversations 角色序列：")
        print("   " + " → ".join(c["from"] for c in train[0]["conversations"]))
    print("\n[完成] 将 sft_train.json / sft_val.json / dataset_info.json 上传 AutoDL 的 LLaMA-Factory/data/")


if __name__ == "__main__":
    main()
