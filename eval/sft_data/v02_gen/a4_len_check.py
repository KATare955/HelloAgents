#!/usr/bin/env python3
"""A4 长度管控核对：用 SemanticPairTemplate 真实渲染 1000 条，统计训练序列 token 长度。

目的：确认 cutoff_len=8192 对 v02 数据零截断（R1 是 240 条 max~4532；v02 桶2 多轮更长）。
输出：train/val 的 max / p99 / p95 / 条数。
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOK_DIR = HERE / ".tokenizer_qwen3"

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(HERE))

from transformers import AutoTokenizer  # noqa: E402

import lf_extensions  # noqa: F401,E402  (注册扩展点)
from llamafactory.data.template import TEMPLATES  # noqa: E402
from llamafactory.data.data_utils import Role  # noqa: E402

tokenizer = AutoTokenizer.from_pretrained(str(TOK_DIR), trust_remote_code=True)
tpl = TEMPLATES["qwen3_semantic_pair"]
tpl.enable_thinking = False


def render(rec):
    """按 converter 输出形态 → SemanticPairTemplate.encode_multiturn → 最终 input_ids。"""
    system = rec.get("system", "")
    msgs = []
    for c in rec["conversations"]:
        role_map = {
            "human": Role.USER.value,
            "gpt": Role.ASSISTANT.value,
            "function_call": Role.FUNCTION.value,
            "observation": Role.OBSERVATION.value,
        }
        m = {"role": role_map[c["from"]], "content": c.get("value") or ""}
        if "loss" in c:
            m["loss"] = bool(c["loss"])
        msgs.append(m)
    prompt, response = msgs[:-1], msgs[-1:]
    encoded_pairs = tpl.encode_multiturn(tokenizer, prompt + response, system, None)
    return sum(len(s) + len(t) for s, t in encoded_pairs)


for name in ("sft_train.json", "sft_val.json"):
    path = HERE / "llamafactory_v02" / name
    recs = json.loads(Path(path).read_text(encoding="utf-8"))
    lengths = [render(r) for r in recs]
    lengths.sort()
    n = len(lengths)
    pct = lambda p: lengths[min(n - 1, int(n * p))]  # noqa: E731
    print(f"{name}: n={n}  max={max(lengths)}  p99={pct(0.99)}  p95={pct(0.95)}  "
          f"min={min(lengths)}  超8192条数={sum(1 for x in lengths if x > 8192)}")
