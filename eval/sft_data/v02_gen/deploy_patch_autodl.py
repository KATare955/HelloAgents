#!/usr/bin/env python3
"""AutoDL 上机部署脚本：llamafactory 0.9.6.dev0 三处 patch（A2 扩展点配套）。

1. data/__init__.py 末尾：from .lf_extensions import *（注册 qwen3_semantic_pair / v02_react）
2. processor/supervised.py 两处段数校验放宽：SemanticPairTemplate 跳过奇偶校验
"""
import re
from pathlib import Path

LF = Path("/root/autodl-tmp/LLaMA-Factory/src/llamafactory")
INIT = LF / "data" / "__init__.py"
SUP = LF / "data" / "processor" / "supervised.py"

# 1. __init__.py
init_text = INIT.read_text(encoding="utf-8")
if "lf_extensions" not in init_text:
    init_text += '\nfrom .lf_extensions import *  # noqa: F401  (v02 语义配对扩展点)\n'
    INIT.write_text(init_text, encoding="utf-8")
    print("[OK] __init__.py import 已添加")
else:
    print("[SKIP] __init__.py 已含 lf_extensions")

# 2. supervised.py 顶部 import（isinstance 检查需要）
sup_text = SUP.read_text(encoding="utf-8")
if "SemanticPairTemplate" not in sup_text:
    # 在最后一个顶层 import 之后插入
    lines = sup_text.split("\n")
    insert_at = 0
    for idx, line in enumerate(lines):
        if line.startswith("from ") or line.startswith("import "):
            insert_at = idx + 1
    lines.insert(insert_at, "from ..lf_extensions import SemanticPairTemplate  # noqa: F401  (v02 语义配对)")
    sup_text = "\n".join(lines)
    print("[OK] supervised.py import 已插入")

# 3. 两处段数校验放宽
OLD = 'if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:'
NEW = (
    'if (len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1) '
    "and not isinstance(self.template, SemanticPairTemplate):"
)
n = sup_text.count(OLD)
if n == 2:
    sup_text = sup_text.replace(OLD, NEW)
    SUP.write_text(sup_text, encoding="utf-8")
    print(f"[OK] supervised.py 段数校验放宽 2 处")
elif n == 0:
    print("[SKIP] supervised.py 校验已 patch 过或结构不同（需人工核对）")
else:
    print(f"[WARN] 校验出现 {n} 处（预期 2），未自动修改，需人工核对")
