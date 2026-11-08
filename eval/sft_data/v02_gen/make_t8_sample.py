# -*- coding: utf-8 -*-
"""T8 矩阵逐格抽检：24 个非空格子每格抽 2-3 条（共 ~52 条）出树状视图 → 人工复核

24 格定义（03 v0.2 的 2.1 矩阵；"跨场景"列落地为具体 scenario 字段）：
    桶1（6 格）: 试驾 / 金融 / 跨场景·优惠 / 跨场景·成交价 / 跨场景·竞品 / 跨场景·金融组合
    桶2（3 格）: 试驾 / 金融 / 竞品
    桶3（4 格）: 试驾 / 金融 / 竞品 / 跨场景
    桶4（5 格）: 试驾 / 金融 / 车型参数 / 跨场景·竞品 / 跨场景·成交价
    桶5（6 格）: 试驾 / 车型参数 / 优惠 / 成交价 / 竞品 / 金融

抽样策略：格内按 sid 前缀（maker 子类）round-robin 分配配额，优先覆盖不同形态；
前缀内均匀间隔取。抽中样本复用 gen_core.render_tree 渲染。

用法（在 HelloAgents 目录下）：
    .venv/bin/python eval/sft_data/v02_gen/make_t8_sample.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_core import SFT_DATA_DIR, render_tree  # noqa: E402

DATA = SFT_DATA_DIR / "v02_full_1000.jsonl"
OUT = SFT_DATA_DIR / "v02_T8_抽检树状视图.md"

# (格名, bucket, scenario, 抽几条, 可选 sid 前缀过滤器)
GRID: List[Tuple[str, str, str, int, List[str]]] = [
    ("桶1×试驾预约", "bucket1_clarify", "试驾预约", 3, []),
    ("桶1×金融测算", "bucket1_clarify", "金融方案测算", 2, ["v02_full_b1fn_"]),
    ("桶1×跨场景·优惠政策", "bucket1_clarify", "优惠政策查询", 2, []),
    ("桶1×跨场景·历史成交价", "bucket1_clarify", "历史成交价", 2, []),
    ("桶1×跨场景·竞品对比", "bucket1_clarify", "竞品对比", 2, []),
    ("桶1×跨场景·金融+优惠组合", "bucket1_clarify", "金融方案测算", 2, ["v02_full_b1cs_d"]),
    ("桶2×试驾预约", "bucket2_correction", "试驾预约", 4, []),
    ("桶2×金融测算", "bucket2_correction", "金融方案测算", 2, []),
    ("桶2×竞品对比", "bucket2_correction", "竞品对比", 2, []),
    ("桶3×试驾预约", "bucket3_multihop", "试驾预约", 2, []),
    ("桶3×金融测算", "bucket3_multihop", "金融方案测算", 3, ["v02_full_b3fn_"]),
    ("桶3×竞品对比", "bucket3_multihop", "竞品对比", 2, []),
    ("桶3×跨场景", "bucket3_multihop", "金融方案测算", 2, ["v02_full_b3cs_"]),
    ("桶4×试驾预约", "bucket4_degrade", "试驾预约", 2, []),
    ("桶4×金融测算", "bucket4_degrade", "金融方案测算", 2, []),
    ("桶4×车型参数", "bucket4_degrade", "车型参数查询", 2, []),
    ("桶4×跨场景·竞品对比", "bucket4_degrade", "竞品对比", 2, []),
    ("桶4×跨场景·历史成交价", "bucket4_degrade", "历史成交价", 2, []),
    ("桶5×试驾预约", "bucket5_coverage", "试驾预约", 3, []),
    ("桶5×车型参数", "bucket5_coverage", "车型参数查询", 2, []),
    ("桶5×优惠政策", "bucket5_coverage", "优惠政策查询", 2, []),
    ("桶5×历史成交价", "bucket5_coverage", "历史成交价", 3, []),
    ("桶5×竞品对比", "bucket5_coverage", "竞品对比", 2, []),
    ("桶5×金融测算", "bucket5_coverage", "金融方案测算", 2, []),
]

# 修复点必选抽检：闭包修复（b2td_z）+ 成交价无数据兜底（b5tp_a02 为30天无数据形态）
PIN = {
    "桶2×试驾预约": "v02_full_b2td_z00",
    "桶5×历史成交价": "v02_full_b5tp_a02",
}

_PREFIX_RE = re.compile(r"(v02_[a-z0-9_]+?)\d{2,3}$")


def _prefix(sid: str) -> str:
    m = _PREFIX_RE.match(sid)
    return m.group(1) if m else sid


def main() -> None:
    samples: List[Dict[str, Any]] = [
        json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    by_id = {s["id"]: s for s in samples}

    picked: List[Tuple[str, str]] = []  # (格名, sid)
    for grid_name, bucket, scenario, k, sid_prefixes in GRID:
        ids = [
            s["id"] for s in samples
            if s["bucket"] == bucket and s["scenario"] == scenario
            and (not sid_prefixes or any(s["id"].startswith(p) for p in sid_prefixes))
        ]
        # 必选样本（修复点）先行入抽，剩余配额 round-robin 分配
        pinned: List[str] = []
        if grid_name in PIN and PIN[grid_name] in ids:
            pinned.append(PIN[grid_name])
            ids = [i for i in ids if i != PIN[grid_name]]
        # 前缀分组（保持出现顺序），配额 round-robin 分配优先覆盖多形态
        groups: Dict[str, List[str]] = defaultdict(list)
        for sid in ids:
            groups[_prefix(sid)].append(sid)
        group_order = sorted(groups, key=lambda p: -len(groups[p]))
        quota: Dict[str, int] = defaultdict(int)
        for i in range(k - len(pinned)):
            quota[group_order[i % len(group_order)]] += 1
        for p in group_order:
            if quota[p] == 0:
                continue
            g = groups[p]
            step = max(1, len(g) // quota[p])
            take = g[::step][: quota[p]]
            for sid in take:
                picked.append((grid_name, sid))
        for sid in pinned:
            picked.append((grid_name, sid))

    # 渲染树状视图：按格分组
    by_grid: Dict[str, List[str]] = defaultdict(list)
    for grid_name, sid in picked:
        by_grid[grid_name].append(sid)

    lines: List[str] = []
    lines.append("# v02 全量 1000 条 — T8 矩阵逐格抽检树状视图")
    lines.append("")
    lines.append(f"抽检 {len(picked)} 条（24 个非空格子，每格 2-3 条）。")
    lines.append("每条样本树状展示消息轨迹；🔴 标记 loss_mask=false 编造段（不计 loss）。")
    lines.append("")
    lines.append("## 抽检清单")
    lines.append("")
    for grid_name, sids in by_grid.items():
        lines.append(f"- **{grid_name}**（{len(sids)} 条）: " + "、".join(f"`{s}`" for s in sids))
    lines.append("")
    lines.append("---")
    lines.append("")

    for grid_name, sid in picked:
        s = by_id[sid]
        lines.append(f"## 格 [{grid_name}] ─ {sid}")
        lines.append("")
        tree = render_tree([s]).splitlines()
        # 跳过 render_tree 首行（样本标题，与格标题重复）
        lines.extend(tree[1:])
        lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"抽检 {len(picked)} 条 / 24 格")
    for grid_name, sids in by_grid.items():
        print(f"  {grid_name}: {len(sids)} 条 {sids}")
    print(f"输出: {OUT}")


if __name__ == "__main__":
    main()
