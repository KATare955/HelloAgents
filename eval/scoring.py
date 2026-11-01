"""评测判分逻辑（单一来源，供 baseline_eval.py 与 rescore.py 共用）

判分逻辑 v2（2026-11-01，B 方案）：
    基础规则：expected_tools ⊆ actual_tools 即正确（宽松子集判定，保持不变）
    新增规则：若题目带 accept_clarification 标记（故意缺必填参数 / 目标不可达），
              则"未调用需要缺失参数的目标工具 + 给出了回应（反问/降级/拒答）"也判正确。

为什么 accept_clarification 不破坏 v1.1 冻结：
    它只是**判分标注**，不改变 question 文本、expected_tools、工具 schema，
    因此模型输入完全不变、模型行为不变，仅影响"如何评判已产出的输出"。
    故 v1.1 评测集对模型而言仍是冻结的，三阶段对比依然有效。
"""

from typing import Any, Dict, List, Optional, Tuple


def judge_tool_correct(
    expected_tools: List[str],
    actual_tools: List[str],
    response: str = "",
    accept_clarification: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str], List[str]]:
    """判定工具调用是否正确。

    返回 (tool_correct, missing_tools, extra_tools)。

    参数：
        expected_tools: 期望工具列表
        actual_tools: 实际调用工具列表
        response: 模型最终文本响应（用于判定"澄清/降级"是否给出了回应）
        accept_clarification: 可选标记 {"target_tool": ..., "reason": ...}，
            表示该题故意缺 target_tool 所需的必填参数（或目标不可达），
            此时"不调用 target_tool 而是反问/降级/拒答"属于正确行为。
    """
    expected_set = set(expected_tools or [])
    actual_set = set(actual_tools or [])

    # 基础规则：期望工具全部被调用即正确
    if expected_set.issubset(actual_set):
        return True, [], []

    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)

    # v2 规则：澄清/降级/拒答也算正确
    if accept_clarification:
        target = accept_clarification.get("target_tool")
        # 正确条件：没有调用需要缺失参数的目标工具（即没有编造参数硬调），
        # 且模型给出了非空回应（反问补全 / 告知不可达 / 推荐替代）。
        if target and target not in actual_set and (response or "").strip():
            return True, [], []

    return False, missing, extra
