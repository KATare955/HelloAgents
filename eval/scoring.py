"""评测判分逻辑（单一来源，供 baseline_eval.py 与 rescore.py 共用）

判分逻辑 v3（2026-11-04，T3 重构）：
    基础规则：expected_tools ⊆ actual_tools 即正确（宽松子集判定，保持不变）
    澄清规则：题目带 accept_clarification 标记（故意缺必填参数 / 目标不可达）时，
              "未调用 target_tool + 回应含澄清意图（反问补全/告知不可达/替代建议）"
              可豁免 target_tool 本身；豁免为精确粒度——移除 target_tool 后剩余期望链路
              仍未命中 → 仍判错误，与 verifier/rules.py 的精确豁免口径对齐
              （v2 的两个缺陷："全有或全无"豁免掩盖双跳题第一跳缺失；只查 response
              非空不查澄清意图，瞎扯非空回复也会被豁免，v3 一并修复）。

v3 变更说明（judge_tool_correct 降级为规则检查项）：
    在新 verify 体系（三维子项 + 7 hard cap）中，本函数不再单独决定最终分数，
    仅作为 tool_call 子项的底子（集合运算逻辑复用）；最终分数由 verifier/rules.py 决定。
    本函数保持纯函数可复算，历史结果可重算（rescore.py 依赖此点）。

为什么 accept_clarification 不破坏 v1.1 冻结：
    它只是**判分标注**，不改变 question 文本、expected_tools、工具 schema，
    因此模型输入完全不变、模型行为不变，仅影响"如何评判已产出的输出"。
    故 v1.1 评测集对模型而言仍是冻结的，三阶段对比依然有效。
"""

from typing import Any, Dict, List, Optional, Tuple
import re

# 判分版本号（env_snapshot 追溯用，判分逻辑变更时必须递增并双文档同步）
JUDGE_VERSION = "v3"

# 澄清意图模式（v3 clarification_policy 消费：豁免要求 response 非空且含澄清意图；
# v2 只查非空是缺陷——模型瞎扯非空回复不该豁免）。
# 语义类别：反问补全 / 告知不可达 / 替代建议；词表来自 rescoreB 4 翻案 case
# 与 8B 校准 case 的真实语料（请提供/请问/需要您提供/不存在/为您找到…）。
CLARIFICATION_PATTERNS = (
    r"请(?:您|你)?(?:提供|告知|补充|确认|留下|输入|完善)",   # 请提供/请告知/请补充
    r"(?:需要|还需)(?:您|你)(?:提供|告知|补充)",             # 需要您提供
    r"请问",                                               # 请问…
    r"是否(?:需要|可以|要)",                                 # 是否需要/是否可以
    r"不存在|未找到|无法(?:找到|查询|预约|完成|获取)",        # 告知不可达
    r"为您(?:找到|推荐|介绍|挑选)",                           # 替代建议
)


def _has_clarification_intent(response: str) -> bool:
    """response 是否含澄清意图（反问补全 / 告知不可达 / 替代建议）。"""
    if not (response or "").strip():
        return False
    return any(re.search(p, response) for p in CLARIFICATION_PATTERNS)


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

    # v3 规则：澄清/降级/拒答豁免 target_tool（精确粒度 + 澄清意图验证）
    if accept_clarification:
        target = accept_clarification.get("target_tool")
        # 豁免条件：没有调用需要缺失参数的目标工具（即没有编造参数硬调），
        # 且回应含澄清意图（反问补全 / 告知不可达 / 替代建议）。
        if target and target not in actual_set and _has_clarification_intent(response):
            # 精确豁免：只豁免 target_tool 本身——移除后剩余期望链路
            # 仍未命中 → 仍判错（如双跳题漏了第一跳，不能被澄清豁免掩盖）。
            remaining = expected_set - {target}
            if remaining.issubset(actual_set):
                return True, [], []
            # target 已豁免，missing 只报剩余缺失（不报 target 本身）
            return False, sorted(remaining - actual_set), extra

    return False, missing, extra
