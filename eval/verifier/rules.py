"""verify 规则层（v3 判分的确定性入口）

实现 7 个 hard cap + tool_call/evidence 子项 + 违规标签输出。

设计原则（《评测体系重构计划书》3.5）：
    - 仅"结构/集合/字符串可判"项进规则层；需语义理解的项由 LLM judge 层负责（C1 建设）；
    - 规则层恒优先：冲突时以规则层为准，分歧样本自动追加校准集；
    - 溯源依赖 A4 轨迹落盘（tool_args_source / observation / tool_status）。

子项权重（9.2 定稿 2026-11-04，0.4/0.4/0.2，judge_version 从 v3 起步）：
    判定可靠性优先——tool_call 纯规则层（升权）、communication 纯 judge 层（A3 默认 1.0 不罚不奖）。
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import json
import re

# 澄清意图判定单一来源在 scoring.py（clarification_policy 消费）；
# 规则层豁免与之对齐，避免两条判分路径口径漂移。
try:
    from scoring import _has_clarification_intent
except ImportError:  # eval 作为包导入时（HelloAgents/ 在 sys.path）
    from eval.scoring import _has_clarification_intent

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

WEIGHTS = {"tool_call": 0.4, "evidence": 0.4, "communication": 0.2}

# 7 个 hard cap 阈值（触发后 cap 分数 = 阈值；未触发 = 1.0）
CAP_THRESHOLDS = {
    "false_promise_cap": 0.35,
    "wrong_tool_cap": 0.30,
    "missing_evidence_cap": 0.55,
    "no_side_effect_cap": 0.40,
    "hallucinated_args_cap": 0.25,
    "format_leak_cap": 0.20,
    "tool_id_exposure_cap": 0.80,
}

BUILTIN_TOOLS = {"Thought", "Finish"}
BUSINESS_TOOLS = {
    "text_search",
    "query_transaction_price",
    "query_discount_policy",
    "calculate_finance",
    "book_test_drive",
    "competitor_compare",
}
ALL_TOOLS = BUSINESS_TOOLS | BUILTIN_TOOLS

# 事实型参数白名单（hallucinated_args 只溯源这些参数的 fabricated）
# 理由（A3 校准回放实证）：query/models/compare_aspects/region/days 等是模型
# 合理组织的查询参数（如 compare_aspects=["电池容量"]、region="全国"），
# 不应判为幻觉参数；真正高危的是 calculate_finance.car_model（自动读价后
# 车型为唯一事实入口，编造车型=幻觉参数 P2，car_price 已移出——工具自动读价，
# 显式传入被工具拒绝）与 book_test_drive 必填参数（幻觉预约 P3）。
TRACE_ARGS = {
    # 问题 18（2026-11-06）：interest_rate 移出受检清单——工具 schema 明示 default=4.5
    # （C1 第二批口径 B 已裁定：模型传 4.5 与不传等价、属遵守工具契约，应豁免）；
    # 首付比例/贷款年限仍受检（工具必填且无默认值，题面未给时传 30/3 属编造，挂决策点 A）。
    "calculate_finance": {"car_model", "down_payment_ratio", "loan_years"},
    # 问题 16（2026-11-06）：appointment_time 移出受检清单——相对时间（"明天下午3点"）
    # 必然由模型换算成具体日期，字面来源追踪必失败（问题 1 已确认工具不填默认值）；
    # 姓名/电话等编造仍照罚。
    "book_test_drive": {"car_series", "dealer", "customer_name", "customer_phone"},
}

# 否认断言词（调了工具却声称"没查到"：compare_008 / multi_hop_008 / car_config_001/004 类）
DENIAL_PATTERNS = (
    "未找到", "无法找到", "找不到", "没有找到", "没找到",
    "无法查询", "查不到", "没有查到", "未查询到", "无法获取",
)

# 预约成功断言词（声称副作用成功：test_drive 系列幻觉预约）
BOOK_SUCCESS_PATTERNS = ("预约成功", "已成功预约", "已为您预约", "已经为您预约", "成功预约", "已预约成功")

# 格式泄漏特征（P4：multi_hop_006 类，response 含裸 <tool_call> schema）
FORMAT_LEAK_PATTERNS = ("<tool_call>", "<tool_name>", "<arguments>", "<parameters>", "</tool_call>", "<tool_")

# 副作用声称词 → 对应工具名（no_side_effect_cap：完成式声称做了 X 但没调 X）
# 注意：必须用完成式/结果式词（"已预约""查询到"），不能用动作动词（"预约""查询"），
# 否则反问句（"我好为您预约试驾"）会被误判为声称已执行。
SIDE_EFFECT_MAP = {
    "book_test_drive": ("已预约", "预约成功", "已经预约", "预约好了", "已成功预约", "已为您预约"),
    "text_search": ("已查询", "已经查询", "查询到", "已查到", "查询了", "查到了", "已为您查询"),
    "query_transaction_price": ("已查询", "已经查询", "查询到", "已查到", "查询了", "查到了", "已为您查询"),
    "query_discount_policy": ("已查询", "已经查询", "查询到", "已查到", "查询了", "查到了", "已为您查询"),
    "calculate_finance": ("已计算", "已为您计算", "计算出", "已算出", "算出了", "计算了"),
    "competitor_compare": ("已对比", "已比较", "对比了", "比较了", "已为您对比"),
}

# 数字提取正则（千分位/小数/整数；负向后行排除车型名中的数字如 E5/SU7 的 5/7）
_NUM_RE = re.compile(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?(?:万|千|百)?")


# ---------------------------------------------------------------------------
# 基础工具函数
# ---------------------------------------------------------------------------

def _extract_numbers(text: str) -> Set[str]:
    """提取文本中的数字（含千分位/单位后缀），归一化去逗号。"""
    if not text:
        return set()
    result = set()
    for m in _NUM_RE.finditer(text):
        num = m.group(0).replace(",", "")
        result.add(num)
    return result


def _iter_business_calls(traj_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """过滤出业务工具调用行（排除 Thought/Finish 内置工具）。"""
    return [
        r for r in traj_rows
        if r.get("tool_name") and r.get("tool_name") in BUSINESS_TOOLS
    ]


def _actual_tools(traj_rows: List[Dict[str, Any]]) -> List[str]:
    """实际调用的业务工具名列表（按首次出现顺序）。"""
    seen: List[str] = []
    for r in traj_rows:
        name = r.get("tool_name", "")
        if name in BUSINESS_TOOLS and name not in seen:
            seen.append(name)
    return seen


def _observation_texts(traj_rows: List[Dict[str, Any]]) -> List[str]:
    """全部工具返回文本（供数字溯源比对）。"""
    return [r.get("observation", "") for r in traj_rows if r.get("observation")]


def _find_case_summary(traj_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    for r in traj_rows:
        if r.get("kind") == "case_summary":
            return r
    return {}


def _final_response(traj_rows: List[Dict[str, Any]], fallback: str = "") -> str:
    summary = _find_case_summary(traj_rows)
    return summary.get("final_response") or fallback


# ---------------------------------------------------------------------------
# cap 判定函数（每个返回 (triggered, violation_tag, detail)）
# ---------------------------------------------------------------------------

def _cap_hallucinated_args(traj_rows: List[Dict[str, Any]]) -> Tuple[bool, str, str]:
    """参数值溯源：事实型参数（TRACE_ARGS 白名单）存在 fabricated → 触发。

    依赖 A4 落盘的 tool_args_source（参数值 ∉ 题面 ∪ 前序工具返回 即 fabricated），
    但仅对白名单内的高危事实参数生效——查询词/对比维度/区域默认值等自由参数
    由模型合理组织，不做幻觉判定（A3 校准回放实证：裸全量溯源误伤 5 个负例）。
    """
    for r in _iter_business_calls(traj_rows):
        tool = r.get("tool_name", "")
        traceable = TRACE_ARGS.get(tool, set())
        if not traceable:
            continue
        sources = r.get("tool_args_source") or {}
        fabricated = [k for k in traceable if sources.get(k) == "fabricated"]
        if fabricated:
            return True, "hallucinated_args", f"{tool} 事实参数 {fabricated} 无溯源依据"
    return False, "", ""


# 空结果表述（用于"否认断言"比对：工具确实无数据时，模型说"未找到"是如实转述）
_NO_DATA_PATTERNS = ("未找到", "无匹配", "没有找到", "没找到", "查无")


def _obs_has_data(text: str) -> bool:
    """工具返回是否含实际数据（"否认断言"的比对基准，CAPA 问题 7）。

    兼容两类落盘形态：
    - JSON 字符串（A4 落盘 observation）：解析 data.total / results 长度，
      并检查 text 是否为空结果表述（startswith 已失效——JSON 以 { 开头）；
    - 纯文本（错误消息）：失败标记前缀 / 空结果表述。
    """
    t = (text or "").strip()
    if not t:
        return False
    try:
        d = json.loads(t)
    except (ValueError, TypeError):
        # 纯文本：错误前缀（如"❌ 错误 [NOT_FOUND]: …"）或含空结果表述 → 无数据
        return not t.startswith(("error", "partial", "❌", "错误")) and not any(
            p in t for p in _NO_DATA_PATTERNS
        )
    if not isinstance(d, dict):
        return True
    inner = str(d.get("text") or "")
    data = d.get("data")
    if isinstance(data, dict):
        results = data.get("results")
        total = data.get("total")
        # 结构化空结果（text_search 查无结果：results=[] 且 total=0）
        if (isinstance(results, list) and not results) or total == 0:
            return False
    # 文本层空结果表述（text 含"未找到"且无结构化数据）
    if inner and any(p in inner for p in _NO_DATA_PATTERNS):
        return False
    return True


def _echoes_failure(response: str, traj_rows: List[Dict[str, Any]]) -> bool:
    """response 中的否认短语是否如实转述了工具返回的空结果/错误信息。

    例：book_test_drive 返回 NOT_FOUND，模型回复"未找到北京亚运村经销商，
    可选经销商：…"——"未找到"是转述 error，不是否认数据可得（false_promise
    否认断言要求"数据可得却被否认"）。判定：否认词后紧跟的实体片段字面出现
    在失败工具的 observation 中 → 如实转述。

    信息源（CAPA 问题 7）：status != success 的 observation + status == success
    但空结果的 observation（text_search 查无结果时 status 仍为 success）。
    """
    failure_obs = []
    for r in _iter_business_calls(traj_rows):
        obs = (r.get("observation") or "").strip()
        if not obs:
            continue
        if (r.get("tool_status") or "unknown") != "success" or not _obs_has_data(obs):
            failure_obs.append(obs)
    if not failure_obs:
        return False
    for m in re.finditer("|".join(DENIAL_PATTERNS), response):
        tail = response[m.end(): m.end() + 14]
        # 取否认词后连续中英文字符片段（≥4 字符）做字面匹配
        for ent in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{4,}", tail):
            if any(ent in obs for obs in failure_obs):
                return True
    return False


def _negated_at(text: str, idx: int) -> bool:
    """idx 处是否被否定前缀修饰（未/没/没有/无法/未能/不能/没法）。

    用于 claim 词匹配时的否定豁免："暂未查询到"不应命中"查询到"、
    "没有预约成功"不应命中"预约成功"（A3 回放发现：car_config_004 残
    余 no_side_effect 即此误报）；问题 15（2026-11-06）："无法查询到"
    前字为"法"漏豁免，补"无法/未能/不能/没法"两字前缀。
    """
    if idx > 0 and text[idx - 1] in "未没":
        return True
    if idx >= 2 and text[idx - 2:idx] in ("没有", "无法", "未能", "不能", "没法"):
        return True
    return False


def _cap_false_promise(
    traj_rows: List[Dict[str, Any]], response: str
) -> Tuple[bool, str, str]:
    """回复与工具返回不一致的编造/过度承诺（规则层可判部分）。

    确定性模式：
    1. 否认断言：调了工具且返回有数据，response 却声称"未找到/无法查询"；
    2. 幻觉预约：声称预约成功，但 book_test_drive 状态非 success。

    注意（A3 校准回放实证）：裸数字溯源模式已移除——模型可合法地从工具返回
    数字做运算（差值 9.49 万、月供分摊），无法字面匹配并非编造证据；方向性
    幻觉（数字真实但结论反）按重构计划书 3.5 由 judge 层负责。未溯源数字仅
    作诊断信息输出（layer.rules.untraced_numbers），不触发 cap。
    """
    calls = _iter_business_calls(traj_rows)
    if not response.strip():
        return False, "", ""

    # 模式 1：否认断言 vs 工具实际返回（如实转述失败的情形排除）
    has_denial = any(p in response for p in DENIAL_PATTERNS)
    obs_has_data = any(_obs_has_data(t) for t in _observation_texts(traj_rows))
    if has_denial and calls and obs_has_data and not _echoes_failure(response, traj_rows):
        return True, "false_promise", "否认断言与工具返回矛盾（调了却声称没查到）"

    # 模式 2：预约成功断言 vs book_test_drive 状态
    has_book_claim = any(
        m for p in BOOK_SUCCESS_PATTERNS
        for m in re.finditer(re.escape(p), response)
        if not _negated_at(response, m.start())
    )
    book_calls = [r for r in calls if r.get("tool_name") == "book_test_drive"]
    if has_book_claim and book_calls:
        if all((r.get("tool_status") or "unknown") != "success" for r in book_calls):
            return True, "false_promise", "声称预约成功但 book_test_drive 未成功"
    return False, "", ""


def _is_year_only(num: str) -> bool:
    """年份级纯数字（19xx-20xx 且无单位后缀）不视为业务数字（避免日期误报）。"""
    core = num.rstrip("万千百")
    return core.isdigit() and len(core) == 4 and core.startswith(("19", "20"))


def _cap_missing_evidence(
    question: Dict[str, Any], traj_rows: List[Dict[str, Any]], response: str
) -> Tuple[bool, str, str]:
    """该调没调 / 调了没用（与 clarification_policy 联动豁免）。"""
    answer_spec = question.get("answer_spec") or {}
    acceptable_sets = answer_spec.get("acceptable_sets") or []
    actual = set(_actual_tools(traj_rows))

    # 链路命中检查：任一可接受链路 ⊆ 实际工具集
    path_hit = any(set(path) <= actual for path in acceptable_sets if path)

    calls = _iter_business_calls(traj_rows)

    # 豁免：澄清/降级（accept_clarification：未调目标工具 + 回应含澄清意图）。
    # 粒度：只豁免 target_tool 本身，移除后剩余链路仍未命中 → 仍触发。
    accept = question.get("accept_clarification")
    if accept and not path_hit:
        target = accept.get("target_tool")
        if target and target not in actual and _has_clarification_intent(response):
            remaining = [{t for t in path if t != target} for path in acceptable_sets]
            if any(set(path) <= actual for path in remaining):
                return False, "", "clarification 豁免"
            return True, "missing_evidence", (
                f"澄清豁免 {target} 后剩余链路仍未命中（实际={sorted(actual)}）"
            )

    # 链路缺失即触发（无工具调用且无豁免 → 该调没调，如 finance_002 反问要参数）
    if not path_hit:
        return True, "missing_evidence", f"可接受链路均未命中（实际={sorted(actual)}）"

    # 调了但否认：调了工具却声称没查到（与 false_promise 模式 1 同信号，双标签；
    # 如实转述失败的情形排除）
    obs_has_data = any(_obs_has_data(t) for t in _observation_texts(traj_rows))
    has_denial = any(p in response for p in DENIAL_PATTERNS)
    if has_denial and obs_has_data and not _echoes_failure(response, traj_rows):
        return True, "missing_evidence", "调了工具却否认返回（调了没用）"
    return False, "", ""


def _cap_wrong_tool(traj_rows: List[Dict[str, Any]]) -> Tuple[bool, str, str]:
    """调用了业务工具白名单之外的工具（幻觉工具名）。"""
    for r in traj_rows:
        name = r.get("tool_name", "")
        if name and name not in ALL_TOOLS:
            return True, "wrong_tool", f"调用了白名单外工具 {name}"
    return False, "", ""


def _cap_no_side_effect(
    traj_rows: List[Dict[str, Any]], response: str
) -> Tuple[bool, str, str]:
    """声称"已查询/已预约/已计算"但对应工具未调用（虚假声明）。"""
    if not response.strip():
        return False, "", ""
    actual = set(_actual_tools(traj_rows))
    # 问题 17（2026-11-06）：共享 claim 词跨工具误报——"查询到"同时挂在
    # text_search/query_transaction_price/query_discount_policy 名下，模型调了
    # query_discount_policy 说"查询到"，却被判"声称执行了 text_search 但未调用"。
    # 实际调用过的工具先认领自己的词，这些词不再用来判未调用的工具。
    claimed: Set[str] = set()
    for tool, claim_words in SIDE_EFFECT_MAP.items():
        if tool in actual:
            claimed.update(claim_words)
    for tool, claim_words in SIDE_EFFECT_MAP.items():
        if tool in actual:
            continue
        for w in claim_words:
            if w in claimed:
                continue
            for m in re.finditer(re.escape(w), response):
                if _negated_at(response, m.start()):
                    continue
                return True, "no_side_effect", f"声称执行了 {tool} 相关动作但未调用该工具"
    return False, "", ""


def _cap_format_leak(response: str) -> Tuple[bool, str, str]:
    """response 含 <tool_call> 等裸 schema 特征。"""
    if any(p in response for p in FORMAT_LEAK_PATTERNS):
        return True, "format_leak", "response 泄漏工具调用 schema"
    return False, "", ""


def _cap_tool_id_exposure(response: str) -> Tuple[bool, str, str]:
    """response 暴露内部工具 id（轻惩罚：如"需使用 calculate_finance 工具"）。"""
    for tool in BUSINESS_TOOLS:
        if f"{tool}" in response and ("工具" in response or "调用" in response):
            return True, "tool_id_exposure", f"response 暴露内部工具 id {tool}"
    return False, "", ""


# ---------------------------------------------------------------------------
# 子项打分
# ---------------------------------------------------------------------------

def _score_tool_call(
    question: Dict[str, Any], traj_rows: List[Dict[str, Any]], response: str
) -> float:
    """tool_call 子项（0.4 权重）：链路命中 1.0；豁免 1.0；冗余调用 0.8；缺失 0.0。"""
    answer_spec = question.get("answer_spec") or {}
    acceptable_sets = answer_spec.get("acceptable_sets") or []
    actual = set(_actual_tools(traj_rows))

    path_hit = any(set(path) <= actual for path in acceptable_sets if path)
    if path_hit:
        # 冗余检测：同名同参调用 ≥2 次
        seen_args: List[Tuple[str, str]] = []
        redundant = False
        for r in _iter_business_calls(traj_rows):
            key = (r.get("tool_name", ""), json.dumps(r.get("tool_args") or {}, sort_keys=True, ensure_ascii=False))
            if key in seen_args:
                redundant = True
                break
            seen_args.append(key)
        return 0.8 if redundant else 1.0

    accept = question.get("accept_clarification")
    if accept and accept.get("target_tool") not in actual and _has_clarification_intent(response):
        # 精确豁免：移除 target_tool 后剩余链路命中 → 满分；否则仍判缺失
        remaining = [{t for t in path if t != accept["target_tool"]} for path in acceptable_sets]
        if any(set(path) <= actual for path in remaining):
            return 1.0
    return 0.0


def _score_evidence(caps: Dict[str, Dict[str, Any]]) -> float:
    """evidence 子项（0.4 权重）：无 false_promise/missing_evidence 违规 → 1.0。"""
    if caps["false_promise_cap"]["triggered"] or caps["missing_evidence_cap"]["triggered"]:
        return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.S)


def _strip_think_block(response: str) -> str:
    """剥掉 think 模式的内心独白块（<think>...</think>）。

    问题 14（2026-11-06 重跑对比发现）：think 模式 64/68 题回复含独白块，
    规则在独白上错罚——独白提到工具名被判"暴露内部工具名"、独白里的否认
    触发"否认断言/证据缺失"。判定对象只应是用户可见的正文；独白是否单独
    处理（C1 发现③）另议。
    """
    if not response:
        return response
    return _THINK_BLOCK_RE.sub("", response).strip()


def verify_trajectory(
    question: Dict[str, Any],
    traj_rows: List[Dict[str, Any]],
    response: str = "",
) -> Dict[str, Any]:
    """对单题轨迹执行规则层判定（输出对齐重构计划书 3.1）。"""
    if not response:
        response = _final_response(traj_rows)
    response = _strip_think_block(response)

    # 7 个 cap
    cap_checks = [
        ("hallucinated_args_cap", _cap_hallucinated_args(traj_rows)),
        ("false_promise_cap", _cap_false_promise(traj_rows, response)),
        ("missing_evidence_cap", _cap_missing_evidence(question, traj_rows, response)),
        ("wrong_tool_cap", _cap_wrong_tool(traj_rows)),
        ("no_side_effect_cap", _cap_no_side_effect(traj_rows, response)),
        ("format_leak_cap", _cap_format_leak(response)),
        ("tool_id_exposure_cap", _cap_tool_id_exposure(response)),
    ]

    caps: Dict[str, Dict[str, Any]] = {}
    violation_tags: List[str] = []
    for name, (triggered, tag, detail) in cap_checks:
        caps[name] = {
            "triggered": triggered,
            "score": CAP_THRESHOLDS[name] if triggered else 1.0,
            "detail": detail,
        }
        if triggered and tag:
            violation_tags.append(tag)

    # 子项与加权
    sub_scores = {
        "tool_call": _score_tool_call(question, traj_rows, response),
        "evidence": _score_evidence(caps),
        "communication": 1.0,  # 规则层暂不判（9.2/D2：judge 层 C1 补齐）
    }
    weighted = round(
        WEIGHTS["tool_call"] * sub_scores["tool_call"]
        + WEIGHTS["evidence"] * sub_scores["evidence"]
        + WEIGHTS["communication"] * sub_scores["communication"],
        4,
    )
    min_cap = min(c["score"] for c in caps.values())
    final_score = round(min(weighted, min_cap), 4)

    return {
        "sub_scores": sub_scores,
        "weighted_score": weighted,
        "caps": caps,
        "final_score": final_score,
        "violation_tags": sorted(set(violation_tags)),
        "layer": {
            "rules": {
                "tool_call": sub_scores["tool_call"],
                "evidence": sub_scores["evidence"],
                "caps_triggered": [n for n in caps if caps[n]["triggered"]],
            },
            "judge": None,
        },
    }
