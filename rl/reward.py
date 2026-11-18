"""Agentic RL reward（阶段二 · 路线 B 手写 GRPO 的奖励函数）

单一来源，离线回归与训练循环共用；纯函数可复算（rescore 精神）。

公式（Reward规格书_2026-11-17_v1.0）：
    r = r_gate × (w_acc·r_acc + w_proc·r_proc)    # 首跑 w_proc=0（λ=0 消融起点）
    r_gate ∈ {0,1}（§2 乘法硬门控）：① 结构不可执行 ② 事实编造
    r_acc ∈ [0,1]（§3）：tool_correct（accept_clarification 语义内置）+ GT 事实匹配
    r_proc ∈ [-1,1]（§4）：r_dep / r_step / r_recover 条件项，首跑禁用

设计边界（2026-11-18 定）：
    - gate 2 只溯源「调用参数」（TRACE_ARGS 白名单 + 预约年份），不做裸数字溯源
      （verify 已实证裸数字溯源误伤计算值）；回复层编造由 r_acc.gt_facts 比对兜底，
      RL prompt 集每题都带 gt_facts 标注。
    - r_acc 子项等权归一（生效子项均分），不引入子项权重超参（审计纪律：
      权重类参数只在非测试集上调）。
    - 数据形态：A4 落盘行与 SFT messages 两种格式经适配器归一为内部 Step 序列；
      参数溯源复用 eval/baseline_eval._trace_args_source（与 verify 同口径单一来源）。
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import re
import sys

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from eval.baseline_eval import _parse_tool_status, _trace_args_source, _value_in_text
from eval.scoring import _has_clarification_intent, judge_tool_correct
from eval.verifier.rules import (
    BUILTIN_TOOLS,
    BUSINESS_TOOLS,
    TRACE_ARGS,
    _car_model_prefix_in_question,
    _strip_think_block,
)

# 阶段二新增工具（与 tools/ 下 mock 落地保持同名）
RL_EXTRA_TOOLS = {"web_search", "follow_up_query"}
RL_BUSINESS_TOOLS = BUSINESS_TOOLS | RL_EXTRA_TOOLS
RL_ALL_TOOLS = RL_BUSINESS_TOOLS | BUILTIN_TOOLS

# 首跑超参（Reward 规格书 §1；网格起点，调用方可覆盖）
DEFAULT_W_ACC = 0.8
DEFAULT_W_PROC = 0.0   # 首跑 λ=0：r_proc 第二轮消融再开
RECOVER_BONUS = 0.15   # r_recover 加分幅度（2026-11-18 拍板）


def _default_eval_year() -> str:
    return os.getenv("EVAL_TODAY", datetime.now().strftime("%Y-%m-%d"))[:4]


# ---------------------------------------------------------------------------
# 格式适配器：A4 落盘行 / SFT messages → 统一 Step 序列
# ---------------------------------------------------------------------------

def from_a4_rows(
    rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """A4 轨迹行（评测落盘 jsonl 逐行 dict）→ (steps, response, meta)。"""
    steps: List[Dict[str, Any]] = []
    response = ""
    meta: Dict[str, Any] = {}
    for r in rows:
        kind = r.get("kind")
        if kind == "model_output":
            response = r.get("content") or response
            continue
        if kind == "case_summary":
            if not response:
                response = r.get("final_response") or ""
            if r.get("env_snapshot"):
                meta["env_snapshot"] = r["env_snapshot"]
            continue
        name = r.get("tool_name")
        if not name:
            continue
        args = r.get("tool_args") or {}
        obs = r.get("observation") or ""
        if name == "Finish" and not response and args.get("answer"):
            response = args["answer"]
        steps.append({
            "tool_name": name,
            "tool_args": args,
            "observation": obs,
            "tool_status": r.get("tool_status") or _parse_tool_status(
                name, r.get("tool_response_raw") or obs
            ),
            "args_source": r.get("tool_args_source") or {},
        })
    return steps, response, meta


def from_sft_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """SFT 轨迹（OpenAI messages 格式）→ (steps, response, meta)。

    meta["question_text"] = 全部 user 轮次拼接（多轮纠错轨迹的后续轮次补充
    参数也是合法题面来源，参数溯源必须覆盖——只用首轮会导致 fabricated 误杀）。
    """
    steps: List[Dict[str, Any]] = []
    response = ""
    user_turns: List[str] = []
    pending: Dict[str, Dict[str, Any]] = {}
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            user_turns.append(msg.get("content") or "")
        elif role == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                raw_args = fn.get("arguments") or ""
                try:
                    args = (
                        json.loads(raw_args)
                        if isinstance(raw_args, str)
                        else (raw_args or {})
                    )
                    parse_err = ""
                except (ValueError, TypeError):
                    args, parse_err = {}, "unparseable_tool_args"
                step: Dict[str, Any] = {
                    "tool_name": name,
                    "tool_args": args,
                    "observation": "",
                    "tool_status": "success",
                    "args_source": {},
                    "parse_err": parse_err,
                }
                steps.append(step)
                pending[tc.get("id")] = step
                if name == "Finish" and args.get("answer"):
                    response = args["answer"]
        elif role == "tool":
            step = pending.get(msg.get("tool_call_id"))
            if step is not None:
                obs = msg.get("content") or ""
                step["observation"] = obs
                step["tool_status"] = _parse_tool_status(step["tool_name"], obs)
    return steps, response, {"question_text": "\n".join(user_turns)}


def _actual_tools(steps: List[Dict[str, Any]]) -> List[str]:
    """实际调用的业务工具名列表（按首次出现顺序，含阶段二新工具）。"""
    seen: List[str] = []
    for s in steps:
        name = s["tool_name"]
        if name in RL_BUSINESS_TOOLS and name not in seen:
            seen.append(name)
    return seen


def _business_calls(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [s for s in steps if s["tool_name"] in RL_BUSINESS_TOOLS]


# ---------------------------------------------------------------------------
# r_gate（§2 乘法硬门控）
# ---------------------------------------------------------------------------

def _gate_structural(
    steps: List[Dict[str, Any]], response: str, require_finish: bool
) -> List[str]:
    """结构不可执行检查：回复为空 / 缺 Finish / 参数不可解析 / 白名单外工具 /
    业务调用缺观察（执行未完成）。"""
    hits: List[str] = []
    if not (response or "").strip():
        hits.append("no_final_answer")
    if require_finish and not any(s["tool_name"] == "Finish" for s in steps):
        hits.append("no_finish")
    for s in steps:
        name = s["tool_name"]
        if s.get("parse_err"):
            hits.append(s["parse_err"])
        if name not in RL_ALL_TOOLS:
            hits.append(f"wrong_tool:{name}")
        elif name in RL_BUSINESS_TOOLS and not (s.get("observation") or "").strip():
            hits.append(f"missing_observation:{name}")
    return hits


def _gate_fabrication(
    steps: List[Dict[str, Any]], question: str, eval_year: str
) -> Tuple[str, str]:
    """事实编造检查：① TRACE_ARGS 白名单参数溯源 fabricated（含题面简称补全豁免）
    ② book_test_drive 预约年份 ≠ 评测基准年份（训练记忆年份幻觉）。"""
    prior_obs: List[str] = []
    for s in steps:
        name = s["tool_name"]
        if name in BUILTIN_TOOLS:
            continue
        traceable = TRACE_ARGS.get(name)
        if traceable:
            sources = s.get("args_source") or _trace_args_source(
                name, s["tool_args"], question, prior_obs
            )
            for k in traceable:
                if sources.get(k) != "fabricated":
                    continue
                if k == "car_model" and _car_model_prefix_in_question(
                    str((s["tool_args"] or {}).get(k, "")), question
                ):
                    continue
                return "fabricated_args", (
                    f"{name}.{k}={s['tool_args'].get(k)!r}"
                )
        prior_obs.append(s.get("observation") or "")
    for s in steps:
        if s["tool_name"] != "book_test_drive":
            continue
        appt = str((s.get("tool_args") or {}).get("appointment_time") or "").strip()
        m = re.match(r"^(\d{4})", appt) if appt else None
        if m and m.group(1) != eval_year:
            return "appt_year_hallucination", appt
    return "", ""


# ---------------------------------------------------------------------------
# r_acc / r_proc
# ---------------------------------------------------------------------------

def _score_r_acc(
    label: Dict[str, Any], steps: List[Dict[str, Any]], response: str
) -> Tuple[float, Dict[str, Any]]:
    """r_acc：tool_correct（accept_clarification 语义内置）+ GT 事实匹配；
    生效子项等权归一，gt_facts 缺失时 r_acc = tool_correct。"""
    expected = label.get("expected_tools") or []
    actual = _actual_tools(steps)
    ok, missing, extra = judge_tool_correct(
        expected, actual, response, label.get("accept_clarification")
    )
    detail: Dict[str, Any] = {
        "tool_correct": bool(ok),
        "missing_tools": missing,
        "extra_tools": extra,
    }
    gt_facts = label.get("gt_facts") or {}
    if gt_facts:
        matched = sum(1 for v in gt_facts.values() if _value_in_text(str(v), response))
        facts = matched / len(gt_facts)
        detail["facts_matched"] = facts
        detail["facts_total"] = len(gt_facts)
        return round(0.5 * float(ok) + 0.5 * facts, 4), detail
    return float(ok), detail


def _score_r_proc(
    label: Dict[str, Any], steps: List[Dict[str, Any]], response: str
) -> Tuple[float, Dict[str, Any]]:
    """r_proc：r_dep（因果边）/ r_step（冗余惩罚）/ r_recover（失败恢复加分）；
    r_dep/r_step 等权归一，r_recover 为加分项封顶 1.0。首跑 w_proc=0 不参与总奖。"""
    dep_edges = label.get("dep_edges") or []
    fail_inject = label.get("fail_inject") or {}
    calls = _business_calls(steps)
    detail: Dict[str, Any] = {"active": []}
    vals: Dict[str, float] = {}

    if dep_edges:
        satisfied = 0
        for pre, suc in dep_edges:
            pre_i = [i for i, c in enumerate(calls) if c["tool_name"] == pre]
            suc_i = [i for i, c in enumerate(calls) if c["tool_name"] == suc]
            if pre_i and suc_i and pre_i[0] < suc_i[0]:
                suc_call = calls[suc_i[0]]
                if any(
                    v == "prior_observation"
                    for v in (suc_call.get("args_source") or {}).values()
                ):
                    satisfied += 1
        vals["r_dep"] = satisfied / len(dep_edges)
        detail["active"].append("r_dep")

    total = len(calls)
    if total:
        unique = len({
            (c["tool_name"], json.dumps(c["tool_args"], sort_keys=True, ensure_ascii=False))
            for c in calls
        })
        vals["r_step"] = min(1.0, unique / total)
    else:
        vals["r_step"] = 1.0
    detail["active"].append("r_step")

    if fail_inject:
        inj_tool = fail_inject.get("tool")
        fail_i = next(
            (i for i, c in enumerate(calls)
             if c["tool_name"] == inj_tool and (c.get("tool_status") or "") != "success"),
            -1,
        )
        if fail_i >= 0:
            later = calls[fail_i + 1:]
            retry = any(c["tool_name"] == inj_tool for c in later)
            switched = any(c["tool_name"] != inj_tool for c in later)
            if retry or switched or _has_clarification_intent(response):
                vals["r_recover"] = RECOVER_BONUS
        detail["active"].append("r_recover")

    base = [v for k, v in vals.items() if k != "r_recover"]
    r_proc = sum(base) / len(base) if base else 0.0
    r_proc = min(1.0, r_proc + vals.get("r_recover", 0.0))
    detail.update({k: round(v, 4) for k, v in vals.items()})
    return round(r_proc, 4), detail


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def compute_reward(
    label: Dict[str, Any],
    steps: List[Dict[str, Any]],
    response: str = "",
    eval_year: Optional[str] = None,
    w_acc: float = DEFAULT_W_ACC,
    w_proc: float = DEFAULT_W_PROC,
    require_finish: bool = False,
) -> Dict[str, Any]:
    """对单条轨迹计算 reward 分解。

    参数：
        label: RL prompt 标注（question / expected_tools / gt_facts /
               accept_clarification / dep_edges / fail_inject / difficulty）；
               question 字段供参数溯源。
        steps: 统一 Step 序列（from_a4_rows / from_sft_messages 产物）。
        response: 最终回复原文（含 think 块时内部剥离，与 verify 同口径）。
        require_finish: SFT messages 格式要求 Finish 结构（A4 落盘由框架保证不查）。
    返回：
        {r, r_gate, gate_hits, r_acc, acc, r_proc, proc, fabrication}
    """
    response = _strip_think_block(response)
    question = str(label.get("question") or "")
    year = eval_year or _default_eval_year()

    gate_hits = _gate_structural(steps, response, require_finish)
    fab_hit, fab_detail = _gate_fabrication(steps, question, year)
    if fab_hit:
        gate_hits.append(fab_hit)
    r_gate = 0 if gate_hits else 1

    r_acc, acc_detail = _score_r_acc(label, steps, response)
    r_proc, proc_detail = _score_r_proc(label, steps, response)

    r = round(r_gate * (w_acc * r_acc + w_proc * r_proc), 4)
    return {
        "r": r,
        "r_gate": r_gate,
        "gate_hits": gate_hits,
        "r_acc": round(r_acc, 4),
        "acc": acc_detail,
        "r_proc": r_proc,
        "proc": proc_detail,
        "fabrication": {"hit": fab_hit, "detail": fab_detail},
    }
