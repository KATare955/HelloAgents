# -*- coding: utf-8 -*-
"""v02 批量生成核心管线（Day 11 批次）

设计原则：
- 正确性由程序保证：工具链序列、工具参数、工具返回（mock 实跑）、时间换算全部程序化；
- 多样性由 Claude 润色提供：只润色三段自然语言（user 问法 / Thought 推理 / Finish 答复），
  润色后程序校验数字 multiset 与白名单实体集合，失败自动回退程序模板；
- 接地：参数值只来自接地表（车型 4 配置 / 门店 7 家白名单 / 优惠 3 项 / 月供由工具实算）。

运行环境：在 HelloAgents 目录下运行（import tools.* 与 hello_agents.*）。
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------- 路径准备 ----------
HELLO_AGENTS_DIR = Path(__file__).resolve().parents[3]  # v02_gen -> sft_data -> eval -> HelloAgents
SFT_DATA_DIR = HELLO_AGENTS_DIR / "eval" / "sft_data"
sys.path.insert(0, str(HELLO_AGENTS_DIR))

from tools.text_search import TextSearchTool
from tools.calculate_finance import CalculateFinanceTool
from tools.book_test_drive import BookTestDriveTool
from tools.query_discount_policy import QueryDiscountPolicyTool
from tools.query_transaction_price import QueryTransactionPriceTool
from tools.competitor_compare import CompetitorCompareTool

# ---------- 系统提示词（与 v02 小样完全一致） ----------
SYSTEM_PROMPT = (
    "你是一个专业的汽车销售顾问助手。\n"
    "你的任务是帮助用户了解车型信息、查询优惠政策、对比竞品、计算金融方案、预约试驾等。\n\n"
    "可用工具：\n"
    "- text_search: 查询车型配置、价格、参数等信息\n"
    "- query_discount_policy: 查询当前可用的优惠政策\n"
    "- query_transaction_price: 查询历史成交价和价格趋势\n"
    "- competitor_compare: 对比同级别车型\n"
    "- calculate_finance: 计算贷款月供和总利息\n"
    "- book_test_drive: 预约试驾\n\n"
    "使用规则：\n"
    "1. 根据用户问题选择合适的工具\n"
    "2. 如果需要多个信息，可以多次调用工具\n"
    "3. 用简洁清晰的语言回复用户\n"
    "4. 提供具体的数字和信息"
)

# ---------- 接地表（以真实 mock 文件为准，2026-11-07 实跑口径） ----------
# 车型 4 配置（text_search.json）
CONFIGS = [
    {"title": "E5 Sportback 2025款 先锋型", "price": 235900, "range": "618km",
     "motor": "单电机 电机最大功率220kW", "battery": "76kWh", "drive": "后驱"},
    {"title": "E5 Sportback 2025款 先锋quattro型", "price": 269900, "range": "623km",
     "motor": "单电机 电机最大功率386kW", "battery": "83kWh", "drive": "quattro智能四驱"},
    {"title": "E5 Sportback 2025款 先锋Plus型", "price": 269900, "range": "773km",
     "motor": "单电机 电机最大功率300kW", "battery": "100kWh", "drive": "后驱"},
    {"title": "E5 Sportback 2025款 旗舰quattro型", "price": 319900, "range": "647km",
     "motor": "单电机 电机最大功率579kW", "battery": "100kWh", "drive": "quattro智能四驱"},
]

# 门店白名单（book_test_drive._DEALER_WHITELIST 7 家）
DEALERS = [
    "上汽奥迪上海郑隆奥翟",
    "上汽奥迪上海大宁音乐广场",
    "上汽奥迪西上海用户中心",
    "上汽奥迪上海东昌用户中心",
    "上汽奥迪上海冠澳用户中心",
    "上汽奥迪上海冠松用户中心",
    "上汽奥迪奥迪之城旗舰店",
]
# 小样曾用“只报 2 家 mock 门店”的反问口径；批量阶段已废弃（2026-11-07 用户裁定：
# 反问门店列表必须报 7 家白名单全量，与工具实跑返回一致，避免 2 家高频重复强化幻觉）
MOCK_DEALERS = DEALERS[:2]  # 仅保留兼容旧小样，新模板不得引用

# 优惠 3 项（query_discount_policy.json：全国 2 项，上海 3 项含贴息）
DISCOUNTS_NATIONWIDE = [
    {"name": "限时优惠", "amount": 20000, "condition": "全款购车，限时优惠", "valid_until": "2026-12-31"},
    {"name": "旧车置换补贴", "amount": 10000, "condition": "需提供旧车行驶证，旧车需使用满3年", "valid_until": "2026-12-31"},
]
DISCOUNT_SHANGHAI_EXTRA = {"name": "低息贷款", "amount": 5000,
                           "condition": "贷款购车，首付不低于30%，贷款期限不低于2年", "valid_until": "2026-11-30"}

# 竞品对（competitor_compare.json）
COMPETITOR_PAIRS = [
    {"a": "E5 Sportback 2025款 旗舰quattro型", "b": "小米SU7 2025款 四驱Max版"},
    {"a": "E5 Sportback 2025款 旗舰quattro型", "b": "帕萨特2024款 330TSI 精英版"},
]

# 编造形态（桶2 turn1 专用，约束 A：只允许真实轨迹已出现的形态）
FABRICATED_NAME = "张三"
FABRICATED_PHONE = "13800138000"
FABRICATED_DEALER = "上海大众4S店"  # 白名单外 → 实跑 NOT_FOUND

# 用户自报姓名库（桶1/桶2 execute 的合法输入，与 assistant 编造无关）
CUSTOMER_NAMES = ["王强", "李娜", "陈晨", "刘洋", "周婷", "吴敏", "郑浩", "黄蕾", "徐磊", "孙悦"]

# 润色实体白名单（校验用：新文本不得引入该集合之外的实体）
KNOWN_ENTITIES = (
    [c["title"] for c in CONFIGS] + DEALERS + [FABRICATED_NAME, FABRICATED_DEALER]
    + ["E5 Sportback", "小米SU7", "帕萨特2024款", "上汽奥迪", "奥迪"]
)

# ---------- 基准日多样化（裁定第 7 项） ----------
WEEKDAY_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

# 基准日池：多周五/周六/周日轮换（评测集基准日 2026-11-06 只占少数）
BASE_DATE_POOL = [
    datetime(2026, 10, 30),  # 周五
    datetime(2026, 11, 6),   # 周五（评测集基准日，限量使用）
    datetime(2026, 11, 13),  # 周五
    datetime(2026, 11, 7),   # 周六
    datetime(2026, 11, 8),   # 周日
]


def base_weekday_cn(base: datetime) -> str:
    """基准日的中文星期（base 均为周五/周六/周日）。"""
    return WEEKDAY_CN[base.weekday()]


def base_date_cn(base: datetime) -> str:
    """基准日中文表述：2026年11月6日（周五）"""
    return f"{base.year}年{base.month}月{base.day}日（{base_weekday_cn(base)}）"


def next_monday(base: datetime) -> datetime:
    """下一个周一（含基准日当天为周日时返回次日）。"""
    delta = (7 - base.weekday()) % 7
    if delta == 0:
        delta = 7
    return base + timedelta(days=delta)


def this_saturday(base: datetime) -> datetime:
    """本周六（基准日所在周的周六）。"""
    return base + timedelta(days=(5 - base.weekday()) % 7)


def this_sunday(base: datetime) -> datetime:
    """本周日。"""
    return this_saturday(base) + timedelta(days=1)


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def date_cn(dt: datetime) -> str:
    return f"{dt.year}年{dt.month}月{dt.day}日（{WEEKDAY_CN[dt.weekday()]}）"


# ---------- mock 实跑 ----------
TOOLS = {
    "text_search": TextSearchTool(),
    "calculate_finance": CalculateFinanceTool(),
    "book_test_drive": BookTestDriveTool(),
    "query_discount_policy": QueryDiscountPolicyTool(),
    "query_transaction_price": QueryTransactionPriceTool(),
    "competitor_compare": CompetitorCompareTool(),
}


def run_tool(name: str, args: Dict[str, Any]) -> str:
    """实跑 mock 工具，返回紧凑单行 JSON（与首轮训练集 tool content 格式一致）。"""
    resp = TOOLS[name].run(args)
    return json.dumps(resp.to_dict(), ensure_ascii=False)


def run_tool_parsed(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return TOOLS[name].run(args).to_dict()


# ---------- 消息构建 ----------
def sys_msg() -> Dict[str, Any]:
    return {"role": "system", "content": SYSTEM_PROMPT}


def user_msg(content: str) -> Dict[str, Any]:
    return {"role": "user", "content": content}


def thought_call(call_id: str, reasoning: str, loss_mask: Optional[bool] = None) -> Dict[str, Any]:
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": "Thought",
                         "arguments": json.dumps({"reasoning": reasoning}, ensure_ascii=False)},
        }],
    }
    if loss_mask is not None:
        msg["loss_mask"] = loss_mask
    return msg


def thought_result(call_id: str, reasoning: str) -> Dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": f"推理: {reasoning}"}


def tool_call(call_id: str, name: str, args: Dict[str, Any],
              loss_mask: Optional[bool] = None) -> Dict[str, Any]:
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        }],
    }
    if loss_mask is not None:
        msg["loss_mask"] = loss_mask
    return msg


def tool_result(call_id: str, content: str) -> Dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def finish_call(call_id: str, answer: str) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": "Finish", "arguments": json.dumps({"answer": answer}, ensure_ascii=False)},
        }],
    }


def make_sample(sid: str, bucket: str, scenario: str, question: str, targets: List[str],
                is_clarify: bool, is_degrade: bool, messages: List[Dict[str, Any]],
                design_intent: str, review_focus: str, stage: str = "v0.2_批量") -> Dict[str, Any]:
    return {
        "id": sid,
        "bucket": bucket,
        "targets": targets,
        "scenario": scenario,
        "question": question,
        "expected_tools": targets,
        "is_clarify": is_clarify,
        "is_degrade": is_degrade,
        "messages": [sys_msg()] + messages,
        "meta": {"design_intent": design_intent, "review_focus": review_focus,
                 "sample_stage": stage},
    }


# ---------- 数字与实体校验 ----------
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def numbers_of(text: str) -> List[str]:
    return sorted(_NUM_RE.findall(text or ""))


def entities_of(text: str) -> set:
    return {e for e in KNOWN_ENTITIES if e in (text or "")}


def validate_polish(orig_q: str, orig_a: str, new_q: str, new_a: str) -> bool:
    """校验润色结果：数字 multiset 逐段相等；白名单实体只减不增；长度不坍缩 ；
    润色后题面不得与评测题面高度重合（零重叠红线，子串或相似度≥0.92 均拒绝）。"""
    if not all(isinstance(x, str) and x.strip() for x in (new_q, new_a)):
        return False
    for o, n in ((orig_q, new_q), (orig_a, new_a)):
        if numbers_of(o) != numbers_of(n):
            return False
        if not entities_of(n).issubset(entities_of(o)):
            return False
        if len(n) < len(o) * 0.5:
            return False
    # 零重叠硬约束：润色不得把题面改得接近评测题面（防训练评测污染）
    q2 = new_q.replace(" ", "")
    for eq in _eval_questions():
        e = eq.replace(" ", "")
        if q2 == e or q2 in e or e in q2:
            return False
        if SequenceMatcher(None, q2, e).ratio() >= 0.92:
            return False
    return True


_EVAL_QS_CACHE: Optional[List[str]] = None


def _eval_questions() -> List[str]:
    """评测题面懒加载缓存（润色校验高频调用）"""
    global _EVAL_QS_CACHE
    if _EVAL_QS_CACHE is None:
        _EVAL_QS_CACHE = load_eval_questions()
    return _EVAL_QS_CACHE


# ---------- 评测集零重叠校验 ----------
def load_eval_questions() -> List[str]:
    qf = HELLO_AGENTS_DIR / "eval" / "eval_data" / "questions.json"
    data = json.loads(qf.read_text(encoding="utf-8"))
    return [q.get("question", "") for q in data.get("questions", [])]


def check_overlap(question: str, eval_questions: List[str]) -> bool:
    """与评测集 68 题零重叠校验：完全/子串/实体组合均不得重合。"""
    q = question.replace(" ", "")
    for eq in eval_questions:
        e = eq.replace(" ", "")
        if q == e or q in e or e in q:
            return True
    return False


# ---------- Claude 润色客户端 ----------
def _load_claude_env() -> Tuple[str, str, int]:
    settings = json.loads((Path.home() / ".claude" / "settings.json").read_text(encoding="utf-8"))
    env = settings.get("env", {})
    return env["ANTHROPIC_AUTH_TOKEN"], env["ANTHROPIC_BASE_URL"], int(env.get("API_TIMEOUT_MS", 120000))


POLISH_SYSTEM = (
    "你是汽车销售对话数据润色助手。请改写【用户问题】与【助手最终答复】两段文本，"
    "使其表达自然、口语化、句式多样，但严格遵守以下规则：\n"
    "1. 不得增删或更改任何数字、日期、时间、金额（ISO时间字符串的每个字符都要原样保留）；\n"
    "2. 不得更改车型名称、门店名称、人名、电话号码；\n"
    "3. 不得增删任何事实信息；用户问题中的每个参数必须原样保留；\n"
    "4. 保持原文的段落结构（如编号列表），只润色措辞；\n"
    '5. 只输出JSON对象：{"question": "...", "answer": "..."}'
)


def _call_claude(user_content: str, token: str, base_url: str, timeout_ms: int,
                 model: str = "claude-sonnet-5-5") -> Optional[str]:
    body = json.dumps({
        "model": model,
        "max_tokens": 2048,
        "system": POLISH_SYSTEM,
        "messages": [{"role": "user", "content": user_content}],
    }, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/messages",
        data=body,
        headers={
            "x-api-key": token,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_ms / 1000) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    for block in data.get("content", []):
        if block.get("type") == "text":
            return block["text"]
    return None


def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None


def polish_qa(question: str, answer: str, retries: int = 2,
              model: str = "claude-sonnet-5-5") -> Optional[Tuple[str, str]]:
    """只润色题面与最终答复两段（用户可见文本）；Thought 推理保持程序模板（含日期换算，
    不交给 LLM 改写以免改动日期）。失败（网络/格式/校验）返回 None → 调用方回退程序模板。"""
    token, base_url, timeout_ms = _load_claude_env()
    payload = f"【用户问题】{question}\n【助手答复】{answer}"
    for _ in range(retries + 1):
        try:
            raw = _call_claude(payload, token, base_url, timeout_ms, model)
            parsed = _parse_json(raw or "")
            if not parsed:
                continue
            q2, a2 = parsed.get("question"), parsed.get("answer")
            if validate_polish(question, answer, q2, a2):
                return q2, a2
        except Exception:
            continue
    return None


# ---------- 树状视图渲染 ----------
def render_tree(samples: List[Dict[str, Any]]) -> str:
    """把 JSONL 样本渲染为人工可读树状视图（只读审核用，不参与训练）。"""
    lines: List[str] = []
    for s in samples:
        bucket_cn = {
            "bucket1_clarify": "桶1 缺参正确处理",
            "bucket2_correction": "桶2 编造纠错多轮",
            "bucket3_multihop": "桶3 多跳链路保持",
            "bucket4_degrade": "桶4 失败降级",
            "bucket5_coverage": "桶5 场景保底",
        }.get(s["bucket"], s["bucket"])
        tags = []
        if s.get("is_clarify"):
            tags.append("缺参反问")
        if s.get("is_degrade"):
            tags.append("失败降级")
        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        lines.append(f"## {s['id']} ─ {bucket_cn}{tag_str}")
        lines.append(f"- 场景: {s['scenario']} | 目标工具: {s['expected_tools'] or '无'}")
        lines.append(f"- 题面: {s['question']}")
        for i, m in enumerate(s["messages"]):
            role = m["role"]
            mask_note = ""
            if m.get("loss_mask") is False:
                mask_note = "  🔴 loss_mask=false（编造段，不计loss）"
            if role == "system":
                lines.append(f"- [{i}] system: （系统提示词，与全体样本一致）")
            elif role == "user":
                lines.append(f"- [{i}] user: {m['content']}")
            elif role == "tool":
                content = m["content"]
                try:
                    obj = json.loads(content)
                    status = obj.get("status")
                    if status == "error":
                        content = f"❌ {obj.get('error', {}).get('code')}: {obj.get('error', {}).get('message', '')}"
                    else:
                        txt = obj.get("text", "")
                        content = f"✅ {txt[:100]}{'…' if len(txt) > 100 else ''}"
                except (json.JSONDecodeError, TypeError):
                    if len(content) > 120:
                        content = content[:120] + "…"
                lines.append(f"- [{i}] tool → {content}")
            elif role == "assistant":
                tc = m.get("tool_calls", [{}])[0]
                fn = tc.get("function", {})
                name = fn.get("name", "?")
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {"<raw>": fn.get("arguments", "")}
                if name == "Thought":
                    lines.append(f"- [{i}] assistant → Thought: {args.get('reasoning', '')}{mask_note}")
                elif name == "Finish":
                    lines.append(f"- [{i}] assistant → Finish: {args.get('answer', '')[:200]}")
                else:
                    lines.append(f"- [{i}] assistant → {name}{json.dumps(args, ensure_ascii=False)}{mask_note}")
        lines.append("")
    return "\n".join(lines)
