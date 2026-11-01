#!/usr/bin/env python3
"""gen_sft_trajectories.py — Claude 合成 SFT 工具轨迹（靶向 7 坏例 + 4 澄清题）

设计要点（与 ReActAgent 推理格式严格对齐，保证 Base↔SFT 同集可比）：
  - 轨迹形态：system → user → [Thought] → (业务tool_call → observation)×n → Finish(answer)
  - observation 内容 = 真实工具的 resp.to_model_text()（与 agent 推理时喂给模型的完全一致）
  - tool_calls 用 OpenAI 风格（function.arguments 为 JSON 字符串）
  - 全部 ≤5 步，匹配 max_steps=5

保真策略（hybrid 两遍，参数接地 + 真实执行 + 真实答复）：
  Pass1: Claude 依据"接地表"生成 {question, reasoning, tool_calls 计划}（参数严格接地，不编造）
  执行 : 用真实 mock 工具执行每个 tool_call，拿真实 observation
  校验 : schema必填 / 桶感知非空(常规要求非空、降级要求空) / 金融价格连贯 / 与评测集零重叠
  Pass2: Claude 依据【真实 observation】写最终自然语言答复（数字与工具结果一致）
  组装 : 写 rich JSONL（messages + 元数据），供 s3 转 LLaMA-Factory sharegpt+tools

安全：token 只在运行时经 _claude_client 从 ~/.claude/settings.json 读取，绝不打印/落盘。

用法（务必用 .venv 的 python，hello_agents 需 3.10+）：
  .venv/bin/python eval/sft_data/gen_sft_trajectories.py --limit 2            # 每桶2条，打通管道(先跑这个)
  .venv/bin/python eval/sft_data/gen_sft_trajectories.py --count 300          # 按配比拼总量放量
  .venv/bin/python eval/sft_data/gen_sft_trajectories.py --bucket clarify --limit 20
"""

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SFT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SFT_DIR.parent.parent  # HelloAgents/
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SFT_DIR))

from _claude_client import call_claude, extract_json, load_config  # noqa: E402
from tools.text_search import TextSearchTool  # noqa: E402
from tools.query_discount_policy import QueryDiscountPolicyTool  # noqa: E402
from tools.query_transaction_price import QueryTransactionPriceTool  # noqa: E402
from tools.competitor_compare import CompetitorCompareTool  # noqa: E402
from tools.calculate_finance import CalculateFinanceTool  # noqa: E402
from tools.book_test_drive import BookTestDriveTool  # noqa: E402

# ---------------------------------------------------------------------------
# 常量：Agent 系统提示（与 baseline_eval.create_sales_agent 完全一致）
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是一个专业的汽车销售顾问助手。
你的任务是帮助用户了解车型信息、查询优惠政策、对比竞品、计算金融方案、预约试驾等。

可用工具：
- text_search: 查询车型配置、价格、参数等信息
- query_discount_policy: 查询当前可用的优惠政策
- query_transaction_price: 查询历史成交价和价格趋势
- competitor_compare: 对比同级别车型
- calculate_finance: 计算贷款月供和总利息
- book_test_drive: 预约试驾

使用规则：
1. 根据用户问题选择合适的工具
2. 如果需要多个信息，可以多次调用工具
3. 用简洁清晰的语言回复用户
4. 提供具体的数字和信息"""

# ReAct 内置工具 schema（取自 react_agent.py，用于 s3 的 tools 字段）
THOUGHT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "Thought",
        "description": "分析问题，制定策略，记录简短的行动说明。在需要思考时调用此工具。",
        "parameters": {
            "type": "object",
            "properties": {"reasoning": {"type": "string", "description": "简短的行动计划或结论"}},
            "required": ["reasoning"],
        },
    },
}
FINISH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "Finish",
        "description": "当你有足够信息得出结论时，使用此工具返回最终答案。",
        "parameters": {
            "type": "object",
            "properties": {"answer": {"type": "string", "description": "最终答案"}},
            "required": ["answer"],
        },
    },
}

# ---------------------------------------------------------------------------
# 工具实例 + 必填参数表
# ---------------------------------------------------------------------------
TOOL_INSTANCES = [
    TextSearchTool(),
    QueryDiscountPolicyTool(),
    QueryTransactionPriceTool(),
    CompetitorCompareTool(),
    CalculateFinanceTool(),
    BookTestDriveTool(),
]
TOOL_MAP = {t.name: t for t in TOOL_INSTANCES}
REQUIRED = {
    "text_search": ["query"],
    "query_discount_policy": ["car_model"],
    "query_transaction_price": ["car_model"],
    "competitor_compare": ["car_model_a", "car_model_b"],
    "calculate_finance": ["car_price", "down_payment_ratio", "loan_years"],
    "book_test_drive": ["car_series", "dealer", "appointment_time", "customer_name", "customer_phone"],
}

# ---------------------------------------------------------------------------
# 接地表（真实 mock 数据，Claude 参数必须严格取自此表；避免编造）
# ---------------------------------------------------------------------------
GROUNDING = {
    "车型与指导价_text_search可查": {
        "E5 Sportback 2025款 先锋型": {"price": 235900, "battery": "76kWh", "range_cltc": "618km", "drive": "后驱"},
        "E5 Sportback 2025款 先锋quattro型": {"price": 269900, "battery": "83kWh", "range_cltc": "623km"},
        "E5 Sportback 2025款 先锋Plus型": {"price": 269900, "battery": "100kWh", "range_cltc": "773km"},
        "E5 Sportback 2025款 旗舰quattro型": {"price": 319900, "battery": "100kWh", "range_cltc": "647km"},
        "帕萨特2024款 280TSI 商务版": {"price": 185900, "engine": "1.4T 150马力"},
        "帕萨特2024款 330TSI 精英版": {"price": 205900, "engine": "2.0T 186马力"},
        "小米SU7 2025款 四驱Max版": {"price": 303900, "battery": "101kWh", "range_cltc": "800km"},
        "小米SU7 2025款 Pro版": {"price": 245900, "battery": "94.3kWh", "range_cltc": "830km"},
    },
    "text_search_query关键词": ["E5 Sportback 2025款", "帕萨特2024款", "小米SU7 2025款"],
    "text_search用法关键": "只传 query(车型名或关键词，可含'价格/续航'等意图词)，category 省略或填 'all'。"
                          "知识库所有结果的 category 均为'车型配置'，价格/续航/电池等在 content 字段内；"
                          "若传 category='价格'或'政策'会过滤成空结果。查价格就用 query=车型名，从 content.price 读。",
    "query_discount_policy_有效参数(仅这些组合有数据)": {
        "E5 Sportback": ["上海(总优惠35000)", "全国(总优惠30000)"],
        "帕萨特2024款": ["上海(总优惠23000)"],
        "小米SU7 2025款": ["上海(总优惠10000)"],
    },
    "query_transaction_price_有效参数(仅这些组合有数据)": {
        "E5 Sportback 2025款 旗舰quattro型": {"days可选": [30, 90], "近90天均价": 289500},
        "E5 Sportback 2025款 先锋型": {"days可选": [90], "近90天均价": 225000},
        "帕萨特2024款 330TSI 精英版": {"days可选": [90], "近90天均价": 195000},
        "小米SU7 2025款 四驱Max版": {"days可选": [90], "近90天均价": 303900},
    },
    "competitor_compare_有效车型对(仅这3对,含反向)": [
        ["E5 Sportback 2025款 旗舰quattro型", "小米SU7 2025款 四驱Max版"],
        ["E5 Sportback 2025款 旗舰quattro型", "帕萨特2024款 330TSI 精英版"],
    ],
    "calculate_finance": "任意 car_price>0 / down_payment_ratio(0-100) / loan_years>0 均可(实时等额本息)。car_price 请用上面真实指导价或优惠后价，严禁编造。",
    "book_test_drive_有效经销商": ["上海郑隆奥翟", "上海大宁音乐广场", "上海浦东奥迪", "上海小米汽车体验中心"],
    "不存在车型_用于降级拒答": ["Model X 2025款", "奥迪A7L", "比亚迪汉EV", "宝马i3 2025款", "特斯拉Model 3"],
}

# ---------------------------------------------------------------------------
# 数据桶（配比见 SFT数据规划§5：工具链40% / 直调25% / 澄清20% / 降级15%）
# ---------------------------------------------------------------------------
BUCKETS: Dict[str, Dict[str, Any]] = {
    "finance_chain": {
        "weight": 0.14, "targets": ["finance_002"], "scenario": "金融方案测算", "include_thought": True,
        "hint": "用户想算某车型贷款月供/首付，但【没有直接给出车价】。正确做法：先 text_search 查该车型真实价格，"
                "再用【查到的真实价格】calculate_finance。严禁凭空编造车价；calculate_finance 的 car_price 必须等于 text_search 返回的价格。"
                "text_search 只传 query(车型名)，【不要传 category】，从返回 content.price 取真实车价。",
        "ref": ["计算E5 Sportback 2025款先锋型的贷款月供", "小米SU7 2025款四驱Max版首付20%贷5年月供多少"],
    },
    "finance_compare": {
        "weight": 0.08, "targets": ["finance_006", "finance_009"], "scenario": "金融方案测算", "include_thought": True,
        "hint": "用户想【对比两种金融方案】(如首付30%vs50%、或3年vs5年哪个利息少)。正确做法：先 text_search 查真实车价(若未给)，"
                "然后【调用 calculate_finance 两次】(两组不同参数)，最后对比给出结论。两次 car_price 一致且为真实价格。",
        "ref": ["如果选择E5 Sportback，首付30%和首付50%哪个更划算？", "E5 Sportback 5年贷款和3年贷款哪个总利息更少？"],
    },
    "competitor_double": {
        "weight": 0.10, "targets": ["compare_005", "compare_006", "compare_010"], "scenario": "竞品对比", "include_thought": True,
        "hint": "用户对比两车，但关注【competitor_compare 未覆盖的细节】(充电速度/具体续航/配置明细)或【预算金融】。"
                "正确做法：先 competitor_compare 拿概览，再 text_search 补两车具体参数(或 calculate_finance 算预算)。"
                "仅用有效对比对：E5旗舰quattro↔小米SU7四驱Max / E5旗舰quattro↔帕萨特330TSI精英。"
                "text_search 只传 query(车型名)，【不要传 category】。",
        "ref": ["E5 Sportback和小米SU7在价格、续航、配置三个方面哪个更有优势？", "E5 Sportback和小米SU7的充电速度哪个更快？",
                "如果预算30万，E5 Sportback和小米SU7应该选哪个？"],
    },
    "multi_hop": {
        "weight": 0.08, "targets": ["multi_hop_004"], "scenario": "历史成交价查询", "include_thought": True,
        "hint": "用户问一个【需要3步串联】的问题：先 query_transaction_price 查成交价，再 query_discount_policy 查优惠，"
                "最后用优惠后价格 calculate_finance 算月供/差价。关键：【末端不能断裂】，必须完成最后一步计算，不能查完就停。仅用有数据的车型/地区。",
        "ref": ["E5 Sportback近90天的平均成交价是多少？现在有什么优惠？优惠后的价格比历史成交价低多少？"],
    },
    "direct_single": {
        "weight": 0.25, "targets": [], "scenario": "综合", "include_thought": False,
        "hint": "【常规单跳】问题，答案只需【一个工具一次调用】(查配置/查优惠/查成交价/一次金融计算/信息齐全的试驾预约)。"
                "不要过度调用工具。参数完整且取自接地表。覆盖6个工具的多种单跳场景，保持多样性。",
        "ref": ["E5 Sportback 2025款先锋型的电池容量是多少？", "E5 Sportback在上海有什么优惠？"],
    },
    "clarify": {
        "weight": 0.20, "targets": ["test_drive_002", "test_drive_003", "multi_hop_005"], "scenario": "试驾预约",
        "include_thought": True, "is_clarify": True,
        "hint": "用户要求【预约试驾但缺必填信息】(customer_name/customer_phone/具体时间/经销商)，或【先查询再预约但预约细节缺失】。"
                "正确做法：【绝不编造缺失参数硬调 book_test_drive】。可先调用能答的查询工具(查配置/优惠/金融)，"
                "最后用 Finish 汇总已知信息并【反问补全缺失的预约细节】。若纯预约缺 name/phone，可直接反问(tool_calls 可为空)。",
        "ref": ["预约明天下午3点在上海浦东店试驾E5 Sportback", "帮我预约周末在北京朝阳店试驾帕萨特",
                "我想预约E5 Sportback的试驾，先帮我查询配置、优惠和金融方案"],
    },
    "degrade_refuse": {
        "weight": 0.15, "targets": ["test_drive_007"], "scenario": "失败恢复", "include_thought": False, "is_degrade": True,
        "hint": "用户询问/操作【知识库中不存在的车型】(取自接地表'不存在车型')。正确做法：调用 text_search(或相应查询工具)确认无数据后，"
                "【如实告知未找到该车型】并【推荐接地表中的真实替代车型】。严禁编造不存在车型的配置/价格/优惠，严禁硬调 book_test_drive。",
        "ref": ["查询一款不存在的车型'Model X 2025款'的配置", "预约一款不存在的车型'Model X 2025款'的试驾"],
    },
}

_PUNCT = re.compile(r"[\s，。、？！,.?!；;：:\"'“”‘’（）()\-—~·]+")


def normalize_q(s: str) -> str:
    return _PUNCT.sub("", (s or "").strip().lower())


def load_eval_questions() -> set:
    """加载评测集所有问题（归一化），用于零重叠校验。"""
    qf = REPO_ROOT / "eval" / "eval_data" / "questions.json"
    out = set()
    try:
        data = json.loads(qf.read_text(encoding="utf-8"))
        for q in data.get("questions", []):
            out.add(normalize_q(q.get("question", "")))
    except Exception as e:
        print(f"[warn] 加载评测集失败(零重叠校验降级): {e}")
    return out


def build_react_schemas() -> List[Dict[str, Any]]:
    """构建 ReActAgent 推理时暴露的 8 工具 schema：Thought + Finish + 6 业务工具。"""
    user_schemas = json.loads((SFT_DIR / "tool_schemas.json").read_text(encoding="utf-8"))
    return [THOUGHT_SCHEMA, FINISH_SCHEMA] + user_schemas


def obs_is_meaningful(name: str, resp) -> bool:
    """按工具判断 observation 是否"有数据"（用于桶感知校验）。"""
    d = resp.data or {}
    if getattr(resp.status, "value", resp.status) != "success":
        return False
    if name == "text_search":
        return d.get("total", 0) > 0
    if name == "query_discount_policy":
        return d.get("total_discount", 0) > 0 or len(d.get("discounts", [])) > 0
    if name == "query_transaction_price":
        return d.get("avg_price", 0) > 0
    if name == "competitor_compare":
        return bool(d.get("comparison"))
    if name == "calculate_finance":
        return d.get("monthly_payment", 0) > 0
    if name == "book_test_drive":
        return bool(d.get("booking_id"))
    return bool(d)


def extract_searched_prices(executed: List[Dict[str, Any]]) -> List[float]:
    prices: List[float] = []
    for s in executed:
        if s["name"] == "text_search":
            for r in (s["resp"].data or {}).get("results", []):
                p = (r.get("content") or {}).get("price")
                if isinstance(p, (int, float)):
                    prices.append(float(p))
    return prices


# ---------------------------------------------------------------------------
# Claude Pass1 / Pass2
# ---------------------------------------------------------------------------
def _pass1_system(schemas_user_json: str) -> str:
    return (
        "你是资深的汽车销售 Agent 训练数据设计专家，为「汽车销售助手」生成高质量的【单轮工具调用轨迹规划】用于 SFT。\n\n"
        f"【Agent 系统提示】\n{SYSTEM_PROMPT}\n\n"
        f"【6个业务工具 JSON Schema】\n{schemas_user_json}\n\n"
        "【接地表】所有车型/价格/优惠/成交价/对比对/经销商等真实数据，工具参数必须严格取自此表，严禁编造：\n"
        f"{json.dumps(GROUNDING, ensure_ascii=False, indent=2)}"
    )


def claude_pass1(bucket_key: str, k: int, schemas_user_json: str, temperature: float) -> List[Dict[str, Any]]:
    b = BUCKETS[bucket_key]
    clarify_extra = ""
    clarify_field = ""
    if b.get("is_clarify"):
        clarify_extra = "\n5. missing_params：列出用户未提供、需反问补全的 book_test_drive 必填参数名(数组)。"
        clarify_field = ',"missing_params":["customer_name"]'
    ref_lines = "\n".join(f"- {r}" for r in b["ref"])
    user_prompt = (
        f"请为下面的【数据类别】生成 {k} 条【互相不同、自然口语、真实多样】的用户问题及工具调用规划。\n\n"
        f"数据类别：{bucket_key}\n场景：{b['scenario']}\n靶向修复的坏例：{b.get('targets') or '常规能力'}\n"
        f"类别要求：\n{b['hint']}\n\n"
        f"参考坏例(理解模式即可，【严禁照抄原句】)：\n{ref_lines}\n\n"
        "【严格约束】\n"
        "1. question 必须是自然中文口语、单轮、彼此不同，且不得与参考坏例句面雷同(换车型/换地区/换参数/换问法)。\n"
        "2. tool_calls 只能用6个业务工具；参数严格取自接地表；多步链参数前后连贯(后一步用前一步能查到的真实值)。\n"
        "3. 不要生成 Thought/Finish(我自动补)；tool_calls 只列业务工具，按执行顺序排列。纯反问类可为空数组。\n"
        "4. reasoning：一句话说明为何这样调用(≤40字)。\n"
        f"{clarify_extra}\n"
        "只输出 JSON 数组，每元素形如：\n"
        f'{{"question":"...","reasoning":"...","tool_calls":[{{"name":"工具名","arguments":{{}}}}]{clarify_field}}}\n'
        "不要输出解释文字、不要 markdown 代码围栏。"
    )
    r = call_claude(
        [{"role": "user", "content": user_prompt}],
        system=_pass1_system(schemas_user_json),
        max_tokens=3500,
        temperature=temperature,
    )
    obj = extract_json(r["text"])
    if isinstance(obj, dict):
        obj = [obj]
    return [p for p in obj if isinstance(p, dict) and p.get("question")]


def claude_pass2(bucket_key: str, items: List[Dict[str, Any]], temperature: float, batch: int = 1) -> List[str]:
    """按 batch 分块调用 _pass2_once（batch=1 逐条，杜绝跨样本串味）。"""
    out: List[str] = []
    for i in range(0, len(items), max(1, batch)):
        out.extend(_pass2_once(bucket_key, items[i:i + batch], temperature))
    return out


def _pass2_once(bucket_key: str, items: List[Dict[str, Any]], temperature: float) -> List[str]:
    """items: [{question, observations:[{name,obs_text}], missing_params?}] → 返回 final_answer 列表。"""
    b = BUCKETS[bucket_key]
    if b.get("is_clarify"):
        rule = "这是【澄清】样本：汇总工具已查到的信息，然后【礼貌反问】missing_params 中缺失的预约细节；绝不编造参数、绝不说已预约成功。"
    elif b.get("is_degrade"):
        rule = "这是【降级拒答】样本：工具已确认无数据，请【如实告知未找到该车型】，并从接地表推荐1-2款真实替代车型；严禁编造该车型任何数据。"
    else:
        rule = "依据工具真实返回数据作答，数字必须与工具结果完全一致，简洁清晰、口语自然、直接回答用户问题。"
    sys = (f"你是汽车销售助手。根据【用户问题】和【工具真实返回】写最终答复(Finish.answer)。要求：{rule} "
           "【铁律】每个样本独立作答，final_answer 只能引用该样本自己的 question 与 observations；"
           "严禁提及该样本之外的其他车型，严禁编造用户未提供的姓名/电话/时间。")
    payload = [
        {
            "idx": i,
            "question": it["question"],
            "observations": it["observations"],
            **({"missing_params": it["missing_params"]} if it.get("missing_params") else {}),
        }
        for i, it in enumerate(items)
    ]
    user = (
        f"下面是 {len(payload)} 个样本，请为每个写 final_answer，只输出 JSON 数组 "
        '[{"final_answer":"..."}, ...]，顺序与输入严格一致，不要多余文字：\n'
        + json.dumps(payload, ensure_ascii=False)
    )
    r = call_claude([{"role": "user", "content": user}], system=sys, max_tokens=3000, temperature=temperature)
    obj = extract_json(r["text"])
    if isinstance(obj, dict):
        obj = [obj]
    answers = [ (x.get("final_answer") if isinstance(x, dict) else str(x)) or "" for x in obj ]
    # 对齐长度（防御）
    while len(answers) < len(items):
        answers.append("")
    return answers[: len(items)]


# ---------------------------------------------------------------------------
# 执行 + 校验 + 组装
# ---------------------------------------------------------------------------
def execute_plan(plan: Dict[str, Any]) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    executed: List[Dict[str, Any]] = []
    for tc in plan.get("tool_calls", []) or []:
        if not isinstance(tc, dict):
            return None, "tool_call 非对象"
        name = tc.get("name")
        args = tc.get("arguments", {}) or {}
        if name not in TOOL_MAP:
            return None, f"未知工具 {name}"
        if not isinstance(args, dict):
            return None, f"{name} arguments 非对象"
        try:
            resp = TOOL_MAP[name].run(args)
        except Exception as e:
            return None, f"{name} 执行异常 {type(e).__name__}: {e}"
        # 自愈：text_search 误传 category 过滤致空 → 去 category 重试（训练数据记录正确调用）
        if name == "text_search" and not obs_is_meaningful(name, resp):
            if args.get("category") not in (None, "", "all"):
                args2 = {k: v for k, v in args.items() if k != "category"}
                resp2 = TOOL_MAP[name].run(args2)
                if obs_is_meaningful(name, resp2):
                    args, resp = args2, resp2
        executed.append({"name": name, "arguments": args, "resp": resp, "obs_text": resp.to_model_text()})
    return executed, None


def validate(bucket_key: str, plan: Dict[str, Any], executed: List[Dict[str, Any]]) -> Optional[str]:
    b = BUCKETS[bucket_key]
    # 1) 必填参数
    for s in executed:
        for r in REQUIRED.get(s["name"], []):
            v = s["arguments"].get(r)
            if v is None or v == "":
                return f"{s['name']} 缺必填参数 {r}"
    # 2) 桶感知非空
    if b.get("is_degrade"):
        if not executed:
            return "降级样本需至少一次探测调用"
        if obs_is_meaningful(executed[0]["name"], executed[0]["resp"]):
            return "降级样本探测却返回了数据(车型其实存在?)"
        return None
    if b.get("is_clarify"):
        if not plan.get("missing_params"):
            return "澄清样本 missing_params 为空"
        for s in executed:
            if not obs_is_meaningful(s["name"], s["resp"]):
                return f"澄清样本 {s['name']} 返回空(接地失败)"
        return None
    # 常规桶：必须有调用且全部非空
    if not executed:
        return "常规样本无工具调用"
    for s in executed:
        if not obs_is_meaningful(s["name"], s["resp"]):
            return f"{s['name']} 返回空(接地失败)"
    # 3) 金融价格连贯：calculate_finance.car_price 必须来自 text_search 查到的真实价
    names = [s["name"] for s in executed]
    if "calculate_finance" in names and "text_search" in names:
        prices = extract_searched_prices(executed)
        if prices:
            for s in executed:
                if s["name"] == "calculate_finance":
                    cp = s["arguments"].get("car_price")
                    if not isinstance(cp, (int, float)) or not any(abs(float(cp) - p) < 1.0 for p in prices):
                        return f"金融链不连贯: car_price={cp} 不在查到的价格 {prices}"
    # 4) 步数上限（Thought + n工具 + Finish ≤ 5）
    n_thought = 1 if b.get("include_thought") else 0
    if n_thought + len(executed) + 1 > 5:
        return f"步数超限({n_thought}+{len(executed)}+1>5)"
    return None


def assemble(bucket_key: str, plan: Dict[str, Any], executed: List[Dict[str, Any]],
             final_answer: str, idx: int) -> Dict[str, Any]:
    b = BUCKETS[bucket_key]
    msgs: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": plan["question"]},
    ]
    cid = 0

    def add_assistant_call(name: str, args_obj: Dict[str, Any]) -> str:
        nonlocal cid
        cid += 1
        tid = f"call_{cid}"
        msgs.append({
            "role": "assistant", "content": None,
            "tool_calls": [{"id": tid, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args_obj, ensure_ascii=False)}}],
        })
        return tid

    if b.get("include_thought") and plan.get("reasoning"):
        tid = add_assistant_call("Thought", {"reasoning": plan["reasoning"]})
        msgs.append({"role": "tool", "tool_call_id": tid, "content": f"推理: {plan['reasoning']}"})

    for s in executed:
        tid = add_assistant_call(s["name"], s["arguments"])
        msgs.append({"role": "tool", "tool_call_id": tid, "content": s["obs_text"]})

    add_assistant_call("Finish", {"answer": final_answer})

    return {
        "id": f"sft_{bucket_key}_{idx:04d}",
        "bucket": bucket_key,
        "targets": b.get("targets", []),
        "scenario": b.get("scenario", ""),
        "question": plan["question"],
        "expected_tools": [s["name"] for s in executed],  # 仅业务工具(判分对齐)
        "is_clarify": bool(b.get("is_clarify")),
        "is_degrade": bool(b.get("is_degrade")),
        "messages": msgs,
        "meta": {
            "synth_model": "claude-sonnet-5",
            "reasoning": plan.get("reasoning", ""),
            "missing_params": plan.get("missing_params"),
            "final_answer": final_answer,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def compute_alloc(args) -> Dict[str, int]:
    if args.bucket:
        return {args.bucket: (args.limit or args.count or 20)}
    if args.limit:
        return {k: args.limit for k in BUCKETS}
    total = args.count or 300
    alloc = {}
    for k, b in BUCKETS.items():
        alloc[k] = max(1, round(total * b["weight"]))
    return alloc


def gen_bucket(bucket_key: str, target_n: int, seen: set, schemas_user_json: str,
               args, rejects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    attempts = 0
    max_attempts = target_n * 5 + 10
    K = args.batch
    while len(kept) < target_n and attempts < max_attempts:
        k = min(K, target_n - len(kept) + 2)
        try:
            plans = claude_pass1(bucket_key, k, schemas_user_json, args.temperature)
        except Exception as e:
            print(f"  [warn] {bucket_key} pass1 失败: {e}")
            time.sleep(3)
            attempts += k
            continue
        attempts += max(1, len(plans))
        # 执行 + 校验
        validated = []
        for p in plans:
            qn = normalize_q(p.get("question", ""))
            if not qn or qn in seen:
                rejects.append({"bucket": bucket_key, "question": p.get("question"), "reason": "重复/与评测集重叠"})
                continue
            executed, err = execute_plan(p)
            if err:
                rejects.append({"bucket": bucket_key, "question": p.get("question"), "reason": err})
                continue
            verr = validate(bucket_key, p, executed)
            if verr:
                rejects.append({"bucket": bucket_key, "question": p.get("question"), "reason": verr})
                continue
            seen.add(qn)
            validated.append((p, executed))
        if not validated:
            continue
        # Pass2：写最终答复
        items = []
        for p, executed in validated:
            items.append({
                "question": p["question"],
                "observations": [{"name": s["name"], "obs_text": s["obs_text"]} for s in executed],
                "missing_params": p.get("missing_params"),
            })
        try:
            answers = claude_pass2(bucket_key, items, args.temperature, args.pass2_batch)
        except Exception as e:
            print(f"  [warn] {bucket_key} pass2 失败: {e}")
            answers = [""] * len(items)
        for (p, executed), ans in zip(validated, answers):
            if not ans or not ans.strip():
                rejects.append({"bucket": bucket_key, "question": p.get("question"), "reason": "final_answer 为空"})
                continue
            kept.append((p, executed, ans))
            if len(kept) >= target_n:
                break
    # 组装
    out = []
    for i, (p, executed, ans) in enumerate(kept):
        out.append(assemble(bucket_key, p, executed, ans, i))
    return out


def main():
    ap = argparse.ArgumentParser(description="Claude 合成 SFT 工具轨迹")
    ap.add_argument("--count", type=int, default=0, help="总量(按配比分配到各桶)")
    ap.add_argument("--limit", type=int, default=0, help="每桶固定条数(冒烟测试用)")
    ap.add_argument("--bucket", type=str, default="", choices=list(BUCKETS.keys()), help="只跑单桶")
    ap.add_argument("--batch", type=int, default=5, help="每次 Claude 生成的规划条数")
    ap.add_argument("--temperature", type=float, default=0.9, help="合成多样性温度")
    ap.add_argument("--pass2-batch", dest="pass2_batch", type=int, default=1, help="写最终答复的批量(1=逐条,最稳)")
    ap.add_argument("--out", type=str, default="", help="输出 JSONL 路径")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    cfg = load_config()
    print(f"[cfg] Claude 端点已就绪 (model={cfg['model']})")

    # 导出 ReAct 8 工具 schema（供 s3 的 tools 字段）
    react_schemas = build_react_schemas()
    (SFT_DIR / "tool_schemas_react.json").write_text(
        json.dumps(react_schemas, ensure_ascii=False, indent=2), encoding="utf-8")
    schemas_user_json = json.dumps(json.loads((SFT_DIR / "tool_schemas.json").read_text(encoding="utf-8")),
                                   ensure_ascii=False, indent=2)
    print(f"[schema] 已导出 tool_schemas_react.json（{len(react_schemas)} 工具：Thought+Finish+6业务）")

    seen = load_eval_questions()
    print(f"[overlap] 已加载 {len(seen)} 条评测集问题用于零重叠校验")

    alloc = compute_alloc(args)
    print(f"[alloc] 目标分配: {alloc} (合计 {sum(alloc.values())})")

    out_path = Path(args.out) if args.out else (SFT_DIR / "sft_trajectories.jsonl")
    rej_path = SFT_DIR / "_rejects.jsonl"
    # 增量落盘：每桶完成即写入，长跑中断也不丢已生成数据
    open(out_path, "w", encoding="utf-8").close()
    open(rej_path, "w", encoding="utf-8").close()
    all_records: List[Dict[str, Any]] = []
    all_rejects: List[Dict[str, Any]] = []
    t0 = time.time()
    for bk, n in alloc.items():
        print(f"\n=== 生成桶 {bk} (目标 {n}) ===", flush=True)
        bucket_rejects: List[Dict[str, Any]] = []
        recs = gen_bucket(bk, n, seen, schemas_user_json, args, bucket_rejects)
        with open(out_path, "a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(rej_path, "a", encoding="utf-8") as f:
            for r in bucket_rejects:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        all_records.extend(recs)
        all_rejects.extend(bucket_rejects)
        print(f"  ✓ {bk}: 得到 {len(recs)}/{n} (累计 {len(all_records)})", flush=True)
    dt = time.time() - t0

    # 报告
    by_bucket: Dict[str, int] = {}
    for r in all_records:
        by_bucket[r["bucket"]] = by_bucket.get(r["bucket"], 0) + 1
    print("\n" + "=" * 56)
    print(f"[完成] 写出 {len(all_records)} 条 → {out_path}")
    print(f"[分布] {by_bucket}")
    print(f"[丢弃] {len(all_rejects)} 条 → {rej_path}")
    rej_reasons: Dict[str, int] = {}
    for r in all_rejects:
        rej_reasons[r["reason"]] = rej_reasons.get(r["reason"], 0) + 1
    if rej_reasons:
        print("[丢弃原因 top]")
        for reason, c in sorted(rej_reasons.items(), key=lambda x: -x[1])[:8]:
            print(f"   {c:3d}  {reason}")
    print(f"[耗时] {dt:.1f}s")
    print("=" * 56)


if __name__ == "__main__":
    main()
