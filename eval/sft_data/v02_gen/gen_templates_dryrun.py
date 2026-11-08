# -*- coding: utf-8 -*-
"""v02 干 run 模板：桶1×试驾（20 条）+ 桶2×试驾（10 条）

骨架结构约定：
- question / answer：润色目标（程序模板文本，参数已代入）
- reasonings：Thought 推理列表（程序模板，不润色——含日期换算）
- build(q_final, a_final) -> messages：闭包捕获实跑结果，组装最终消息序列
- 润色失败时 q_final/a_final 直接用程序模板原文

桶1 覆盖形态：缺姓名+电话 / 缺姓名 / 缺电话 / 缺时间+姓名+电话 / 全缺 / 门店未指定 / 门店未指定+时刻不明确 / 门店未指定+缺电话
桶2 覆盖形态：clarify 结局 4 条（编造 dealer / 编造姓名电话 / 编造姓名 / 编造+时间错）
               execute 结局 6 条（下周一→明天 / 本周日→下周日 / 明天→今天 / 后天多算 / 门店写错 / 时间越界被工具拦截）
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Any

from gen_core import (
    BASE_DATE_POOL, CONFIGS, DEALERS, CUSTOMER_NAMES,
    FABRICATED_NAME, FABRICATED_PHONE, FABRICATED_DEALER,
    date_cn, iso, next_monday, this_saturday, this_sunday, run_tool,
    user_msg, thought_call, thought_result, tool_call, tool_result, finish_call,
)


def _dealer_list() -> str:
    """全量 7 家白名单门店列表（与 book_test_drive 工具契约一致，避免子集呈现强化幻觉）。"""
    return "\n".join(f"{i}. {d}" for i, d in enumerate(DEALERS, 1))


def _pick(seq, i):
    return seq[i % len(seq)]


def _phone(i: int) -> str:
    """生成合法 11 位手机号（137 开头，随 i 变化；任意 i 均保证 11 位，
    避免 book_test_drive 手机号格式校验误触发 INVALID_PARAM）。"""
    return f"137{i:04d}{i:04d}"


# ============================================================
# 桶1 × 试驾（20 条，全部 clarify，不调用 book_test_drive）
# ============================================================

def _b1_common(i: int, sid: str, question: str, reasoning: str, answer: str,
               design_intent: str, review_focus: str) -> Dict[str, Any]:
    def build(q_final: str, a_final: str) -> List[Dict[str, Any]]:
        return [
            user_msg(q_final),
            thought_call("call_1", reasoning),
            thought_result("call_1", reasoning),
            finish_call("call_2", a_final),
        ]
    return {
        "sid": sid, "bucket": "bucket1_clarify", "scenario": "试驾预约", "targets": [],
        "is_clarify": True, "is_degrade": False,
        "question": question, "answer": answer, "reasonings": [reasoning],
        "build": build, "design_intent": design_intent, "review_focus": review_focus,
    }


def make_bucket1_test_drive(rng: random.Random) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    d1, d2 = DEALERS[0], DEALERS[1]  # 题面指定门店时仍用白名单门店轮换

    # A 组：缺姓名+电话（时间+门店+车型齐）3 条
    specs_a = [
        (BASE_DATE_POOL[1], "明天上午10点", d2, CONFIGS[0]),
        (BASE_DATE_POOL[0], "下周一上午9点", d1, CONFIGS[1]),
        (BASE_DATE_POOL[2], "本周六下午3点", DEALERS[4], CONFIGS[2]),
    ]
    qs_a = [
        "帮我预约{time}在{dealer}试驾{car}",
        "我想约{time}的试驾，去{dealer}，车型{car}",
        "{car}能约在{time}试驾吗？我想去{dealer}",
    ]
    for j, (base, t_expr, dealer, cfg) in enumerate(specs_a, start=1):
        q = qs_a[j - 1].format(time=t_expr, dealer=dealer, car=cfg["title"])
        r = "用户已提供车型、门店和时间，但缺少预约必需的姓名和联系电话。不能编造客户信息，先向用户询问这两个缺失参数。"
        a = (f"好的，可以为您预约{cfg['title']}的试驾。目前还差两项信息，麻烦您补充一下：\n"
             f"1. 您的姓名\n2. 联系电话\n\n补充后我会立即为您安排{dealer}的试驾。")
        out.append(_b1_common(
            j, f"v02_dry_001_{j:02d}", q, r, a,
            f"试驾缺参→反问正例（缺姓名+电话）：时间/门店/车型已给，不编造、不硬调",
            f"①是否补全姓名+电话两个缺失参数 ②未调用 book_test_drive ③时间表达'{t_expr}'未硬换算"))

    # B 组：缺姓名（电话+时间+门店齐）2 条
    specs_b = [
        (BASE_DATE_POOL[1], "明天上午10点", d1, CONFIGS[1]),
        (BASE_DATE_POOL[2], "下周一上午9点", d2, CONFIGS[3]),
    ]
    for j, (base, t_expr, dealer, cfg) in enumerate(specs_b, start=1):
        phone = _phone(10 + j)
        q = f"你好，电话{phone}，想约{t_expr}在{dealer}试驾{cfg['title']}"
        r = "用户提供了电话、时间和门店，但缺少预约必需的姓名。不能编造客户姓名，先向用户询问。"
        a = (f"可以的，帮您预约{cfg['title']}试驾还差您的姓名，请提供一下姓名，"
             f"我马上为您安排{dealer}的试驾。")
        out.append(_b1_common(
            j, f"v02_dry_001_b{j:02d}", q, r, a,
            "试驾缺参→反问正例（只缺姓名）：电话/时间/门店齐，只问姓名一个参数",
            "①只问姓名这一个缺失参数 ②未调用工具 ③电话/时间/门店如实保留"))

    # C 组：缺电话（姓名+时间+门店齐）2 条
    specs_c = [
        (BASE_DATE_POOL[2], "本周六下午2点", DEALERS[3], CONFIGS[2]),
        (BASE_DATE_POOL[4], "明天下午4点", d1, CONFIGS[0]),
    ]
    for j, (base, t_expr, dealer, cfg) in enumerate(specs_c, start=1):
        name = _pick(CUSTOMER_NAMES, 10 + j)
        q = f"我是{name}，想约{t_expr}在{dealer}试驾{cfg['title']}"
        r = "用户提供了姓名、时间和门店，但缺少预约必需的联系电话。不能编造电话号码，先向用户询问。"
        a = (f"您好{name}，为您预约{cfg['title']}试驾还差联系电话，请提供您的手机号，"
             f"我马上为您安排{dealer}的试驾。")
        out.append(_b1_common(
            j, f"v02_dry_001_c{j:02d}", q, r, a,
            "试驾缺参→反问正例（只缺电话）：姓名/时间/门店齐，只问电话一个参数",
            "①只问电话这一个缺失参数 ②未调用工具 ③姓名/时间/门店如实保留"))

    # D 组：缺时间+姓名+电话（车型+门店齐）3 条
    specs_d = [d2, d1, DEALERS[5]]
    qs_d = [
        "帮我预约在{dealer}的{car}试驾",
        "我想去{dealer}试驾{car}，帮我约一下",
        "能帮我约{dealer}的{car}试驾吗？",
    ]
    for j, dealer in enumerate(specs_d, start=1):
        cfg = _pick(CONFIGS, j)
        q = qs_d[j - 1].format(dealer=dealer, car=cfg["title"])
        r = "用户提供了车型和门店，但时间、姓名、联系电话均缺失，需要一次性向用户问清这三项信息。"
        a = (f"没问题，帮您预约{cfg['title']}试驾需要确认几项信息：\n"
             f"1. 您希望什么时间试驾？\n2. 您的姓名\n3. 联系电话\n\n"
             f"告诉我后马上为您安排在{dealer}的试驾。")
        out.append(_b1_common(
            j, f"v02_dry_001_d{j:02d}", q, r, a,
            "试驾缺参→反问正例（缺时间+姓名+电话）：车型/门店已给，一次问清三项",
            "①三项缺失信息是否全部问到 ②未调用工具 ③门店如实保留"))

    # E 组：全缺（只给车型/车系）2 条
    qs_e = [
        "我想约个{car}的试驾，你帮我安排下",
        "帮我安排一下试驾，就{car}这款",
    ]
    for j, q_t in enumerate(qs_e, start=1):
        cfg = _pick(CONFIGS, 3 + j)
        q = q_t.format(car=cfg["title"])
        r = "用户想预约试驾但未提供任何预约所需信息：时间、门店、姓名、电话均缺失，需要一次性向用户问清。"
        a = (f"没问题，帮您预约{cfg['title']}试驾需要确认几项信息：\n"
             f"1. 您希望什么时间试驾？\n2. 倾向哪家门店？（可选：\n{_dealer_list()}）\n"
             f"3. 您的姓名和联系电话？\n\n告诉我后马上为您安排。")
        out.append(_b1_common(
            j, f"v02_dry_001_e{j:02d}", q, r, a,
            "试驾全缺参→反问正例：时间/门店/姓名/电话全部缺失，一次性问全",
            "①四项缺失信息是否全部问到 ②门店列表=7家白名单全量（与工具契约一致） ③未调用工具"))

    # F 组：门店未指定（姓名+电话+时间齐）3 条
    specs_f = [
        (BASE_DATE_POOL[1], "明天上午10点", CONFIGS[0]),
        (BASE_DATE_POOL[2], "下周一上午9点", CONFIGS[1]),
        (BASE_DATE_POOL[0], "本周六下午3点", CONFIGS[3]),
    ]
    qs_f = [
        "我叫{name}，电话{phone}，想约{time}试驾{car}，附近有什么门店可选？",
        "我是{name}，电话{phone}，约{time}试驾{car}，门店你推荐一下",
        "我叫{name}，电话{phone}，想{time}试驾{car}，都有哪些门店？",
    ]
    for j, (base, t_expr, cfg) in enumerate(specs_f, start=1):
        name = _pick(CUSTOMER_NAMES, 20 + j)
        phone = _phone(20 + j)
        q = qs_f[j - 1].format(name=name, phone=phone, time=t_expr, car=cfg["title"])
        r = "用户已提供姓名、电话和试驾时间，但未指定门店。系统内可预约的门店共有七家，如实列出让用户选择，不能替用户编造或随意指定。"
        a = (f"{name}您好，目前可预约{cfg['title']}试驾的门店共有7家：\n{_dealer_list()}\n\n"
             f"请问您想选择哪家？确认后我会为您预约{t_expr}的试驾。")
        out.append(_b1_common(
            j, f"v02_dry_001_f{j:02d}", q, r, a,
            "门店未指定→报门店列表正例：报 7 家白名单全量门店，请用户选择，不编店名",
            "①门店列表=7家白名单全量（与工具契约一致） ②姓名/电话/时间如实保留 ③未调用工具"))

    # G 组：门店未指定+时刻不明确（"上午"）2 条
    specs_g = [
        (BASE_DATE_POOL[3], "本周日上午", CONFIGS[1]),
        (BASE_DATE_POOL[1], "明天上午", CONFIGS[2]),
    ]
    for j, (base, t_expr, cfg) in enumerate(specs_g, start=1):
        name = _pick(CUSTOMER_NAMES, 30 + j)
        phone = _phone(30 + j)
        q = f"预约{cfg['title']}试驾，{t_expr}，我叫{name}，电话{phone}"
        r = "用户给了姓名电话和时间范围（上午），但没给具体时刻和门店。先列出可选门店并请用户确认具体时间点，再完成预约。"
        a = (f"{name}您好，为您预约{t_expr}的试驾，还需要确认两点：\n"
             f"1. 门店选择：\n{_dealer_list()}\n2. 具体时间：上午几点方便？（营业时间 9:00-18:00）\n\n"
             f"确认后立即为您锁定。")
        out.append(_b1_common(
            j, f"v02_dry_001_g{j:02d}", q, r, a,
            "门店未指定+时间未到时刻→反问正例：报门店列表并确认具体时刻",
            "①门店列表=7家白名单全量 ②时间范围'上午'未硬换算（先确认） ③不编时间"))

    # H 组：门店未指定+缺电话 3 条
    specs_h = [
        (BASE_DATE_POOL[1], "明天下午2点", CONFIGS[0]),
        (BASE_DATE_POOL[2], "下周一上午10点", CONFIGS[3]),
        (BASE_DATE_POOL[0], "本周六下午4点", CONFIGS[2]),
    ]
    qs_h = [
        "我是{name}，想约{time}试驾{car}，去哪家店方便？",
        "我叫{name}，约{time}试驾{car}，哪家店都行，你定",
        "我{name}，{time}想试驾{car}，门店有哪些可选？",
    ]
    for j, (base, t_expr, cfg) in enumerate(specs_h, start=1):
        name = _pick(CUSTOMER_NAMES, 40 + j)
        q = qs_h[j - 1].format(name=name, time=t_expr, car=cfg["title"])
        r = "用户提供姓名和时间，但未指定门店，也缺少联系电话。先列出可选门店，并请用户补充联系电话。"
        a = (f"{name}您好，可预约的门店共有7家：\n{_dealer_list()}\n\n"
             f"另外还需要您的联系电话，请一并告知，我马上为您预约。")
        out.append(_b1_common(
            j, f"v02_dry_001_h{j:02d}", q, r, a,
            "门店未指定+缺电话→反问正例：报门店列表并补问电话",
            "①门店列表=7家白名单全量 ②电话作为缺失参数被问到 ③未调用工具"))

    return out


# ============================================================
# 桶2 × 试驾（10 条：clarify 结局 4 条 + execute 结局 6 条）
# ============================================================

def _b2_common(sid: str, question: str, r1: str, err_args: Dict[str, Any],
               correction_user: str, r2: str, answer: str,
               turn2_calls: List[Dict[str, Any]], design_intent: str,
               review_focus: str, call_ok: bool = True,
               targets: List[str] = None) -> Dict[str, Any]:
    """turn2_calls: [(tool_name, args)] 修正轮的工具调用序列（clarify 结局为空 → Finish 反问）"""
    if targets is None:
        targets = ["book_test_drive"] if call_ok else []
    # 实跑 turn1 错误调用（骨架构建时一次性完成，保证可复现）
    err_obs = run_tool("book_test_drive", err_args)
    # 实跑 turn2 修正调用
    turn2_obs: List[str] = []
    for name, args in turn2_calls:
        turn2_obs.append(run_tool(name, args))

    def build(q_final: str, a_final: str) -> List[Dict[str, Any]]:
        msgs: List[Dict[str, Any]] = [
            user_msg(q_final),
            thought_call("call_1", r1, loss_mask=False),
            thought_result("call_1", r1),
            tool_call("call_2", "book_test_drive", err_args, loss_mask=False),
            tool_result("call_2", err_obs),
        ]
        if correction_user:
            msgs.append(user_msg(correction_user))
        cid = 3
        msgs.append(thought_call(f"call_{cid}", r2))
        msgs.append(thought_result(f"call_{cid}", r2))
        cid += 1
        for (name, args), obs in zip(turn2_calls, turn2_obs):
            msgs.append(tool_call(f"call_{cid}", name, args))
            msgs.append(tool_result(f"call_{cid}", obs))
            cid += 1
        msgs.append(finish_call(f"call_{cid}", a_final))
        return msgs

    return {
        "sid": sid, "bucket": "bucket2_correction", "scenario": "试驾预约", "targets": targets,
        "is_clarify": not call_ok, "is_degrade": False,
        "question": question, "answer": answer, "reasonings": [r1, r2],
        "build": build, "design_intent": design_intent, "review_focus": review_focus,
    }


def make_bucket2_test_drive(rng: random.Random) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    d1, d2 = DEALERS[0], DEALERS[1]

    # ---- clarify 结局（turn2 反问，不 execute）----
    # K1: 缺姓名+电话+门店 → turn1 编造 dealer+姓名+电话 → NOT_FOUND → 指正 → 报 7 家+反问
    base = BASE_DATE_POOL[1]  # 11-06 周五，明天=11-07
    t_correct = base + timedelta(days=1)
    q = f"预约明天上午10点试驾{CONFIGS[0]['title']}"
    r1 = "用户只给了时间，姓名、电话、门店都缺。为尽快完成预约，先按常见信息补全参数尝试预约。"
    err_args = {"car_series": "E5 Sportback", "dealer": FABRICATED_DEALER,
                "appointment_time": iso(t_correct.replace(hour=10)),
                "customer_name": FABRICATED_NAME, "customer_phone": FABRICATED_PHONE}
    cu = "我没有告诉过你姓名电话，也没说去哪家店，你怎么自己编了？请重新处理。"
    r2 = "用户指出我擅自编造了姓名、电话和门店。正确做法是列出可预约的真实门店，并请用户提供姓名和联系电话，不再擅自补全参数。"
    a = (f"非常抱歉，我此前不应该擅自填写您的个人信息和门店。预约明天上午10点的试驾，还需要您确认：\n"
         f"1. 门店（共7家可选）：\n{_dealer_list()}\n2. 您的姓名\n3. 联系电话\n\n补充后我立即为您预约，不会再自行编造信息。")
    out.append(_b2_common(
        "v02_dry_002_k01", q, r1, err_args, cu, r2, a, [],
        "纠错多轮 clarify 结局：turn1 编造张三/13800138000/上海大众4S店硬调→工具 NOT_FOUND+用户指正→turn2 报真实门店+反问",
        "①turn1 编造形态=真实形态 ②turn1 编造段 loss_mask=false（Thought+调用 2 段）③turn2 修正完整：道歉+7家白名单全量+反问姓名电话",
        call_ok=False))

    # K2: 缺姓名+电话（门店给）→ turn1 编造姓名+电话 → SUCCESS → 指正 → 反问
    base = BASE_DATE_POOL[2]  # 11-13 周五，下周一=11-16
    t_correct = next_monday(base)
    q = f"我想约下周一上午9点在{d1}试驾{CONFIGS[1]['title']}"
    r1 = "用户没给姓名和电话，为了尽快约上，先按常见信息补全后直接预约。"
    err_args = {"car_series": "E5 Sportback", "dealer": d1,
                "appointment_time": iso(t_correct.replace(hour=9)),
                "customer_name": FABRICATED_NAME, "customer_phone": FABRICATED_PHONE}
    cu = "我没说过姓名和电话，你怎么就给我约上了？请重新处理。"
    r2 = "用户指出我编造了姓名和电话。预约必须使用用户提供的真实信息，应撤回编造操作，向用户询问缺失的姓名和联系电话。"
    a = (f"抱歉，我之前不应擅自填写您的姓名和电话。为您预约下周一上午9点在{d1}的{CONFIGS[1]['title']}试驾，"
         f"还需要您提供：\n1. 您的姓名\n2. 联系电话\n\n补充后我立即为您预约。")
    out.append(_b2_common(
        "v02_dry_002_k02", q, r1, err_args, cu, r2, a, [],
        "纠错多轮 clarify 结局：turn1 编造姓名+电话预约成功（工具不校验个人信息）→用户指正→turn2 反问",
        "①turn1 编造段 loss_mask=false ②turn2 修正为反问（不执行）③反问只问姓名+电话两个缺失参数",
        call_ok=False))

    # K3: 缺姓名（电话+时间+门店齐）→ turn1 编造姓名 → SUCCESS → 指正 → 反问姓名
    base = BASE_DATE_POOL[3]  # 11-07 周六，本周日=11-08
    t_correct = this_sunday(base)
    phone = _phone(55)
    q = f"电话{phone}，想约周日下午2点在{d2}试驾{CONFIGS[2]['title']}"
    r1 = "用户给了电话、时间和门店，缺姓名。先补一个常见姓名尽快预约。"
    err_args = {"car_series": "E5 Sportback", "dealer": d2,
                "appointment_time": iso(t_correct.replace(hour=14)),
                "customer_name": FABRICATED_NAME, "customer_phone": phone}
    cu = "我还没告诉你姓名，你编了个名字就预约了，重新处理。"
    r2 = "用户指出我编造了姓名。应请用户提供真实姓名后再预约，不再擅自补全。"
    a = (f"抱歉，我不应擅自填写姓名。为完成周日下午2点在{d2}的{CONFIGS[2]['title']}试驾预约，"
         f"请提供您的姓名，我马上为您安排。")
    out.append(_b2_common(
        "v02_dry_002_k03", q, r1, err_args, cu, r2, a, [],
        "纠错多轮 clarify 结局：turn1 只编造姓名（电话/门店真实）→用户指正→turn2 反问姓名",
        "①turn1 编造段 loss_mask=false ②turn2 只问姓名 ③电话/门店参数全程可溯源",
        call_ok=False))

    # K4': 缺姓名+电话，时间"明天上午10点" → turn1 编造姓名电话+时间换算错（明天→后天）→ SUCCESS → 指正 → 反问
    base = BASE_DATE_POOL[1]  # 11-06 周五，明天=11-07；错误写成后天 11-08
    t_correct = base + timedelta(days=1)
    t_wrong = base + timedelta(days=2)
    q = f"帮我约明天上午10点在{DEALERS[6]}试驾{CONFIGS[3]['title']}"
    r1 = "用户缺姓名电话。先按常见信息补全，并把'明天'理解为后天进行预约。"
    err_args = {"car_series": "E5 Sportback", "dealer": DEALERS[6],
                "appointment_time": iso(t_wrong.replace(hour=10)),
                "customer_name": FABRICATED_NAME, "customer_phone": FABRICATED_PHONE}
    cu = f"首先我没给你姓名电话，另外我说的明天是{date_cn(t_correct)}，不是{date_cn(t_wrong)}，请重新处理。"
    r2 = "用户指出两点：一是我不应编造姓名电话，二是'明天'应指11月7日而非11月8日。撤回编造操作，向用户询问缺失信息。"
    a = (f"抱歉，我此前擅自填写了您的个人信息，时间也理解错了。为您预约明天（{date_cn(t_correct)}）上午10点"
         f"在{DEALERS[6]}的{CONFIGS[3]['title']}试驾，还需要您提供：\n1. 您的姓名\n2. 联系电话\n\n补充后我立即为您预约。")
    out.append(_b2_common(
        "v02_dry_002_k04", q, r1, err_args, cu, r2, a, [],
        "纠错多轮 clarify 结局：turn1 编造姓名电话+时间换算错（明天→后天）→用户两点指正→turn2 反问（日期按正确口径呈现）",
        "①turn1 编造段 loss_mask=false ②turn2 中'明天'正确呈现为11月7日 ③反问姓名+电话",
        call_ok=False))

    # ---- execute 结局（turn2 修正后成功预约）----
    def _execute(seq_no: str, base, t_expr, t_correct: datetime, t_wrong: datetime,
                 name, phone, dealer: str, dealer_wrong: str, cfg, r1: str,
                 cu: str, r2: str, a: str, intent: str, focus: str):
        err_args = {"car_series": "E5 Sportback", "dealer": dealer_wrong,
                    "appointment_time": iso(t_wrong),
                    "customer_name": name, "customer_phone": phone}
        fix_args = {"car_series": "E5 Sportback", "dealer": dealer,
                    "appointment_time": iso(t_correct),
                    "customer_name": name, "customer_phone": phone}
        out.append(_b2_common(
            f"v02_dry_002_x{seq_no}", t_expr, r1, err_args, cu, r2, a,
            [("book_test_drive", fix_args)], intent, focus, call_ok=True))

    # X1: 下周一→明天（base 11-06，正确 11-09，错 11-07）
    base = BASE_DATE_POOL[1]
    name, phone = CUSTOMER_NAMES[0], _phone(61)
    t_correct = next_monday(base).replace(hour=9)
    t_wrong = (base + timedelta(days=1)).replace(hour=9)
    q = f"我是{name}，电话{phone}，想约下周一上午9点在{d1}试驾{CONFIGS[0]['title']}"
    _execute("01", base, q, t_correct, t_wrong, name, phone, d1, d1, CONFIGS[0],
        "用户信息齐全。'下周一'理解为明天11月7日，直接预约。",
        f"不对，我说的下周一应该是{date_cn(t_correct)}，你约成{date_cn(t_wrong)}了，请重新预约。",
        "用户指出'下周一'应指下一周的周一，即2026年11月9日，而非11月7日。按11月9日上午9点重新预约。",
        (f"抱歉之前把日期理解错了。已为您重新预约：{date_cn(t_correct)}上午9:00，{d1}，"
         f"试驾{CONFIGS[0]['title']}。请携带驾驶证原件，提前15分钟到店。"),
        "纠错多轮 execute 结局：turn1 时间换算错（下周一→11-07）→用户指正→turn2 修正为11-09并成功预约",
        "①turn1 错误=时间换算错（参数未编造）②turn2 修正后参数全部可溯源 ③最终 execute 成功")

    # X2: 本周日→下周日（base 11-07 周六，正确 11-08，错 11-15）
    base = BASE_DATE_POOL[3]
    name, phone = CUSTOMER_NAMES[1], _phone(62)
    t_correct = this_sunday(base).replace(hour=14)
    t_wrong = (this_sunday(base) + timedelta(days=7)).replace(hour=14)
    q = f"我叫{name}，电话{phone}，本周日下午2点去{d2}试驾{CONFIGS[1]['title']}"
    _execute("02", base, q, t_correct, t_wrong, name, phone, d2, d2, CONFIGS[1],
        "用户信息齐全。'本周日'按下一周的周日理解，预约在11月15日下午2点。",
        f"我明明说的是本周日{date_cn(t_correct)}，你怎么约到下周日去了？请重新预约。",
        "用户指出'本周日'指本周的周日11月8日，而非下周的周日11月15日。按11月8日下午2点重新预约。",
        (f"抱歉理解错了。已为您重新预约：{date_cn(t_correct)}下午2:00，{d2}，"
         f"试驾{CONFIGS[1]['title']}。请携带驾驶证原件，提前15分钟到店。"),
        "纠错多轮 execute 结局：turn1 本周日→下周日错算→用户指正→turn2 修正成功",
        "①turn1 时间换算错 ②turn2 修正为本周日 ③最终 execute 成功")

    # X3: 明天→今天（base 11-06，正确 11-07，错 11-06）
    base = BASE_DATE_POOL[1]
    name, phone = CUSTOMER_NAMES[2], _phone(63)
    t_correct = (base + timedelta(days=1)).replace(hour=15)
    t_wrong = base.replace(hour=15)
    q = f"我是{name}，电话{phone}，明天下午3点想在{DEALERS[5]}试驾{CONFIGS[2]['title']}"
    _execute("03", base, q, t_correct, t_wrong, name, phone, DEALERS[5], DEALERS[5], CONFIGS[2],
        "用户信息齐全。'明天'按今天理解，预约在今天下午3点。",
        f"我说的是明天{date_cn(t_correct)}下午3点，你约成今天{date_cn(t_wrong)}了，重新约。",
        "用户指出'明天'指11月7日而非今天11月6日。按11月7日下午3点重新预约。",
        (f"抱歉理解错了。已为您重新预约：{date_cn(t_correct)}下午3:00，{DEALERS[5]}，"
         f"试驾{CONFIGS[2]['title']}。请携带驾驶证原件，提前15分钟到店。"),
        "纠错多轮 execute 结局：turn1 明天→今天错算→用户指正→turn2 修正成功",
        "①turn1 时间换算错 ②turn2 修正为明天 ③最终 execute 成功")

    # X4: 后天→多算一天（base 11-08 周日，正确 11-10 周二，错 11-11 周三）
    base = BASE_DATE_POOL[4]
    name, phone = CUSTOMER_NAMES[3], _phone(64)
    t_correct = (base + timedelta(days=2)).replace(hour=10)
    t_wrong = (base + timedelta(days=3)).replace(hour=10)
    q = f"我叫{name}，电话{phone}，想约后天上午10点在{d1}试驾{CONFIGS[3]['title']}"
    _execute("04", base, q, t_correct, t_wrong, name, phone, d1, d1, CONFIGS[3],
        f"用户信息齐全。'后天'理解为{date_cn(t_wrong)}，直接预约。",
        f"后天应该是{date_cn(t_correct)}，你约成{date_cn(t_wrong)}了，请重新预约。",
        f"用户指出'后天'指{date_cn(t_correct)}而非{date_cn(t_wrong)}。按{date_cn(t_correct)}上午10点重新预约。",
        (f"抱歉日期算错了。已为您重新预约：{date_cn(t_correct)}上午10:00，{d1}，"
         f"试驾{CONFIGS[3]['title']}。请携带驾驶证原件，提前15分钟到店。"),
        "纠错多轮 execute 结局：turn1 后天多算一天→用户指正→turn2 修正成功",
        "①turn1 时间换算错 ②turn2 修正为后天 ③最终 execute 成功")

    # X5: 门店写错（冠松→冠澳，全参数齐，base 11-13，下周一 11-16）
    base = BASE_DATE_POOL[2]
    name, phone = CUSTOMER_NAMES[4], _phone(65)
    t_correct = next_monday(base).replace(hour=9)
    dealer_ok, dealer_wrong = DEALERS[5], DEALERS[4]  # 冠松 / 冠澳
    q = (f"我是{name}，电话{phone}，想约下周一上午9点在{dealer_ok}试驾{CONFIGS[0]['title']}")
    _execute("05", base, q, t_correct, t_correct, name, phone, dealer_ok, dealer_wrong, CONFIGS[0],
        "用户信息齐全。下周一即2026年11月16日，按用户指定门店预约。",
        f"我选的是{dealer_ok}，不是{dealer_wrong}，你约错店了，请重新预约。",
        "用户指出门店写错了，应预约上汽奥迪上海冠松用户中心而非冠澳用户中心。按正确门店重新预约。",
        (f"抱歉把门店弄错了。已为您重新预约：{date_cn(t_correct)}上午9:00，{dealer_ok}，"
         f"试驾{CONFIGS[0]['title']}。请携带驾驶证原件，提前15分钟到店。"),
        "纠错多轮 execute 结局：turn1 门店写错（冠松→冠澳，均为真实门店）→用户指正→turn2 修正成功",
        "①turn1 错误=选错真实门店（非编造）②turn2 修正后门店与题面一致 ③最终 execute 成功")

    # X6: 时间越界被工具拦截（上午9点写成21:00 → INVALID_PARAM → 工具纠错后修正，无用户指正）
    base = BASE_DATE_POOL[2]
    name, phone = CUSTOMER_NAMES[5], _phone(66)
    t_correct = next_monday(base).replace(hour=9)
    t_wrong = next_monday(base).replace(hour=21)
    q = f"我是{name}，电话{phone}，下周一上午9点在{d2}试驾{CONFIGS[1]['title']}"
    err_args = {"car_series": "E5 Sportback", "dealer": d2, "appointment_time": iso(t_wrong),
                "customer_name": name, "customer_phone": phone}
    fix_args = {"car_series": "E5 Sportback", "dealer": d2, "appointment_time": iso(t_correct),
                "customer_name": name, "customer_phone": phone}
    err_obs = run_tool("book_test_drive", err_args)
    fix_obs = run_tool("book_test_drive", fix_args)
    r1 = "用户信息齐全。下周一即2026年11月16日，按21:00（晚9点）预约。"
    r2 = "工具提示预约时间需在工作时间内（9:00-18:00）。用户说的是上午9点，应修正为09:00重新预约。"
    a = (f"已为您预约成功：{date_cn(t_correct)}上午9:00，{d2}，试驾{CONFIGS[1]['title']}。"
         f"请携带驾驶证原件，提前15分钟到店。")

    def build_x6(q_final: str, a_final: str) -> List[Dict[str, Any]]:
        return [
            user_msg(q_final),
            thought_call("call_1", r1, loss_mask=False),
            thought_result("call_1", r1),
            tool_call("call_2", "book_test_drive", err_args, loss_mask=False),
            tool_result("call_2", err_obs),
            thought_call("call_3", r2),
            thought_result("call_3", r2),
            tool_call("call_4", "book_test_drive", fix_args),
            tool_result("call_4", fix_obs),
            finish_call("call_5", a_final),
        ]

    out.append({
        "sid": "v02_dry_002_x06", "bucket": "bucket2_correction", "scenario": "试驾预约",
        "targets": ["book_test_drive"], "is_clarify": False, "is_degrade": False,
        "question": q, "answer": a, "reasonings": [r1, r2], "build": build_x6,
        "design_intent": "工具纠错 execute 结局：turn1 时间越界（21:00）被工具 INVALID_PARAM 拦截→turn2 自查修正为09:00成功预约（无需用户指正）",
        "review_focus": "①turn1 调用段 loss_mask=false ②工具报错为真实实跑返回 ③turn2 修正后预约成功",
    })

    return out


def make_all_dryrun_templates(rng: random.Random) -> List[Dict[str, Any]]:
    return make_bucket1_test_drive(rng) + make_bucket2_test_drive(rng)
