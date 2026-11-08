# -*- coding: utf-8 -*-
"""v02 全量模板 · 桶1（250 条）+ 桶2（250 条）

桶1：试驾 90（dry-run 已并入 20，本文件新增 70）、金融 80、跨场景 80
桶2：试驾 200（dry-run 已并入 10，本文件新增 190）、金融 30、竞品 20

骨架结构约定与 gen_templates_dryrun 一致：question/answer 润色，reasonings 程序模板，
build(q_final, a_final) 闭包组装消息（工具实跑）。
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List

from gen_core import (
    BASE_DATE_POOL, CONFIGS, DEALERS, CUSTOMER_NAMES,
    FABRICATED_NAME, FABRICATED_PHONE, FABRICATED_DEALER,
    date_cn, iso, next_monday, this_saturday, this_sunday,
    run_tool, run_tool_parsed,
    user_msg, thought_call, thought_result, tool_call, tool_result, finish_call,
)
from gen_templates_dryrun import _b1_common, _b2_common, _dealer_list, _phone, _pick

# 相对时间表达池：(表达文本, 换算函数, 小时, 分钟)
def _rel_times(base: datetime):
    return [
        ("明天上午10点", base + timedelta(days=1, hours=10 - base.hour)),
        ("明天下午2点", base + timedelta(days=1, hours=14 - base.hour)),
        ("后天上午10点半", base + timedelta(days=2, hours=10 - base.hour, minutes=30)),
        ("下周一上午9点", next_monday(base).replace(hour=9, minute=0)),
        ("下周二下午3点", (next_monday(base) + timedelta(days=1)).replace(hour=15, minute=0)),
        ("本周六上午10点", this_saturday(base).replace(hour=10, minute=0)),
        ("本周日下午2点", this_sunday(base).replace(hour=14, minute=0)),
    ]


# ============================================================
# 桶1 · 试驾（新增 70 条；dry-run 20 条已并入）
# 子类型：A缺姓名+电话 12 / B缺姓名 8 / C缺电话 8 / D缺时间+姓名+电话 12
#         E全缺 8 / F门店未指定 12 / G门店未指定+时刻不明确 3 / H门店未指定+缺电话 7
# ============================================================

def make_b1_test_drive_extra(rng: random.Random) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    bases = BASE_DATE_POOL

    # A: 缺姓名+电话（时间+门店+车型齐）12
    for j in range(12):
        base = bases[j % 5]
        rel = _rel_times(base)[j % 7]
        dealer = DEALERS[(j * 3 + 1) % 7]
        cfg = CONFIGS[j % 4]
        tpls = [
            "麻烦把{time}{dealer}的{car}试驾约上",
            "我想约{time}试驾{car}，去{dealer}",
            "{car}能约在{time}试驾吗？我想去{dealer}",
            "麻烦帮我预约{dealer}的{car}试驾，时间{time}",
        ]
        q = tpls[j % 4].format(time=rel[0], dealer=dealer, car=cfg["title"])
        r = "用户已提供车型、门店和时间，但缺少预约必需的姓名和联系电话。不能编造客户信息，先向用户询问这两个缺失参数。"
        a = (f"好的，可以为您预约{cfg['title']}的试驾。目前还差两项信息，麻烦您补充一下：\n"
             f"1. 您的姓名\n2. 联系电话\n\n补充后我会立即为您安排{dealer}的试驾。")
        out.append(_b1_common(j, f"v02_full_b1td_a{j:02d}", q, r, a,
            "试驾缺参→反问正例（缺姓名+电话）：不编造、不硬调",
            "①补全姓名+电话两个缺失参数 ②未调用 book_test_drive"))

    # B: 缺姓名 8
    for j in range(8):
        base = bases[(j + 2) % 5]
        rel = _rel_times(base)[(j + 1) % 7]
        dealer = DEALERS[(j * 2 + 2) % 7]
        cfg = CONFIGS[(j + 1) % 4]
        phone = _phone(100 + j)
        q = f"你好，电话{phone}，想约{rel[0]}在{dealer}试驾{cfg['title']}"
        r = "用户提供了电话、时间和门店，但缺少预约必需的姓名。不能编造客户姓名，先向用户询问。"
        a = (f"可以的，帮您预约{cfg['title']}试驾还差您的姓名，请提供一下姓名，"
             f"我马上为您安排{dealer}的试驾。")
        out.append(_b1_common(j, f"v02_full_b1td_b{j:02d}", q, r, a,
            "试驾缺参→反问正例（只缺姓名）", "①只问姓名 ②未调用工具"))

    # C: 缺电话 8
    for j in range(8):
        base = bases[(j + 4) % 5]
        rel = _rel_times(base)[(j + 2) % 7]
        dealer = DEALERS[(j * 5 + 3) % 7]
        cfg = CONFIGS[(j + 2) % 4]
        name = _pick(CUSTOMER_NAMES, 110 + j)
        q = f"我是{name}，想约{rel[0]}在{dealer}试驾{cfg['title']}"
        r = "用户提供了姓名、时间和门店，但缺少预约必需的联系电话。不能编造电话号码，先向用户询问。"
        a = (f"您好{name}，为您预约{cfg['title']}试驾还差联系电话，请提供您的手机号，"
             f"我马上为您安排{dealer}的试驾。")
        out.append(_b1_common(j, f"v02_full_b1td_c{j:02d}", q, r, a,
            "试驾缺参→反问正例（只缺电话）", "①只问电话 ②未调用工具"))

    # D: 缺时间+姓名+电话 12
    for j in range(12):
        dealer = DEALERS[(j * 3 + 4) % 7]
        cfg = CONFIGS[(j + 1) % 4]
        tpls = [
            "帮我预约在{dealer}的{car}试驾",
            "想试驾{car}，麻烦安排在{dealer}",
            "能帮我约{dealer}的{car}试驾吗？",
            "{car}的试驾我想在{dealer}进行，帮忙安排",
        ]
        q = tpls[j % 4].format(dealer=dealer, car=cfg["title"])
        r = "用户提供了车型和门店，但时间、姓名、联系电话均缺失，需要一次性向用户问清这三项信息。"
        a = (f"没问题，帮您预约{cfg['title']}试驾需要确认几项信息：\n"
             f"1. 您希望什么时间试驾？\n2. 您的姓名\n3. 联系电话\n\n"
             f"告诉我后马上为您安排在{dealer}的试驾。")
        out.append(_b1_common(j, f"v02_full_b1td_d{j:02d}", q, r, a,
            "试驾缺参→反问正例（缺时间+姓名+电话）", "①三项缺失信息全部问到 ②未调用工具"))

    # E: 全缺 8
    for j in range(8):
        cfg = CONFIGS[(j * 3 + 1) % 4]
        tpls = [
            "我想约个{car}的试驾，你帮我安排下",
            "帮我安排一下试驾，就{car}这款",
            "{car}的试驾能帮我约一下吗？",
            "我想试驾{car}，怎么约？",
            "请问{car}可以试驾吗？帮我约个时间",
            "想试试{car}，麻烦帮我预约试驾",
            "你好，我想体验一下{car}的试驾",
            "{car}试驾怎么预约？帮我处理一下",
        ]
        q = tpls[j % 8].format(car=cfg["title"])
        r = "用户想预约试驾但未提供任何预约所需信息：时间、门店、姓名、电话均缺失，需要一次性向用户问清。"
        a = (f"没问题，帮您预约{cfg['title']}试驾需要确认几项信息：\n"
             f"1. 您希望什么时间试驾？\n2. 倾向哪家门店？（可选：\n{_dealer_list()}）\n"
             f"3. 您的姓名和联系电话？\n\n告诉我后马上为您安排。")
        out.append(_b1_common(j, f"v02_full_b1td_e{j:02d}", q, r, a,
            "试驾全缺参→反问正例", "①四项缺失信息全部问到 ②门店列表=7家白名单全量 ③未调用工具"))

    # F: 门店未指定 12
    for j in range(12):
        base = bases[j % 5]
        rel = _rel_times(base)[(j + 3) % 7]
        cfg = CONFIGS[(j + 1) % 4]
        name = _pick(CUSTOMER_NAMES, 120 + j)
        phone = _phone(120 + j)
        tpls = [
            "我叫{name}，电话{phone}，想约{time}试驾{car}，附近有什么门店可选？",
            "我是{name}，电话{phone}，约{time}试驾{car}，门店你推荐一下",
            "我叫{name}，电话{phone}，想{time}试驾{car}，都有哪些门店？",
            "{name}，电话{phone}，约{time}试驾{car}，哪家店方便？",
        ]
        q = tpls[j % 4].format(name=name, phone=phone, time=rel[0], car=cfg["title"])
        r = "用户已提供姓名、电话和试驾时间，但未指定门店。系统内可预约的门店共有七家，如实列出让用户选择，不能替用户编造或随意指定。"
        a = (f"{name}您好，目前可预约{cfg['title']}试驾的门店共有7家：\n{_dealer_list()}\n\n"
             f"请问您想选择哪家？确认后我会为您预约{rel[0]}的试驾。")
        out.append(_b1_common(j, f"v02_full_b1td_f{j:02d}", q, r, a,
            "门店未指定→报 7 家白名单全量正例", "①门店列表=7家全量 ②姓名/电话/时间如实保留 ③未调用工具"))

    # G: 门店未指定+时刻不明确 3
    for j in range(3):
        base = bases[(j + 1) % 5]
        cfg = CONFIGS[(j + 2) % 4]
        name = _pick(CUSTOMER_NAMES, 130 + j)
        phone = _phone(130 + j)
        t_exprs = ["本周日上午", "明天上午", "周六上午"]
        q = f"预约{cfg['title']}试驾，{t_exprs[j]}，我叫{name}，电话{phone}"
        r = "用户给了姓名电话和时间范围（上午），但没给具体时刻和门店。先列出可选门店并请用户确认具体时间点，再完成预约。"
        a = (f"{name}您好，为您预约{t_exprs[j]}的试驾，还需要确认两点：\n"
             f"1. 门店选择：\n{_dealer_list()}\n2. 具体时间：上午几点方便？（营业时间 9:00-18:00）\n\n"
             f"确认后立即为您锁定。")
        out.append(_b1_common(j, f"v02_full_b1td_g{j:02d}", q, r, a,
            "门店未指定+时间未到时刻→反问正例", "①门店列表=7家全量 ②时间范围未硬换算（先确认）"))

    # H: 门店未指定+缺电话 7
    for j in range(7):
        base = bases[j % 5]
        rel = _rel_times(base)[(j + 5) % 7]
        cfg = CONFIGS[j % 4]
        name = _pick(CUSTOMER_NAMES, 140 + j)
        tpls = [
            "我是{name}，想约{time}试驾{car}，去哪家店方便？",
            "我叫{name}，约{time}试驾{car}，哪家店都行，你定",
            "我{name}，{time}想试驾{car}，门店有哪些可选？",
        ]
        q = tpls[j % 3].format(name=name, time=rel[0], car=cfg["title"])
        r = "用户提供姓名和时间，但未指定门店，也缺少联系电话。先列出可选门店，并请用户补充联系电话。"
        a = (f"{name}您好，可预约的门店共有7家：\n{_dealer_list()}\n\n"
             f"另外还需要您的联系电话，请一并告知，我马上为您预约。")
        out.append(_b1_common(j, f"v02_full_b1td_h{j:02d}", q, r, a,
            "门店未指定+缺电话→反问正例", "①门店列表=7家全量 ②电话被问到 ③未调用工具"))

    return out  # 70 条


# ============================================================
# 桶1 · 金融（80 条：缺参→默认值执行并标注）
# 子类型：缺首付+年限 25 / 缺首付 20 / 缺年限 15 / 车系级→清单反问 20
# ============================================================

def _fin_sample(sid: str, q: str, r: str, args: Dict[str, Any], cfg_label: str,
                clarify: bool, intent: str, focus: str) -> Dict[str, Any]:
    """金融单轮样本：实跑 calculate_finance，Finish 转述实跑数字。clarify=True 时为
    车系级 AMBIGUOUS 反问（无工具调用或调用后反问）。"""
    obs = run_tool_parsed("calculate_finance", args)

    if clarify:
        # 车系级 → partial 多配置清单 → 反问
        listing = "；".join(f"{x['title']} {x['price']} 元" for x in obs.get("data", {}).get("results", []))
        a = (f"您说的{cfg_label}包含多个配置，请明确具体配置后我为您计算：{listing}。\n"
             f"请问您考虑哪个配置？")
    else:
        d = obs.get("data", {})
        ratio = args.get("down_payment_ratio", 30)
        years = args.get("loan_years", 3)
        rate = args.get("interest_rate", 4.5)
        a = (f"按{cfg_label}为您测算：指导价{int(d.get('guide_price', 0))}元，"
             f"首付{ratio}%（约{int(d.get('down_payment', 0))}元），贷款{int(d.get('loan_amount', 0))}元"
             f"分{years}年期、年利率{rate}%，每月月供约{round(float(d.get('monthly_payment', 0)))}元。\n\n"
             f"以上为首付{ratio}%、{years}年期的测算，如需调整首付比例或贷款年限，随时告诉我。")

    def build(q_final: str, a_final: str) -> List[Dict[str, Any]]:
        msgs = [user_msg(q_final), thought_call("call_1", r), thought_result("call_1", r)]
        cid = 2
        if not clarify:
            msgs.append(tool_call(f"call_{cid}", "calculate_finance", args))
            msgs.append(tool_result(f"call_{cid}", json.dumps(obs, ensure_ascii=False)))
            cid += 1
        else:
            # 调用工具拿到清单后再反问（AMBIGUOUS 由工具返回）
            msgs.append(tool_call(f"call_{cid}", "calculate_finance", args))
            msgs.append(tool_result(f"call_{cid}", json.dumps(obs, ensure_ascii=False)))
            cid += 1
        msgs.append(finish_call(f"call_{cid}", a_final))
        return msgs

    return {
        "sid": sid, "bucket": "bucket1_clarify", "scenario": "金融方案测算",
        "targets": [] if clarify else ["calculate_finance"],
        "is_clarify": clarify, "is_degrade": False,
        "question": q, "answer": a, "reasonings": [r], "build": build,
        "design_intent": intent, "review_focus": focus,
    }


def make_b1_finance(rng: random.Random) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # 缺首付+年限（只给车型）25：默认 30%/3年/4.5%
    for j in range(25):
        cfg = CONFIGS[(j * 3 + 1) % 4]
        tpls = [
            "{car}做贷款的话，月供大概多少？",
            "帮我算下{car}的月供",
            "{car}分期付款每个月要还多少？",
            "想了解{car}的贷款月供情况",
            "贷款买{car}月供多少？",
            "{car}月供怎么算？帮我算一下",
            "如果分期买{car}，每月还款多少？",
        ]
        q = tpls[j % 7].format(car=cfg["title"])
        r = "用户未指定首付比例、贷款年限和利率，按默认方案（首付30%、3年期、年利率4.5%）计算，车价由工具自动读取指导价，回复中标注默认方案并说明可调整。"
        out.append(_fin_sample(
            f"v02_full_b1fn_a{j:02d}", q, r, {"car_model": cfg["title"]}, cfg["title"],
            False, "金融缺参→默认值执行正例（决策点 A）：首付30%/3年/4.5% 并标注可调整",
            "①未反问、直接计算 ②未传 car_price 废弃参数 ③回复标注默认方案且提示可调整"))

    # 缺首付（给年限）20
    for j in range(20):
        cfg = CONFIGS[(j * 3 + 2) % 4]
        years = [2, 3, 5][j % 3]
        tpls = [
            "{car}分{y}年贷款，每月要还多少？",
            "{car}贷款{y}年的话月供多少？",
            "帮我算{car}分{y}年还的月供",
            "{car}贷{y}年，每月还款多少？",
            "分期{y}年买{car}，月供大概多少？",
        ]
        q = tpls[j % 5].format(car=cfg["title"], y=years)
        r = f"用户指定{years}年贷款，未给首付比例和利率，按默认首付30%、年利率4.5%计算；车价由工具自动读取指导价。"
        out.append(_fin_sample(
            f"v02_full_b1fn_b{j:02d}", q, r, {"car_model": cfg["title"], "loan_years": years}, cfg["title"],
            False, "金融缺参→默认值执行正例：只给年限，首付/利率按默认",
            "①car_model 传全称自动读价 ②首付默认30% ③回复标注默认项"))

    # 缺年限（给首付）15
    for j in range(15):
        cfg = CONFIGS[(j * 3 + 3) % 4]
        ratio = [40, 50, 30][j % 3]
        tpls = [
            "{car}首付{r}成的话，月供多少？",
            "首付{r}成买{car}，每个月还多少？",
            "{car}首付比例{r}%，帮我算月供",
            "想首付{r}成贷款买{car}，月供多少？",
            "{car}首付{r}成，每月还款大概多少？",
        ]
        q = tpls[j % 5].format(car=cfg["title"], r=ratio)
        r = f"用户给出首付比例{ratio}%，贷款年限与利率缺省按3年、4.5%处理。calculate_finance 会自动读取车型指导价，直接调用计算。"
        out.append(_fin_sample(
            f"v02_full_b1fn_c{j:02d}", q, r, {"car_model": cfg["title"], "down_payment_ratio": ratio}, cfg["title"],
            False, "金融缺参→默认值执行正例：只给首付，年限/利率按默认",
            "①未前置查价 ②年限默认3年 ③首付如实传参"))

    # 车系级名称→AMBIGUOUS 清单反问 20
    # q 无实体变量，用场景修饰池×措辞池保证 20 条唯一
    scenes = ["我打算买家用车", "最近在看车", "准备换车了", "想给家里添辆车", "看中这款车了"]
    for j in range(20):
        tpls = [
            "{scene}，E5 Sportback贷款月供多少？",
            "{scene}，帮我看下E5 Sportback的分期月供",
            "{scene}，E5 Sportback月供大概什么水平？",
            "{scene}，想贷款买E5 Sportback，月供多少？",
        ]
        q = tpls[j % 4].format(scene=scenes[(j // 4) % 5])
        r = "用户只给了车系级名称'E5 Sportback'，未明确具体配置。先调用工具获取配置清单，再请用户确认具体配置后计算。"
        out.append(_fin_sample(
            f"v02_full_b1fn_d{j:02d}", q, r, {"car_model": "E5 Sportback"}, "E5 Sportback",
            True, "车系级名称→工具 AMBIGUOUS 清单→反问配置正例",
            "①调用后如实呈现配置清单 ②反问具体配置而非硬算 ③价格数字与 mock 一致"))

    return out  # 80 条


# ============================================================
# 桶1 · 跨场景（80 条：其他缺参→默认值/如实说明）
# 优惠缺region 20 / 成交价缺days 20 / 竞品缺对比对象 20 / 金融+优惠组合缺参 20
# ============================================================

def make_b1_cross(rng: random.Random) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def _simple(sid: str, scenario: str, q: str, r: str, targets: List[str],
                tool_seq: List[tuple], a: str, intent: str, focus: str) -> Dict[str, Any]:
        def build(q_final: str, a_final: str) -> List[Dict[str, Any]]:
            msgs = [user_msg(q_final), thought_call("call_1", r), thought_result("call_1", r)]
            cid = 2
            for name, args in tool_seq:
                obs = run_tool_parsed(name, args)
                msgs.append(tool_call(f"call_{cid}", name, args))
                msgs.append(tool_result(f"call_{cid}", json.dumps(obs, ensure_ascii=False)))
                cid += 1
            msgs.append(finish_call(f"call_{cid}", a_final))
            return msgs
        return {
            "sid": sid, "bucket": "bucket1_clarify", "scenario": scenario, "targets": targets,
            "is_clarify": False, "is_degrade": False,
            "question": q, "answer": a, "reasonings": [r], "build": build,
            "design_intent": intent, "review_focus": focus,
        }

    # 优惠缺 region（默认全国）20
    for j in range(20):
        car = ["E5 Sportback", "帕萨特2024款", "小米SU7 2025款"][j % 3]
        tpls = [
            "想入手{car}，能介绍下现在有什么优惠吗？",
            "{car}目前有哪些优惠活动？",
            "想问下{car}的优惠政策",
            "{car}现在有补贴吗？",
            "{car}近期优惠力度怎么样？",
            "帮我查下{car}的优惠信息",
            "{car}有什么购车优惠？",
        ]
        q = tpls[j % 7].format(car=car)
        r = "用户未指定地区，query_discount_policy 的 region 缺省为全国，直接查询全国通用优惠政策。"
        obs = run_tool_parsed("query_discount_policy", {"car_model": car})
        ds = obs["data"]["discounts"]
        a = (f"目前{car}的全国通用优惠共{len(ds)}项：\n" +
             "\n".join(f"{i + 1}. {d['name']}：{d['amount']}元（{d['condition']}，有效期至{d['valid_until']}）"
                       for i, d in enumerate(ds)) +
             f"\n\n（未指定地区，按全国口径查询；如需了解地区专属政策，可告诉我您的城市。）")
        out.append(_simple(f"v02_full_b1cs_a{j:02d}", "优惠政策查询", q, r,
            ["query_discount_policy"], [("query_discount_policy", {"car_model": car})], a,
            "优惠缺 region→默认全国执行正例", "①未追问地区、默认全国 ②优惠条目与 mock 一致"))

    # 成交价缺 days（默认 90 天）20
    for j in range(20):
        car = ["E5 Sportback 2025款 旗舰quattro型", "E5 Sportback 2025款 先锋型",
               "帕萨特2024款 330TSI 精英版"][j % 3]
        tpls = [
            "{car}最近成交价怎么样？",
            "{car}目前成交行情怎么样？",
            "帮我看下{car}的历史成交价",
            "{car}近期价格走势如何？",
            "{car}现在买大概多少钱？",
            "查下{car}的成交行情",
            "{car}最近卖得怎么样？价格如何？",
        ]
        q = tpls[j % 7].format(car=car)
        r = "用户未指定时间范围，query_transaction_price 的 days 缺省为 90 天，直接查询近 90 天成交价。"
        obs = run_tool_parsed("query_transaction_price", {"car_model": car})
        d = obs["data"]
        a = (f"{car}近90天：平均成交价{d['avg_price']}元（最低{d['min_price']}元/最高{d['max_price']}元），"
             f"趋势{d['trend']}，样本{d['sample_count']}条。建议：{d['recommendation']}。\n\n"
             f"（未指定时间范围，按默认近90天查询；也可为您查近30天。）")
        out.append(_simple(f"v02_full_b1cs_b{j:02d}", "历史成交价", q, r,
            ["query_transaction_price"], [("query_transaction_price", {"car_model": car})], a,
            "成交价缺 days→默认 90 天执行正例", "①未追问时间范围、默认90天 ②数字与 mock 一致"))

    # 竞品缺对比对象（只有本车）20：反问对比对象
    # q 无实体变量，用场景修饰池×措辞池保证 20 条唯一
    scenes = ["最近在看车", "准备换车", "看了几个品牌", "想多了解一下", "在选车阶段"]
    for j in range(20):
        tpls = [
            "{scene}，帮我对比一下E5 Sportback 2025款 旗舰quattro型",
            "{scene}，E5 Sportback 2025款 旗舰quattro型和别的车比怎么样？",
            "{scene}，给我对比下E5 Sportback 2025款 旗舰quattro型这车",
            "{scene}，E5 Sportback 2025款 旗舰quattro型有竞品对比吗？",
        ]
        q = tpls[j % 4].format(scene=scenes[(j // 4) % 5])
        r = "用户只给了本车车型，未说明要与哪款竞品对比。不能替用户猜测竞品，先询问对比对象。"
        a = ("好的，可以为您对比。目前支持以下竞品对比：\n"
             "1. 小米SU7 2025款 四驱Max版\n2. 帕萨特2024款 330TSI 精英版\n\n"
             "请问您想和哪款车对比？也可以告诉我其他想了解的车型。")
        def build(q_final: str, a_final: str, r=r) -> List[Dict[str, Any]]:
            return [user_msg(q_final), thought_call("call_1", r), thought_result("call_1", r),
                    finish_call("call_2", a_final)]
        out.append({
            "sid": f"v02_full_b1cs_c{j:02d}", "bucket": "bucket1_clarify", "scenario": "竞品对比",
            "targets": [], "is_clarify": True, "is_degrade": False,
            "question": q, "answer": a, "reasonings": [r], "build": build,
            "design_intent": "竞品缺对比对象→反问正例：不替用户猜竞品",
            "review_focus": "①反问对比对象 ②竞品列表 ∈ 接地表（SU7/帕萨特）③未调用工具",
        })

    # 金融+优惠组合缺参 20：查优惠（全国默认）+算月供（默认方案）
    for j in range(20):
        cfg = CONFIGS[(j * 3 + 1) % 4]
        tpls = [
            "{car}现在有优惠吗？贷款月供多少？",
            "想了解{car}的优惠和月供情况",
            "{car}优惠政策和分期月供一起看下",
            "{car}有优惠吗？顺便算下月供",
            "{car}的优惠和月供都帮我查下",
        ]
        q = tpls[j % 5].format(car=cfg["title"])
        r = "用户想了解优惠和月供。先查全国通用优惠政策，再按默认方案（首付30%、3年期）计算月供，车价由工具自动读取。"
        obs_d = run_tool_parsed("query_discount_policy", {"car_model": "E5 Sportback"})
        ds = obs_d["data"]["discounts"]
        obs_f = run_tool_parsed("calculate_finance", {"car_model": cfg["title"]})
        d = obs_f["data"]
        a = (f"为您整理{cfg['title']}的优惠与月供：\n\n【全国通用优惠】\n" +
             "\n".join(f"{i + 1}. {x['name']}：{x['amount']}元（{x['condition']}）" for i, x in enumerate(ds)) +
             f"\n\n【贷款月供】指导价{int(d['guide_price'])}元，首付30%（约{int(d['down_payment'])}元），"
             f"贷款3年、年利率4.5%，月供约{round(float(d['monthly_payment']))}元（默认方案，可调整）。")
        out.append(_simple(f"v02_full_b1cs_d{j:02d}", "金融方案测算", q, r,
            ["query_discount_policy", "calculate_finance"],
            [("query_discount_policy", {"car_model": "E5 Sportback"}),
             ("calculate_finance", {"car_model": cfg["title"]})], a,
            "优惠+月供组合：region 缺省全国、金融缺参按默认方案",
            "①两跳工具链完整 ②优惠与月供数字均为实跑值 ③默认方案标注"))

    return out  # 80 条


# ============================================================
# 桶2 · 试驾（新增 190 条；dry-run 10 条已并入）
# clarify 96 + execute 94
# ============================================================

def make_b2_test_drive_extra(rng: random.Random) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # ---- clarify 结局 96：turn1 编造 → 报错/成功 → 用户指正 → turn2 反问 ----
    # C1: 编造 dealer+姓名+电话（缺姓名电话门店）40
    for j in range(40):
        base = BASE_DATE_POOL[j % 5]
        rel = _rel_times(base)[(j * 5) % 7]
        cfg = CONFIGS[(j * 3 + 1) % 4]
        t_correct = rel[1].replace(second=0)
        tpls = [
            "预约{time}试驾{car}",
            "帮我预约{time}的{car}试驾",
            "我想约{time}试驾{car}",
            "{time}能试驾{car}吗？帮我约",
            "预约一下{time}的{car}试驾",
            "{car}试驾约在{time}可以吗？",
            "你好，{time}想试驾{car}，帮我预约",
            "{time}的{car}试驾帮我安排",
        ]
        q = tpls[j % 8].format(time=rel[0], car=cfg["title"])
        r1 = "用户只给了时间，姓名、电话、门店都缺。为尽快完成预约，先按常见信息补全参数尝试预约。"
        err_args = {"car_series": "E5 Sportback", "dealer": FABRICATED_DEALER,
                    "appointment_time": iso(t_correct), "customer_name": FABRICATED_NAME,
                    "customer_phone": FABRICATED_PHONE}
        cus = [
            "我没有告诉过你姓名电话，也没说去哪家店，你怎么自己编了？请重新处理。",
            "你编造了我的姓名、电话还有门店，请重新处理。",
            "我根本没给你这些信息，你从哪编出来的？重新来。",
        ]
        r2 = "用户指出我擅自编造了姓名、电话和门店。正确做法是列出可预约的真实门店，并请用户提供姓名和联系电话，不再擅自补全参数。"
        a = (f"非常抱歉，我此前不应该擅自填写您的个人信息和门店。预约{rel[0]}的试驾，还需要您确认：\n"
             f"1. 门店（共7家可选）：\n{_dealer_list()}\n2. 您的姓名\n3. 联系电话\n\n"
             f"补充后我立即为您预约，不会再自行编造信息。")
        out.append(_b2_common(f"v02_full_b2td_c{j:02d}", q, r1, err_args, cus[j % 3], r2, a, [],
            "纠错多轮 clarify：编造 dealer 硬调→NOT_FOUND+指正→报 7 家+反问",
            "①turn1 编造段 loss_mask=false ②turn2 修正完整", call_ok=False))

    # C2: 编造姓名+电话（门店时间给）30
    for j in range(30):
        base = BASE_DATE_POOL[(j + 1) % 5]
        rel = _rel_times(base)[(j + 1) % 7]
        cfg = CONFIGS[(j + 1) % 4]
        dealer = DEALERS[(j * 3) % 7]
        t_correct = rel[1].replace(second=0)
        tpls = [
            "我想约{time}在{dealer}试驾{car}",
            "帮我预约{time}在{dealer}的{car}试驾",
            "{time}在{dealer}试驾{car}，帮我约一下",
        ]
        q = tpls[j % 3].format(time=rel[0], dealer=dealer, car=cfg["title"])
        r1 = "用户没给姓名和电话，为了尽快约上，先按常见信息补全后直接预约。"
        err_args = {"car_series": "E5 Sportback", "dealer": dealer,
                    "appointment_time": iso(t_correct), "customer_name": FABRICATED_NAME,
                    "customer_phone": FABRICATED_PHONE}
        cus = [
            "我没说过姓名和电话，你怎么就给我约上了？请重新处理。",
            "我的姓名电话你从哪知道的？我没有提供过，重新处理。",
            "你编了我的姓名和电话，请重新处理。",
        ]
        r2 = "用户指出我编造了姓名和电话。预约必须使用用户提供的真实信息，应撤回编造操作，向用户询问缺失的姓名和联系电话。"
        a = (f"抱歉，我之前不应擅自填写您的姓名和电话。为您预约{rel[0]}在{dealer}的{cfg['title']}试驾，"
             f"还需要您提供：\n1. 您的姓名\n2. 联系电话\n\n补充后我立即为您预约。")
        out.append(_b2_common(f"v02_full_b2td_d{j:02d}", q, r1, err_args, cus[j % 3], r2, a, [],
            "纠错多轮 clarify：编造姓名电话预约成功→指正→反问", "①turn1 编造段 loss_mask=false ②turn2 反问姓名电话",
            call_ok=False))

    # C3: 编造姓名（电话时间门店给）16
    for j in range(16):
        base = BASE_DATE_POOL[(j + 2) % 5]
        rel = _rel_times(base)[(j + 2) % 7]
        cfg = CONFIGS[(j + 2) % 4]
        dealer = DEALERS[(j * 2 + 1) % 7]
        phone = _phone(200 + j)
        t_correct = rel[1].replace(second=0)
        q = f"电话{phone}，想约{rel[0]}在{dealer}试驾{cfg['title']}"
        r1 = "用户给了电话、时间和门店，缺姓名。先补一个常见姓名尽快预约。"
        err_args = {"car_series": "E5 Sportback", "dealer": dealer,
                    "appointment_time": iso(t_correct), "customer_name": FABRICATED_NAME,
                    "customer_phone": phone}
        cus = ["我还没告诉你姓名，你编了个名字就预约了，重新处理。",
               "我没说过我叫什么，你替我编了名字，重新处理。"]
        r2 = "用户指出我编造了姓名。应请用户提供真实姓名后再预约，不再擅自补全。"
        a = (f"抱歉，我不应擅自填写姓名。为完成{rel[0]}在{dealer}的{cfg['title']}试驾预约，"
             f"请提供您的姓名，我马上为您安排。")
        out.append(_b2_common(f"v02_full_b2td_e{j:02d}", q, r1, err_args, cus[j % 2], r2, a, [],
            "纠错多轮 clarify：只编造姓名→指正→反问姓名", "①turn1 编造段 loss_mask=false ②turn2 只问姓名",
            call_ok=False))

    # C4: 编造姓名电话+时间换算错 10
    for j in range(10):
        base = BASE_DATE_POOL[j % 5]
        rel = _rel_times(base)[(j + 3) % 7]
        cfg = CONFIGS[(j + 3) % 4]
        dealer = DEALERS[(j * 4 + 2) % 7]
        t_correct = rel[1].replace(second=0)
        t_wrong = t_correct + timedelta(days=1)
        q = f"帮我约{rel[0]}在{dealer}试驾{cfg['title']}"
        r1 = "用户缺姓名电话。先按常见信息补全，并把时间往后错算一天进行预约。"
        err_args = {"car_series": "E5 Sportback", "dealer": dealer,
                    "appointment_time": iso(t_wrong), "customer_name": FABRICATED_NAME,
                    "customer_phone": FABRICATED_PHONE}
        cus = (f"首先我没给你姓名电话，另外我说的是{rel[0]}（{date_cn(t_correct)}），"
               f"你约成{date_cn(t_wrong)}了，请重新处理。")
        r2 = (f"用户指出两点：一是我不应编造姓名电话，二是时间应指{date_cn(t_correct)}。"
              f"撤回编造操作，向用户询问缺失信息。")
        a = (f"抱歉，我此前擅自填写了您的个人信息，时间也理解错了。为您预约{rel[0]}"
             f"（{date_cn(t_correct)}）在{dealer}的{cfg['title']}试驾，还需要您提供：\n"
             f"1. 您的姓名\n2. 联系电话\n\n补充后我立即为您预约。")
        out.append(_b2_common(f"v02_full_b2td_f{j:02d}", q, r1, err_args, cus, r2, a, [],
            "纠错多轮 clarify：编造姓名电话+时间错→两点指正→反问", "①turn1 编造段 loss_mask=false ②turn2 日期按正确口径呈现",
            call_ok=False))

    # ---- execute 结局 94：信息齐全，turn1 单一参数错 → 指正 → 修正成功 ----
    def _ex(seq: str, base, rel, cfg, dealer_ok, dealer_wrong, t_correct, t_wrong,
            name, phone, r1, cu, r2, a, intent, focus):
        err_args = {"car_series": "E5 Sportback", "dealer": dealer_wrong,
                    "appointment_time": iso(t_wrong), "customer_name": name,
                    "customer_phone": phone}
        fix_args = {"car_series": "E5 Sportback", "dealer": dealer_ok,
                    "appointment_time": iso(t_correct), "customer_name": name,
                    "customer_phone": phone}
        out.append(_b2_common(f"v02_full_b2td_{seq}", rel, r1, err_args, cu, r2, a,
            [("book_test_drive", fix_args)], intent, focus, call_ok=True))

    # E1: 下周一→明天/本周一/上周一 等错算 30
    for j in range(30):
        base = BASE_DATE_POOL[j % 5]
        rel = _rel_times(base)[j % 7]
        cfg = CONFIGS[j % 4]
        dealer = DEALERS[(j * 5 + 3) % 7]
        name = _pick(CUSTOMER_NAMES, 300 + j)
        phone = _phone(300 + j)
        t_correct = rel[1].replace(second=0)
        t_wrong = t_correct + timedelta(days=[-1, 1, 2, 7, -7][j % 5])
        if t_wrong.hour < 9:
            t_wrong = t_wrong.replace(hour=10)
        q = f"我是{name}，电话{phone}，想约{rel[0]}在{dealer}试驾{cfg['title']}"
        _ex(f"x{j:02d}", base, q, cfg, dealer, dealer, t_correct, t_wrong, name, phone,
            f"用户信息齐全。'{rel[0]}'理解有误，按{date_cn(t_wrong)}预约。",
            f"不对，我说的是{rel[0]}，应该是{date_cn(t_correct)}，你约成{date_cn(t_wrong)}了，请重新预约。",
            f"用户指出时间理解错误，'{rel[0]}'应为{date_cn(t_correct)}。按正确日期重新预约。",
            (f"抱歉日期理解错了。已为您重新预约：{date_cn(t_correct)}{t_correct.strftime('%H:%M')}，{dealer}，"
             f"试驾{cfg['title']}。请携带驾驶证原件，提前15分钟到店。"),
            "纠错多轮 execute：时间换算错→指正→修正成功", "①turn1 时间错 ②turn2 修正后参数可溯源 ③execute 成功")

    # E2: 门店写错（相近店名）30
    for j in range(30):
        base = BASE_DATE_POOL[(j + 2) % 5]
        rel = _rel_times(base)[(j + 4) % 7]
        cfg = CONFIGS[(j + 1) % 4]
        # 相近店名对：冠澳↔冠松、郑隆奥翟↔大宁、西上海↔东昌、奥迪之城↔冠澳
        pairs = [(DEALERS[4], DEALERS[5]), (DEALERS[0], DEALERS[1]),
                 (DEALERS[2], DEALERS[3]), (DEALERS[6], DEALERS[4])]
        d_ok, d_wrong = pairs[j % 4]
        name = _pick(CUSTOMER_NAMES, 330 + j)
        phone = _phone(330 + j)
        t_correct = rel[1].replace(second=0)
        q = f"我是{name}，电话{phone}，想约{rel[0]}在{d_ok}试驾{cfg['title']}"
        _ex(f"y{j:02d}", base, q, cfg, d_ok, d_wrong, t_correct, t_correct, name, phone,
            f"用户信息齐全。'{rel[0]}'即{date_cn(t_correct)}，按用户指定门店预约。",
            f"我选的是{d_ok}，不是{d_wrong}，你约错店了，请重新预约。",
            f"用户指出门店写错了，应为{d_ok}。按正确门店重新预约。",
            (f"抱歉把门店弄错了。已为您重新预约：{date_cn(t_correct)}{t_correct.strftime('%H:%M')}，{d_ok}，"
             f"试驾{cfg['title']}。请携带驾驶证原件，提前15分钟到店。"),
            "纠错多轮 execute：门店写错（均为真实门店）→指正→修正成功", "①turn1 选错真实门店 ②turn2 修正与题面一致 ③execute 成功")

    # E3: 时间越界被工具拦截（无用户指正）19
    for j in range(19):
        base = BASE_DATE_POOL[(j + 4) % 5]
        rel = _rel_times(base)[(j + 6) % 7]
        cfg = CONFIGS[(j + 2) % 4]
        dealer = DEALERS[(j * 2 + 5) % 7]
        name = _pick(CUSTOMER_NAMES, 360 + j)
        phone = _phone(360 + j)
        t_correct = rel[1].replace(second=0)
        t_wrong = t_correct.replace(hour=[20, 21, 22, 8, 7][j % 5])
        q = f"我是{name}，电话{phone}，想约{rel[0]}在{dealer}试驾{cfg['title']}"
        err_args = {"car_series": "E5 Sportback", "dealer": dealer, "appointment_time": iso(t_wrong),
                    "customer_name": name, "customer_phone": phone}
        fix_args = {"car_series": "E5 Sportback", "dealer": dealer, "appointment_time": iso(t_correct),
                    "customer_name": name, "customer_phone": phone}
        err_obs = run_tool("book_test_drive", err_args)
        fix_obs = run_tool("book_test_drive", fix_args)
        r1 = (f"用户信息齐全。'{rel[0]}'即{date_cn(t_correct)}，按{t_wrong.strftime('%H:%M')}预约。")
        r2 = (f"工具提示预约时间需在工作时间内（9:00-18:00）。用户说的是{rel[0]}，"
              f"应修正为{t_correct.strftime('%H:%M')}重新预约。")
        a = (f"已为您预约成功：{date_cn(t_correct)}{t_correct.strftime('%H:%M')}，{dealer}，"
             f"试驾{cfg['title']}。请携带驾驶证原件，提前15分钟到店。")

        def build_x(q_final: str, a_final: str, r1=r1, r2=r2, err_args=err_args,
                    fix_args=fix_args, err_obs=err_obs, fix_obs=fix_obs) -> List[Dict[str, Any]]:
            return [
                user_msg(q_final),
                thought_call("call_1", r1, loss_mask=False), thought_result("call_1", r1),
                tool_call("call_2", "book_test_drive", err_args, loss_mask=False),
                tool_result("call_2", err_obs),
                thought_call("call_3", r2), thought_result("call_3", r2),
                tool_call("call_4", "book_test_drive", fix_args), tool_result("call_4", fix_obs),
                finish_call("call_5", a_final),
            ]
        out.append({
            "sid": f"v02_full_b2td_z{j:02d}", "bucket": "bucket2_correction", "scenario": "试驾预约",
            "targets": ["book_test_drive"], "is_clarify": False, "is_degrade": False,
            "question": q, "answer": a, "reasonings": [r1, r2], "build": build_x,
            "design_intent": "工具纠错 execute：时间越界被工具 INVALID_PARAM 拦截→自查修正成功（无用户指正）",
            "review_focus": "①turn1 调用段 loss_mask=false ②工具报错为真实实跑返回 ③turn2 修正后预约成功",
        })

    # E4: 下周一 vs 本周一混淆、时刻 AM/PM 错等（补充 execute）15
    for j in range(15):
        base = BASE_DATE_POOL[j % 5]
        rel = _rel_times(base)[j % 7]
        cfg = CONFIGS[(j + 3) % 4]
        dealer = DEALERS[(j * 3 + 1) % 7]
        name = _pick(CUSTOMER_NAMES, 380 + j)
        phone = _phone(380 + j)
        t_correct = rel[1].replace(second=0)
        # 错到下周同一天
        t_wrong = t_correct + timedelta(days=7)
        q = f"我叫{name}，电话{phone}，想约{rel[0]}在{dealer}试驾{cfg['title']}"
        _ex(f"w{j:02d}", base, q, cfg, dealer, dealer, t_correct, t_wrong, name, phone,
            f"用户信息齐全。'{rel[0]}'理解为下一周同一天，按{date_cn(t_wrong)}预约。",
            f"我要的是{rel[0]}（{date_cn(t_correct)}），不是{date_cn(t_wrong)}，请重新预约。",
            f"用户指出'{rel[0]}'应为{date_cn(t_correct)}。按正确日期重新预约。",
            (f"抱歉日期理解错了。已为您重新预约：{date_cn(t_correct)}{t_correct.strftime('%H:%M')}，{dealer}，"
             f"试驾{cfg['title']}。请携带驾驶证原件，提前15分钟到店。"),
            "纠错多轮 execute：时间错到下周→指正→修正成功", "①turn1 时间错 ②turn2 修正 ③execute 成功")

    return out  # 40+30+16+10+30+30+19+15 = 190


# ============================================================
# 桶2 · 金融（30 条：编造参数纠错）
# car_price 废弃参数 10 / 编造首付 10 / 编造利率 5 / 编造年限 5
# ============================================================

def make_b2_finance(rng: random.Random) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # F1: turn1 传废弃 car_price → INVALID_PARAM（工具纠错，无用户指正）10
    for j in range(10):
        cfg = CONFIGS[(j * 3 + 1) % 4]
        tpls = [
            "帮我算下{car}首付三成的月供",
            "{car}首付三成，月供帮我算一下",
            "首付三成买{car}，每月要还多少？",
        ]
        q = tpls[j % 3].format(car=cfg["title"])
        r1 = f"用户给出车型和首付比例。直接调用计算，并附带传入自估车价{int(cfg['price'] * 0.95)}。"
        err_args = {"car_model": cfg["title"], "down_payment_ratio": 30,
                    "car_price": int(cfg["price"] * 0.95)}
        fix_args = {"car_model": cfg["title"], "down_payment_ratio": 30}
        err_obs = run_tool("calculate_finance", err_args)
        fix_obs = run_tool_parsed("calculate_finance", fix_args)
        d = fix_obs["data"]
        r2 = "工具拒绝显式传入 car_price：车价由工具自动读取车型指导价。移除该参数重新计算。"
        a = (f"{cfg['title']}指导价{int(d['guide_price'])}元，首付三成（约{int(d['down_payment'])}元），"
             f"贷款{int(d['loan_amount'])}元分{d['loan_years']}年、年利率{d['interest_rate']}%，"
             f"每月月供约{round(float(d['monthly_payment']))}元。")

        def build(q_final: str, a_final: str, r1=r1, err_args=err_args, err_obs=err_obs,
                  r2=r2, fix_args=fix_args, fix_obs=fix_obs) -> List[Dict[str, Any]]:
            return [
                user_msg(q_final),
                thought_call("call_1", r1, loss_mask=False), thought_result("call_1", r1),
                tool_call("call_2", "calculate_finance", err_args, loss_mask=False),
                tool_result("call_2", err_obs),
                thought_call("call_3", r2), thought_result("call_3", r2),
                tool_call("call_4", "calculate_finance", fix_args), tool_result("call_4", json.dumps(fix_obs, ensure_ascii=False)),
                finish_call("call_5", a_final),
            ]
        out.append({
            "sid": f"v02_full_b2fn_a{j:02d}", "bucket": "bucket2_correction", "scenario": "金融方案测算",
            "targets": ["calculate_finance"], "is_clarify": False, "is_degrade": False,
            "question": q, "answer": a, "reasonings": [r1, r2], "build": build,
            "design_intent": "工具纠错：turn1 传废弃 car_price→INVALID_PARAM 拒绝→turn2 移除参数自动读价成功",
            "review_focus": "①turn1 调用段 loss_mask=false ②turn2 未再传 car_price ③月供数字=实跑值",
        })

    # ---- 编造可选参数 → 成功但错 → 用户指正 → 默认方案重算 ----
    def _b2_fin_corr(sid: str, q: str, r1: str, err_args: Dict[str, Any], cu: str,
                     r2: str, a: str, intent: str, focus: str) -> Dict[str, Any]:
        """金融纠错多轮：turn1 编造可选参数调用成功（结果错）→ 用户指正 → turn2 默认方案重算。
        _b2_common 的 turn1 错误调用写死 book_test_drive，金融场景需专用构建器。"""
        err_obs = run_tool("calculate_finance", err_args)
        fix_args = {"car_model": err_args["car_model"]}
        fix_obs = run_tool_parsed("calculate_finance", fix_args)

        def build(q_final: str, a_final: str) -> List[Dict[str, Any]]:
            return [
                user_msg(q_final),
                thought_call("call_1", r1, loss_mask=False), thought_result("call_1", r1),
                tool_call("call_2", "calculate_finance", err_args, loss_mask=False),
                tool_result("call_2", err_obs),
                user_msg(cu),
                thought_call("call_3", r2), thought_result("call_3", r2),
                tool_call("call_4", "calculate_finance", fix_args),
                tool_result("call_4", json.dumps(fix_obs, ensure_ascii=False)),
                finish_call("call_5", a_final),
            ]
        return {
            "sid": sid, "bucket": "bucket2_correction", "scenario": "金融方案测算",
            "targets": ["calculate_finance"], "is_clarify": False, "is_degrade": False,
            "question": q, "answer": a, "reasonings": [r1, r2], "build": build,
            "design_intent": intent, "review_focus": focus,
        }

    # F2: turn1 编造首付 50%（用户未指定）→ 成功但错 → 用户指正 → 默认 30% 重算 10
    for j in range(10):
        cfg = CONFIGS[(j * 3 + 2) % 4]
        tpls = [
            "{car}贷款月供多少？",
            "想贷款入手{car}，每月大概要还多少？",
            "{car}分期的话月供多少？",
            "想贷款买{car}，月供大概多少？",
            "{car}月供是什么水平？",
        ]
        q = tpls[j % 5].format(car=cfg["title"])
        r1 = "用户未指定首付，按五成首付计算。"
        err_args = {"car_model": cfg["title"], "down_payment_ratio": 50}
        cus = ["我只问月供，没说要首付五成，按你们的默认方案重新算。",
               "我没指定首付比例，你怎么按五成算了？重新算。"]
        r2 = "用户指出不应擅自假设首付五成，应按默认方案（首付30%、3年期）重新计算。"
        obs_f = run_tool_parsed("calculate_finance", {"car_model": cfg["title"]})
        d = obs_f["data"]
        a = (f"抱歉，之前按五成首付算的作废。按默认方案重新测算：{cfg['title']}指导价{int(d['guide_price'])}元，"
             f"首付30%（约{int(d['down_payment'])}元），贷款3年、年利率4.5%，月供约{round(float(d['monthly_payment']))}元。")
        out.append(_b2_fin_corr(
            f"v02_full_b2fn_b{j:02d}", q, r1, err_args, cus[j % 2], r2, a,
            "纠错多轮：编造首付五成→用户指正→默认方案重算", "①turn1 编造段 loss_mask=false ②turn2 按默认30%重算"))

    # F3: turn1 编造利率 3%（用户未指定）→ 成功但错 → 指正 → 默认 4.5% 重算 5
    for j in range(5):
        cfg = CONFIGS[(j * 3 + 3) % 4]
        tpls = [
            "{car}贷月供大概多少？",
            "{car}每个月还多少钱？",
            "{car}贷款每月还多少？",
        ]
        q = tpls[j % 3].format(car=cfg["title"])
        r1 = "用户未指定利率，按3%年利率计算。"
        err_args = {"car_model": cfg["title"], "interest_rate": 3}
        cus = "我没说过利率，你怎么按3%算了？按默认重新算。"
        r2 = "用户指出不应擅自设定利率，按默认年利率4.5%重新计算。"
        obs_f = run_tool_parsed("calculate_finance", {"car_model": cfg["title"]})
        d = obs_f["data"]
        a = (f"抱歉，之前按3%利率算的作废。按默认方案重新测算：{cfg['title']}指导价{int(d['guide_price'])}元，"
             f"首付30%（约{int(d['down_payment'])}元），贷款3年、年利率4.5%，月供约{round(float(d['monthly_payment']))}元。")
        out.append(_b2_fin_corr(
            f"v02_full_b2fn_c{j:02d}", q, r1, err_args, cus, r2, a,
            "纠错多轮：编造利率3%→用户指正→默认4.5%重算", "①turn1 编造段 loss_mask=false ②turn2 年利率回默认4.5%"))

    # F4: turn1 编造年限 5 年（用户未指定）→ 成功但错 → 指正 → 默认 3 年重算 5
    for j in range(5):
        cfg = CONFIGS[(j * 3 + 1) % 4]
        tpls = [
            "{car}按揭月供怎么算？",
            "{car}贷款的话月供多少？",
            "帮我看看{car}的月供",
        ]
        q = tpls[j % 3].format(car=cfg["title"])
        r1 = "用户未指定贷款年限，按5年期计算。"
        err_args = {"car_model": cfg["title"], "loan_years": 5}
        cus = "我没说贷几年，你怎么按5年算了？按默认重新算。"
        r2 = "用户指出不应擅自设定贷款年限，按默认3年期重新计算。"
        obs_f = run_tool_parsed("calculate_finance", {"car_model": cfg["title"]})
        d = obs_f["data"]
        a = (f"抱歉，之前按5年期算的作废。按默认方案重新测算：{cfg['title']}指导价{int(d['guide_price'])}元，"
             f"首付30%（约{int(d['down_payment'])}元），贷款3年、年利率4.5%，月供约{round(float(d['monthly_payment']))}元。")
        out.append(_b2_fin_corr(
            f"v02_full_b2fn_d{j:02d}", q, r1, err_args, cus, r2, a,
            "纠错多轮：编造贷款年限5年→用户指正→默认3年重算", "①turn1 编造段 loss_mask=false ②turn2 年限回默认3年"))

    return out  # 10 + 10 + 5 + 5 = 30


# ============================================================
# 桶2 · 竞品（20 条：编造参数纠错）
# 编造具体配置 10（clarify）/ 对比对象写错 10（execute）
# ============================================================

def make_b2_competitor(rng: random.Random) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # G1: 只给车系级名称 → turn1 擅自替用户选定具体配置对比 → 指正 → 反问配置 10
    # 车系级题面无实体变量，用 10 种措辞保证唯一
    tpls = [
        "帮我对比一下E5 Sportback和小米SU7",
        "E5 Sportback和小米SU7怎么选？",
        "我想看看E5 Sportback和小米SU7的对比",
        "对比下E5 Sportback和帕萨特",
        "E5 Sportback和帕萨特哪个更值得买？",
        "E5 Sportback跟小米SU7比怎么样？",
        "帮我分析下E5 Sportback和帕萨特",
        "E5 Sportback、小米SU7怎么对比？",
        "想了解一下E5 Sportback和帕萨特的差异",
        "E5 Sportback和小米SU7帮我看看选哪个",
    ]
    pairs = [("E5 Sportback 2025款 旗舰quattro型", "小米SU7 2025款 四驱Max版")] * 3 + \
            [("E5 Sportback 2025款 旗舰quattro型", "帕萨特2024款 330TSI 精英版")] * 2
    for j in range(10):
        a_cfg, b_cfg = pairs[j % 5]
        q = tpls[j % 10]
        r1 = "用户只给车系名，未说明具体配置。为快速给结论，直接替用户选定旗舰quattro型与竞品高配版进行对比。"
        err_args = {"car_model_a": a_cfg, "car_model_b": b_cfg}
        err_obs = run_tool("competitor_compare", err_args)
        cu = "我没说具体配置，你怎么替我选了旗舰quattro和别的款？先问清楚要对比哪款再说。"
        r2 = "用户指出不应擅自替其选定具体配置。应先请用户明确两款车的具体型号，确认后再对比。"
        a = (f"抱歉，我刚才擅自替您选定了具体配置，这样对比不准确。请确认要对比的具体型号：\n"
             f"1. E5 Sportback 可选配置：\n" +
             "\n".join(f"   - {c['title']}（{c['price']}元）" for c in CONFIGS) +
             f"\n2. 竞品可选：小米SU7 2025款 四驱Max版、帕萨特2024款 330TSI 精英版\n\n"
             f"确认后我为您提供准确对比。")

        def build(q_final: str, a_final: str, r1=r1, err_args=err_args, err_obs=err_obs,
                  cu=cu, r2=r2) -> List[Dict[str, Any]]:
            return [
                user_msg(q_final),
                thought_call("call_1", r1, loss_mask=False), thought_result("call_1", r1),
                tool_call("call_2", "competitor_compare", err_args, loss_mask=False),
                tool_result("call_2", err_obs),
                user_msg(cu),
                thought_call("call_3", r2), thought_result("call_3", r2),
                finish_call("call_4", a_final),
            ]
        out.append({
            "sid": f"v02_full_b2cp_a{j:02d}", "bucket": "bucket2_correction", "scenario": "竞品对比",
            "targets": [], "is_clarify": True, "is_degrade": False,
            "question": q, "answer": a, "reasonings": [r1, r2], "build": build,
            "design_intent": "纠错多轮 clarify：车系级名称→turn1 擅自替用户选配置对比→指正→反问具体型号",
            "review_focus": "①turn1 编造段 loss_mask=false ②turn2 列出可选配置/竞品 ③配置清单∈接地表",
        })

    # G2: 对比对象写错（同车系不同竞品）→ 指正 → 修正成功 10
    # 注：配置级"E5旗舰quattro vs 小米SU7四驱Max"组合与评测集 compare_002 整题子串重叠，
    # 题面只允许帕萨特对；小米SU7 仅出现在 turn1 错误工具调用参数（工具参数不参与零重叠校验）。
    a_cfg = "E5 Sportback 2025款 旗舰quattro型"
    b_ok = "帕萨特2024款 330TSI 精英版"
    b_wrong = "小米SU7 2025款 四驱Max版"
    tpls_g2 = [
        "帮我对比{pa}和{b}",
        "{pa}和{b}哪个更值得买？",
        "对比下{pa}与{b}",
        "{pa}跟{b}怎么选？",
        "帮我分析下{pa}和{b}的差异",
        "{pa}对比{b}，哪个好？",
        "想了解{pa}和{b}的优劣",
        "{pa}和{b}帮我对比一下",
        "{pa}与{b}该怎么选？",
        "帮我看看{pa}和{b}哪个合适",
    ]
    for j in range(10):
        q = tpls_g2[j % 10].format(pa=a_cfg, b=b_ok)
        r1 = f"用户要求对比{a_cfg}与{b_ok}，调用竞品对比工具。"
        err_args = {"car_model_a": a_cfg, "car_model_b": b_wrong}
        err_obs = run_tool("competitor_compare", err_args)
        fix_args = {"car_model_a": a_cfg, "car_model_b": b_ok}
        fix_obs = run_tool_parsed("competitor_compare", fix_args)
        cu = f"我要对比的是{b_ok}，不是{b_wrong}，你搞错了，重新对比。"
        r2 = f"用户指出对比对象写错，应为{b_ok}。按正确竞品重新调用对比工具。"
        d = fix_obs["data"]
        s = d["summary"]
        a = (f"抱歉，刚才对比对象弄错了。已重新为您对比{a_cfg}与{b_ok}：\n"
             f"整体结论：{s['overall_winner']}更胜一筹（优势项{s['car_a_advantages']}:{s['car_b_advantages']}）。\n"
             f"建议：{d['recommendation']}")

        def build(q_final: str, a_final: str, r1=r1, err_args=err_args, err_obs=err_obs,
                  cu=cu, r2=r2, fix_args=fix_args, fix_obs=fix_obs) -> List[Dict[str, Any]]:
            return [
                user_msg(q_final),
                thought_call("call_1", r1, loss_mask=False), thought_result("call_1", r1),
                tool_call("call_2", "competitor_compare", err_args, loss_mask=False),
                tool_result("call_2", err_obs),
                user_msg(cu),
                thought_call("call_3", r2), thought_result("call_3", r2),
                tool_call("call_4", "competitor_compare", fix_args),
                tool_result("call_4", json.dumps(fix_obs, ensure_ascii=False)),
                finish_call("call_5", a_final),
            ]
        out.append({
            "sid": f"v02_full_b2cp_b{j:02d}", "bucket": "bucket2_correction", "scenario": "竞品对比",
            "targets": ["competitor_compare"], "is_clarify": False, "is_degrade": False,
            "question": q, "answer": a, "reasonings": [r1, r2], "build": build,
            "design_intent": "纠错多轮 execute：对比对象写错（均为真实车型）→指正→修正成功",
            "review_focus": "①turn1 错误段 loss_mask=false ②turn2 对比对象与题面一致 ③结论数字=实跑值",
        })

    return out  # 20 条


# 桶1 补 dry-run 后的 230 条（试驾 70/金融 80/跨场景 80）
# + 桶2 补 dry-run 后的 240 条（试驾 190/金融 30/竞品 20） = 470 条
ALL_MAKERS = [
    make_b1_test_drive_extra, make_b1_finance, make_b1_cross,
    make_b2_test_drive_extra, make_b2_finance, make_b2_competitor,
]
