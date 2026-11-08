# -*- coding: utf-8 -*-
"""v02 全量模板 · 桶3（150 条）+ 桶4（100 条）+ 桶5（250 条）

桶3 多跳链路保持：试驾 40（末尾反问 20/执行 20）、金融 45、竞品 35、
                  跨场景 30（地区无专属优惠→全国兜底+自动读价 15 / 同工具双调用对比 15）
桶4 失败降级：试驾 40（车型不存在→告知+推荐替代）、金融 20（查不到→如实降级）、
              车型参数 20（text_search 假车型→如实说明）、跨场景 20（工具报错→重试修正）
桶5 场景保底：试驾 80（60 相对时间换算规范 + 20 显式日期）、车型参数 40、
              优惠 40、成交价 40、竞品 25、金融 25

骨架约定与 b1b2 一致：question/answer 由 Claude 润色，reasonings 程序模板，
build(q_final, a_final) 闭包组装消息（工具实跑）。
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

from gen_core import (
    BASE_DATE_POOL, CONFIGS, DEALERS, CUSTOMER_NAMES,
    base_date_cn, date_cn, iso, run_tool, run_tool_parsed,
    user_msg, thought_call, thought_result, tool_call, tool_result, finish_call,
)
from gen_templates_dryrun import _phone, _pick
from gen_templates_full_b1b2 import _rel_times


def _future_rel(base: datetime, j: int) -> Tuple[str, datetime]:
    """取第 j 个落在未来的相对时间表达（跳过过去时刻，保证预约日期不早于基准日）。"""
    opts = [(expr, t) for expr, t in _rel_times(base) if t > base]
    return opts[j % len(opts)]


def _simple(sid: str, scenario: str, q: str, r: str, tool_seq: List[Tuple[str, Dict[str, Any]]],
            a: str, targets: List[str], bucket: str, is_clarify: bool = False,
            is_degrade: bool = False, intent: str = "", focus: str = "") -> Dict[str, Any]:
    """单轮（可多跳）通用构建器：Thought → 工具链实跑 → Finish。"""
    def build(q_final: str, a_final: str) -> List[Dict[str, Any]]:
        msgs: List[Dict[str, Any]] = [
            user_msg(q_final),
            thought_call("call_1", r), thought_result("call_1", r),
        ]
        cid = 2
        for name, args in tool_seq:
            obs = run_tool(name, args)
            msgs.append(tool_call(f"call_{cid}", name, args))
            msgs.append(tool_result(f"call_{cid}", obs))
            cid += 1
        msgs.append(finish_call(f"call_{cid}", a_final))
        return msgs
    return {
        "sid": sid, "bucket": bucket, "scenario": scenario, "targets": targets,
        "is_clarify": is_clarify, "is_degrade": is_degrade,
        "question": q, "answer": a, "reasonings": [r], "build": build,
        "design_intent": intent, "review_focus": focus,
    }


# ============================================================
# 桶3 · 多跳链路保持
# ============================================================

def make_b3_test_drive(rng: random.Random) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # A: 末尾反问 20（text_search 查价 → 缺电话 → 反问）
    for j in range(20):
        base = BASE_DATE_POOL[j % 5]
        rel_expr, _t = _future_rel(base, j)
        cfg = CONFIGS[j % 4]
        dealer = DEALERS[(j * 3) % 7]
        name = _pick(CUSTOMER_NAMES, 400 + j)
        tpls = [
            "你好，我想看看{car}的价格，合适的话约{time}在{dealer}试驾，我叫{name}",
            "我是{name}，先帮我查下{car}的价格，再约{time}在{dealer}的试驾",
            "{car}现在什么价？价格合适我想约{time}在{dealer}试驾，我叫{name}",
            "我叫{name}，查下{car}价格，顺便约{time}的试驾，门店选{dealer}",
        ]
        q = tpls[j % 4].format(car=cfg["title"], time=rel_expr, dealer=dealer, name=name)
        obs_ts = run_tool_parsed("text_search", {"query": cfg["title"]})
        price = obs_ts["data"]["results"][0]["content"]["price"]
        r = "用户想先了解价格再决定是否预约试驾。先查询车型价格；预约信息中缺少联系电话，查价后向用户询问电话。"
        a = (f"{name}您好，{cfg['title']}指导价{price}元。若要预约{rel_expr}在{dealer}的试驾，"
             f"还需要您的联系电话，请提供一下，我马上为您安排。")
        out.append(_simple(f"v02_full_b3td_a{j:02d}", "试驾预约",
            q, r, [("text_search", {"query": cfg["title"]})], a,
            ["text_search"], "bucket3_multihop", is_clarify=True,
            intent="多跳末尾反问：查价→缺电话→反问（价格数字如实转述，不编造电话硬约）",
            focus="①先 text_search 查价 ②未调用 book_test_drive ③价格数字=实跑值"))

    # B: 执行 20（text_search 查参数 → 相对时间换算 → 预约成功）
    for j in range(20):
        base = BASE_DATE_POOL[(j + 2) % 5]
        rel_expr, t_c = _future_rel(base, (j + 3))
        t_book = t_c.replace(second=0)
        cfg = CONFIGS[(j + 1) % 4]
        dealer = DEALERS[(j * 5 + 1) % 7]
        name = _pick(CUSTOMER_NAMES, 420 + j)
        phone = _phone(420 + j)
        tpls = [
            "帮我查下{car}的续航，然后约{time}在{dealer}试驾，我叫{name}，电话{phone}",
            "我是{name}，电话{phone}，先查{car}续航，再约{time}在{dealer}试驾",
            "{car}续航多少？顺便帮我约{time}在{dealer}的试驾，{name}，电话{phone}",
            "查下{car}的续航，然后约{time}试驾，门店{dealer}，我叫{name}，电话{phone}",
        ]
        q = tpls[j % 4].format(car=cfg["title"], time=rel_expr, dealer=dealer,
                               name=name, phone=phone)
        obs_ts = run_tool_parsed("text_search", {"query": cfg["title"]})
        c = obs_ts["data"]["results"][0]["content"]
        args_bt = {"car_series": "E5 Sportback", "dealer": dealer,
                   "appointment_time": iso(t_book), "customer_name": name,
                   "customer_phone": phone}
        r = f"今天是{base_date_cn(base)}，用户说的'{rel_expr}'指{date_cn(t_book)}。先查续航，再按该日期预约试驾。"
        a = (f"{name}您好，{cfg['title']}CLTC续航{c['range_cltc']}。试驾已预约：{date_cn(t_book)}"
             f"{t_book.strftime('%H:%M')}，{dealer}。请携带驾驶证原件，提前15分钟到店。")
        out.append(_simple(f"v02_full_b3td_b{j:02d}", "试驾预约",
            q, r, [("text_search", {"query": cfg["title"]}), ("book_test_drive", args_bt)], a,
            ["text_search", "book_test_drive"], "bucket3_multihop",
            intent="多跳执行：查参数→相对时间换算→预约成功（两跳信息都进答复）",
            focus="①时间换算与基准日自洽 ②续航=实跑值 ③预约参数完整"))
    return out  # 40


def make_b3_finance(rng: random.Random) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # A: text_search → calculate_finance（默认方案）20
    for j in range(20):
        cfg = CONFIGS[(j * 3 + 1) % 4]
        tpls = [
            "帮我看看{car}的参数，再算下月供",
            "查下{car}的配置，顺便算下贷款月供",
            "{car}的参数和月供都帮我查下",
            "了解下{car}的参数，再帮我算算分期月供",
            "{car}的配置和月供情况都帮我看看",
        ]
        q = tpls[j % 5].format(car=cfg["title"])
        obs_ts = run_tool_parsed("text_search", {"query": cfg["title"]})
        c = obs_ts["data"]["results"][0]["content"]
        obs_fn = run_tool_parsed("calculate_finance", {"car_model": cfg["title"]})
        d = obs_fn["data"]
        r = "用户想了解参数并计算月供。先检索车型参数，再按默认方案（首付30%、3年期）计算月供，两跳信息都要在答复中呈现。"
        a = (f"{cfg['title']}核心参数：{c['motor']}，电池{c['battery_capacity']}，CLTC续航{c['range_cltc']}。\n"
             f"月供测算（默认方案）：指导价{int(d['guide_price'])}元，首付30%（约{int(d['down_payment'])}元），"
             f"贷款3年、年利率4.5%，月供约{round(float(d['monthly_payment']))}元。")
        out.append(_simple(f"v02_full_b3fn_a{j:02d}", "金融方案测算",
            q, r, [("text_search", {"query": cfg["title"]}),
                   ("calculate_finance", {"car_model": cfg["title"]})], a,
            ["text_search", "calculate_finance"], "bucket3_multihop",
            intent="多跳：查参数→算月供（默认方案），参数与月供均为实跑",
            focus="①两跳工具链完整 ②参数数字=text_search 实跑 ③月供=calculate_finance 实跑"))

    # B: query_discount_policy → calculate_finance（默认方案）15
    for j in range(15):
        cfg = CONFIGS[(j * 3 + 1) % 4]
        tpls = [
            "{car}现在有什么优惠？贷款月供怎么算？",
            "查下{car}的优惠政策，再算下月供",
            "{car}有优惠吗？顺便算下贷款月供",
            "{car}的优惠和月供帮我一起看下",
            "先看{car}的优惠，再帮我算月供",
        ]
        q = tpls[j % 5].format(car=cfg["title"])
        obs_dc = run_tool_parsed("query_discount_policy", {"car_model": "E5 Sportback"})
        ds = obs_dc["data"]["discounts"]
        obs_fn = run_tool_parsed("calculate_finance", {"car_model": cfg["title"]})
        d = obs_fn["data"]
        r = "用户想了解优惠并计算月供。先查全国通用优惠，再按默认方案计算月供，优惠与月供都要在答复中呈现。"
        a = (f"【优惠】E5 Sportback 全国通用优惠当前{len(ds)}项：\n" +
             "\n".join(f"{i + 1}. {x['name']}：{x['amount']}元（{x['condition']}）" for i, x in enumerate(ds)) +
             f"\n【月供】{cfg['title']}指导价{int(d['guide_price'])}元，首付30%（约{int(d['down_payment'])}元），"
             f"贷款3年、年利率4.5%，月供约{round(float(d['monthly_payment']))}元。")
        out.append(_simple(f"v02_full_b3fn_b{j:02d}", "金融方案测算",
            q, r, [("query_discount_policy", {"car_model": "E5 Sportback"}),
                   ("calculate_finance", {"car_model": cfg["title"]})], a,
            ["query_discount_policy", "calculate_finance"], "bucket3_multihop",
            intent="多跳：查优惠（全国口径，未指定地区不代入）→算月供，两跳结果合并呈现",
            focus="①优惠条目=实跑（全国口径）②月供=默认方案实跑"))

    # C: text_search → calculate_finance（用户参数跨跳传递）10
    for j in range(10):
        cfg = CONFIGS[(j + 2) % 4]
        ratio = [40, 30, 50][j % 3]
        years = [5, 3, 2][j % 3]
        tpls = [
            "查下{car}的价格，然后按首付{r}成、贷{y}年算月供",
            "{car}现在卖多少钱？按首付{r}%、{y}年期帮我算月供",
            "先看下{car}的价格，再按首付{r}成贷{y}年算下月供",
        ]
        q = tpls[j % 3].format(car=cfg["title"], r=ratio, y=years)
        obs_ts = run_tool_parsed("text_search", {"query": cfg["title"]})
        price = obs_ts["data"]["results"][0]["content"]["price"]
        obs_fn = run_tool_parsed("calculate_finance", {
            "car_model": cfg["title"], "down_payment_ratio": ratio, "loan_years": years})
        d = obs_fn["data"]
        r = f"用户指定首付{ratio}%、贷款{years}年。先查价格，再按用户参数计算月供。"
        a = (f"{cfg['title']}指导价{price}元。按首付{ratio}%、贷款{years}年测算："
             f"首付约{int(d['down_payment'])}元，贷款{int(d['loan_amount'])}元，年利率4.5%，"
             f"月供约{round(float(d['monthly_payment']))}元。")
        out.append(_simple(f"v02_full_b3fn_c{j:02d}", "金融方案测算",
            q, r, [("text_search", {"query": cfg["title"]}),
                   ("calculate_finance", {"car_model": cfg["title"],
                                          "down_payment_ratio": ratio, "loan_years": years})], a,
            ["text_search", "calculate_finance"], "bucket3_multihop",
            intent="多跳：查价→按用户参数算月供（首付/年限跨跳正确传递）",
            focus="①首付/年限如实传参 ②月供=实跑值 ③指导价与 text_search 一致"))
    return out  # 45


def make_b3_competitor(rng: random.Random) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    PAIR_A = "E5 Sportback 2025款 旗舰quattro型"

    # A: text_search → competitor_compare 20
    # 注："E5旗舰quattro vs 小米SU7四驱Max"与评测集 compare_002 整题子串重叠，
    # 竞品题面统一用帕萨特对（零重叠优先）。题面用场景修饰池×措辞池保证唯一。
    B_CFG = "帕萨特2024款 330TSI 精英版"
    scenes = ["我打算买家用车", "准备换车了", "想给家里添辆车", "最近在看车", "第一辆车想选好点"]
    for j in range(20):
        tpls = [
            "{scene}，帮我看看{pa}，和{b}比怎么样？",
            "{scene}，{pa}和{b}哪个更值得买？先查下参数再对比",
            "{scene}，查下{pa}的参数，再和{b}对比下",
            "{scene}，{pa}跟{b}怎么选？帮我分析下",
        ]
        q = tpls[j % 4].format(scene=scenes[(j // 4) % 5], pa=PAIR_A, b=B_CFG)
        obs_ts = run_tool_parsed("text_search", {"query": PAIR_A})
        c = obs_ts["data"]["results"][0]["content"]
        obs_cc = run_tool_parsed("competitor_compare", {"car_model_a": PAIR_A, "car_model_b": B_CFG})
        s = obs_cc["data"]["summary"]
        rec = obs_cc["data"]["recommendation"]
        r = "用户想了解本车参数并与竞品对比。先检索本车参数，再调用竞品对比工具，综合两跳信息作答。"
        a = (f"{PAIR_A}：指导价{c['price']}元，CLTC续航{c['range_cltc']}，{c['motor']}。\n"
             f"与{B_CFG}对比：{s['overall_winner']}更胜一筹（优势项{s['car_a_advantages']}:{s['car_b_advantages']}）。\n{rec}")
        out.append(_simple(f"v02_full_b3cp_a{j:02d}", "竞品对比",
            q, r, [("text_search", {"query": PAIR_A}),
                   ("competitor_compare", {"car_model_a": PAIR_A, "car_model_b": B_CFG})], a,
            ["text_search", "competitor_compare"], "bucket3_multihop",
            intent="多跳：查本车参数→竞品对比，两跳信息合并作答",
            focus="①两跳工具链完整 ②参数=实跑 ③对比结论=实跑"))

    # B: query_transaction_price → competitor_compare 15（竞品同用帕萨特对，零重叠优先）
    scenes = ["我打算买家用车", "准备换车了", "想给家里添辆车", "最近在看车", "第一辆车想选好点"]
    for j in range(15):
        tpls = [
            "{scene}，{pa}最近成交价多少？和{b}比怎么样？",
            "{scene}，查下{pa}的成交价，顺便和{b}对比一下",
            "{scene}，{pa}现在成交价如何？跟{b}比哪个划算？",
        ]
        q = tpls[j % 3].format(scene=scenes[(j // 3) % 5], pa=PAIR_A, b=B_CFG)
        obs_tp = run_tool_parsed("query_transaction_price", {"car_model": PAIR_A})
        tp = obs_tp["data"]
        obs_cc = run_tool_parsed("competitor_compare", {"car_model_a": PAIR_A, "car_model_b": B_CFG})
        s = obs_cc["data"]["summary"]
        rec = obs_cc["data"]["recommendation"]
        r = "用户想了解成交价并对比竞品。先查近90天成交价，再调用竞品对比工具，两跳信息都要呈现。"
        a = (f"{PAIR_A}近90天平均成交价{tp['avg_price']}元，趋势{tp['trend']}。\n"
             f"与{B_CFG}对比：{s['overall_winner']}更胜一筹（优势项{s['car_a_advantages']}:{s['car_b_advantages']}）。\n{rec}")
        out.append(_simple(f"v02_full_b3cp_b{j:02d}", "竞品对比",
            q, r, [("query_transaction_price", {"car_model": PAIR_A}),
                   ("competitor_compare", {"car_model_a": PAIR_A, "car_model_b": B_CFG})], a,
            ["query_transaction_price", "competitor_compare"], "bucket3_multihop",
            intent="多跳：查成交价→竞品对比，两跳信息合并作答",
            focus="①两跳工具链完整 ②成交价=实跑 ③对比结论=实跑"))
    return out  # 35


def make_b3_cross(rng: random.Random) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # A: 地区无专属优惠→如实说明+全国口径兜底→按工具读价算月供（multi_hop_008 正向）15
    cities = ["北京", "深圳", "广州", "杭州", "成都"]
    for j in range(15):
        city = cities[j % 5]
        cfg = CONFIGS[j % 4]
        tpls = [
            "{car}在{city}有什么优惠？优惠后首付30%贷款月供多少？",
            "{city}买{car}有优惠吗？首付30%的月供帮我算下",
            "查下{car}在{city}的优惠，顺便算下首付30%的月供",
        ]
        q = tpls[j % 3].format(car=cfg["title"], city=city)
        obs_dc = run_tool_parsed("query_discount_policy", {"car_model": "E5 Sportback", "region": city})
        ds = obs_dc["data"]["discounts"]
        obs_fn = run_tool_parsed("calculate_finance", {
            "car_model": cfg["title"], "down_payment_ratio": 30})
        d = obs_fn["data"]
        r = (f"用户问{city}地区优惠并要算月供。先查{city}优惠；若该地区暂无专属政策，"
             f"如实说明当前查到的是全国通用政策，再按首付30%计算月供（车价由工具自动读取）。")
        a = (f"{city}地区暂无专属优惠政策数据，当前查到的是全国通用优惠（共{len(ds)}项）：\n" +
             "\n".join(f"{i + 1}. {x['name']}：{x['amount']}元（{x['condition']}）" for i, x in enumerate(ds)) +
             f"\n\n优惠不直接抵扣指导价，最终成交价以经销商报价为准。按指导价{int(d['guide_price'])}元、首付30%测算："
             f"贷款3年、年利率4.5%，月供约{round(float(d['monthly_payment']))}元。")
        out.append(_simple(f"v02_full_b3cs_a{j:02d}", "金融方案测算",
            q, r, [("query_discount_policy", {"car_model": "E5 Sportback", "region": city}),
                   ("calculate_finance", {"car_model": cfg["title"], "down_payment_ratio": 30})], a,
            ["query_discount_policy", "calculate_finance"], "bucket3_multihop", is_degrade=True,
            intent="跨场景降级：地区无专属优惠→如实说明+全国口径兜底→按工具读价算月供（不编造价）",
            focus="①未编造地区专属优惠 ②如实标注全国口径 ③月供按工具读价实算"))

    # B: 同工具双调用对比：3年 vs 5年总利息（finance_009 正向）15
    for j in range(15):
        cfg = CONFIGS[(j + 1) % 4]
        tpls = [
            "{car}5年贷款和3年贷款哪个总利息更少？",
            "{car}贷3年和贷5年，总利息差多少？",
            "帮我对比下{car}3年期和5年期贷款的总利息",
            "{car}分3年和分5年还，利息差多少？",
            "想了解{car}3年贷和5年贷的总利息区别",
        ]
        q = tpls[j % 5].format(car=cfg["title"])
        obs3 = run_tool_parsed("calculate_finance", {"car_model": cfg["title"], "loan_years": 3})
        obs5 = run_tool_parsed("calculate_finance", {"car_model": cfg["title"], "loan_years": 5})
        d3, d5 = obs3["data"], obs5["data"]
        r = "用户想对比不同贷款年限的总利息。分别按3年期和5年期调用计算工具，用实算数字对比作答。"
        i3, i5 = int(d3["total_interest"]), int(d5["total_interest"])
        less = "3年期" if i3 < i5 else "5年期"
        a = (f"按{cfg['title']}（指导价{int(d3['guide_price'])}元，首付30%、年利率4.5%）测算：\n"
             f"3年期：月供约{round(float(d3['monthly_payment']))}元，总利息约{i3}元\n"
             f"5年期：月供约{round(float(d5['monthly_payment']))}元，总利息约{i5}元\n\n"
             f"总利息更少的是{less}（少约{abs(i5 - i3)}元），但月供相应更高，可按您的现金流偏好选择。")
        out.append(_simple(f"v02_full_b3cs_b{j:02d}", "金融方案测算",
            q, r, [("calculate_finance", {"car_model": cfg["title"], "loan_years": 3}),
                   ("calculate_finance", {"car_model": cfg["title"], "loan_years": 5})], a,
            ["calculate_finance"], "bucket3_multihop",
            intent="跨场景：同工具双调用对比（finance_009 正向），总利息全部实算不编造",
            focus="①两次调用参数正确 ②总利息/月供=实跑值 ③结论与数字一致"))
    return out  # 30


# ============================================================
# 桶4 · 失败降级
# ============================================================
# 白名单外车系（book_test_drive/text_search/calculate_finance 均无数据 → 实跑 NOT_FOUND/未找到）
FAKE_SERIES = ["特斯拉Model Y", "宝马3系", "奔驰C级", "理想L7",
               "问界M7", "比亚迪汉EV", "蔚来ET5", "小鹏P7"]


def make_b4_test_drive(rng: random.Random) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # 车型不存在→告知+推荐替代（推荐 ∈ 接地表 4 配置）40
    for j in range(40):
        base = BASE_DATE_POOL[j % 5]
        rel_expr, t_c = _future_rel(base, j)
        t_book = t_c.replace(second=0)
        fake = FAKE_SERIES[j % 8]
        dealer = DEALERS[(j * 3) % 7]
        name = _pick(CUSTOMER_NAMES, 500 + j)
        phone = _phone(500 + j)
        tpls = [
            "我是{name}，电话{phone}，想约{time}在{dealer}试驾{fake}",
            "帮我约{time}在{dealer}试驾{fake}，我叫{name}，电话{phone}",
            "{name}，电话{phone}，{time}想试驾{fake}，门店{dealer}",
            "预约{fake}试驾，{time}，{dealer}，{name}，{phone}",
        ]
        q = tpls[j % 4].format(name=name, phone=phone, time=rel_expr, dealer=dealer, fake=fake)
        args_bt = {"car_series": fake, "dealer": dealer, "appointment_time": iso(t_book),
                   "customer_name": name, "customer_phone": phone}
        r = "用户预约的车型需先经工具校验是否在可预约车系内。调用预约工具，若车型不存在则如实告知并推荐可预约车系。"
        a = (f"抱歉，{fake}目前不在我们的试驾预约车系内（可预约车系：E5 Sportback）。\n\n"
             f"为您推荐 E5 Sportback 的四个配置：\n" +
             "\n".join(f"{i + 1}. {c['title']}（{c['price']}元，CLTC续航{c['range']}）" for i, c in enumerate(CONFIGS)) +
             f"\n\n如感兴趣，我可以为您预约{dealer}的试驾。")
        out.append(_simple(f"v02_full_b4td_a{j:02d}", "试驾预约",
            q, r, [("book_test_drive", args_bt)], a,
            ["book_test_drive"], "bucket4_degrade", is_degrade=True,
            intent="失败降级：车型不在白名单→工具 NOT_FOUND→如实告知+推荐接地表车型",
            focus="①如实转述可预约车系 ②推荐车型∈接地表 4 配置 ③不编造预约成功"))
    return out  # 40


def make_b4_finance(rng: random.Random) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # 车型未收录→NOT_FOUND→如实说明+引导可用车系（不编造价）20
    for j in range(20):
        fake = FAKE_SERIES[(j + 2) % 8]
        tpls = [
            "帮我算下{fake}的贷款月供",
            "{fake}贷款月供大概多少？",
            "想贷款买{fake}，月供多少？",
            "{fake}分3年贷款，每月还多少？",
            "{fake}月供怎么算？帮我算下",
        ]
        q = tpls[j % 5].format(fake=fake)
        r = "先调用金融计算工具查询该车型指导价；若车型不在知识库，如实告知无法计算并说明可用车系，不编造价格计算。"
        a = (f"抱歉，{fake}暂未收录在我们的知识库中，无法为您计算月供。\n\n"
             f"目前可支持的车系：E5 Sportback 2025款、帕萨特2024款、小米SU7 2025款。\n"
             f"您可以告诉我具体配置（如 E5 Sportback 2025款 先锋型），我立即为您测算。")
        out.append(_simple(f"v02_full_b4fn_a{j:02d}", "金融方案测算",
            q, r, [("calculate_finance", {"car_model": fake})], a,
            ["calculate_finance"], "bucket4_degrade", is_degrade=True,
            intent="失败降级：车型未收录→工具 NOT_FOUND→如实说明+引导可用车系（不编造价）",
            focus="①如实说明无法计算 ②可用车系与工具返回一致 ③未编造月供数字"))
    return out  # 20


def make_b4_params(rng: random.Random) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # text_search 假车型→未找到→如实说明+列可用车型（car_config_008 正向）20
    for j in range(20):
        fake = FAKE_SERIES[(j + 4) % 8]
        tpls = [
            "帮我查下{fake}的配置参数",
            "{fake}的续航和价格是多少？",
            "查一下{fake}的参数信息",
            "{fake}有什么配置？帮我看看",
            "{fake}的参数帮我查一下",
        ]
        q = tpls[j % 5].format(fake=fake)
        r = "先检索该车型参数；若知识库无此车型，如实说明未找到并列出可用车型，不编造参数。"
        a = (f"抱歉，知识库中未找到{fake}的相关参数信息。当前知识库可用车型：\n"
             f"1. E5 Sportback 2025款（4个配置）\n2. 帕萨特2024款\n3. 小米SU7 2025款\n\n"
             f"如需了解具体车型的参数价格，请告诉我车型名称。")
        out.append(_simple(f"v02_full_b4pm_a{j:02d}", "车型参数查询",
            q, r, [("text_search", {"query": fake})], a,
            ["text_search"], "bucket4_degrade", is_degrade=True,
            intent="失败降级：假车型查询→text_search 未找到→如实说明+列可用车型",
            focus="①如实说明未找到 ②不编造参数数字 ③可用车型=知识库实际"))
    return out  # 20


def make_b4_cross(rng: random.Random) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    PAIR_A = "E5 Sportback 2025款 旗舰quattro型"

    # A: 漏传 car_model_b→INVALID_PARAM→补全重试成功 10（竞品用帕萨特对，零重叠优先）
    B_CFG = "帕萨特2024款 330TSI 精英版"
    for j in range(10):
        tpls = [
            "{pa}和{b}麻烦帮我比一比",
            "{pa}跟{b}哪个更值得买？",
            "{pa}、{b}这两款帮我对比看看",
            "{pa}和{b}帮我分析下",
            "想了解{pa}与{b}的差异",
            "{pa}对比{b}怎么样？",
            "{pa}和{b}怎么选？",
            "帮我看看{pa}跟{b}的优劣",
            "{pa}、{b}对比一下",
            "{pa}和{b}哪个更合适？",
        ]
        q = tpls[j % 10].format(pa=PAIR_A, b=B_CFG)
        r1 = f"用户要求对比{PAIR_A}与{B_CFG}。调用竞品对比工具。"
        err_args = {"car_model_a": PAIR_A}
        err_obs = run_tool("competitor_compare", err_args)
        fix_args = {"car_model_a": PAIR_A, "car_model_b": B_CFG}
        fix_obs = run_tool_parsed("competitor_compare", fix_args)
        r2 = "工具提示 car_model_b 参数不能为空。补全竞品车型参数后重新调用。"
        s = fix_obs["data"]["summary"]
        rec = fix_obs["data"]["recommendation"]
        a = (f"对比结果：{PAIR_A} vs {B_CFG}，{s['overall_winner']}更胜一筹"
             f"（优势项{s['car_a_advantages']}:{s['car_b_advantages']}）。\n{rec}")

        def build(q_final: str, a_final: str, r1=r1, err_args=err_args, err_obs=err_obs,
                  r2=r2, fix_args=fix_args, fix_obs=fix_obs) -> List[Dict[str, Any]]:
            return [
                user_msg(q_final),
                thought_call("call_1", r1), thought_result("call_1", r1),
                tool_call("call_2", "competitor_compare", err_args), tool_result("call_2", err_obs),
                thought_call("call_3", r2), thought_result("call_3", r2),
                tool_call("call_4", "competitor_compare", fix_args),
                tool_result("call_4", json.dumps(fix_obs, ensure_ascii=False)),
                finish_call("call_5", a_final),
            ]
        out.append({
            "sid": f"v02_full_b4cs_a{j:02d}", "bucket": "bucket4_degrade", "scenario": "竞品对比",
            "targets": ["competitor_compare"], "is_clarify": False, "is_degrade": False,
            "question": q, "answer": a, "reasonings": [r1, r2], "build": build,
            "design_intent": "工具报错→重试修正：漏传 car_model_b→INVALID_PARAM→补全重试成功",
            "review_focus": "①工具报错=实跑 INVALID_PARAM ②turn2 补全参数 ③结论=实跑值",
        })

    # B: days=45 非法→INVALID_PARAM→改 90 天口径重试 10
    for j in range(10):
        car = ["E5 Sportback 2025款 先锋型", "E5 Sportback 2025款 旗舰quattro型",
               "帕萨特2024款 330TSI 精英版", "小米SU7 2025款 四驱Max版"][j % 4]
        tpls = [
            "帮我查下{car}近一个半月的成交价",
            "{car}最近45天成交价怎么样？",
            "查下{car}近45天的历史成交价",
        ]
        q = tpls[j % 3].format(car=car)
        r1 = "用户想查近一个半月的成交价，按45天调用查询工具。"
        err_args = {"car_model": car, "days": 45}
        err_obs = run_tool("query_transaction_price", err_args)
        fix_args = {"car_model": car, "days": 90}
        fix_obs = run_tool_parsed("query_transaction_price", fix_args)
        r2 = "工具提示 days 仅支持 30/90/180/365。45 天最接近的官方口径是 90 天，按近90天重新查询。"
        d = fix_obs["data"]
        a = (f"成交价查询仅支持30/90/180/365天口径，已按近90天为您查询：\n{car}近90天平均成交价"
             f"{d['avg_price']}元（最低{d['min_price']}元/最高{d['max_price']}元），趋势{d['trend']}。\n"
             f"建议：{d['recommendation']}")

        def build(q_final: str, a_final: str, r1=r1, err_args=err_args, err_obs=err_obs,
                  r2=r2, fix_args=fix_args, fix_obs=fix_obs) -> List[Dict[str, Any]]:
            return [
                user_msg(q_final),
                thought_call("call_1", r1), thought_result("call_1", r1),
                tool_call("call_2", "query_transaction_price", err_args), tool_result("call_2", err_obs),
                thought_call("call_3", r2), thought_result("call_3", r2),
                tool_call("call_4", "query_transaction_price", fix_args),
                tool_result("call_4", json.dumps(fix_obs, ensure_ascii=False)),
                finish_call("call_5", a_final),
            ]
        out.append({
            "sid": f"v02_full_b4cs_b{j:02d}", "bucket": "bucket4_degrade", "scenario": "历史成交价",
            "targets": ["query_transaction_price"], "is_clarify": False, "is_degrade": False,
            "question": q, "answer": a, "reasonings": [r1, r2], "build": build,
            "design_intent": "工具报错→重试修正：days=45 非法→INVALID_PARAM→按90天官方口径重试成功",
            "review_focus": "①工具报错=实跑 INVALID_PARAM ②turn2 用合法 days ③成交价=实跑值",
        })
    return out  # 20


# ============================================================
# 桶5 · 场景保底
# ============================================================

def make_b5_test_drive(rng: random.Random) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # A: 信息齐全直接预约（相对时间换算规范，Thought 含基准日换算）60
    for j in range(60):
        base = BASE_DATE_POOL[j % 5]
        rel_expr, t_c = _future_rel(base, j)
        t_book = t_c.replace(second=0)
        cfg = CONFIGS[j % 4]
        dealer = DEALERS[(j * 2 + 1) % 7]
        name = _pick(CUSTOMER_NAMES, 600 + j)
        phone = _phone(600 + j)
        tpls = [
            "我是{name}，电话{phone}，想约{time}在{dealer}试驾{car}",
            "帮我约{time}在{dealer}试驾{car}，我叫{name}，电话{phone}",
            "{name}，{phone}，{time}想去{dealer}试驾{car}，帮我安排",
            "预约{car}试驾：{time}，{dealer}，{name}，{phone}",
        ]
        q = tpls[j % 4].format(name=name, phone=phone, time=rel_expr, dealer=dealer, car=cfg["title"])
        args_bt = {"car_series": "E5 Sportback", "dealer": dealer, "appointment_time": iso(t_book),
                   "customer_name": name, "customer_phone": phone}
        r = f"今天是{base_date_cn(base)}，用户说的'{rel_expr}'指{date_cn(t_book)}。信息齐全，直接预约。"
        a = (f"{name}您好，试驾已预约成功：{date_cn(t_book)}{t_book.strftime('%H:%M')}，{dealer}，"
             f"试驾{cfg['title']}。请携带驾驶证原件，提前15分钟到店。")
        out.append(_simple(f"v02_full_b5td_a{j:02d}", "试驾预约",
            q, r, [("book_test_drive", args_bt)], a,
            ["book_test_drive"], "bucket5_coverage",
            intent="场景保底：信息齐全直接预约（相对时间换算规范）",
            focus="①换算与基准日自洽 ②参数完整 ③预约成功如实转述"))

    # B: 显式日期 20
    for j in range(20):
        base = BASE_DATE_POOL[(j + 1) % 5]
        t_book = (base + timedelta(days=3 + j % 7)).replace(
            hour=[10, 14, 16][j % 3], minute=0, second=0)
        ampm = "下午" if t_book.hour >= 12 else "上午"
        hh = t_book.hour if t_book.hour <= 12 else t_book.hour - 12
        date_txt = f"{t_book.month}月{t_book.day}日{ampm}{hh}点"
        cfg = CONFIGS[(j + 2) % 4]
        dealer = DEALERS[(j * 4 + 3) % 7]
        name = _pick(CUSTOMER_NAMES, 660 + j)
        phone = _phone(660 + j)
        tpls = [
            "我是{name}，电话{phone}，想约{date}在{dealer}试驾{car}",
            "帮我约{date}的{car}试驾，{dealer}，我叫{name}，电话{phone}",
            "{name}，{phone}，{date}在{dealer}试驾{car}，麻烦安排下",
            "想约{date}试驾{car}，门店{dealer}，{name}，{phone}",
        ]
        q = tpls[j % 4].format(name=name, phone=phone, date=date_txt, dealer=dealer, car=cfg["title"])
        args_bt = {"car_series": "E5 Sportback", "dealer": dealer, "appointment_time": iso(t_book),
                   "customer_name": name, "customer_phone": phone}
        r = f"用户指定{date_txt}。信息齐全，直接预约。"
        a = (f"{name}您好，试驾已预约成功：{date_cn(t_book)}{t_book.strftime('%H:%M')}，{dealer}，"
             f"试驾{cfg['title']}。请携带驾驶证原件，提前15分钟到店。")
        out.append(_simple(f"v02_full_b5td_b{j:02d}", "试驾预约",
            q, r, [("book_test_drive", args_bt)], a,
            ["book_test_drive"], "bucket5_coverage",
            intent="场景保底：显式日期直接预约",
            focus="①日期解析正确 ②参数完整 ③预约成功如实转述"))
    return out  # 80


def make_b5_params(rng: random.Random) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    TITLES = [c["title"] for c in CONFIGS] + [
        "帕萨特2024款 280TSI 商务版", "帕萨特2024款 330TSI 精英版",
        "小米SU7 2025款 四驱Max版", "小米SU7 2025款 Pro版"]
    EV_QTYPES = ["配置", "价格", "续航", "电池容量", "快充"]
    ICE_QTYPES = ["配置", "价格", "油耗", "动力"]

    # text_search 单跳保底 40（8 车型 × 查询类轮换；帕萨特为燃油车，用油耗/动力口径）
    # 唯一性：tpl 索引 (j%8 + j//8)%8 每 8 条平移一轮，同车型 5 次出现的 tpl 全异；
    # ICE qtype 用 seq=[0,1,2,3,1] 按轮次错位，避免第 5 轮（j=36/37）与第 1 轮重复
    for j in range(40):
        title = TITLES[j % 8]
        is_ice = title.startswith("帕萨特")
        if is_ice:
            # 帕萨特在 TITLES 索引 4/5（j%8∈{4,5}），j%4 恒为 0/1，
            # 用 j%len 会永远轮不到油耗/动力——按帕萨特出现轮次独立轮换
            qtype = ICE_QTYPES[[0, 1, 2, 3, 1][(j - 4) // 8]]
        else:
            qtype = EV_QTYPES[j % 5]
        tpls = [
            "帮我查下{car}的{qtype}",
            "{car}的{qtype}是什么？",
            "了解下{car}的{qtype}",
            "{car}{qtype}怎么样？",
            "请问{car}的{qtype}数据",
            "{car}的{qtype}能介绍下吗？",
            "查下{car}{qtype}",
            "想知道{car}的{qtype}",
        ]
        q = tpls[(j % 8 + j // 8) % 8].format(car=title, qtype=qtype)
        obs = run_tool_parsed("text_search", {"query": title})
        c = obs["data"]["results"][0]["content"]
        r = f"用户查询{title}的{qtype}，直接检索知识库。"
        if is_ice:
            if qtype == "配置":
                a = (f"{title}：{c['engine']}，{c['transmission']}，综合油耗{c['fuel_consumption']}，"
                     f"车身尺寸{c['dimensions']}，轴距{c['wheelbase']}。")
            elif qtype == "价格":
                a = f"{title}指导价{c['price']}元。"
            elif qtype == "油耗":
                a = f"{title}综合油耗{c['fuel_consumption']}。"
            else:  # 动力
                a = f"{title}搭载{c['engine']}，{c['transmission']}。"
        else:
            if qtype == "配置":
                a = (f"{title}：{c['motor']}，电池{c['battery_capacity']}，CLTC续航{c['range_cltc']}，"
                     f"{c['drive_type']}，快充{c['charge_speed']}，车身尺寸{c['dimensions']}，轴距{c['wheelbase']}。")
            elif qtype == "价格":
                a = f"{title}指导价{c['price']}元。"
            elif qtype == "续航":
                a = f"{title}CLTC续航{c['range_cltc']}，快充{c['charge_speed']}。"
            elif qtype == "电池容量":
                a = f"{title}电池容量{c['battery_capacity']}，CLTC续航{c['range_cltc']}。"
            else:  # 快充
                a = f"{title}快充：{c['charge_speed']}。"
        out.append(_simple(f"v02_full_b5pm_a{j:02d}", "车型参数查询",
            q, r, [("text_search", {"query": title})], a,
            ["text_search"], "bucket5_coverage",
            intent="场景保底：text_search 单跳（电动车：配置/价格/续航/电池/快充；燃油车：配置/价格/油耗/动力）",
            focus="①参数数字=实跑 ②车型=知识库真实条目 ③油电字段口径正确"))
    return out  # 40


def make_b5_discount(rng: random.Random) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    CAR_KEYS = ["E5 Sportback", "帕萨特2024款", "小米SU7 2025款"]

    # query_discount_policy 单跳保底 40（全国/上海两口径，各 20 条）
    # 唯一性：car 用互质乘子 5 解耦（(j*5)%3 周期 3，与口径/措辞不再同频），
    # 口径按 j%2 分、措辞池 7 种按 (j//2)%7 → 同口径联合周期 lcm(14,6)=42 ≥ 20
    for j in range(40):
        car = CAR_KEYS[(j * 5) % 3]
        tpls_none = [
            "最近买{car}能享受哪些优惠？",
            "{car}现在下单有什么购车福利？",
            "想买{car}，最近店里有什么活动？",
            "{car}目前购车有什么补贴政策？",
            "{car}的优惠幅度大不大？帮我看看",
            "今年{car}的促销政策有哪些？",
            "{car}现在买有没有什么特别优惠？",
        ]
        tpls_sh = [
            "{car}在上海的购车优惠有哪些？",
            "上海地区买{car}有优惠吗？",
            "{car}上海的优惠政策有哪些？",
            "上海的{car}购车补贴怎么样？",
            "帮我查{car}在上海的优惠活动",
            "{car}在上海现在有什么补贴？",
            "上海买{car}有哪些促销？",
        ]
        if j % 2 == 0:
            q = tpls_none[(j // 2) % 7].format(car=car)
            args = {"car_model": car}
        else:
            q = tpls_sh[(j // 2) % 7].format(car=car)
            args = {"car_model": car, "region": "上海"}
        obs = run_tool_parsed("query_discount_policy", args)
        ds = obs["data"]["discounts"]
        reg = obs["data"]["region"]
        r = f"用户查询{car}的优惠政策，调用优惠查询工具。"
        a = (f"{car}在{reg}的优惠共{len(ds)}项：\n" +
             "\n".join(f"{i + 1}. {x['name']}：{x['amount']}元（{x['condition']}，有效期至{x['valid_until']}）"
                       for i, x in enumerate(ds)) +
             f"\n\n合计优惠{obs['data']['total_discount']}元。")
        out.append(_simple(f"v02_full_b5dc_a{j:02d}", "优惠政策查询",
            q, r, [("query_discount_policy", args)], a,
            ["query_discount_policy"], "bucket5_coverage",
            intent="场景保底：优惠查询单跳（全国/上海口径）",
            focus="①优惠条目=实跑 ②合计金额=实跑 ③region 与题面一致"))
    return out  # 40


def make_b5_transaction(rng: random.Random) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    CARS = ["E5 Sportback 2025款 旗舰quattro型", "E5 Sportback 2025款 先锋型",
            "帕萨特2024款 330TSI 精英版", "小米SU7 2025款 四驱Max版"]

    # query_transaction_price 单跳保底 40（30/90/180 天口径轮换）
    # 唯一性：car 用互质乘子 3 错位（周期 4）、tpl 周期 5、days 周期 3
    # → 联合周期 lcm(5,4,3)=60 ≥ 40；tpls 全部带 {days}，删除无 days 分支
    # 无数据兜底：mock 仅 4 车 90 天 + 旗舰quattro型 30 天有数据，其余 (车, 30/180) 组合
    # 实跑返回 sample_count=0 → 不直接播报 0 元，改查官方口径近90天（两跳兜底）。
    def _tp_sample(sid: str, q: str, r: str, r2: str, tool_seq: List[Tuple[str, Dict[str, Any]]],
                   a: str, intent: str, focus: str) -> Dict[str, Any]:
        """成交价样本构建器：单跳（有数据）或两跳兜底（无数据），参数绑定无闭包风险。"""
        def build(q_final: str, a_final: str) -> List[Dict[str, Any]]:
            msgs: List[Dict[str, Any]] = [
                user_msg(q_final),
                thought_call("call_1", r), thought_result("call_1", r),
            ]
            cid = 2
            for i, (name, args) in enumerate(tool_seq):
                obs = run_tool(name, args)
                if i >= 1:
                    r_next = r2 if r2 else r
                    msgs.append(thought_call(f"call_{cid}", r_next))
                    msgs.append(thought_result(f"call_{cid}", r_next))
                    cid += 1
                msgs.append(tool_call(f"call_{cid}", name, args))
                msgs.append(tool_result(f"call_{cid}", obs))
                cid += 1
            msgs.append(finish_call(f"call_{cid}", a_final))
            return msgs
        return {
            "sid": sid, "bucket": "bucket5_coverage", "scenario": "历史成交价",
            "targets": ["query_transaction_price"], "is_clarify": False, "is_degrade": False,
            "question": q, "answer": a, "reasonings": [x for x in [r, r2] if x], "build": build,
            "design_intent": intent, "review_focus": focus,
        }

    for j in range(40):
        car = CARS[(j * 3 + 1) % 4]
        days = [30, 90, 180][j % 3]
        tpls = [
            "{car}近{days}天成交价怎么样？",
            "帮我看下{car}近{days}天的历史成交价",
            "{car}近{days}天一般卖什么价？",
            "{car}近{days}天的价格走势如何？",
            "查下{car}近{days}天的成交行情",
        ]
        q = tpls[j % 5].format(car=car, days=days)
        obs = run_tool_parsed("query_transaction_price", {"car_model": car, "days": days})
        d = obs["data"]
        r = f"用户查询{car}的成交价，调用历史成交价查询工具。"
        r2 = ""
        if d.get("sample_count", 0) == 0:
            # 兜底：改查官方口径近90天（mock 中 4 车均有 90 天数据）
            obs90 = run_tool_parsed("query_transaction_price", {"car_model": car, "days": 90})
            d90 = obs90["data"]
            r2 = f"近{days}天暂无成交数据。为提供有效参考，改查官方口径的近90天成交价。"
            a = (f"{car}近{days}天暂无历史成交记录（样本0条）。\n"
                 f"已为您改查近90天：平均成交价{d90['avg_price']}元（最低{d90['min_price']}元/"
                 f"最高{d90['max_price']}元），趋势{d90['trend']}，样本{d90['sample_count']}条。\n"
                 f"建议：{d90['recommendation']}")
            tool_seq = [("query_transaction_price", {"car_model": car, "days": days}),
                        ("query_transaction_price", {"car_model": car, "days": 90})]
            out.append(_tp_sample(f"v02_full_b5tp_a{j:02d}", q, r, r2, tool_seq, a,
                "场景保底：成交价单跳（30/90/180天口径），无数据时兜底改查近90天",
                "①days 合法取值 ②无数据不播报 0 元 ③兜底成交价=实跑"))
        else:
            a = (f"{car}近{days}天：平均成交价{d['avg_price']}元（最低{d['min_price']}元/最高{d['max_price']}元），"
                 f"趋势{d['trend']}，样本{d['sample_count']}条。建议：{d['recommendation']}")
            out.append(_tp_sample(f"v02_full_b5tp_a{j:02d}", q, r, r2,
                [("query_transaction_price", {"car_model": car, "days": days})], a,
                "场景保底：成交价单跳（30/90/180天口径）",
                "①days 合法取值 ②成交价数字=实跑"))
    return out  # 40


def make_b5_competitor(rng: random.Random) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    # competitor_compare 单跳保底 25（与评测集 compare_002 零重叠，统一用帕萨特对）
    # 唯一性：对比对固定 → 场景修饰池 3 × 措辞池 10 = 30 组合 ≥ 25
    A_CFG = "E5 Sportback 2025款 旗舰quattro型"
    B_CFG = "帕萨特2024款 330TSI 精英版"
    scenes = ["我打算买家用车", "准备换车了", "最近在看车"]
    for j in range(25):
        tpls = [
            "{scene}，帮我对比{pa}和{b}",
            "{scene}，{pa}和{b}哪个更值得买？",
            "{scene}，对比下{pa}与{b}",
            "{scene}，{pa}跟{b}比有哪些优劣？",
            "{scene}，帮我分析下{pa}和{b}的差异",
            "{scene}，{pa}和{b}谁性价比高？",
            "{scene}，{pa}对比{b}怎么样？",
            "{scene}，麻烦比较一下{pa}和{b}",
            "{scene}，{pa}和{b}选哪个好？帮我对比",
            "{scene}，{pa}与{b}的优缺点对比一下",
        ]
        q = tpls[j % 10].format(scene=scenes[(j // 10) % 3], pa=A_CFG, b=B_CFG)
        obs = run_tool_parsed("competitor_compare", {"car_model_a": A_CFG, "car_model_b": B_CFG})
        s = obs["data"]["summary"]
        rec = obs["data"]["recommendation"]
        r = f"用户想对比{A_CFG}与{B_CFG}，调用竞品对比工具。"
        a = (f"对比结论：{s['overall_winner']}更胜一筹（优势项{s['car_a_advantages']}:{s['car_b_advantages']}，"
             f"相当项{s['equal_count']}）。\n{rec}")
        out.append(_simple(f"v02_full_b5cp_a{j:02d}", "竞品对比",
            q, r, [("competitor_compare", {"car_model_a": A_CFG, "car_model_b": B_CFG})], a,
            ["competitor_compare"], "bucket5_coverage",
            intent="场景保底：竞品对比单跳",
            focus="①对比对∈mock ②结论数字=实跑"))
    return out  # 25


def make_b5_finance(rng: random.Random) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # calculate_finance 单跳保底 25（首付/年限组合轮换）
    # 唯一性：cfg 用互质乘子 3 错位（周期 4）、tpl 周期 5、ratio/years 按 (j//5) 分档
    # → 组内 5 条 tpl 全异；组间 ratio/years 全异；cfg 重复仅发生在 j 与 j+20，其 ratio 必异
    for j in range(25):
        cfg = CONFIGS[(j * 3 + 1) % 4]
        ratio = [30, 40, 50, 30, 20][(j // 5) % 5]
        years = [3, 5, 3, 2, 5][(j // 5) % 5]
        tpls = [
            "{car}首付{r}成、贷{y}年，月供多少？",
            "帮我算{car}首付{r}%、{y}年贷款的月供",
            "{car}首付比例{r}%、分{y}年还，每月多少？",
            "首付{r}成买{car}，贷{y}年，月供怎么算？",
            "{car}贷款{y}年、首付{r}%，帮我算算月供",
        ]
        q = tpls[j % 5].format(car=cfg["title"], r=ratio, y=years)
        obs = run_tool_parsed("calculate_finance", {
            "car_model": cfg["title"], "down_payment_ratio": ratio, "loan_years": years})
        d = obs["data"]
        r = f"用户指定首付{ratio}%、贷款{years}年，直接调用金融计算工具（车价自动读取）。"
        a = (f"{cfg['title']}：指导价{int(d['guide_price'])}元，首付{ratio}%（约{int(d['down_payment'])}元），"
             f"贷款{int(d['loan_amount'])}元、{years}年期、年利率4.5%，月供约{round(float(d['monthly_payment']))}元，"
             f"总利息约{int(d['total_interest'])}元。")
        out.append(_simple(f"v02_full_b5fn_a{j:02d}", "金融方案测算",
            q, r, [("calculate_finance", {"car_model": cfg["title"],
                                          "down_payment_ratio": ratio, "loan_years": years})], a,
            ["calculate_finance"], "bucket5_coverage",
            intent="场景保底：金融单跳（首付/年限组合轮换）",
            focus="①参数如实传参 ②月供/总利息=实算"))
    return out  # 25


# 桶3 150 + 桶4 100 + 桶5 250 = 500
ALL_MAKERS = [
    make_b3_test_drive, make_b3_finance, make_b3_competitor, make_b3_cross,
    make_b4_test_drive, make_b4_finance, make_b4_params, make_b4_cross,
    make_b5_test_drive, make_b5_params, make_b5_discount, make_b5_transaction,
    make_b5_competitor, make_b5_finance,
]
