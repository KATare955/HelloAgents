"""RL prompt 集生成器 + 三条质量红线断言（RL数据规格书 2026-11-17 §2/§3）

配比：多跳 100 / 失败恢复 62 / 新工具 50 / 回归 38（共 250 条）
输出：
- eval/eval_data/questions_v1.3_rl.json   15 题 RL 专属评测子集（先冻结，训练集不得复用其实体组合）
- rl/data/rl_prompts_v1_2026-11-18.jsonl  250 条训练 prompt（spec §1 schema）
- rl/data/quality_report_2026-11-18.json   红线断言报告

红线：
R1 与 questions_v1.2.1（68）+ questions_v1.3_rl（15）精确 + 归一化零重叠（双向子串级）
R2 gt_facts 与 mock 数据接地一致（指导价/优惠/月供/成交价/对比价/web 摘要/跟进任务/试驾预约）
R3 accept_clarification 标注语义与 v1.2.1 同口径（{target_tool, reason} 字典，题面真实缺参才豁免）

用法：python rl/gen_rl_prompts.py        生成 + 断言（任一红线失败 → 退出码 1）
      python rl/gen_rl_prompts.py --check 对已落盘产物重跑 R1/R3 + R2（正则重推导）
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = HERE / "data"
EVAL_DATA = ROOT / "eval" / "eval_data"

sys.path.insert(0, str(ROOT))

from tools._matching import score_match, _norm  # noqa: E402
from tools.calculate_finance import CalculateFinanceTool  # noqa: E402
from tools.book_test_drive import BookTestDriveTool  # noqa: E402
from tools.follow_up_query import FollowUpQueryTool  # noqa: E402

EVAL_YEAR = "2026"
OUT_JSONL = DATA_DIR / "rl_prompts_v1_2026-11-18.jsonl"
OUT_REPORT = DATA_DIR / "quality_report_2026-11-18.json"
OUT_SUBSET = EVAL_DATA / "questions_v1.3_rl.json"

# ---------------------------------------------------------------------------
# 接地事实表（mock 数据）
# ---------------------------------------------------------------------------

def _load(name: str):
    with open(ROOT / "tools" / "mock_data" / name, encoding="utf-8") as f:
        return json.load(f)

TS = _load("text_search.json")              # 车系 -> {results:[{title, content:{price}}]}
DISCOUNT = _load("query_discount_policy.json")
TX = _load("query_transaction_price.json")
CMP = _load("competitor_compare.json")
WEB = _load("web_search.json")
FU = _load("follow_up_query.json")
BK = _load("book_test_drive.json")

TRIMS = [  # (series, title, price)
    (s, r["title"], r["content"]["price"])
    for s, d in TS.items() for r in d["results"]
]
PRICE_OF = {t: p for _, t, p in TRIMS}
SERIES_OF = {t: s for s, t, _ in TRIMS}

DISCOUNT_NATIONAL = {
    k: (v.get("全国") or next(iter(v.values())) or {}).get("total_discount")
    for k, v in DISCOUNT.items()
}

TX_AVG = {  # 车型 -> 默认期（90 优先）成交均价
    k: (v.get("90") or next(iter(v.values())))["avg_price"]
    for k, v in TX.items()
}

CMP_PAIRS = []  # (model_a, model_b, price_a, price_b)
for outer, inner in CMP.items():
    for b_key, pair in inner.items():
        p = pair["comparison"]["价格"]
        CMP_PAIRS.append((pair["car_model_a"], pair["car_model_b"],
                          p["car_a_value"], p["car_b_value"]))

WEB_ITEMS = [
    (r["id"], r["snippet"])
    for d in WEB.values() for r in d["results"]
]

FU_CUSTOMERS = {k: [t["task_type"] for t in v["tasks"]] for k, v in FU.items()}

DEALERS = list(BK["E5 Sportback"].keys())
SERIES_SHORT = ["E5 Sportback", "小米SU7", "帕萨特"]
TRIM_NAMES = [t for _, t, _ in TRIMS]

_cf_tool = CalculateFinanceTool()


def _monthly(price: float, ratio: int, years: int, rate: float = 4.5) -> float:
    return round(_cf_tool._calculate_finance(price, ratio, years, rate)["monthly_payment"], 2)


# 试驾槽位池（姓名/电话/日期与子集隔离：不用李强/13800138001/浦东金桥店/11月21日）
_BK_NAMES = ["张伟", "李娜", "刘敏", "陈杰", "吴磊", "徐强", "马丽", "高翔", "韩雪",
             "郭涛", "曹阳", "彭飞", "蒋丽", "沈涛", "冯雪", "何军", "宋雨", "唐芳",
             "魏东", "邓琳"]
_BK_DATES = ["11月22日", "11月23日", "11月24日", "11月25日", "11月26日",
             "11月27日", "11月28日", "11月29日", "11月30日"]
_BK_SLOTS = [
    (d, "下午", "3", "15:00:00") for d in _BK_DATES
] + [
    (d, "上午", "10", "10:00:00") for d in _BK_DATES
]  # 18 个 (date, daypart, hour_cn, hour_iso)

_bk_counter = {"i": 0}


def bk_slot(i: int):
    """第 i 个试驾槽位：dealer/date/time/name/phone 循环取值。"""
    name = _BK_NAMES[i % len(_BK_NAMES)]
    phone = f"13{i % 10}1234{i % 10:04d}"
    date, daypart, hour_cn, hour_iso = _BK_SLOTS[i % len(_BK_SLOTS)]
    dealer = DEALERS[i % 2]
    iso = f"{EVAL_YEAR}-{date[0:2]}-{date[3:5]}T{hour_iso}"
    return {
        "dealer": dealer, "date": date, "daypart": daypart, "hour": hour_cn,
        "name": name, "phone": phone, "iso": iso,
    }


def bk_text(slot) -> str:
    return f"{slot['date']}{slot['daypart']}{slot['hour']}点{slot['dealer']}"


# ---------------------------------------------------------------------------
# prompt 构造
# ---------------------------------------------------------------------------

PROMPTS = []  # (bucket, dict)


def P(bucket, question, expected_tools, gt_facts=None, accept_clarification=False,
      fail_inject=None, dep_edges=None, difficulty="easy", ground=None):
    d = {
        "question": question,
        "expected_tools": expected_tools,
        "gt_facts": gt_facts or {},
        "accept_clarification": accept_clarification,
        "fail_inject": fail_inject,
        "dep_edges": dep_edges or [],
        "difficulty": difficulty,
    }
    PROMPTS.append((bucket, d, ground))


def F(tool, times, code):
    return {"tool": tool, "fail_times": times, "error_code": code}


def AC(target, reason):
    return {"target_tool": target, "reason": reason}


# ---------------------------------------------------------------------------
# 评测子集 15 题（先冻结；训练集实体组合隔离）
# ---------------------------------------------------------------------------

def build_subset() -> list:
    qs = []
    qs.append(dict(id="rl_nt_001", scenario="新工具-网页搜索", type="单跳",
        question="我听说E5 Sportback最近有召回，帮我查一下涉及多少辆车",
        expected_tools=["web_search"], difficulty="medium",
        gt_facts={"召回数量": "1482"}, accept_clarification=False,
        fail_inject=None, dep_edges=[]))
    qs.append(dict(id="rl_nt_002", scenario="新工具-网页搜索", type="单跳",
        question="A5L 2026款是不是已经上市了？卖多少钱？",
        expected_tools=["web_search"], difficulty="medium",
        gt_facts={"价格区间": "29.98万"}, accept_clarification=False,
        fail_inject=None, dep_edges=[]))
    qs.append(dict(id="rl_nt_003", scenario="新工具-网页搜索", type="单跳",
        question="帮我搜一下Model 3现在卖多少钱",
        expected_tools=["web_search"], difficulty="medium",
        gt_facts={"价格": "23.19万"}, accept_clarification=False,
        fail_inject=None, dep_edges=[]))
    qs.append(dict(id="rl_nt_004", scenario="新工具-网页搜索", type="单跳",
        question="2026年新能源车购置税政策是什么样的？",
        expected_tools=["web_search"], difficulty="medium",
        gt_facts={"税率": "5%"}, accept_clarification=False,
        fail_inject=None, dep_edges=[]))
    qs.append(dict(id="rl_nt_005", scenario="新工具-跟进任务", type="单跳",
        question="帮我看下王建国的跟进任务有哪些",
        expected_tools=["follow_up_query"], difficulty="easy",
        gt_facts={"任务1": "试驾提醒", "任务2": "报价跟进"}, accept_clarification=False,
        fail_inject=None, dep_edges=[]))
    qs.append(dict(id="rl_nt_006", scenario="新工具-跟进任务", type="单跳",
        question="孙丽那边有什么待跟进的事？",
        expected_tools=["follow_up_query"], difficulty="easy",
        gt_facts={"任务1": "试驾提醒", "任务2": "报价跟进", "任务3": "交车确认"},
        accept_clarification=False, fail_inject=None, dep_edges=[]))
    qs.append(dict(id="rl_nt_007", scenario="新工具-跟进任务", type="单跳",
        question="查一下周伟还有没有没做完的跟进任务",
        expected_tools=["follow_up_query"], difficulty="easy",
        gt_facts={"任务": "没有未完成"}, accept_clarification=False,
        fail_inject=None, dep_edges=[]))
    qs.append(dict(id="rl_nt_008", scenario="新工具-跟进任务", type="单跳",
        question="帮我查下李明的跟进任务",
        expected_tools=["follow_up_query"], difficulty="medium",
        gt_facts={},
        accept_clarification=AC("follow_up_query",
            "客户'李明'不在CRM中（目标不可达）；正确行为=告知未找到并请用户确认姓名"),
        fail_inject=None, dep_edges=[]))

    qs.append(dict(id="rl_fr_001", scenario="失败恢复", type="失败恢复",
        question="帕萨特2024款330TSI精英版现在卖多少钱？",
        expected_tools=["text_search"], difficulty="hard",
        gt_facts={"指导价": 205900}, accept_clarification=False,
        fail_inject=F("text_search", 1, "TIMEOUT"), dep_edges=[]))
    qs.append(dict(id="rl_fr_002", scenario="失败恢复", type="失败恢复",
        question="帮我算一下E5 Sportback 2025款先锋型首付30%贷3年的月供",
        expected_tools=["calculate_finance"], difficulty="hard",
        gt_facts={"指导价": 235900, "月供": _monthly(235900, 30, 3)},
        accept_clarification=False,
        fail_inject=F("calculate_finance", 1, "PARAM_ERROR"), dep_edges=[]))
    qs.append(dict(id="rl_fr_003", scenario="失败恢复", type="失败恢复",
        question="我想约E5 Sportback试驾，浦东金桥店，11月21日下午3点，我叫李强，电话13800138001",
        expected_tools=["book_test_drive"], difficulty="hard",
        gt_facts={},
        accept_clarification=AC("book_test_drive",
            "经销商'浦东金桥店'不在白名单（目标不可达）；正确行为=告知可选门店并反问确认"),
        fail_inject=F("book_test_drive", 1, "NOT_FOUND"), dep_edges=[]))
    qs.append(dict(id="rl_fr_004", scenario="失败恢复", type="失败恢复",
        question="现在买小米SU7 2025款有什么优惠？",
        expected_tools=["query_discount_policy"], difficulty="hard",
        gt_facts={"优惠总额": 10000}, accept_clarification=False,
        fail_inject=F("query_discount_policy", 1, "TIMEOUT"), dep_edges=[]))
    qs.append(dict(id="rl_fr_005", scenario="失败恢复", type="失败恢复",
        question="帮我查下帕萨特2024款330TSI精英版最近成交价",
        expected_tools=["query_transaction_price"], difficulty="hard",
        gt_facts={"成交均价": 195000}, accept_clarification=False,
        fail_inject=F("query_transaction_price", 1, "NOT_FOUND"), dep_edges=[]))
    qs.append(dict(id="rl_fr_006", scenario="失败恢复", type="失败恢复",
        question="帮我搜下汉EV现在什么价格",
        expected_tools=["web_search"], difficulty="hard",
        gt_facts={"价格": "21.98万"}, accept_clarification=False,
        fail_inject=F("web_search", 1, "TIMEOUT"), dep_edges=[]))
    qs.append(dict(id="rl_fr_007", scenario="失败恢复", type="失败恢复",
        question="查下王芳的跟进任务",
        expected_tools=["follow_up_query"], difficulty="hard",
        gt_facts={},
        accept_clarification=AC("follow_up_query",
            "客户'王芳'不在CRM中（目标不可达）；正确行为=告知未找到并请用户确认姓名"),
        fail_inject=F("follow_up_query", 1, "NOT_FOUND"), dep_edges=[]))
    return qs


# ---------------------------------------------------------------------------
# 多跳 100
# ---------------------------------------------------------------------------

def mh_a():
    """3跳：配置发现→金融(顶配)→试驾。dep: text_search→calculate_finance。"""
    top = ("E5 Sportback 2025款 旗舰quattro型", 319900)
    ph = [
        "我想买E5 Sportback，帮我看看有哪些配置和价格？算下顶配首付{r}%贷{y}年的月供，"
        "合适的话帮我约{s}的试驾，我叫{name}，电话{phone}",
        "帮我查下E5 Sportback的配置表，用最贵的那款算算首付{r}%贷{y}年月供多少？"
        "可以的话约{s}试驾，{name}，{phone}",
    ]
    params = [(30, 5), (50, 3), (20, 5), (30, 3)]
    for i, (r, y) in enumerate(params):
        s = bk_slot(i)
        q = ph[i % 2].format(r=r, y=y, s=bk_text(s), name=s["name"], phone=s["phone"])
        P("多跳", q, ["text_search", "calculate_finance", "book_test_drive"],
          gt_facts={"指导价": top[1], "月供": _monthly(top[1], r, y), "试驾预约": "预约成功"},
          dep_edges=[["text_search", "calculate_finance"]], difficulty="multi_hop_3",
          ground={"finance": (top[0], top[1], r, y, 4.5), "booking": s})


def mh_b():
    """3跳：优惠→金融→试驾。dep: query_discount_policy→calculate_finance（贴息）。"""
    trims = [("E5 Sportback 2025款 先锋型", 235900),
             ("E5 Sportback 2025款 先锋Plus型", 269900),
             ("E5 Sportback 2025款 旗舰quattro型", 319900)]
    params = [(30, 5), (30, 3), (50, 3), (20, 3)]
    ph = [
        "现在买E5 Sportback有什么优惠？{trim}首付{r}%贷{y}年月供多少？"
        "合适的话帮我约{s}试驾，{name}，{phone}",
        "买E5 Sportback现在能优惠多少？帮我算下{trim}首付{r}%贷{y}年的月供，"
        "可以就约{s}的试驾，{name}，{phone}",
    ]
    for i, (trim, price) in enumerate(trims):
        for j, (r, y) in enumerate(params):
            s = bk_slot(i * 4 + j)
            q = ph[(i + j) % 2].format(trim=trim, r=r, y=y, s=bk_text(s),
                                       name=s["name"], phone=s["phone"])
            P("多跳", q, ["query_discount_policy", "calculate_finance", "book_test_drive"],
              gt_facts={"优惠总额": 30000, "指导价": price,
                        "月供": _monthly(price, r, y), "试驾预约": "预约成功"},
              dep_edges=[["query_discount_policy", "calculate_finance"]],
              difficulty="multi_hop_3",
              ground={"discount": "E5 Sportback", "finance": (trim, price, r, y, 4.5),
                      "booking": s})


def mh_c():
    """3跳：对比→成交价→优惠。"""
    combos = [  # (A, B, B 车系)——对比价从 CMP 表运行时取
        ("E5 Sportback 2025款 旗舰quattro型", "小米SU7 2025款 四驱Max版",
         "小米SU7 2025款"),
        ("E5 Sportback 2025款 旗舰quattro型", "帕萨特2024款 330TSI 精英版",
         "帕萨特2024款"),
        ("小米SU7 2025款 四驱Max版", "E5 Sportback 2025款 旗舰quattro型",
         "E5 Sportback"),
    ]
    # 运行时从 CMP 表取价，避免硬编码错误
    pair_price = {}
    for a, b, va, vb in CMP_PAIRS:
        pair_price[(a, b)] = (va, vb)
        pair_price[(b, a)] = (vb, va)
    ph = [
        "我纠结{A}和{B}，帮我对比一下；另外{B}最近成交价怎么样？现在买{series}的话有什么优惠？",
        "帮我对比{A}跟{B}，再查查{B}近期的成交价，{series}现在有什么优惠活动？",
        "{A}跟{B}哪个更值得买？顺便看看{B}最近成交价和{series}目前的优惠",
        "{A}和{B}选哪个好？帮我对比后查下{B}的近期成交价和{series}现在的优惠",
    ]
    n = 0
    for ci, (a, b, series) in enumerate(combos):
        va, vb = pair_price[(a, b)]
        for j in range([4, 3, 2][ci]):
            q = ph[j].format(A=a, B=b, series=series)
            P("多跳", q, ["competitor_compare", "query_transaction_price",
                          "query_discount_policy"],
              gt_facts={"对比价A": va, "对比价B": vb,
                        "成交均价": TX_AVG[b], "优惠总额": DISCOUNT_NATIONAL[series]},
              difficulty="multi_hop_3",
              ground={"compare": (a, b), "tx": b, "discount": series})
            n += 1
    assert n == 9, n


def mh_d():
    """4跳：对比→成交价→优惠→金融。"""
    combos = [
        ("E5 Sportback 2025款 旗舰quattro型", "帕萨特2024款 330TSI 精英版",
         "帕萨特2024款", 205900, 195000, 23000, (30, 3)),
        ("E5 Sportback 2025款 旗舰quattro型", "小米SU7 2025款 四驱Max版",
         "小米SU7 2025款", 303900, 303900, 10000, (30, 5)),
        ("小米SU7 2025款 四驱Max版", "E5 Sportback 2025款 旗舰quattro型",
         "E5 Sportback", 319900, 289500, 30000, (50, 3)),
    ]
    pair_price = {}
    for a, b, va, vb in CMP_PAIRS:
        pair_price[(a, b)] = (va, vb)
        pair_price[(b, a)] = (vb, va)
    ph = [
        "帮我对比{A}跟{B}，看看{B}最近成交价，现在买{series}的话有什么优惠？"
        "算下{B}首付{r}%贷{y}年月供",
        "先对比下{A}跟{B}，再查{B}的近期成交价和{series}的优惠，"
        "最后算算{B}首付{r}%贷{y}年的月供",
        "{A}和{B}哪个好？帮我对比完再查{B}近期成交价、{series}当前优惠，"
        "算下{B}首付{r}%贷{y}年月供",
    ]
    n = 0
    for ci, (a, b, series, bprice, btx, bdisc, (r, y)) in enumerate(combos):
        va, vb = pair_price[(a, b)]
        rep = 3 if ci < 2 else 2
        for j in range(rep):
            q = ph[j].format(A=a, B=b, series=series, r=r, y=y)
            P("多跳", q, ["competitor_compare", "query_transaction_price",
                          "query_discount_policy", "calculate_finance"],
              gt_facts={"对比价A": va, "对比价B": vb, "成交均价": btx,
                        "优惠总额": bdisc, "指导价": bprice,
                        "月供": _monthly(bprice, r, y)},
              difficulty="multi_hop_4",
              ground={"compare": (a, b), "tx": b, "discount": series,
                      "finance": (b, bprice, r, y, 4.5)})
            n += 1
    assert n == 8, n


def mh_e():
    """4跳：配置→优惠→金融→试驾。"""
    trims = [("E5 Sportback 2025款 先锋型", 235900),
             ("E5 Sportback 2025款 旗舰quattro型", 319900),
             ("E5 Sportback 2025款 先锋Plus型", 269900)]
    params = [(30, 5), (30, 3), (50, 3), (20, 3)]
    ph = [
        "帮我查下{trim}的配置和价格，现在有什么优惠？首付{r}%贷{y}年月供多少？"
        "合适的话帮我约{s}试驾，{name}，{phone}",
        "查查{trim}的价格配置，再看看目前的优惠，算下首付{r}%贷{y}年的月供，"
        "可以就约{s}试驾，我叫{name}，电话{phone}",
    ]
    n = 0
    for i, (trim, price) in enumerate(trims):
        for j, (r, y) in enumerate(params):
            if i == 2 and j == 3:
                continue  # 11 条
            s = bk_slot(20 + i * 4 + j)
            q = ph[(i + j) % 2].format(trim=trim, r=r, y=y, s=bk_text(s),
                                       name=s["name"], phone=s["phone"])
            P("多跳", q, ["text_search", "query_discount_policy",
                          "calculate_finance", "book_test_drive"],
              gt_facts={"指导价": price, "优惠总额": 30000,
                        "月供": _monthly(price, r, y), "试驾预约": "预约成功"},
              difficulty="multi_hop_4",
              ground={"discount": "E5 Sportback",
                      "finance": (trim, price, r, y, 4.5), "booking": s})
            n += 1
    assert n == 11, n


def mh_f():
    """5跳：配置发现→对比(最贵款)→成交价→金融→试驾。dep: text_search→{compare,tx,finance}。"""
    top = ("E5 Sportback 2025款 旗舰quattro型", 319900)
    rival = ("小米SU7 2025款 四驱Max版", 303900)
    params = [(30, 5), (50, 3), (30, 3)]
    ph = [
        "我想买E5 Sportback，帮我看看有哪些配置和价格？拿最贵的那个配置跟{rival}对比一下，"
        "再查查它最近成交价，算下它首付{r}%贷{y}年的月供，合适就约{s}试驾，{name}，{phone}",
        "帮我查E5 Sportback的全部配置，用最高配和{rival}做个对比，然后查它的近期成交价，"
        "算首付{r}%贷{y}年月供，可以的话约{s}试驾，{name}，{phone}",
        "E5 Sportback有哪些配置？最贵那款对比{rival}怎么样？它最近成交价多少？"
        "算下首付{r}%贷{y}年月供，合适就约{s}的试驾，{name}，{phone}",
    ]
    for i, (r, y) in enumerate(params):
        s = bk_slot(30 + i)
        q = ph[i].format(rival=rival[0], r=r, y=y, s=bk_text(s),
                         name=s["name"], phone=s["phone"])
        P("多跳", q, ["text_search", "competitor_compare", "query_transaction_price",
                      "calculate_finance", "book_test_drive"],
          gt_facts={"指导价": top[1], "对比价A": 289900, "对比价B": rival[1],
                    "成交均价": TX_AVG[top[0]], "月供": _monthly(top[1], r, y),
                    "试驾预约": "预约成功"},
          dep_edges=[["text_search", "competitor_compare"],
                     ["text_search", "query_transaction_price"],
                     ["text_search", "calculate_finance"]],
          difficulty="multi_hop_5",
          ground={"compare": (top[0], rival[0]), "tx": top[0],
                  "finance": (top[0], top[1], r, y, 4.5), "booking": s})


def mh_g():
    """3跳：虚构车型否认→替代车系优惠→金融。"""
    fake = [("星曜X7 2025款", "E5 Sportback", "E5 Sportback 2025款 先锋型", 235900, 30000),
            ("幻影V8 2026款", "帕萨特2024款", "帕萨特2024款 280TSI 商务版", 185900, 23000)]
    params = [(30, 3), (30, 5)]
    ph = [
        "帮我查下{fake}的价格和优惠，如果没有这款车，看看{series}有什么优惠，"
        "再算下{alt}首付{r}%贷{y}年月供",
        "查查{fake}卖多少钱？要是没这个车型，就帮我看看{series}的优惠，"
        "顺便算{alt}首付{r}%贷{y}年的月供",
    ]
    n = 0
    for fi, (fake_name, series, alt, alt_price, disc) in enumerate(fake):
        for j, (r, y) in enumerate(params):
            q = ph[(fi + j) % 2].format(fake=fake_name, series=series, alt=alt, r=r, y=y)
            P("多跳", q, ["text_search", "query_discount_policy", "calculate_finance"],
              gt_facts={"优惠总额": disc, "指导价": alt_price,
                        "月供": _monthly(alt_price, r, y)},
              difficulty="multi_hop_3",
              ground={"discount": series,
                      "finance": (alt, alt_price, r, y, 4.5)})
            n += 1
    # 补充 6 条变体（多车型替代）
    extra = [
        ("星曜X7 2025款", "E5 Sportback", "E5 Sportback 2025款 先锋Plus型", 269900, 30000, 30, 3),
        ("幻影V8 2026款", "小米SU7 2025款", "小米SU7 2025款 Pro版", 245900, 10000, 30, 5),
        ("星曜X7 2025款", "帕萨特2024款", "帕萨特2024款 330TSI 精英版", 205900, 23000, 30, 5),
        ("幻影V8 2026款", "E5 Sportback", "E5 Sportback 2025款 旗舰quattro型", 319900, 30000, 30, 3),
        ("星曜X7 2025款", "小米SU7 2025款", "小米SU7 2025款 四驱Max版", 303900, 10000, 30, 3),
        ("幻影V8 2026款", "帕萨特2024款", "帕萨特2024款 280TSI 商务版", 185900, 23000, 30, 5),
    ]
    for fake_name, series2, alt2, alt2_price, disc2, r, y in extra:
        q = (f"帮我看看{fake_name}这款车有没有？查不到的话看看{series2}有什么优惠，"
             f"再算下{alt2}首付{r}%贷{y}年月供")
        P("多跳", q, ["text_search", "query_discount_policy", "calculate_finance"],
          gt_facts={"优惠总额": disc2, "指导价": alt2_price,
                    "月供": _monthly(alt2_price, r, y)},
          difficulty="multi_hop_3",
          ground={"discount": series2,
                  "finance": (alt2, alt2_price, r, y, 4.5)})
        n += 1
    assert n == 10, n


def mh_h():
    """3跳：成交价→金融→试驾。"""
    trim = "E5 Sportback 2025款 先锋型"
    price, tx = 235900, 225000
    ph = [
        "{trim}最近成交价怎么样？按首付{r}%贷{y}年算算月供，合适的话帮我约{s}试驾，"
        "{name}，{phone}",
        "帮我查查{trim}近期的成交价格，再算下首付{r}%贷{y}年月供，可以就约{s}试驾，"
        "{name}，{phone}",
    ]
    for i in range(6):
        r, y = [(30, 3), (30, 5)][i % 2]
        s = bk_slot(40 + i)
        q = ph[i % 2].format(trim=trim, r=r, y=y, s=bk_text(s),
                             name=s["name"], phone=s["phone"])
        P("多跳", q, ["query_transaction_price", "calculate_finance", "book_test_drive"],
          gt_facts={"成交均价": tx, "月供": _monthly(price, r, y),
                    "试驾预约": "预约成功"},
          difficulty="multi_hop_3",
          ground={"tx": trim, "finance": (trim, price, r, y, 4.5), "booking": s})


def mh_i():
    """4跳：配置→成交价→优惠→试驾。"""
    trim = "E5 Sportback 2025款 先锋型"
    price, tx = 235900, 225000
    ph = [
        "帮我查下{trim}的价格，最近成交价怎么样？现在买有什么优惠？合适就约{s}试驾，"
        "{name}，{phone}",
        "查查{trim}的价格配置和近期成交价，再看看当前优惠，合适的话约{s}试驾，"
        "{name}，{phone}",
        "想知道{trim}的价格，以及最近成交行情和优惠力度，可以的话约{s}试驾，"
        "{name}，{phone}",
    ]
    for i in range(3):
        s = bk_slot(50 + i)
        q = ph[i].format(trim=trim, s=bk_text(s), name=s["name"], phone=s["phone"])
        P("多跳", q, ["text_search", "query_transaction_price",
                      "query_discount_policy", "book_test_drive"],
          gt_facts={"指导价": price, "成交均价": tx, "优惠总额": 30000,
                    "试驾预约": "预约成功"},
          difficulty="multi_hop_4",
          ground={"tx": trim, "discount": "E5 Sportback", "booking": s})


def mh_j():
    """3跳：web改款→优惠→金融。"""
    trim, price = "E5 Sportback 2025款 先锋quattro型", 269900
    ph = [
        "听说E5 Sportback要出2026改款了，帮我搜下有什么变化？我们店现在买E5有什么优惠？"
        "算下{trim}首付30%贷5年月供",
        "帮我搜搜E5 Sportback改款的消息，另外现在买E5优惠多少？"
        "{trim}首付30%贷5年月供是多少？",
        "E5 Sportback明年改款续航能到多少？帮我搜一下；现在买E5有什么优惠？"
        "顺便算{trim}首付30%贷5年月供",
    ]
    for i, q in enumerate(ph):
        q = q.format(trim=trim)
        P("多跳", q, ["web_search", "query_discount_policy", "calculate_finance"],
          gt_facts={"续航": "720km", "优惠总额": 30000, "指导价": price,
                    "月供": _monthly(price, 30, 5)},
          difficulty="multi_hop_3",
          ground={"web": ("news_e5_2026_facelift", "720km"),
                  "discount": "E5 Sportback", "finance": (trim, price, 30, 5, 4.5)})


def mh_k():
    """3跳：web召回→优惠→试驾。"""
    ph = [
        "听说帕萨特最近有召回，帮我查查怎么回事；现在买E5 Sportback有什么优惠？"
        "想约{s}试驾，{name}，{phone}",
        "网上说帕萨特有召回公告，帮我搜一下；E5 Sportback现在有什么优惠活动？"
        "顺便约{s}的试驾，{name}，{phone}",
        "帮我查查帕萨特召回涉及多少辆车；另外现在买E5 Sportback优惠多少？"
        "合适就约{s}试驾，{name}，{phone}",
    ]
    for i in range(3):
        s = bk_slot(60 + i)
        q = ph[i].format(s=bk_text(s), name=s["name"], phone=s["phone"])
        P("多跳", q, ["web_search", "query_discount_policy", "book_test_drive"],
          gt_facts={"召回数量": "3206", "优惠总额": 30000, "试驾预约": "预约成功"},
          difficulty="multi_hop_3",
          ground={"web": ("recall_passat_2024_018", "3206"),
                  "discount": "E5 Sportback", "booking": s})


def mh_l():
    """3跳：优惠→成交价→金融。"""
    combos = [
        ("帕萨特2024款", "帕萨特2024款 330TSI 精英版", 205900, 195000, 23000, (30, 3)),
        ("小米SU7 2025款", "小米SU7 2025款 四驱Max版", 303900, 303900, 10000, (30, 5)),
        ("E5 Sportback", "E5 Sportback 2025款 先锋型", 235900, 225000, 30000, (30, 3)),
    ]
    ph = [
        "现在买{series}的话有什么优惠？{trim}最近成交价怎么样？算下{trim}首付{r}%贷{y}年月供",
        "{series}目前有什么优惠？帮我查下{trim}近期成交价，再算算首付{r}%贷{y}年的月供",
    ]
    n = 0
    for ci, (series, trim, price, tx, disc, (r, y)) in enumerate(combos):
        for j in range(2):
            q = ph[(ci + j) % 2].format(series=series, trim=trim, r=r, y=y)
            P("多跳", q, ["query_discount_policy", "query_transaction_price",
                          "calculate_finance"],
              gt_facts={"优惠总额": disc, "成交均价": tx, "指导价": price,
                        "月供": _monthly(price, r, y)},
              difficulty="multi_hop_3",
              ground={"discount": series, "tx": trim,
                      "finance": (trim, price, r, y, 4.5)})
            n += 1
    assert n == 6, n


def mh_m():
    """4跳：对比→优惠→金融→试驾。"""
    a, b = "E5 Sportback 2025款 旗舰quattro型", "小米SU7 2025款 四驱Max版"
    va, vb = 289900, 303900
    params = [(30, 5), (50, 3)]
    ph = [
        "帮我对比{a}跟{b}，现在买E5有什么优惠？算下{a}首付{r}%贷{y}年月供，"
        "合适约{s}试驾，{name}，{phone}",
        "{a}和{b}怎么选？帮我对比下；E5现在优惠多少？再算{a}首付{r}%贷{y}年月供，"
        "可以就约{s}试驾，{name}，{phone}",
        "先对比{a}跟{b}，再查E5当前优惠，算{a}首付{r}%贷{y}年月供，"
        "合适的话约{s}试驾，{name}，{phone}",
    ]
    n = 0
    for i, (r, y) in enumerate(params):
        for j in range(3):
            s = bk_slot(70 + n)
            q = ph[j].format(a=a, b=b, r=r, y=y, s=bk_text(s),
                             name=s["name"], phone=s["phone"])
            P("多跳", q, ["competitor_compare", "query_discount_policy",
                          "calculate_finance", "book_test_drive"],
              gt_facts={"对比价A": va, "对比价B": vb, "优惠总额": 30000,
                        "指导价": 319900, "月供": _monthly(319900, r, y),
                        "试驾预约": "预约成功"},
              difficulty="multi_hop_4",
              ground={"compare": (a, b), "discount": "E5 Sportback",
                      "finance": (a, 319900, r, y, 4.5), "booking": s})
            n += 1
    assert n == 6, n


def mh_n():
    """5跳：虚构否认→对比→成交价→金融→试驾。"""
    a, b = "E5 Sportback 2025款 旗舰quattro型", "小米SU7 2025款 四驱Max版"
    ph = [
        "帮我查下幻影V8 2026款有没有这款车？没有的话对比下{a}和{b}，"
        "查查{a}最近成交价，算下它首付30%贷5年月供，合适约{s}试驾，{name}，{phone}",
        "幻影V8 2026款有卖吗？没有就帮我对比{a}跟{b}，再查{a}的近期成交价，"
        "算{a}首付30%贷5年月供，可以就约{s}试驾，{name}，{phone}",
    ]
    for i in range(2):
        s = bk_slot(80 + i)
        q = ph[i].format(a=a, b=b, s=bk_text(s), name=s["name"], phone=s["phone"])
        P("多跳", q, ["text_search", "competitor_compare", "query_transaction_price",
                      "calculate_finance", "book_test_drive"],
          gt_facts={"对比价A": 289900, "对比价B": 303900, "成交均价": 289500,
                    "指导价": 319900, "月供": _monthly(319900, 30, 5),
                    "试驾预约": "预约成功"},
          difficulty="multi_hop_5",
          ground={"compare": (a, b), "tx": a,
                  "finance": (a, 319900, 30, 5, 4.5), "booking": s})


def mh_o():
    """3跳：配置→成交价→金融。"""
    combos = [
        ("E5 Sportback 2025款 旗舰quattro型", 319900, 289500),
        ("E5 Sportback 2025款 先锋型", 235900, 225000),
        ("帕萨特2024款 330TSI 精英版", 205900, 195000),
        ("小米SU7 2025款 四驱Max版", 303900, 303900),
    ]
    params = [(30, 3), (30, 5), (50, 3)]
    ph = [
        "帮我查下{trim}的配置和价格，最近成交价怎么样？算下首付{r}%贷{y}年月供",
        "查查{trim}的价格，再看看近期成交价，算下首付{r}%贷{y}年的月供",
    ]
    n = 0
    for i, (trim, price, tx) in enumerate(combos):
        for j, (r, y) in enumerate(params):
            if i == 3 and j == 2:
                continue  # 11 条
            q = ph[(i + j) % 2].format(trim=trim, r=r, y=y)
            P("多跳", q, ["text_search", "query_transaction_price", "calculate_finance"],
              gt_facts={"指导价": price, "成交均价": tx, "月供": _monthly(price, r, y)},
              difficulty="multi_hop_3",
              ground={"tx": trim, "finance": (trim, price, r, y, 4.5)})
            n += 1
    assert n == 11, n


def mh_p():
    """3跳：优惠→对比→试驾。"""
    a, b = "E5 Sportback 2025款 旗舰quattro型", "帕萨特2024款 330TSI 精英版"
    va, vb = 0, 0
    for x, y, p1, p2 in CMP_PAIRS:
        if (x, y) == (a, b):
            va, vb = p1, p2
    ph = [
        "现在买E5有什么优惠？帮我对比下{a}和{b}，想约{s}试驾，{name}，{phone}",
        "E5现在有什么优惠活动？顺便对比{a}和{b}，合适的话约{s}试驾，{name}，{phone}",
        "先查查买E5的优惠，再对比下{a}跟{b}，可以就约{s}的试驾，{name}，{phone}",
    ]
    for i in range(3):
        s = bk_slot(90 + i)
        q = ph[i].format(a=a, b=b, s=bk_text(s), name=s["name"], phone=s["phone"])
        P("多跳", q, ["query_discount_policy", "competitor_compare", "book_test_drive"],
          gt_facts={"优惠总额": 30000, "对比价A": va, "对比价B": vb,
                    "试驾预约": "预约成功"},
          difficulty="multi_hop_3",
          ground={"discount": "E5 Sportback", "compare": (a, b), "booking": s})


# ---------------------------------------------------------------------------
# 失败恢复 62
# ---------------------------------------------------------------------------

def fr_build():
    def add(q, tools, facts=None, ac=False, inj=None, ground=None):
        P("失败恢复", q, tools, gt_facts=facts, accept_clarification=ac,
          fail_inject=inj, difficulty="recovery", ground=ground)

    # text_search
    add("帮我查下小米SU7 2025款四驱Max版的价格", ["text_search"],
        {"指导价": 303900}, inj=F("text_search", 1, "TIMEOUT"),
        ground={"price": "小米SU7 2025款 四驱Max版"})
    add("查一下小米SU7 2025款四驱Max版现在卖多少钱", ["text_search"],
        {"指导价": 303900}, inj=F("text_search", 1, "TIMEOUT"),
        ground={"price": "小米SU7 2025款 四驱Max版"})
    add("帮我查下E5 Sportback 2025款旗舰quattro型的配置", ["text_search"],
        {"指导价": 319900}, inj=F("text_search", 2, "TIMEOUT"),
        ground={"price": "E5 Sportback 2025款 旗舰quattro型"})
    add("查查E5 Sportback 2025款旗舰quattro型都有哪些配置", ["text_search"],
        {"指导价": 319900}, inj=F("text_search", 2, "TIMEOUT"),
        ground={"price": "E5 Sportback 2025款 旗舰quattro型"})
    add("查下E5 Sportback 2025款先锋Plus型的价格", ["text_search"],
        {"指导价": 269900}, inj=F("text_search", 1, "PARAM_ERROR"),
        ground={"price": "E5 Sportback 2025款 先锋Plus型"})
    add("帮我查下E5 Sportback 2025款先锋Plus型多少钱", ["text_search"],
        {"指导价": 269900}, inj=F("text_search", 1, "PARAM_ERROR"),
        ground={"price": "E5 Sportback 2025款 先锋Plus型"})
    add("帮我查下帕萨特2024款280TSI商务版的价格，如果第一次没查到就换个说法再查一次",
        ["text_search"], {"指导价": 185900}, inj=F("text_search", 1, "NOT_FOUND"),
        ground={"price": "帕萨特2024款 280TSI 商务版"})
    add("查下帕萨特2024款280TSI商务版卖多少钱", ["text_search"],
        {"指导价": 185900}, inj=F("text_search", 1, "NOT_FOUND"),
        ground={"price": "帕萨特2024款 280TSI 商务版"})

    # query_discount_policy
    add("现在买E5 Sportback有什么优惠？", ["query_discount_policy"],
        {"优惠总额": 30000}, inj=F("query_discount_policy", 1, "TIMEOUT"),
        ground={"discount": "E5 Sportback"})
    add("买E5 Sportback能享受哪些优惠？", ["query_discount_policy"],
        {"优惠总额": 30000}, inj=F("query_discount_policy", 1, "TIMEOUT"),
        ground={"discount": "E5 Sportback"})
    add("买帕萨特2024款现在有啥优惠？", ["query_discount_policy"],
        {"优惠总额": 23000}, inj=F("query_discount_policy", 1, "PARAM_ERROR"),
        ground={"discount": "帕萨特2024款"})
    add("帕萨特2024款现在有什么优惠活动？", ["query_discount_policy"],
        {"优惠总额": 23000}, inj=F("query_discount_policy", 1, "PARAM_ERROR"),
        ground={"discount": "帕萨特2024款"})
    add("理想L6现在有什么优惠？如果查不到优惠，帮我搜下它最近的价格",
        ["query_discount_policy", "web_search"], {"价格": "25.98万"},
        inj=F("query_discount_policy", 1, "NOT_FOUND"),
        ground={"web": ("rival_l6_price_2026", "25.98万")})
    add("理想L6有什么优惠活动？查不到的话搜搜它现在的行情",
        ["query_discount_policy", "web_search"], {"价格": "25.98万"},
        inj=F("query_discount_policy", 1, "NOT_FOUND"),
        ground={"web": ("rival_l6_price_2026", "25.98万")})
    add("帮我查下理想L6的优惠，没有的话就搜下它的最新价格",
        ["query_discount_policy", "web_search"], {"价格": "25.98万"},
        inj=F("query_discount_policy", 1, "NOT_FOUND"),
        ground={"web": ("rival_l6_price_2026", "25.98万")})

    # query_transaction_price
    add("E5 Sportback 2025款先锋型最近成交价怎么样？", ["query_transaction_price"],
        {"成交均价": 225000}, inj=F("query_transaction_price", 1, "TIMEOUT"),
        ground={"tx": "E5 Sportback 2025款 先锋型"})
    add("帮我查查E5 Sportback 2025款先锋型近期的成交价格", ["query_transaction_price"],
        {"成交均价": 225000}, inj=F("query_transaction_price", 1, "TIMEOUT"),
        ground={"tx": "E5 Sportback 2025款 先锋型"})
    add("E5 Sportback 2025款旗舰quattro型近90天成交均价是多少？",
        ["query_transaction_price"], {"成交均价": 289500},
        inj=F("query_transaction_price", 1, "PARAM_ERROR"),
        ground={"tx": "E5 Sportback 2025款 旗舰quattro型"})
    add("查下E5 Sportback 2025款旗舰quattro型90天内的成交价",
        ["query_transaction_price"], {"成交均价": 289500},
        inj=F("query_transaction_price", 1, "PARAM_ERROR"),
        ground={"tx": "E5 Sportback 2025款 旗舰quattro型"})
    add("小米SU7 2025款四驱Max版最近成交价怎么样？", ["query_transaction_price"],
        {"成交均价": 303900}, inj=F("query_transaction_price", 1, "NOT_FOUND"),
        ground={"tx": "小米SU7 2025款 四驱Max版"})
    add("帮我查下小米SU7 2025款四驱Max版的成交均价", ["query_transaction_price"],
        {"成交均价": 303900}, inj=F("query_transaction_price", 1, "NOT_FOUND"),
        ground={"tx": "小米SU7 2025款 四驱Max版"})

    # competitor_compare
    add("帮我对比一下E5 Sportback 2025款旗舰quattro型和小米SU7 2025款四驱Max版",
        ["competitor_compare"], {"对比价A": 289900, "对比价B": 303900},
        inj=F("competitor_compare", 1, "TIMEOUT"),
        ground={"compare": ("E5 Sportback 2025款 旗舰quattro型", "小米SU7 2025款 四驱Max版")})
    add("E5 Sportback 2025款旗舰quattro型跟小米SU7 2025款四驱Max版哪个好？",
        ["competitor_compare"], {"对比价A": 289900, "对比价B": 303900},
        inj=F("competitor_compare", 1, "TIMEOUT"),
        ground={"compare": ("E5 Sportback 2025款 旗舰quattro型", "小米SU7 2025款 四驱Max版")})
    add("帮我对比下小米SU7 2025款四驱Max版和E5 Sportback 2025款旗舰quattro型",
        ["competitor_compare"], {"对比价A": 303900, "对比价B": 289900},
        inj=F("competitor_compare", 1, "NOT_FOUND"),
        ground={"compare": ("小米SU7 2025款 四驱Max版", "E5 Sportback 2025款 旗舰quattro型")})
    add("小米SU7 2025款四驱Max版对比E5 Sportback 2025款旗舰quattro型怎么样？",
        ["competitor_compare"], {"对比价A": 303900, "对比价B": 289900},
        inj=F("competitor_compare", 1, "NOT_FOUND"),
        ground={"compare": ("小米SU7 2025款 四驱Max版", "E5 Sportback 2025款 旗舰quattro型")})

    # calculate_finance
    add("算下E5 Sportback 2025款旗舰quattro型首付30%贷5年的月供",
        ["calculate_finance"], {"指导价": 319900, "月供": _monthly(319900, 30, 5)},
        inj=F("calculate_finance", 1, "TIMEOUT"),
        ground={"finance": ("E5 Sportback 2025款 旗舰quattro型", 319900, 30, 5, 4.5)})
    add("帮我算算E5 Sportback 2025款旗舰quattro型首付30%贷款5年月供多少",
        ["calculate_finance"], {"指导价": 319900, "月供": _monthly(319900, 30, 5)},
        inj=F("calculate_finance", 1, "TIMEOUT"),
        ground={"finance": ("E5 Sportback 2025款 旗舰quattro型", 319900, 30, 5, 4.5)})
    add("算下帕萨特2024款280TSI商务版首付30%贷3年月供",
        ["calculate_finance"], {"指导价": 185900, "月供": _monthly(185900, 30, 3)},
        inj=F("calculate_finance", 1, "PARAM_ERROR"),
        ground={"finance": ("帕萨特2024款 280TSI 商务版", 185900, 30, 3, 4.5)})
    add("帕萨特2024款280TSI商务版首付30%贷3年月供是多少？",
        ["calculate_finance"], {"指导价": 185900, "月供": _monthly(185900, 30, 3)},
        inj=F("calculate_finance", 1, "PARAM_ERROR"),
        ground={"finance": ("帕萨特2024款 280TSI 商务版", 185900, 30, 3, 4.5)})
    for q in ["算下星曜X7 2025款首付30%贷3年月供",
              "帮我算算星曜X7 2025款首付30%贷3年的月供",
              "星曜X7 2025款首付30%贷3年月供多少？"]:
        add(q, ["calculate_finance"], ac=AC(
            "calculate_finance",
            "车型'星曜X7 2025款'不存在（目标不可达）；正确行为=告知不可达并反问替代车型"),
            inj=F("calculate_finance", 1, "NOT_FOUND"))

    # book_test_drive
    for i in range(2):
        s = bk_slot(100 + i)
        add(f"帮我约E5 Sportback试驾，{bk_text(s)}，我叫{s['name']}，电话{s['phone']}",
            ["book_test_drive"], {"试驾预约": "预约成功"},
            inj=F("book_test_drive", 1, "TIMEOUT"), ground={"booking": s})
    for i in range(2):
        s = bk_slot(102 + i)
        add(f"我想预约E5 Sportback试驾，{bk_text(s)}，{s['name']}，{s['phone']}",
            ["book_test_drive"], {"试驾预约": "预约成功"},
            inj=F("book_test_drive", 1, "PARAM_ERROR"), ground={"booking": s})
    for i in range(3):
        s = bk_slot(104 + i)
        add(f"我想约E5 Sportback试驾，北京朝阳北路店，{s['date']}{s['daypart']}{s['hour']}点，"
            f"我叫{s['name']}，电话{s['phone']}",
            ["book_test_drive"], ac=AC(
                "book_test_drive",
                "经销商'北京朝阳北路店'不在白名单（目标不可达）；正确行为=告知可选门店并反问确认"),
            inj=F("book_test_drive", 1, "NOT_FOUND"))

    # web_search
    add("帮我搜下理想L6现在什么价格", ["web_search"], {"价格": "25.98万"},
        inj=F("web_search", 1, "TIMEOUT"),
        ground={"web": ("rival_l6_price_2026", "25.98万")})
    add("查下理想L6最近的行情", ["web_search"], {"价格": "25.98万"},
        inj=F("web_search", 1, "TIMEOUT"),
        ground={"web": ("rival_l6_price_2026", "25.98万")})
    add("搜下海豹06的上市信息", ["web_search"], {"价格": "9.98万"},
        inj=F("web_search", 1, "PARAM_ERROR"),
        ground={"web": ("news_seal06_2026_launch", "9.98万")})
    add("帮我查查比亚迪海豹06上市的消息", ["web_search"], {"价格": "9.98万"},
        inj=F("web_search", 1, "PARAM_ERROR"),
        ground={"web": ("news_seal06_2026_launch", "9.98万")})
    for q in ["帮我搜下量子引擎汽车的消息", "网上有量子引擎汽车吗？帮我查查",
              "帮我搜搜量子引擎汽车相关的资讯"]:
        add(q, ["web_search"], inj=F("web_search", 1, "NOT_FOUND"))

    # follow_up_query
    add("查下陈静的跟进任务", ["follow_up_query"],
        {"任务1": "交车确认", "任务2": "回访"}, inj=F("follow_up_query", 1, "TIMEOUT"),
        ground={"fu": "陈静"})
    add("帮我看下陈静有哪些待跟进事项", ["follow_up_query"],
        {"任务1": "交车确认", "任务2": "回访"}, inj=F("follow_up_query", 1, "TIMEOUT"),
        ground={"fu": "陈静"})
    add("查下赵强的待办事项", ["follow_up_query"],
        {"任务1": "报价跟进", "任务2": "试驾提醒", "任务3": "保养提醒"},
        inj=F("follow_up_query", 1, "PARAM_ERROR"), ground={"fu": "赵强"})
    add("帮我看看赵强名下未完成的跟进任务", ["follow_up_query"],
        {"任务1": "报价跟进", "任务2": "试驾提醒", "任务3": "保养提醒"},
        inj=F("follow_up_query", 1, "PARAM_ERROR"), ground={"fu": "赵强"})
    for q in ["查下李伟的跟进任务", "帮我看看李伟有哪些待跟进事项", "李伟那边有什么要跟进的？"]:
        add(q, ["follow_up_query"], ac=AC(
            "follow_up_query",
            "客户'李伟'不在CRM中（目标不可达）；正确行为=告知未找到并请用户确认姓名"),
            inj=F("follow_up_query", 1, "NOT_FOUND"))

    # 纯澄清（无注入）：题面真实缺参
    add("帮我约个E5 Sportback试驾，11月28日下午3点，上汽奥迪上海郑隆奥翟店",
        ["book_test_drive"], ac=AC(
            "book_test_drive",
            "故意缺 customer_name/customer_phone；正确行为=反问补全，而非编造参数硬调"))
    add("我想约E5 Sportback的试驾，11月29日下午2点，上汽奥迪上海大宁音乐广场店",
        ["book_test_drive"], ac=AC(
            "book_test_drive",
            "故意缺 customer_name/customer_phone；正确行为=反问补全，而非编造参数硬调"))
    add("我想约E5 Sportback试驾，11月29日下午2点，我叫徐强，电话13412340006",
        ["book_test_drive"], ac=AC(
            "book_test_drive",
            "题面未提供经销商；正确行为=列出白名单门店并反问用户选择"))
    add("帮我看下我的跟进任务", ["follow_up_query"], ac=AC(
        "follow_up_query",
        "题面未提供客户姓名；正确行为=反问客户姓名，而非编造姓名硬调"))
    add("我名下有哪些待办？", ["follow_up_query"], ac=AC(
        "follow_up_query",
        "题面未提供客户姓名；正确行为=反问客户姓名，而非编造姓名硬调"))
    add("算下E5 Sportback首付30%贷3年的月供", ["calculate_finance"], ac=AC(
        "calculate_finance",
        "题面未指明具体配置（E5 Sportback 有 4 款配置）；正确行为=列出配置并反问确认后再计算"))
    add("算下帕萨特2024款首付30%贷3年的月供", ["calculate_finance"], ac=AC(
        "calculate_finance",
        "题面未指明具体配置（帕萨特2024款有 2 款配置）；正确行为=列出配置并反问确认后再计算"))
    add("帮我算算小米SU7首付30%贷5年月供", ["calculate_finance"], ac=AC(
        "calculate_finance",
        "题面未指明具体配置；正确行为=列出配置并反问确认后再计算"))
    add("约E5 Sportback试驾，上汽奥迪上海大宁音乐广场店，我叫马丽，电话13312340007",
        ["book_test_drive"], ac=AC(
            "book_test_drive",
            "题面未提供预约时间；正确行为=给出可选时间段并反问用户选择"))


# ---------------------------------------------------------------------------
# 新工具 50
# ---------------------------------------------------------------------------

def nt_build():
    web_topics = [
        # (questions, facts, ground item id)
        (["听说帕萨特最近有召回，涉及多少辆车？",
          "帮我查查帕萨特召回的事",
          "帕萨特的召回公告说了什么？",
          "网上说帕萨特要召回，是真的吗？",
          "帮我搜一下帕萨特最近是不是有召回"],
         {"召回数量": "3206"}, "recall_passat_2024_018"),
        (["E5 Sportback明年要改款吗？续航能到多少？",
          "听说E5要出800V改款了，帮我查查",
          "E5 Sportback 2026改款有什么变化？",
          "帮我搜下E5改款的最新消息",
          "E5 Sportback改款后快充和续航怎么样？"],
         {"续航": "720km"}, "news_e5_2026_facelift"),
        (["理想L6现在什么价？",
          "帮我搜下理想L6的价格",
          "理想L6最近降价了吗？",
          "网上理想L6卖多少钱？",
          "理想L6现在有优惠吗？帮我搜搜"],
         {"价格": "25.98万"}, "rival_l6_price_2026"),
        (["极氪007现在多少钱？",
          "帮我查查极氪007的价格",
          "极氪007最近什么行情？",
          "网上极氪007报价多少？"],
         {"价格": "20.99万"}, "rival_zeekr007_price_2026"),
        (["上海现在有什么以旧换新补贴？",
          "帮我查下上海以旧换新的政策",
          "上海购车以旧换新补贴多少？",
          "上海以旧换新有什么优惠？",
          "上海置换买电车有补贴吗？帮我搜下"],
         {"补贴": "5000元"}, "policy_sh_tradein_2026"),
        (["海豹06上市了吗？什么价格？",
          "帮我搜下海豹06的上市信息",
          "比亚迪海豹06卖多少钱？",
          "海豹06 2026款什么价？"],
         {"价格": "9.98万"}, "news_seal06_2026_launch"),
    ]
    for questions, facts, item_id in web_topics:
        for q in questions:
            P("新工具", q, ["web_search"], gt_facts=dict(facts),
              difficulty="easy", ground={"web": (item_id, list(facts.values())[0])})

    fu_topics = [
        # (customer, questions, facts)
        ("陈静", ["帮我查下陈静的跟进任务",
                  "陈静那边有什么待办？",
                  "看看陈静还有哪些事要跟进",
                  "陈静的跟进任务有哪些？",
                  "帮我看看陈静最近要跟进什么",
                  "查一下陈静名下未完成的跟进事项",
                  "我这周要跟进陈静的哪些事？",
                  "陈静有什么任务快到期了？"],
         {"任务1": "交车确认", "任务2": "回访"}),
        ("赵强", ["帮我查下赵强的跟进任务",
                  "赵强那边有哪些待办事项？",
                  "看看赵强还有什么要跟进的",
                  "赵强的跟进任务有哪些？",
                  "查一下赵强名下未完成的跟进事项",
                  "赵强最近要跟进什么？",
                  "我这周要跟进赵强的哪些事？",
                  "赵强有什么任务快到期了？"],
         {"任务1": "报价跟进", "任务2": "试驾提醒", "任务3": "保养提醒"}),
        ("刘洋", ["帮我查下刘洋的跟进任务",
                  "刘洋那边有什么待办？",
                  "看看刘洋还有哪些事要跟进",
                  "刘洋的跟进任务有哪些？",
                  "查一下刘洋名下未完成的跟进事项",
                  "我这周要跟进刘洋的哪些事？"],
         {"任务1": "回访"}),
    ]
    for customer, questions, facts in fu_topics:
        for q in questions:
            P("新工具", q, ["follow_up_query"], gt_facts=dict(facts),
              difficulty="easy", ground={"fu": customer})


# ---------------------------------------------------------------------------
# 回归 38
# ---------------------------------------------------------------------------

def rg_build():
    # text_search 7
    ts_ph = ["{t}现在卖多少钱？", "{t}的官方指导价是多少", "{t}目前报价多少",
             "帮我看看{t}现在什么价", "{t}裸车价多少", "查下{t}卖多少钱", "{t}现在落地多少"]
    ts_trims = ["E5 Sportback 2025款 先锋quattro型", "E5 Sportback 2025款 先锋Plus型",
                "E5 Sportback 2025款 旗舰quattro型", "帕萨特2024款 280TSI 商务版",
                "帕萨特2024款 330TSI 精英版", "小米SU7 2025款 四驱Max版",
                "小米SU7 2025款 Pro版"]
    for i, t in enumerate(ts_trims):
        q = ts_ph[i].format(t=t)
        P("回归", q, ["text_search"], gt_facts={"指导价": PRICE_OF[t]},
          difficulty="easy", ground={"price": t})

    # query_discount_policy 6
    dp_ph = [("E5 Sportback", "现在买E5 Sportback的话有什么优惠？"),
             ("E5 Sportback", "E5 Sportback目前有哪些优惠活动？"),
             ("帕萨特2024款", "现在买帕萨特2024款有什么优惠？"),
             ("帕萨特2024款", "帕萨特2024款目前有哪些优惠活动？"),
             ("小米SU7 2025款", "现在买小米SU7 2025款的话有什么优惠？"),
             ("小米SU7 2025款", "小米SU7 2025款目前有哪些优惠活动？")]
    for series, q in dp_ph:
        P("回归", q, ["query_discount_policy"],
          gt_facts={"优惠总额": DISCOUNT_NATIONAL[series]},
          difficulty="easy", ground={"discount": series})

    # query_transaction_price 5
    tp_ph = [
        ("E5 Sportback 2025款 旗舰quattro型", "E5 Sportback 2025款旗舰quattro型最近成交价怎么样？"),
        ("E5 Sportback 2025款 先锋型", "E5 Sportback 2025款先锋型近期的成交行情如何？"),
        ("帕萨特2024款 330TSI 精英版", "帕萨特2024款330TSI精英版最近成交价怎么样？"),
        ("小米SU7 2025款 四驱Max版", "小米SU7 2025款四驱Max版最近的成交价格是多少？"),
        ("E5 Sportback 2025款 旗舰quattro型", "E5 Sportback 2025款旗舰quattro型近30天成交均价是多少？"),
    ]
    for t, q in tp_ph:
        P("回归", q, ["query_transaction_price"], gt_facts={"成交均价": TX_AVG[t]},
          difficulty="easy", ground={"tx": t})

    # competitor_compare 5
    cc_pairs = [("E5 Sportback 2025款 旗舰quattro型", "小米SU7 2025款 四驱Max版"),
                ("E5 Sportback 2025款 旗舰quattro型", "帕萨特2024款 330TSI 精英版"),
                ("小米SU7 2025款 四驱Max版", "E5 Sportback 2025款 旗舰quattro型")]
    cc_ph = ["{a}和{b}怎么选？", "帮我做个{a}和{b}的对比分析", "{a}跟{b}哪个更值得买？",
             "纠结{a}还是{b}", "{a}跟{b}做个对比看看"]
    pair_price = {}
    for a, b, va, vb in CMP_PAIRS:
        pair_price[(a, b)] = (va, vb)
        pair_price[(b, a)] = (vb, va)
    n = 0
    for pi, (a, b) in enumerate(cc_pairs):
        va, vb = pair_price[(a, b)]
        for j in range([2, 2, 1][pi]):
            q = cc_ph[n].format(a=a, b=b)
            P("回归", q, ["competitor_compare"], gt_facts={"对比价A": va, "对比价B": vb},
              difficulty="easy", ground={"compare": (a, b)})
            n += 1
    assert n == 5, n

    # calculate_finance 8
    cf_combos = [("E5 Sportback 2025款 先锋quattro型", 269900, 20, 3),
                 ("E5 Sportback 2025款 先锋Plus型", 269900, 30, 5),
                 ("E5 Sportback 2025款 旗舰quattro型", 319900, 30, 3),
                 ("帕萨特2024款 280TSI 商务版", 185900, 30, 5),
                 ("帕萨特2024款 330TSI 精英版", 205900, 50, 3),
                 ("小米SU7 2025款 四驱Max版", 303900, 30, 5),
                 ("小米SU7 2025款 Pro版", 245900, 40, 3)]
    cf_ph = ["{t}首付{r}%贷{y}年月供多少？", "帮我算下{t}首付{r}%贷{y}年的月供",
             "算算{t}首付{r}%贷款{y}年的月供是多少"]
    n = 0
    for t, price, r, y in cf_combos:
        q = cf_ph[n % 3].format(t=t, r=r, y=y)
        P("回归", q, ["calculate_finance"],
          gt_facts={"指导价": price, "月供": _monthly(price, r, y)},
          difficulty="easy", ground={"finance": (t, price, r, y, 4.5)})
        n += 1
    # 默认方案（不传首付/年限）
    q = "帮我算下帕萨特2024款280TSI商务版的贷款方案"
    P("回归", q, ["calculate_finance"],
      gt_facts={"指导价": 185900, "月供": _monthly(185900, 30, 3)},
      difficulty="easy", ground={"finance": ("帕萨特2024款 280TSI 商务版", 185900, 30, 3, 4.5)})

    # book_test_drive 7
    for i in range(7):
        s = bk_slot(200 + i)
        q = (f"帮我约个E5 Sportback的试驾，{bk_text(s)}，"
             f"我叫{s['name']}，电话{s['phone']}")
        P("回归", q, ["book_test_drive"], gt_facts={"试驾预约": "预约成功"},
          difficulty="easy", ground={"booking": s})


# ---------------------------------------------------------------------------
# 红线断言
# ---------------------------------------------------------------------------

def r1_check(prompts: list, v121_questions: list, subset: list):
    """R1：与 v1.2.1 + v1.3_rl 精确 + 归一化零重叠（双向子串）。"""
    violations = []
    eval_texts = [q["question"] for q in v121_questions]
    subset_texts = [q["question"] for q in subset]

    def norm(s):
        return _norm(s)

    def overlap(a, b):
        na, nb = norm(a), norm(b)
        return na == nb or na in nb or nb in na

    for p in prompts:
        for eq in eval_texts:
            if overlap(p["question"], eq):
                violations.append(f"[prompts vs v1.2.1] {p['id']} '{p['question'][:40]}' ~ '{eq[:40]}'")
    for p in prompts:
        for sq in subset_texts:
            if overlap(p["question"], sq):
                violations.append(f"[prompts vs v1.3_rl] {p['id']} '{p['question'][:40]}' ~ '{sq[:40]}'")
    for sq in subset_texts:
        for eq in eval_texts:
            if overlap(sq, eq):
                violations.append(f"[v1.3_rl vs v1.2.1] '{sq[:40]}' ~ '{eq[:40]}'")
    return violations


def _web_snippet(item_id: str) -> str:
    for iid, sn in WEB_ITEMS:
        if iid == item_id:
            return sn
    return ""


def r2_check_ground(prompts: list, ground_map: dict):
    """R2（生成期，ground 上下文全量校验）：gt_facts 与 mock 数据一致。"""
    violations = []
    bk_tool = BookTestDriveTool()
    fu_tool = FollowUpQueryTool()
    for p in prompts:
        g = ground_map.get(p["id"]) or {}
        facts = p["gt_facts"]
        if "price" in g:
            if facts.get("指导价") != PRICE_OF.get(g["price"]):
                violations.append(f"{p['id']} 指导价 {facts.get('指导价')} != mock {PRICE_OF.get(g['price'])}")
        if "discount" in g:
            if facts.get("优惠总额") != DISCOUNT_NATIONAL.get(g["discount"]):
                violations.append(f"{p['id']} 优惠总额 {facts.get('优惠总额')} != mock {DISCOUNT_NATIONAL.get(g['discount'])}")
        if "finance" in g:
            _, price, r, y, rate = g["finance"]
            if "指导价" in facts and facts.get("指导价") != price:
                violations.append(f"{p['id']} finance 指导价 {facts.get('指导价')} != {price}")
            if facts.get("月供") != _monthly(price, r, y, rate):
                violations.append(f"{p['id']} 月供 {facts.get('月供')} != 复算 {_monthly(price, r, y, rate)}")
        if "tx" in g:
            if facts.get("成交均价") != TX_AVG.get(g["tx"]):
                violations.append(f"{p['id']} 成交均价 {facts.get('成交均价')} != mock {TX_AVG.get(g['tx'])}")
        if "compare" in g:
            a, b = g["compare"]
            va = vb = None
            for x, y, p1, p2 in CMP_PAIRS:
                if (x, y) == (a, b):
                    va, vb = p1, p2
                elif (x, y) == (b, a):
                    vb, va = p1, p2
            if facts.get("对比价A") != va or facts.get("对比价B") != vb:
                violations.append(f"{p['id']} 对比价 {facts} != mock ({va},{vb})")
        if "web" in g:
            iid, val = g["web"]
            if str(val) not in _web_snippet(iid):
                violations.append(f"{p['id']} web 事实 '{val}' 不在摘要 {_web_snippet(iid)[:40]}")
        if "fu" in g:
            customer = g["fu"]
            for k, v in facts.items():
                if v not in FU_CUSTOMERS.get(customer, []):
                    violations.append(f"{p['id']} 任务事实 {v} 不在 {customer} 任务表")
            r = fu_tool.run({"customer_name": customer})
            if r.status.value == "success":
                text = r.to_dict().get("text", "")
                for v in facts.values():
                    if v not in text:
                        violations.append(f"{p['id']} 任务事实 '{v}' 不在工具返回文本")
        if "booking" in g:
            s = g["booking"]
            r = bk_tool.run({"car_series": "E5 Sportback", "dealer": s["dealer"],
                             "appointment_time": s["iso"],
                             "customer_name": s["name"], "customer_phone": s["phone"]})
            if r.status.value != "success":
                violations.append(f"{p['id']} 试驾 dry-run 失败: {r.to_dict().get('text','')[:60]}")
            elif facts.get("试驾预约") != "预约成功":
                violations.append(f"{p['id']} 试驾预约 fact 缺失")
    return violations


PHONE_RE = re.compile(r"1[3-9]\d{9}")


def _calc_missing(question: str, tool: str, expected_tools: list) -> bool:
    """题面是否真实缺少 target 工具的必填信息（R3 判定）。

    例外（不算缺失）：
    - calculate_finance 的候选串为空：操作数在题面他处已显式给出（如
      「{trim}最近成交价…再算下首付…」），正则无法回填不算缺参；
    - 候选为「顶配/它」等代指且 expected 含 text_search：发现链里
      操作数由搜索观察决定（与 verify 的 prior_observation 溯源同口径）。
    """
    if tool == "book_test_drive":
        has_phone = bool(PHONE_RE.search(question))
        has_dealer = any(d in question for d in DEALERS)
        has_time = bool(re.search(r"\d{1,2}月\d{1,2}日", question))
        return not (has_phone and has_dealer and has_time)
    if tool == "follow_up_query":
        return not any(name in question for name in FU_CUSTOMERS)
    if tool == "calculate_finance":
        m = re.search(r"(?:算下|算一下|算算|帮我算算)(.*?)(?:首付|的月供|贷款方案)", question)
        if m:
            cand = m.group(1).strip()
        else:
            m = re.match(r"([^，。？!]{0,60}?)(?:首付)", question)
            if not m:
                return False
            cand = m.group(1).strip()
        hits = [t for t in TRIM_NAMES if score_match(cand, t) >= 2.5]
        if not hits and (not cand or "text_search" in expected_tools):
            return False
        return len(hits) != 1  # 0 个=目标不可达；>1 个=配置歧义
    return False


def r3_check(prompts: list):
    """R3：accept_clarification 语义与 v1.2.1 同口径。"""
    violations = []
    for p in prompts:
        ac = p["accept_clarification"]
        if ac and isinstance(ac, dict):
            target = ac.get("target_tool")
            if target not in p["expected_tools"]:
                violations.append(f"{p['id']} accept target {target} 不在 expected_tools")
            elif not _calc_missing(p["question"], target, p["expected_tools"]):
                violations.append(f"{p['id']} accept={target} 但题面不缺该工具必填信息")
        elif ac:
            violations.append(f"{p['id']} accept_clarification 非字典形态（v1.2.1 口径为 {target_tool, reason}）")
        else:
            for t in p["expected_tools"]:
                if _calc_missing(p["question"], t, p["expected_tools"]):
                    violations.append(f"{p['id']} 无 accept 标注但 {t} 缺必填信息")
    return violations


# ---------------------------------------------------------------------------
# R2 检查模式（对落盘产物正则重推导，无 ground 上下文）
# ---------------------------------------------------------------------------

def r2_check_standalone(prompts: list):
    violations, skipped = [], 0

    def trim_in_question(q):
        hits = [t for _, t, _ in TRIMS if score_match(t, q) >= 2.5]
        # 取唯一强匹配
        return hits[0] if len(hits) == 1 else None

    for p in prompts:
        facts, q = p["gt_facts"], p["question"]
        if not facts:
            continue
        deps = {(pre, suc) for pre, suc in (p.get("dep_edges") or [])}
        if "指导价" in facts:
            t = trim_in_question(q)
            if t is None:
                skipped += 1  # 顶配/替代类无显式车名，生成期 ground 已校验
            elif ("text_search", "calculate_finance") in deps:
                skipped += 1  # 发现链：金融操作数由搜索决定，题面车名不可套用
            elif facts["指导价"] != PRICE_OF[t]:
                violations.append(f"{p['id']} 指导价 {facts['指导价']} != mock {PRICE_OF[t]}")
        if "优惠总额" in facts:
            series_hits = [k for k in DISCOUNT_NATIONAL if _norm(k) in _norm(q)]
            if series_hits and facts["优惠总额"] not in {
                    DISCOUNT_NATIONAL[k] for k in series_hits}:
                violations.append(
                    f"{p['id']} 优惠总额 {facts['优惠总额']} != mock 任一题面车系 {series_hits}")
        if "月供" in facts:
            m = re.search(r"首付(\d+)%[^，。]{0,30}?贷(?:款)?(\d+)年", q)
            if m and ("text_search", "calculate_finance") in deps:
                skipped += 1  # 发现链同 指导价
            elif m:
                t = trim_in_question(q)
                if t:
                    price = PRICE_OF[t]
                    if facts["月供"] != _monthly(price, int(m.group(1)), int(m.group(2))):
                        violations.append(f"{p['id']} 月供 {facts['月供']} != 复算 {_monthly(price, int(m.group(1)), int(m.group(2)))}")
                else:
                    skipped += 1
        if "成交均价" in facts:
            if ("text_search", "query_transaction_price") in deps:
                skipped += 1  # 发现链同 指导价
            else:
                t = trim_in_question(q)
                if t and t in TX_AVG and facts["成交均价"] != TX_AVG[t]:
                    violations.append(f"{p['id']} 成交均价 {facts['成交均价']} != mock {TX_AVG[t]}")
        for k, v in facts.items():
            if k in ("指导价", "优惠总额", "月供", "成交均价", "对比价A", "对比价B"):
                continue
            if str(v) not in "".join(sn for _, sn in WEB_ITEMS) and v not in (
                    t for tasks in FU_CUSTOMERS.values() for t in tasks) and v != "预约成功" and v != "没有未完成":
                violations.append(f"{p['id']} 事实 {k}={v} 无法在 mock 中溯源")
    return violations, skipped


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    check_mode = "--check" in sys.argv

    subset = build_subset()

    if check_mode:
        prompts = [json.loads(l) for l in OUT_JSONL.read_text(encoding="utf-8").splitlines()]
        v121 = json.load(open(EVAL_DATA / "questions_v1.2.1.json", encoding="utf-8"))["questions"]
        v13 = json.load(open(OUT_SUBSET, encoding="utf-8"))["questions"]
        r1 = r1_check(prompts, v121, v13)
        r2, skipped = r2_check_standalone(prompts)
        r3 = r3_check(prompts)
        report = {
            "mode": "check", "generated_at": None,
            "r1_violations": r1, "r2_violations": r2, "r2_skipped": skipped,
            "r3_violations": r3,
            "pass": not (r1 or r2 or r3),
        }
        OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"R1 违规 {len(r1)} / R2 违规 {len(r2)} (跳过 {skipped}) / R3 违规 {len(r3)}")
        for v in (r1 + r2 + r3)[:30]:
            print(" -", v)
        sys.exit(0 if report["pass"] else 1)

    # 生成训练集
    mh_a(); mh_b(); mh_c(); mh_d(); mh_e(); mh_f(); mh_g(); mh_h(); mh_i()
    mh_j(); mh_k(); mh_l(); mh_m(); mh_n(); mh_o(); mh_p()
    fr_build(); nt_build(); rg_build()

    # 分配 id 并剥离内部 ground 字段
    bucket_codes = {"多跳": "mh", "失败恢复": "fr", "新工具": "nt", "回归": "rg"}
    seq = {}
    prompts = []
    ground_map = {}
    for bucket, d, ground in PROMPTS:
        seq[bucket] = seq.get(bucket, 0) + 1
        d["id"] = f"rl_{bucket_codes[bucket]}_{seq[bucket]:03d}"
        d["bucket"] = bucket
        ground_map[d["id"]] = ground
        prompts.append(d)
    for d in prompts:
        d.pop("_ground", None)

    counts = {}
    for d in prompts:
        counts[d["bucket"]] = counts.get(d["bucket"], 0) + 1
    print("配比:", counts, "合计:", len(prompts))
    assert len(prompts) == 250, len(prompts)
    assert counts == {"多跳": 100, "失败恢复": 62, "新工具": 50, "回归": 38}, counts

    # 题面去重
    norms = [_norm(d["question"]) for d in prompts]
    dup = len(norms) - len(set(norms))
    assert dup == 0, f"训练集题面重复 {dup} 条"

    # 冻结子集（先于红线断言落盘）
    OUT_SUBSET.write_text(json.dumps({
        "metadata": {
            "total_questions": 15, "scenarios": 3,
            "question_types": ["单跳", "失败恢复"],
            "note": "RL 专属评测子集（2026-11-18 冻结）：新工具 8 + 失败恢复 7；"
                    "由 reward 函数判分（verify 68 题 harness 不覆盖新工具）；"
                    "训练集实体组合与本子集隔离，冻结后不得再修改",
        },
        "questions": subset,
        "scenario_distribution": {"新工具-网页搜索": 4, "新工具-跟进任务": 4, "失败恢复": 7},
        "type_distribution": {"单跳": 8, "失败恢复": 7},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 训练集落盘
    DATA_DIR.mkdir(exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for d in prompts:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # 红线断言
    v121 = json.load(open(EVAL_DATA / "questions_v1.2.1.json", encoding="utf-8"))["questions"]
    r1 = r1_check(prompts, v121, subset)
    r2_full = r2_check_ground(prompts, ground_map)
    r3 = r3_check(prompts)
    # 补跑独立模式 R2（对落盘条目）做交叉验证
    r2_alt, skipped = r2_check_standalone(prompts)
    r2_alt = [v for v in r2_alt]

    report = {
        "generated_at": "2026-11-18",
        "counts": counts, "total": len(prompts),
        "subset_frozen": str(OUT_SUBSET),
        "r1": {"violations": r1, "pass": not r1,
               "note": "与 questions_v1.2.1(68) + questions_v1.3_rl(15) 精确+归一化(小写去空格)零重叠，双向子串级"},
        "r2": {"violations": r2_full, "pass": not r2_full,
               "note": "gt_facts 与 mock 数据接地一致（生成期 ground 全量校验）",
               "standalone_crosscheck": {"violations": r2_alt, "skipped": skipped,
                                         "pass": not r2_alt}},
        "r3": {"violations": r3, "pass": not r3,
               "note": "accept_clarification 与 v1.2.1 同口径：{target_tool, reason} 字典，题面真实缺参才豁免"},
        "pass": not (r1 or r2_full or r3),
    }
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"R1 违规 {len(r1)} / R2 违规 {len(r2_full)}（独立交叉 {len(r2_alt)}，跳过 {skipped}）/ R3 违规 {len(r3)}")
    for v in (r1 + r2_full + r3)[:30]:
        print(" -", v)
    print("PASS" if report["pass"] else "FAIL")
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
