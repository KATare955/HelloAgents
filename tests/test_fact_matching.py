"""_fact_in_text GT 事实匹配容差单测（2026-11-21 多跳根因诊断修复）

双指标验收（校准集方法论）：
- 正例：合理舍入/单位换算/状态同义必须命中（防口径过严误伤自然话术）
- 负例：错误数字/否定预约/跨量级误命中必须拒绝（防口径过松放走真错误）
- 隔离：参数溯源 _value_in_text/_trace_args_source 保持精确口径不受影响
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.baseline_eval import _fact_in_text, _value_in_text, _trace_args_source


# --- 正例：合理容差必须命中 ---

def test_exact_still_hits():
    assert _fact_in_text("4758.02", "月供4758.02元")


def test_integer_rounding():
    assert _fact_in_text("4758.02", "月供大约4758元")


def test_one_decimal_rounding():
    assert _fact_in_text("4174.73", "月供约4174.7元")


def test_wan_to_yuan():
    assert _fact_in_text("269900", "指导价26.99万元")


def test_yuan_to_wan():
    assert _fact_in_text("21.99", "指导价219900元")


def test_thousands_separator():
    assert _fact_in_text("269900", "指导价269,900元")


def test_booking_ok_synonym():
    assert _fact_in_text("预约成功", "试驾这边帮您安排好了")
    assert _fact_in_text("预约成功", "已经约好了，周一到店")


def test_booking_fail_synonym():
    assert _fact_in_text("预约失败", "抱歉，未能预约成功")


def test_non_numeric_fact():
    assert _fact_in_text("上汽奥迪上海郑隆奥翟", "地点在上汽奥迪上海郑隆奥翟")


# --- 负例：真错误必须拒绝 ---

def test_wrong_number_rejected():
    assert not _fact_in_text("4758.02", "月供大概5620元")


def test_hundred_rounding_rejected():
    # 百位舍入偏差 0.88% > 0.5% 容差，拒绝
    assert not _fact_in_text("4758.02", "月供4800元左右")


def test_prefix_overlap_rejected():
    # token 级比对：14758 不是 4758.02 的舍入
    assert not _fact_in_text("4758.02", "总价14758元")


def test_booking_negation_rejected():
    assert not _fact_in_text("预约成功", "时间还没约好，请确认")
    assert not _fact_in_text("预约成功", "目前无法预约")


def test_non_numeric_fact_absent():
    assert not _fact_in_text("上汽奥迪上海郑隆奥翟", "地点在另一家店")


# --- 隔离：参数溯源口径不受影响 ---

def test_value_in_text_still_strict():
    # 参数溯源仍精确：舍入值不算"出现在题面/obs"
    assert not _value_in_text("4758.02", "月供大约4758元")


def test_trace_args_source_unaffected():
    src = _trace_args_source(
        "book_test_drive",
        {"dealer": "凭空 invented 店名", "customer_phone": "13800000000"},
        "帮我预约试驾",
        ['{"dealer": "上汽奥迪上海郑隆奥翟"}'],
    )
    assert src["dealer"] == "fabricated"
    assert src["customer_phone"] == "fabricated"