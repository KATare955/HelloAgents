"""车型名称健壮匹配工具（供各 Mock 工具共享）

背景：各工具的 mock_data key 粒度不一致（有的带年款、有的带具体配置、有的只有车系），
而用户/模型的查询词往往带有配置后缀（如"帕萨特2024款280TSI商务版"）或查询意图词
（如"E5 Sportback 价格"）。原始的单向子串匹配（query in key）会在这些情况下漏检，
导致检索返回空、下游工具链断裂（典型如金融测算拿不到车价）。

匹配策略（优先级递减）：
1. 精确匹配：归一化（小写+去空格）后完全相等
2. 双向子串：query ⊆ key 或 key ⊆ query（处理查询比 key 长/短）
3. token 重叠：分词后核心 token 重叠（处理 query 与 key 后缀分叉，
   如"先锋型" vs "2025款"，或查询夹带"价格/配置"等意图词）
"""

import re
from typing import Iterable, List, Optional

# 归一化：合并所有空白字符
_WS = re.compile(r"\s+")
# 分词：英文+数字连写串（如 su7、2025）、中文单字
_TOKEN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def _norm(s: str) -> str:
    """归一化：转小写 + 去除所有空格"""
    return _WS.sub("", s.lower())


def _tokens(s: str) -> set:
    """分词：英文数字串按整体切分，中文按单字切分"""
    return set(_TOKEN.findall(s.lower()))


def is_match(query: str, key: str, min_overlap: int = 2) -> bool:
    """判断单个 query 与 key 是否匹配（三级策略）

    Args:
        query: 用户/模型查询词
        key: mock_data 的键
        min_overlap: token 重叠匹配所需的最小重叠 token 数

    Returns:
        是否匹配
    """
    qn, kn = _norm(query), _norm(key)
    if not qn or not kn:
        return False

    # 1 & 2. 精确匹配 or 双向子串
    if qn == kn or qn in kn or kn in qn:
        return True

    # 3. token 重叠（重叠数达标，且覆盖 key 一半以上 token）
    qt, kt = _tokens(query), _tokens(key)
    if not kt:
        return False
    overlap = len(qt & kt)
    return overlap >= min_overlap and overlap / len(kt) >= 0.5


def match_key(query: str, keys: Iterable[str], min_overlap: int = 2) -> Optional[str]:
    """从 keys 中找出与 query 最匹配的单个 key

    优先级：精确 > 双向子串（取最具体/最长）> token 重叠（取覆盖率最高）。

    Args:
        query: 用户/模型查询词
        keys: 候选键集合（如 mock_data.keys()）
        min_overlap: token 重叠匹配所需的最小重叠 token 数

    Returns:
        最佳匹配的 key；若无匹配返回 None
    """
    if not query:
        return None
    keys = list(keys)
    qn = _norm(query)

    # 1. 精确匹配
    for key in keys:
        if _norm(key) == qn:
            return key

    # 2. 双向子串：取归一化后最长的 key（最具体）
    subs = [k for k in keys if qn in _norm(k) or _norm(k) in qn]
    if subs:
        return max(subs, key=lambda k: len(_norm(k)))

    # 3. token 重叠：取覆盖率最高的 key
    qt = _tokens(query)
    best, best_score = None, 0.0
    for key in keys:
        kt = _tokens(key)
        if not kt:
            continue
        overlap = len(qt & kt)
        score = overlap / len(kt)
        if overlap >= min_overlap and score >= 0.5 and score > best_score:
            best, best_score = key, score
    return best


def match_keys(query: str, keys: Iterable[str], min_overlap: int = 2) -> List[str]:
    """从 keys 中找出所有与 query 匹配的 key（用于聚合型检索，如 text_search）

    Args:
        query: 用户/模型查询词
        keys: 候选键集合
        min_overlap: token 重叠匹配所需的最小重叠 token 数

    Returns:
        所有匹配的 key 列表（可能为空）
    """
    if not query:
        return []
    return [k for k in keys if is_match(query, k, min_overlap)]
