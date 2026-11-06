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
# 年份 token（2023/2024/2025…）：不构成实质匹配证据（问题 8 修复）
_YEAR = re.compile(r"^\d{4}$")
# 中文通用单字 token（款/型/价/配…意图词与后缀词）：不构成实质匹配证据（问题 8 修复）
_GENERIC_ZH = set("款型车价配置参数信息优惠政策对比指导金融方案月供试驾预约多少钱新车现在查一下有没在与和或")


def has_substantive_overlap(query: str, key: str) -> bool:
    """query 与 key 是否存在"实质 token 重叠"（问题 8 修复配套，2026-11-06）。

    兜底检索（无强相关条目时）的判定：共享 token 中至少一个既不是纯年份
    （2023/2024…），也不是中文通用单字（款/型/价/配…）。

    反例（假车型噪声根因）："星曜X7 2025款" 与 "E5 Sportback 2025款 先锋型"
    共享 {2025, 款}，全是通用 token → 不算实质匹配，不再被兜底捞回。
    正例："E5 Sportback 配置" 与 "E5 Sportback 2025款 先锋型" 共享 {e5, sportback}
    → 实质匹配成立，正常查询不受影响。
    """
    qt, kt = _tokens(query), _tokens(key)
    for t in qt & kt:
        if _YEAR.match(t):
            continue
        if len(t) == 1 and t in _GENERIC_ZH:
            continue
        return True
    return False


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


def match_candidates(query: str, keys: Iterable[str]) -> List[str]:
    """收集 query 在 keys 中的精确/子串级命中候选（用于歧义检测）

    只统计前两级命中（精确相等、双向子串），不含 token 重叠。
    用途：经销商等「实体选择」场景——候选多于一说明 query 有歧义（如只给品牌前缀），
    调用方应要求提供更完整的名称而非任选其一。

    Args:
        query: 用户/模型查询词
        keys: 候选键集合

    Returns:
        所有精确/子串命中的 key 列表（可能为空、唯一或多个）
    """
    if not query:
        return []
    qn = _norm(query)
    return [k for k in keys if _norm(k) == qn or qn in _norm(k) or _norm(k) in qn]


def score_match(query: str, key: str) -> float:
    """计算 query 与 key 的匹配得分（用于相关度排序，不用于匹配判定）

    得分区间 [0, 3]：
    - 3.0：归一化后完全相等
    - 2.5：双向子串
    - 0~2.0：token 重叠覆盖率（overlap / len(key_tokens) * 2）
    - 0.0：无重叠

    Note:
        与 is_match 不同，本函数永远返回得分而非布尔，
        供 text_search 等聚合类工具对多条结果做相关度排序。
    """
    qn, kn = _norm(query), _norm(key)
    if not qn or not kn:
        return 0.0
    if qn == kn:
        return 3.0
    if qn in kn or kn in qn:
        return 2.5
    qt, kt = _tokens(query), _tokens(key)
    if not kt:
        return 0.0
    overlap = len(qt & kt)
    return min(overlap / len(kt) * 2.0, 2.0)
