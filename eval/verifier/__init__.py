"""verify 验证器包（v3 判分）

规则层（rules.py）为本轮实现；LLM judge 层暂缓（C1 建设）。
"""

from .rules import verify_trajectory, WEIGHTS, CAP_THRESHOLDS

__all__ = ["verify_trajectory", "WEIGHTS", "CAP_THRESHOLDS"]
