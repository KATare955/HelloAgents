"""助销Agent工具包

包含6个核心工具：
- text_search: 销售知识库检索
- query_discount_policy: 优惠政策查询
- query_transaction_price: 历史成交价查询
- competitor_compare: 竞品对比
- calculate_finance: 金融方案测算
- book_test_drive: 试驾预约
"""

from .text_search import TextSearchTool
from .query_discount_policy import QueryDiscountPolicyTool
from .query_transaction_price import QueryTransactionPriceTool
from .competitor_compare import CompetitorCompareTool
from .calculate_finance import CalculateFinanceTool
from .book_test_drive import BookTestDriveTool

__all__ = [
    "TextSearchTool",
    "QueryDiscountPolicyTool",
    "QueryTransactionPriceTool",
    "CompetitorCompareTool",
    "CalculateFinanceTool",
    "BookTestDriveTool",
]
