"""综合测试：测试所有6个工具"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hello_agents import ToolRegistry
from tools import (
    TextSearchTool,
    QueryDiscountPolicyTool,
    QueryTransactionPriceTool,
    CompetitorCompareTool,
    CalculateFinanceTool,
    BookTestDriveTool
)

def main():
    print("=" * 80)
    print("助销Agent工具综合测试")
    print("=" * 80)
    
    # 1. 创建工具注册表
    registry = ToolRegistry()
    
    # 2. 注册所有工具
    print("\n【步骤1】注册所有工具")
    print("-" * 80)
    registry.register_tool(TextSearchTool())
    registry.register_tool(QueryDiscountPolicyTool())
    registry.register_tool(QueryTransactionPriceTool())
    registry.register_tool(CompetitorCompareTool())
    registry.register_tool(CalculateFinanceTool())
    registry.register_tool(BookTestDriveTool())
    
    # 3. 查看已注册的工具
    print(f"\n已注册的工具: {registry.list_tools()}")
    print(f"工具数量: {len(registry.list_tools())}")
    
    # 4. 测试每个工具
    print("\n【步骤2】测试每个工具")
    print("-" * 80)
    
    # 测试1: text_search
    print("\n1. 测试 text_search（销售知识库检索）")
    response = registry.execute_tool(
        "text_search",
        {"query": "E5 Sportback 2025款", "category": "车型配置"}
    )
    print(f"   状态: {response.status.value}")
    print(f"   结果: {response.text}")
    
    # 测试2: query_discount_policy
    print("\n2. 测试 query_discount_policy（优惠政策查询）")
    response = registry.execute_tool(
        "query_discount_policy",
        {"car_model": "E5 Sportback", "region": "上海"}
    )
    print(f"   状态: {response.status.value}")
    print(f"   结果: {response.text}")
    
    # 测试3: query_transaction_price
    print("\n3. 测试 query_transaction_price（历史成交价查询）")
    response = registry.execute_tool(
        "query_transaction_price",
        {"car_model": "E5 Sportback 2025款 旗舰quattro型", "days": 90}
    )
    print(f"   状态: {response.status.value}")
    print(f"   结果: {response.text}")
    
    # 测试4: competitor_compare
    print("\n4. 测试 competitor_compare（竞品对比）")
    response = registry.execute_tool(
        "competitor_compare",
        {
            "car_model_a": "E5 Sportback 2025款 旗舰quattro型",
            "car_model_b": "小米SU7 2025款 四驱Max版"
        }
    )
    print(f"   状态: {response.status.value}")
    print(f"   结果: {response.text}")
    
    # 测试5: calculate_finance
    print("\n5. 测试 calculate_finance（金融方案测算）")
    response = registry.execute_tool(
        "calculate_finance",
        {
            "car_price": 289900,
            "down_payment_ratio": 30,
            "loan_years": 5,
            "interest_rate": 4.5
        }
    )
    print(f"   状态: {response.status.value}")
    print(f"   结果: {response.text}")
    
    # 测试6: book_test_drive
    print("\n6. 测试 book_test_drive（试驾预约）")
    response = registry.execute_tool(
        "book_test_drive",
        {
            "car_series": "E5 Sportback",
            "dealer": "上海郑隆奥翟",
            "appointment_time": "2026-10-29T14:00:00",
            "customer_name": "张先生",
            "customer_phone": "13800138000"
        }
    )
    print(f"   状态: {response.status.value}")
    print(f"   结果: {response.text}")
    
    # 5. 测试工具调用链路（多跳）
    print("\n【步骤3】测试工具调用链路（多跳）")
    print("-" * 80)
    
    print("\n场景：用户问'E5 Sportback 2025款旗舰quattro型现在有什么优惠？首付30%，5年贷款月供多少？'")
    print("\n调用链路:")
    print("  1. text_search → 查询车型信息")
    print("  2. query_discount_policy → 查询优惠政策")
    print("  3. calculate_finance → 计算金融方案")
    
    # Step 1: 查询车型信息
    print("\n  Step 1: 查询车型信息")
    response = registry.execute_tool(
        "text_search",
        {"query": "E5 Sportback 2025款", "category": "车型配置"}  # 修改：使用正确的关键词
    )
    car_price = 319900  # 默认值（旗舰quattro型的价格）
    if response.data and response.data.get("results"):
        # 查找旗舰quattro型
        for car_info in response.data["results"]:
            if "旗舰quattro型" in car_info["title"]:
                car_price = car_info["content"]["price"]
                print(f"    车型: {car_info['title']}")
                print(f"    指导价: {car_price}元")
                break
        else:
            # 如果没找到旗舰quattro型，使用第一个结果
            car_info = response.data["results"][0]
            car_price = car_info["content"]["price"]
            print(f"    车型: {car_info['title']}")
            print(f"    指导价: {car_price}元")
    else:
        print(f"    未找到车型信息，使用默认价格: {car_price}元")
    
    # Step 2: 查询优惠政策
    print("\n  Step 2: 查询优惠政策")
    response = registry.execute_tool(
        "query_discount_policy",
        {"car_model": "E5 Sportback", "region": "上海"}
    )
    if response.data and response.data.get("discounts"):
        total_discount = response.data["total_discount"]
        print(f"    总优惠: {total_discount}元")
        final_price = car_price - total_discount
        print(f"    优惠后价格: {final_price}元")
    
    # Step 3: 计算金融方案
    print("\n  Step 3: 计算金融方案")
    response = registry.execute_tool(
        "calculate_finance",
        {
            "car_price": final_price,
            "down_payment_ratio": 30,
            "loan_years": 5,
            "interest_rate": 4.5
        }
    )
    if response.data:
        print(f"    首付: {response.data['down_payment']}元")
        print(f"    贷款金额: {response.data['loan_amount']}元")
        print(f"    月供: {response.data['monthly_payment']}元")
        print(f"    总利息: {response.data['total_interest']}元")
    
    # 6. 获取工具描述
    print("\n【步骤4】工具描述")
    print("-" * 80)
    print(registry.get_tools_description())
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
