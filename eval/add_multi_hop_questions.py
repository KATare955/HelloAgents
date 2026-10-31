"""添加多跳问题到评测集

多跳问题：需要调用3个或更多不同的工具才能回答的问题。
"""

import json
from pathlib import Path


def add_multi_hop_questions():
    """添加多跳问题"""
    
    # 加载问题集
    questions_file = Path(__file__).parent / "eval_data" / "questions.json"
    with open(questions_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print("=" * 80)
    print("添加多跳问题到评测集")
    print("=" * 80)
    
    # 定义多跳问题（需要调用3个或更多工具）
    multi_hop_questions = [
        {
            "id": "multi_hop_001",
            "scenario": "金融方案测算",
            "type": "多跳",
            "question": "E5 Sportback 2025款旗舰quattro型现在有什么优惠？首付30%，5年贷款月供多少？近90天的平均成交价是多少？",
            "expected_tools": ["query_discount_policy", "calculate_finance", "query_transaction_price"],
            "difficulty": "hard",
            "note": "需要调用3个工具：查询优惠、计算金融、查询成交价"
        },
        {
            "id": "multi_hop_002",
            "scenario": "竞品对比",
            "type": "多跳",
            "question": "对比E5 Sportback和小米SU7的价格、续航、优惠政策，哪个更值得买？",
            "expected_tools": ["competitor_compare", "text_search", "query_discount_policy"],
            "difficulty": "hard",
            "note": "需要调用3个工具：竞品对比、查询配置、查询优惠"
        },
        {
            "id": "multi_hop_003",
            "scenario": "车型参数查询",
            "type": "多跳",
            "question": "E5 Sportback 2025款所有车型配置中，哪款续航里程最长？这款车的优惠政策是什么？首付30%贷款月供多少？",
            "expected_tools": ["text_search", "query_discount_policy", "calculate_finance"],
            "difficulty": "hard",
            "note": "需要调用3个工具：查询配置、查询优惠、计算金融"
        },
        {
            "id": "multi_hop_004",
            "scenario": "历史成交价查询",
            "type": "多跳",
            "question": "E5 Sportback近90天的平均成交价是多少？现在有什么优惠？优惠后的价格比历史成交价低多少？",
            "expected_tools": ["query_transaction_price", "query_discount_policy", "calculate_finance"],
            "difficulty": "hard",
            "note": "需要调用3个工具：查询成交价、查询优惠、计算差价"
        },
        {
            "id": "multi_hop_005",
            "scenario": "试驾预约",
            "type": "多跳",
            "question": "我想预约E5 Sportback的试驾，先帮我查询一下这款车的配置、优惠政策和金融方案",
            "expected_tools": ["text_search", "query_discount_policy", "calculate_finance", "book_test_drive"],
            "difficulty": "hard",
            "note": "需要调用4个工具：查询配置、查询优惠、计算金融、预约试驾"
        },
        {
            "id": "multi_hop_006",
            "scenario": "竞品对比",
            "type": "多跳",
            "question": "对比E5 Sportback和小米SU7的价格、续航、优惠政策、金融方案，然后帮我预约试驾E5 Sportback",
            "expected_tools": ["competitor_compare", "text_search", "query_discount_policy", "calculate_finance", "book_test_drive"],
            "difficulty": "hard",
            "note": "需要调用5个工具：竞品对比、查询配置、查询优惠、计算金融、预约试驾"
        },
        {
            "id": "multi_hop_007",
            "scenario": "金融方案测算",
            "type": "多跳",
            "question": "E5 Sportback 2025款旗舰quattro型的配置是什么？现在有什么优惠？首付30%，5年贷款月供多少？近90天的平均成交价是多少？",
            "expected_tools": ["text_search", "query_discount_policy", "calculate_finance", "query_transaction_price"],
            "difficulty": "hard",
            "note": "需要调用4个工具：查询配置、查询优惠、计算金融、查询成交价"
        },
        {
            "id": "multi_hop_008",
            "scenario": "优惠政策查询",
            "type": "多跳",
            "question": "E5 Sportback在上海和北京的优惠政策有什么区别？哪个地区的优惠力度更大？优惠后的价格首付30%贷款月供多少？",
            "expected_tools": ["query_discount_policy", "calculate_finance"],
            "difficulty": "hard",
            "note": "需要调用2个工具：查询优惠（2次）、计算金融"
        },
    ]
    
    # 添加多跳问题
    print(f"\n添加 {len(multi_hop_questions)} 个多跳问题：")
    print(f"{'ID':<20} {'场景':<15} {'工具数量':<10} {'问题':<40}")
    print("-" * 85)
    
    for question in multi_hop_questions:
        data["questions"].append(question)
        tool_count = len(question["expected_tools"])
        print(f"{question['id']:<20} {question['scenario']:<15} {tool_count:<10} {question['question'][:35]}...")
    
    # 更新统计信息
    data["metadata"]["total_questions"] = len(data["questions"])
    
    # 重新统计类型分布
    type_distribution = {"单跳": 0, "双跳": 0, "多跳": 0, "失败恢复": 0}
    for question in data["questions"]:
        type_distribution[question["type"]] += 1
    
    data["type_distribution"] = type_distribution
    
    # 重新统计场景分布
    scenario_distribution = {}
    for question in data["questions"]:
        scenario = question["scenario"]
        scenario_distribution[scenario] = scenario_distribution.get(scenario, 0) + 1
    
    data["scenario_distribution"] = scenario_distribution
    
    # 保存更新后的问题集
    with open(questions_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print("添加完成！")
    print("=" * 80)
    
    print(f"\n更新后的统计信息：")
    print(f"  总问题数: {data['metadata']['total_questions']}")
    print(f"\n类型分布：")
    for q_type, count in type_distribution.items():
        percentage = count / data['metadata']['total_questions'] * 100
        print(f"  {q_type}: {count} 个 ({percentage:.1f}%)")
    print(f"\n场景分布：")
    for scenario, count in scenario_distribution.items():
        print(f"  {scenario}: {count} 个")
    
    print(f"\n✅ 更新后的问题集已保存到: {questions_file}")


if __name__ == "__main__":
    add_multi_hop_questions()
