"""修复 questions.json 中的 type 标注

根据 expected_tools 的数量，重新标注 type：
- 1个工具 → 单跳
- 2个工具 → 双跳
- 3+个工具 → 多跳
- 失败恢复问题保持不变
"""

import json
from pathlib import Path


def fix_question_types():
    """修复问题类型标注"""
    
    # 加载问题集
    questions_file = Path(__file__).parent / "eval_data" / "questions.json"
    with open(questions_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print("=" * 80)
    print("修复 questions.json 中的 type 标注")
    print("=" * 80)
    
    # 统计修复情况
    fixed_count = 0
    type_changes = {"单跳→双跳": 0, "单跳→多跳": 0, "双跳→单跳": 0, "双跳→多跳": 0, "多跳→单跳": 0, "多跳→双跳": 0}
    
    print("\n修复详情：")
    print(f"{'ID':<20} {'原Type':<10} {'新Type':<10} {'Expected Tools':<40}")
    print("-" * 80)
    
    for question in data["questions"]:
        # 失败恢复问题保持不变
        if question["type"] == "失败恢复":
            continue
        
        # 根据 expected_tools 数量确定正确的 type
        tool_count = len(question["expected_tools"])
        
        if tool_count == 1:
            correct_type = "单跳"
        elif tool_count == 2:
            correct_type = "双跳"
        else:  # tool_count >= 3
            correct_type = "多跳"
        
        # 如果 type 不正确，修复它
        if question["type"] != correct_type:
            old_type = question["type"]
            question["type"] = correct_type
            fixed_count += 1
            
            # 记录变化
            change_key = f"{old_type}→{correct_type}"
            if change_key in type_changes:
                type_changes[change_key] += 1
            
            print(f"{question['id']:<20} {old_type:<10} {correct_type:<10} {str(question['expected_tools']):<40}")
    
    # 更新类型分布统计
    type_distribution = {"单跳": 0, "双跳": 0, "多跳": 0, "失败恢复": 0}
    for question in data["questions"]:
        type_distribution[question["type"]] += 1
    
    data["type_distribution"] = type_distribution
    
    # 保存修复后的问题集
    with open(questions_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print("修复完成！")
    print("=" * 80)
    
    print(f"\n修复统计：")
    print(f"  总共修复: {fixed_count} 个问题")
    print(f"\n类型变化统计：")
    for change, count in type_changes.items():
        if count > 0:
            print(f"  {change}: {count} 个")
    
    print(f"\n新的类型分布：")
    for q_type, count in type_distribution.items():
        print(f"  {q_type}: {count} 个")
    
    print(f"\n✅ 修复后的问题集已保存到: {questions_file}")


if __name__ == "__main__":
    fix_question_types()
