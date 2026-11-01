"""Baseline评测脚本（带版本管理）

支持通过环境变量配置模型后端，兼容 Ollama（本地）和 vLLM（AutoDL）。

环境变量：
    EVAL_MODEL_NAME  - 模型名称，用于结果目录命名（默认: qwen3-8b）
    LLM_MODEL_ID     - LLM API 的 model 参数（默认: qwen3:8b）
    LLM_API_KEY      - API Key（默认: ollama）
    LLM_BASE_URL     - API Base URL（默认: http://localhost:11434/v1）

用法：
    # 本地 Ollama（默认，无需设置环境变量）
    cd /Users/kay/Desktop/hello-agents-family/HelloAgents
    uv run python eval/baseline_eval.py

    # AutoDL vLLM + Qwen3-14B
    export EVAL_MODEL_NAME=qwen3-14b
    export LLM_MODEL_ID=Qwen3-14B
    export LLM_API_KEY=token-abc123
    export LLM_BASE_URL=http://localhost:8000/v1
    python eval/baseline_eval.py

版本管理：
- 评测集版本：v1.1（68题）
- 结果文件命名：{阶段}_{模型}_{评测集版本}_{日期}.json
- 不覆盖原则：历史结果必须保留
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM, ReActAgent, ToolRegistry
from tools import (
    TextSearchTool,
    QueryDiscountPolicyTool,
    QueryTransactionPriceTool,
    CompetitorCompareTool,
    CalculateFinanceTool,
    BookTestDriveTool,
)

# 加载环境变量
load_dotenv()

# ============================================================
# 版本管理配置
# ============================================================

EVAL_VERSION = "v1.1"  # 评测集版本
MODEL_NAME = os.getenv("EVAL_MODEL_NAME", "qwen3-8b")  # 模型名称（环境变量可覆盖）
STAGE = "baseline"  # 评测阶段


def get_output_dir() -> Path:
    """获取输出目录（带版本管理）"""
    output_dir = Path(__file__).parent / "eval_data" / "results" / STAGE / MODEL_NAME / EVAL_VERSION
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_output_filename() -> str:
    """生成输出文件名（带版本管理）"""
    date_str = datetime.now().strftime("%Y%m%d")
    return f"{STAGE}_{MODEL_NAME}_{EVAL_VERSION}_{date_str}.json"


def get_bad_cases_filename() -> str:
    """生成bad cases文件名（带版本管理）"""
    date_str = datetime.now().strftime("%Y%m%d")
    return f"{STAGE}_{MODEL_NAME}_{EVAL_VERSION}_{date_str}_bad_cases.json"


def check_file_exists(filepath: Path) -> bool:
    """检查文件是否已存在（不覆盖保护）"""
    if filepath.exists():
        print(f"\n⚠️  警告：文件已存在！")
        print(f"   路径: {filepath}")
        print(f"   为避免覆盖历史结果，将创建新文件（添加时间戳）")
        return True
    return False


def get_unique_filename(filepath: Path) -> Path:
    """获取唯一文件名（避免覆盖）"""
    if not filepath.exists():
        return filepath
    
    # 添加时间戳（小时分钟秒）
    timestamp = datetime.now().strftime("%H%M%S")
    stem = filepath.stem
    suffix = filepath.suffix
    new_filename = f"{stem}_{timestamp}{suffix}"
    return filepath.parent / new_filename


def log_evaluation(accuracy: float, total_questions: int, bad_cases_count: int):
    """记录评测到版本日志"""
    log_file = Path(__file__).parent / "version_log.md"
    
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = f"""
## {date_str} - {STAGE} ({MODEL_NAME}, {EVAL_VERSION})

- **准确率**: {accuracy:.1f}%
- **总问题数**: {total_questions}
- **Bad case数量**: {bad_cases_count}
- **评测集版本**: {EVAL_VERSION}
- **模型**: {MODEL_NAME}
- **阶段**: {STAGE}
- **结果文件**: `results/{STAGE}/{MODEL_NAME}/{EVAL_VERSION}/{get_output_filename()}`
"""
    
    # 如果日志文件不存在，创建并添加标题
    if not log_file.exists():
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("# 评测版本日志\n\n")
            f.write(log_entry)
    else:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
    
    print(f"✅ 评测记录已保存到: {log_file}")


def create_sales_agent():
    """创建汽车销售 Agent"""
    
    # 1. 创建 LLM 实例（通过环境变量配置，兼容 Ollama / vLLM）
    llm = HelloAgentsLLM(
        model=os.getenv("LLM_MODEL_ID", "qwen3:8b"),
        api_key=os.getenv("LLM_API_KEY", "ollama"),
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        temperature=0.7,
    )
    
    # 2. 创建工具注册表
    registry = ToolRegistry()
    
    # 3. 注册所有工具
    tools = [
        TextSearchTool(),
        QueryDiscountPolicyTool(),
        QueryTransactionPriceTool(),
        CompetitorCompareTool(),
        CalculateFinanceTool(),
        BookTestDriveTool(),
    ]
    
    for tool in tools:
        registry.register_tool(tool)
    
    # 4. 创建 ReActAgent
    system_prompt = """你是一个专业的汽车销售顾问助手。
你的任务是帮助用户了解车型信息、查询优惠政策、对比竞品、计算金融方案、预约试驾等。

可用工具：
- text_search: 查询车型配置、价格、参数等信息
- query_discount_policy: 查询当前可用的优惠政策
- query_transaction_price: 查询历史成交价和价格趋势
- competitor_compare: 对比同级别车型
- calculate_finance: 计算贷款月供和总利息
- book_test_drive: 预约试驾

使用规则：
1. 根据用户问题选择合适的工具
2. 如果需要多个信息，可以多次调用工具
3. 用简洁清晰的语言回复用户
4. 提供具体的数字和信息"""
    
    agent = ReActAgent(
        name="SalesAgent",
        llm=llm,
        tool_registry=registry,
        system_prompt=system_prompt,
        max_steps=5,
    )
    
    return agent


def load_questions() -> Dict[str, Any]:
    """加载评测问题集"""
    questions_file = Path(__file__).parent / "eval_data" / "questions.json"
    with open(questions_file, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_single_question(agent, question: Dict[str, Any]) -> Dict[str, Any]:
    """评测单个问题"""
    
    # 清空消息历史
    if hasattr(agent, 'history_manager'):
        agent.history_manager.clear()
    
    result = {
        "id": question["id"],
        "scenario": question["scenario"],
        "type": question["type"],
        "question": question["question"],
        "expected_tools": question["expected_tools"],
        "difficulty": question["difficulty"],
    }
    
    try:
        # 运行 Agent
        response = agent.run(question["question"])
        
        # 记录响应
        result["response"] = response
        result["status"] = "success"
        
        # 提取实际调用的工具
        actual_tools = []
        if hasattr(agent, '_history') and agent._history:
            for msg in agent._history:
                if hasattr(msg, 'metadata') and msg.metadata:
                    model_msg = msg.metadata.get('model_message', {})
                    if model_msg.get('role') == 'assistant':
                        tool_calls = model_msg.get('tool_calls', [])
                        for tc in tool_calls:
                            if tc.get('function', {}).get('name'):
                                tool_name = tc['function']['name']
                                if tool_name not in ['Thought', 'Finish']:
                                    actual_tools.append(tool_name)
        
        result["actual_tools"] = actual_tools
        result["tool_calls_count"] = len(actual_tools)
        
        # 判断工具调用是否正确（简化版：只检查是否调用了预期工具）
        expected_set = set(question["expected_tools"])
        actual_set = set(actual_tools)
        
        # 如果实际调用的工具包含所有预期工具，则认为正确
        if expected_set.issubset(actual_set):
            result["tool_correct"] = True
        else:
            result["tool_correct"] = False
            result["missing_tools"] = list(expected_set - actual_set)
            result["extra_tools"] = list(actual_set - expected_set)
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["actual_tools"] = []
        result["tool_calls_count"] = 0
        result["tool_correct"] = False
    
    return result


def run_baseline_eval():
    """运行 baseline 评测"""
    
    print("=" * 80)
    print(f"Baseline评测：使用 {MODEL_NAME}（后端: {os.getenv('LLM_BASE_URL', 'http://localhost:11434/v1')}）")
    print("=" * 80)
    
    # 加载问题集
    print("\n【步骤1】加载评测问题集")
    print("-" * 80)
    questions_data = load_questions()
    questions = questions_data["questions"]
    print(f"✅ 加载了 {len(questions)} 个评测问题")
    print(f"   场景分布: {questions_data['scenario_distribution']}")
    print(f"   类型分布: {questions_data['type_distribution']}")
    
    # 创建 Agent
    print("\n【步骤2】创建 Agent")
    print("-" * 80)
    agent = create_sales_agent()
    print(f"✅ Agent 创建成功：{agent.name}")
    print(f"   已注册工具数量: {len(agent.tool_registry.list_tools())}")
    
    # 运行评测
    print("\n【步骤3】运行评测")
    print("-" * 80)
    
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] 评测问题: {question['id']}")
        print(f"   场景: {question['scenario']}")
        print(f"   类型: {question['type']}")
        print(f"   问题: {question['question'][:50]}...")
        
        result = evaluate_single_question(agent, question)
        results.append(result)
        
        # 打印结果
        if result["status"] == "success":
            print(f"   ✅ 响应成功")
            print(f"   工具调用: {result['actual_tools']}")
            print(f"   工具正确: {'✅' if result['tool_correct'] else '❌'}")
        else:
            print(f"   ❌ 错误: {result['error']}")
    
    # 统计结果
    print("\n【步骤4】统计结果")
    print("-" * 80)
    
    # 按场景统计
    scenario_stats = {}
    for result in results:
        scenario = result["scenario"]
        if scenario not in scenario_stats:
            scenario_stats[scenario] = {
                "total": 0,
                "success": 0,
                "tool_correct": 0,
                "errors": []
            }
        
        scenario_stats[scenario]["total"] += 1
        if result["status"] == "success":
            scenario_stats[scenario]["success"] += 1
        if result.get("tool_correct", False):
            scenario_stats[scenario]["tool_correct"] += 1
        else:
            scenario_stats[scenario]["errors"].append(result["id"])
    
    # 打印统计结果
    print("\n场景准确率统计:")
    print(f"{'场景':<20} {'总数':<8} {'成功':<8} {'工具正确':<10} {'准确率':<10}")
    print("-" * 60)
    
    total_questions = 0
    total_tool_correct = 0
    
    for scenario, stats in scenario_stats.items():
        accuracy = stats["tool_correct"] / stats["total"] * 100
        print(f"{scenario:<20} {stats['total']:<8} {stats['success']:<8} {stats['tool_correct']:<10} {accuracy:.1f}%")
        total_questions += stats["total"]
        total_tool_correct += stats["tool_correct"]
    
    overall_accuracy = total_tool_correct / total_questions * 100
    print("-" * 60)
    print(f"{'总体':<20} {total_questions:<8} {total_tool_correct:<10} {overall_accuracy:.1f}%")
    
    # 按类型统计
    type_stats = {}
    for result in results:
        q_type = result["type"]
        if q_type not in type_stats:
            type_stats[q_type] = {"total": 0, "tool_correct": 0}
        
        type_stats[q_type]["total"] += 1
        if result.get("tool_correct", False):
            type_stats[q_type]["tool_correct"] += 1
    
    print("\n问题类型准确率统计:")
    print(f"{'类型':<15} {'总数':<8} {'工具正确':<10} {'准确率':<10}")
    print("-" * 45)
    
    for q_type, stats in type_stats.items():
        accuracy = stats["tool_correct"] / stats["total"] * 100
        print(f"{q_type:<15} {stats['total']:<8} {stats['tool_correct']:<10} {accuracy:.1f}%")
    
    # 保存结果（带版本管理）
    print("\n【步骤5】保存结果（带版本管理）")
    print("-" * 80)
    
    # 获取输出目录和文件名
    output_dir = get_output_dir()
    output_filename = get_output_filename()
    bad_cases_filename = get_bad_cases_filename()
    
    output_file = output_dir / output_filename
    bad_case_file = output_dir / bad_cases_filename
    
    # 检查文件是否已存在（不覆盖保护）
    if check_file_exists(output_file):
        output_file = get_unique_filename(output_file)
        bad_case_file = get_unique_filename(bad_case_file)
    
    # 构建输出数据
    output = {
        "metadata": {
            "eval_type": STAGE,
            "model": MODEL_NAME,
            "eval_version": EVAL_VERSION,
            "timestamp": datetime.now().isoformat(),
            "total_questions": total_questions,
            "overall_accuracy": overall_accuracy,
            "bad_cases_count": len([r for r in results if not r.get("tool_correct", False)]),
        },
        "scenario_stats": scenario_stats,
        "type_stats": type_stats,
        "results": results,
    }
    
    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 结果已保存到: {output_file}")
    
    # 导出 bad case
    bad_cases = [r for r in results if not r.get("tool_correct", False)]
    
    with open(bad_case_file, "w", encoding="utf-8") as f:
        json.dump(bad_cases, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Bad cases 已保存到: {bad_case_file}")
    print(f"   Bad case 数量: {len(bad_cases)}")
    
    # 记录到版本日志
    log_evaluation(overall_accuracy, total_questions, len(bad_cases))
    
    print("\n" + "=" * 80)
    print("Baseline评测完成！")
    print("=" * 80)
    
    return output


if __name__ == "__main__":
    run_baseline_eval()
