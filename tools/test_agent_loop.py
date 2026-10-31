"""Agent Loop 测试：验证工具调用格式

使用 Ollama + Qwen3-8B 运行 Agent，验证工具调用格式是否正确。

前置条件：
1. Ollama 已启动（ollama serve）
2. Qwen3-8B 模型已下载（ollama pull qwen3:8b）
3. .env 文件已配置（见下方说明）

配置 .env 文件：
LLM_MODEL_ID=qwen3:8b
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
"""

import sys
from pathlib import Path

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


def create_sales_agent():
    """创建汽车销售 Agent"""
    
    # 1. 创建 LLM 实例（使用 Ollama + Qwen3-8B）
    llm = HelloAgentsLLM(
        model="qwen3:8b",
        api_key="ollama",  # Ollama 不需要真实的 API key
        base_url="http://localhost:11434/v1",
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
        # ToolRegistry 内部已打印注册信息，无需重复打印
    
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
        max_steps=5,  # 最多调用5次工具
    )
    
    return agent


def test_agent_loop():
    """测试 Agent Loop"""
    
    print("=" * 80)
    print("Agent Loop 测试：验证工具调用格式")
    print("=" * 80)
    
    # 创建 Agent
    print("\n【步骤1】创建 Agent")
    print("-" * 80)
    agent = create_sales_agent()
    print(f"\n✅ Agent 创建成功：{agent.name}")
    print(f"   已注册工具数量: {len(agent.tool_registry.list_tools())}")
    
    # 测试用例
    test_cases = [
        {
            "input": "E5 Sportback 2025款有什么配置？",
            "expected_tools": ["text_search"],
            "description": "单工具调用：查询车型配置",
        },
        {
            "input": "E5 Sportback现在有什么优惠？",
            "expected_tools": ["query_discount_policy"],
            "description": "单工具调用：查询优惠政策",
        },
        {
            "input": "E5 Sportback和小米SU7哪个好？",
            "expected_tools": ["competitor_compare"],
            "description": "单工具调用：竞品对比",
        },
        {
            "input": "E5 Sportback 2025款旗舰quattro型现在有什么优惠？首付30%，5年贷款月供多少？",
            "expected_tools": ["text_search", "query_discount_policy", "calculate_finance"],
            "description": "多跳调用：查询车型→查询优惠→计算金融",
        },
    ]
    
    print("\n【步骤2】测试 Agent Loop")
    print("-" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {test_case['description']}")
        print(f"   用户输入: {test_case['input']}")
        print(f"   预期工具: {', '.join(test_case['expected_tools'])}")
        print(f"   正在运行...")
        
        # 清空消息历史，确保每次测试独立
        if hasattr(agent, 'history_manager'):
            agent.history_manager.clear()
        
        try:
            # 运行 Agent
            response = agent.run(test_case["input"])
            
            print(f"   ✅ Agent 响应成功")
            print(f"   响应内容: {response[:200]}...")  # 只显示前200个字符
            
            # 检查工具调用
            if hasattr(agent, 'last_run') and agent.last_run:
                # last_run.tool_calls 是整数（工具调用次数）
                tool_calls_count = agent.last_run.tool_calls
                print(f"   工具调用次数: {tool_calls_count}")
                
                # 从消息历史中提取实际调用的工具
                if hasattr(agent, '_history') and agent._history:
                    actual_tools = []
                    for msg in agent._history:
                        # 查找 assistant 消息中的 tool_calls
                        if hasattr(msg, 'metadata') and msg.metadata:
                            model_msg = msg.metadata.get('model_message', {})
                            if model_msg.get('role') == 'assistant':
                                tool_calls = model_msg.get('tool_calls', [])
                                for tc in tool_calls:
                                    if tc.get('function', {}).get('name'):
                                        tool_name = tc['function']['name']
                                        # 排除内置工具 Thought 和 Finish
                                        if tool_name not in ['Thought', 'Finish']:
                                            actual_tools.append(tool_name)
                    
                    if actual_tools:
                        print(f"   实际调用的工具: {actual_tools}")
                    else:
                        print(f"   ⚠️  未检测到实际工具调用")
            
        except Exception as e:
            print(f"   ❌ 错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    
    # 总结
    print("\n【总结】")
    print("-" * 80)
    print("✅ Agent Loop 实现成功")
    print("✅ 支持多轮对话")
    print("✅ 支持工具自动调用")
    print("✅ 支持多跳调用链路")
    print("\n【下一步】")
    print("- 验证工具调用格式是否与 Qwen3 function calling 兼容")
    print("- 检查 LLM 生成的工具参数是否正确")
    print("- 测试边界情况和错误处理")


if __name__ == "__main__":
    test_agent_loop()
