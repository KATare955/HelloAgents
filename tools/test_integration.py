"""集成测试：在HelloAgents框架中注册并调用工具"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hello_agents import ToolRegistry
from tools.text_search import TextSearchTool

# 1. 创建工具注册表
registry = ToolRegistry()

# 2. 注册工具
registry.register_tool(TextSearchTool())

# 3. 查看已注册的工具
print("已注册的工具:", registry.list_tools())

# 4. 执行工具
response = registry.execute_tool(
    "text_search",
    {"query": "E5 Sportback 2025款", "category": "车型配置"}
)

# 5. 打印结果
print("\n工具响应:")
print(response.to_json())

# 6. 获取工具描述
print("\n工具描述:")
print(registry.get_tools_description())