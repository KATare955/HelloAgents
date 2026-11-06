"""Baseline评测脚本（带版本管理）

支持通过环境变量配置模型后端，兼容 Ollama（本地）和 vLLM（AutoDL）。

环境变量：
    EVAL_STAGE       - 评测阶段，用于结果目录/文件命名（默认: baseline；SFT 阶段设 sft，RL 阶段设 rl）
    EVAL_MODEL_NAME  - 模型名称，用于结果目录命名（默认: qwen3-8b）
    LLM_MODEL_ID     - LLM API 的 model 参数（默认: qwen3:8b）
    LLM_API_KEY      - API Key（默认: ollama）
    LLM_BASE_URL     - API Base URL（默认: http://localhost:11434/v1）
    EVAL_TEMPERATURE - 采样温度（默认: 0，贪婪解码确保可复现；如需模拟真实波动可设0.7并多次取平均）
    EVAL_TODAY       - 评测基准日期 YYYY-MM-DD（默认: 运行当天）。注入 system prompt 供模型换算
                      相对时间（问题 19：此前未注入日期，模型凭训练记忆猜成 2023 年→系统性年份幻觉）

用法：
    # 本地 Ollama（默认，无需设置环境变量）
    cd /Users/kay/Desktop/hello-agents-family/HelloAgents
    uv run python eval/baseline_eval.py

    # AutoDL vLLM + Qwen3-14B（Base 阶段）
    export EVAL_MODEL_NAME=qwen3-14b
    export LLM_MODEL_ID=Qwen3-14B
    export LLM_API_KEY=token-abc123
    export LLM_BASE_URL=http://localhost:8000/v1
    python eval/baseline_eval.py

    # SFT 阶段：同一 harness + 同一 v1.1 冻结评测集，仅换阶段名与模型后端
    # 保证 Base/SFT/RL 三行结果严格可比（评测集、判分、工具 schema 完全不变）
    export EVAL_STAGE=sft
    export EVAL_MODEL_NAME=qwen3-14b-sft
    export LLM_MODEL_ID=qwen3-14b-sales-sft
    export LLM_BASE_URL=http://localhost:8000/v1
    python eval/baseline_eval.py

版本管理：
- 评测集版本：v1.2（68题，含 answer_spec GT schema）
- 结果文件命名：{阶段}_{模型}_{评测集版本}_{日期}.json
- 不覆盖原则：历史结果必须保留
- 轨迹落盘：每题一个 {case_id}.jsonl 存于 results/{stage}/{model}/{version}/trajectories/（重构计划书第 4 章）
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM, ReActAgent, ToolRegistry
from hello_agents.core.runtime import RunResult, sync_events, turn_events
from tools import (
    TextSearchTool,
    QueryDiscountPolicyTool,
    QueryTransactionPriceTool,
    CompetitorCompareTool,
    CalculateFinanceTool,
    BookTestDriveTool,
)

# 判分逻辑 v2（单一来源，含 accept_clarification 澄清规则）
try:
    from .scoring import judge_tool_correct, JUDGE_VERSION
except ImportError:  # 直接运行脚本时
    from scoring import judge_tool_correct, JUDGE_VERSION

# 加载环境变量
load_dotenv()

# ============================================================
# 版本管理配置
# ============================================================

EVAL_VERSION = "v1.2"  # 评测集版本（三阶段共用，冻结不变）
MODEL_NAME = os.getenv("EVAL_MODEL_NAME", "qwen3-8b")  # 模型名称（环境变量可覆盖）
STAGE = os.getenv("EVAL_STAGE", "baseline")  # 评测阶段：baseline / sft / rl（环境变量可覆盖，同一 harness 跑三阶段）


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


def _get_eval_today() -> tuple:
    """评测基准日期（问题 19 修复，2026-11-06）：EVAL_TODAY 环境变量优先，默认运行当天。

    返回 (date_str, weekday_cn)。system prompt 注入与 env_snapshot 记录共用同一来源，
    判分器年份检查（rules.py）读同一环境变量，保证"模型被告知的今天"与"判分基准"一致。
    """
    raw = os.getenv("EVAL_TODAY", "").strip()
    if raw:
        try:
            d = datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError:
            print(f"[警告] EVAL_TODAY={raw} 格式非法（应为 YYYY-MM-DD），回退到运行当天")
            d = datetime.now().date()
    else:
        d = datetime.now().date()
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return d.strftime("%Y-%m-%d"), weekdays[d.weekday()]


def create_sales_agent():
    """创建汽车销售 Agent"""
    
    # 1. 创建 LLM 实例（通过环境变量配置，兼容 Ollama / vLLM）
    llm = HelloAgentsLLM(
        model=os.getenv("LLM_MODEL_ID", "qwen3:8b"),
        api_key=os.getenv("LLM_API_KEY", "ollama"),
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        # 评测默认 temperature=0（贪婪解码，确保三阶段可复现对比）；可用 EVAL_TEMPERATURE 覆盖
        temperature=float(os.getenv("EVAL_TEMPERATURE", "0")),
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
    # 问题 19（2026-11-06）：注入评测基准日期——此前未注入日期，模型凭训练记忆
    # 把"今天"猜成 2023 年（系统性年份幻觉），"明天下午3点"换算成 2023-10-06；
    # 注入后模型有换算依据，判分器恢复年份正确性检查（见 verifier/rules.py）。
    eval_today, eval_weekday = _get_eval_today()
    system_prompt = f"""你是一个专业的汽车销售顾问助手。
你的任务是帮助用户了解车型信息、查询优惠政策、对比竞品、计算金融方案、预约试驾等。

当前日期：{eval_today}（{eval_weekday}）。用户提到"明天/下周/周末"等相对时间时，请以当前日期为基准换算。

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


# ============================================================
# A4: trajectory 落盘辅助（重构计划书第 4 章字段清单）
# ============================================================

def _get_git_hash() -> str:
    """获取当前代码 git hash（env_snapshot 追溯工具版本，避免“+1 修复”归因存疑）"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(project_root), timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _parse_tool_status(tool_name: str, raw: str) -> str:
    """解析工具返回状态；内置工具（Thought/Finish）无失败语义，恒为 success"""
    if tool_name in ("Thought", "Finish"):
        return "success"
    try:
        data = json.loads(raw)
        return str(data.get("status", "unknown"))
    except (ValueError, TypeError):
        return "unknown"


# 数字 token 提取（参数溯源精确匹配用，问题 12 修法①；负向后行排除
# 车型名中的数字，与 rules.py _NUM_RE 口径一致：E8/SU7 的 8/7 不单独成 token）
_NUM_TOKEN_RE = re.compile(r"(?<![A-Za-z])\d[\d,]*")


def _value_in_text(value: str, text: str) -> bool:
    """参数值是否出现在文本中（问题 12 修法：数字精确 + 短词词边界）。

    - 纯数字值（含千分位）：提取目标文本数字 token 集做精确匹配
      （"0" 不再误命中 "90天" 中的 90；"8" 不命中 "E8"）；
    - 短 ASCII 词（<4 字符）：左右非字母数字边界匹配；
    - 其余（中文短词/长串）：保留子串匹配。
    """
    if not value or not text:
        return False
    if _NUM_TOKEN_RE.fullmatch(value):
        tokens = {m.group(0).replace(",", "") for m in _NUM_TOKEN_RE.finditer(text)}
        return value.replace(",", "") in tokens
    if len(value) < 4 and re.fullmatch(r"[A-Za-z0-9]+", value):
        return re.search(
            r"(?<![A-Za-z0-9])" + re.escape(value) + r"(?![A-Za-z0-9])", text
        ) is not None
    return value in text


def _trace_args_source(tool_name: str, args: Dict[str, Any], question_text: str, prior_obs: List[str]) -> Dict[str, str]:
    """参数值溯源（v2 精确匹配，供 verify 规则层 hallucinated_args_cap 细化）

    取值：question（值来自题面）/ prior_observation（值来自此前工具返回）/
    fabricated（两个来源都没有，疑似编造）。Thought/Finish 参数不溯源。

    问题 12 修法（2026-11-06）：① 数字值改精确 token 匹配（提取题面数字 token
    集，"0" 不再命中 "90天"）；② 短 ASCII 词改词边界匹配（避免 "E5" 类误命中）；
    ③ 长串保留子串。
    """
    if tool_name in ("Thought", "Finish"):
        return {}
    sources = {}
    for key, value in (args or {}).items():
        text = str(value)
        if _value_in_text(text, question_text):
            sources[key] = "question"
        elif any(_value_in_text(text, obs) for obs in prior_obs):
            sources[key] = "prior_observation"
        else:
            sources[key] = "fabricated"
    return sources


def _extract_step_reasoning(agent) -> Dict[int, str]:
    """从 history 提取每步 reasoning_content（think 模式下的思考链）"""
    reasoning = {}
    step = 0
    for msg in getattr(agent, "_history", []):
        model_message = (msg.metadata or {}).get("model_message") or {}
        if model_message.get("role") == "assistant":
            step += 1
            rc = model_message.get("reasoning_content")
            if isinstance(rc, str) and rc.strip():
                reasoning[step] = rc
    return reasoning


def evaluate_single_question(
    agent, question: Dict[str, Any], trajectories_dir: Path, env_snapshot: Dict[str, Any]
) -> Dict[str, Any]:
    """评测单个问题（消费事件流，逐工具调用落盘轨迹 jsonl）"""
    
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
    
    traj_rows = []      # 轨迹行（工具调用行 + 最终回复行）
    pending = {}        # tool_call_id -> {"row": 行骨架, "started": 起始时间}
    step_usage = {}     # step -> 该步 LLM 调用 usage
    prior_obs = []      # 已完成工具返回文本（供参数溯源）
    response = ""
    case_errors = []
    
    try:
        # 消费事件流（直接使用框架 turn_events 事件源，与 agent.run 同源；
        # 可捕获 model 事件的 usage、tool_start/tool_finish 完整字段）
        agent.last_run = RunResult()
        messages = agent._build_messages(question["question"])
        packets = agent._context_packets({})
        start_index = len(messages) - 1
        for event in sync_events(
            turn_events(agent, messages, packets, stream=False, kwargs={})
        ):
            kind = event["kind"]
            if kind == "model":
                step_usage[event.get("step", 0)] = event.get("usage") or {}
                # A4.1 增强（2026-11-05）：模型每步文本输出落盘——
                # 补齐"中间纯文本回合（反问/说明）不落盘"的可观测性缺口；
                # think 模式下 content 含 <think> 块（泄漏形态如实记录，供 C1 分析）
                model_text = (event.get("content") or "").strip()
                if model_text:
                    traj_rows.append({
                        "case_id": question["id"],
                        "kind": "model_output",
                        "turn_index": event.get("step", 0),
                        "content": model_text,
                        "finish_reason": event.get("finish_reason", ""),
                        "tool_count": event.get("tool_count", 0),
                        "usage": event.get("usage") or {},
                    })
            elif kind == "tool_start":
                try:
                    args = json.loads(event.get("arguments") or "{}")
                    if not isinstance(args, dict):
                        args = {}
                except (ValueError, TypeError):
                    args = {}
                pending[event["id"]] = {
                    "row": {
                        "case_id": question["id"],
                        "turn_index": event.get("step", 0),
                        "thought": "",
                        "tool_name": event.get("name", ""),
                        "tool_args": args,
                        "tool_response_raw": "",
                        "tool_status": "",
                        "observation": "",
                        "tool_args_source": {},
                        "latency_ms": 0,
                        "tokens": step_usage.get(event.get("step", 0), {}),
                        "is_final_answer": False,
                    },
                    "started": time.time(),
                }
            elif kind == "tool_finish":
                info = pending.pop(event.get("id"), None)
                if info is None:
                    continue
                row = info["row"]
                raw = event.get("result", "")
                # 完整落盘（不截断）：2048 字符截断会切断多字节字符与 JSON 结构，
                # 致 rule 层 JSON 解析失效（CAPA 问题 7/落盘截断发现）；
                # 框架默认 50KB 上限，全量落盘体积可接受（136 份轨迹 ~10MB）
                row["tool_response_raw"] = raw
                row["observation"] = raw
                row["tool_status"] = _parse_tool_status(row["tool_name"], raw)
                row["tool_args_source"] = _trace_args_source(
                    row["tool_name"], row["tool_args"], question["question"], prior_obs
                )
                row["latency_ms"] = int((time.time() - info["started"]) * 1000)
                if row["tool_name"] == "Finish":
                    row["is_final_answer"] = True
                    row["final_answer"] = (row["tool_args"] or {}).get("answer", "")
                traj_rows.append(row)
                prior_obs.append(raw)
        # 按 _events 收尾逻辑提交消息历史（供 actual_tools/reasoning 提取）
        agent._commit_messages(messages[start_index:])
        
        response = agent.last_run.answer if agent.last_run else ""
        result["response"] = response
        result["status"] = "success"
        
        # think 模式思考链回填
        step_reasoning = _extract_step_reasoning(agent)
        for row in traj_rows:
            if not row.get("thought"):
                row["thought"] = step_reasoning.get(row["turn_index"], "")
        
        # 无 Finish 工具的最终回复补行（模型直接文本回复/超步数）
        if not any(r.get("is_final_answer") for r in traj_rows) and response.strip():
            traj_rows.append({
                "case_id": question["id"],
                "turn_index": max((r["turn_index"] for r in traj_rows), default=0),
                "thought": "",
                "tool_name": "",
                "tool_args": {},
                "tool_response_raw": "",
                "tool_status": "",
                "observation": "",
                "tool_args_source": {},
                "latency_ms": 0,
                "tokens": {},
                "is_final_answer": True,
                "final_answer": response,
            })
        
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
        
        # 判断工具调用是否正确（判分逻辑 v2：见 eval/scoring.py，含 accept_clarification 澄清/降级规则）
        correct, missing, extra = judge_tool_correct(
            question["expected_tools"],
            actual_tools,
            response=response,
            accept_clarification=question.get("accept_clarification"),
        )
        result["tool_correct"] = correct
        if not correct:
            result["missing_tools"] = missing
            result["extra_tools"] = extra
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["actual_tools"] = []
        result["tool_calls_count"] = 0
        result["tool_correct"] = False
        case_errors.append(str(e))
    
    # case 汇总行 + 落盘（异常时保留部分轨迹，有诊断价值）
    last_run = agent.last_run if agent.last_run else None
    stop_reason = result["status"] if result["status"] == "error" else (last_run.status if last_run else "unknown")
    snapshot = dict(env_snapshot)
    snapshot["thinking_observed"] = "on" if any(r.get("thought") for r in traj_rows) else "off"
    traj_rows.append({
        "case_id": question["id"],
        "kind": "case_summary",
        "stop_reason": stop_reason,
        "final_response": response,
        "env_snapshot": snapshot,
        "agent_loop_events": [],
        "errors": case_errors,
    })
    
    traj_file = trajectories_dir / f"{question['id']}.jsonl"
    with open(traj_file, "w", encoding="utf-8") as f:
        for row in traj_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    result["trajectory_ref"] = f"trajectories/{question['id']}.jsonl"
    
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
    
    # A4: 轨迹目录与 env_snapshot（case 级元数据，重构计划书 4.1）
    trajectories_dir = get_output_dir() / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    env_snapshot = {
        "tools_git_hash": _get_git_hash(),
        "judge_version": JUDGE_VERSION,
        "model": MODEL_NAME,
        "llm_model_id": os.getenv("LLM_MODEL_ID", "qwen3:8b"),
        "thinking_observed": "unknown",  # 由 evaluate 按实际轨迹观测覆盖
        "temperature": float(os.getenv("EVAL_TEMPERATURE", "0")),
        "eval_version": EVAL_VERSION,
        "eval_today": _get_eval_today()[0],  # 问题 19：注入 system prompt 的基准日期，供判分/比对溯源
    }
    print(f"   轨迹落盘目录: {trajectories_dir}")
    
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] 评测问题: {question['id']}")
        print(f"   场景: {question['scenario']}")
        print(f"   类型: {question['type']}")
        print(f"   问题: {question['question'][:50]}...")
        
        result = evaluate_single_question(agent, question, trajectories_dir, env_snapshot)
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
