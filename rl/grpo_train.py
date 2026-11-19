"""手写 GRPO 训练循环（阶段二 · 路线 B）

Reward 规格书_2026-11-17_v1.0 的 §1/§5/§7 落地，不依赖 veRL/TRL：

    loss   = −advantage · Σ_m log π_θ(t_m) − β · Σ_m (log π_θ(t_m) − log π_ref(t_m))
    advantage = clip((r − group_mean) / group_std, max=1.0)   # 单侧 clip（Dr. GRPO）
    loss 仅作用于模型生成 token（assistant 各轮，含工具调用 JSON）；
          prompt / tool 返回 token 全 mask（diff 法逐消息定界，不依赖模板内部实现）

Rollout 沙盒（§7）：
    1. 每条样本 = 全新 Agent + 全新 ToolRegistry（不共享历史，rollout 后即弃）
    2. fail_inject 走 per-sample 包装器实例计数（禁用模块级全局状态）
    3. reward 事后离线计算（rollout 只收集不判分）——同轨迹重算同分
    4. 原始轨迹全量落盘 out_dir/rollouts_*.jsonl（A4 行格式，可离线重放）
    5. 连续失败 K=3 截断轨迹（§1 max_turns=10 / 连续失败截断 K=3）

已知取舍（首跑口径，2026-11-18）：
    - rollout 由独立推理服务（vLLM 起点 = SFT4B）采样，logp/KL 在训练进程内前向；
      训练 LoRA 从 SFT4B 合并权重起步（--sft-adapter merge 后挂新 LoRA），
      ref = 关掉本 LoRA 的底座（= SFT4B 权重，零额外显存）。
      步内策略漂移属 off-policy 近似，30~50 步短跑可接受。
    - 首跑 w_proc=0（λ=0 消融起点），r_proc 第二轮再开。

用法（GPU 日）：
    python rl/grpo_train.py \
        --data rl/data/rl_prompts_filtered.jsonl \
        --base-model /root/autodl-tmp/models/Qwen3-4B \
        --sft-adapter /root/autodl-tmp/HelloAgents/saves/qwen3-4b-sales/lora/sft_v02 \
        --llm-base-url http://127.0.0.1:8000/v1 --llm-model sft4b \
        --steps 40 --out-dir rl/data/grpo_run_2026-11-19
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from hello_agents import HelloAgentsLLM, ReActAgent, ToolRegistry
from hello_agents.core.runtime import RunResult, sync_events, turn_events
from hello_agents.tools import Tool, ToolResponse

from tools import (
    TextSearchTool,
    QueryDiscountPolicyTool,
    QueryTransactionPriceTool,
    CompetitorCompareTool,
    CalculateFinanceTool,
    BookTestDriveTool,
)
from tools.web_search import WebSearchTool
from tools.follow_up_query import FollowUpQueryTool

from eval.baseline_eval import _parse_tool_status, _trace_args_source
from rl.reward import compute_reward

# ---------------------------------------------------------------------------
# 首跑超参（Reward 规格书 §1；网格起点，命令行可覆盖）
# ---------------------------------------------------------------------------

DEFAULT_BETA = 0.05
DEFAULT_GROUP_SIZE = 8
DEFAULT_LR = 1e-5
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_STEPS = 10          # ReActAgent max_steps
DEFAULT_MAX_CONSEC_FAIL = 3     # 连续失败截断 K
ADV_STD_FLOOR = 1e-4            # 组内零方差 → advantage 全零（不产生梯度）
ADV_CLIP_UPPER = 1.0            # 单侧 clip：只压正侧，负侧保留惩罚强度

BUSINESS_TOOL_CLASSES = [
    TextSearchTool, QueryDiscountPolicyTool, QueryTransactionPriceTool,
    CompetitorCompareTool, CalculateFinanceTool, BookTestDriveTool,
    WebSearchTool, FollowUpQueryTool,
]


def tool_instances() -> List[Tool]:
    return [cls() for cls in BUSINESS_TOOL_CLASSES]


# ---------------------------------------------------------------------------
# fail_inject 包装器（§7 规则 2：per-sample 实例计数，无全局态）
# ---------------------------------------------------------------------------

class FailInjectTool(Tool):
    """包装目标工具：前 fail_times 次调用返回注入错误，之后正常委托。"""

    def __init__(self, inner: Tool, fail_times: int, error_code: str):
        super().__init__(name=inner.name, description=inner.description)
        self._inner = inner
        self._fail_times = fail_times
        self._error_code = error_code
        self._calls = 0

    def get_parameters(self):
        return self._inner.get_parameters()

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        self._calls += 1
        if self._calls <= self._fail_times:
            return ToolResponse.error(
                code=self._error_code,
                message=(
                    f"模拟故障注入（{self._error_code}）：服务暂时不可用，"
                    f"请稍后重试或改用其他方式。"
                ),
            )
        return self._inner.run(parameters)


def build_rl_registry(fail_inject: Optional[Dict[str, Any]] = None) -> ToolRegistry:
    """8 工具注册表；fail_inject={'tool','fail_times','error_code'} 时替换目标实例。"""
    registry = ToolRegistry()
    inj = fail_inject or {}
    inj_tool = inj.get("tool")
    for inst in tool_instances():
        if inst.name == inj_tool:
            inst = FailInjectTool(inst, inj.get("fail_times", 1),
                                  inj.get("error_code", "TIMEOUT"))
        registry.register_tool(inst)
    return registry


# ---------------------------------------------------------------------------
# Agent 构造（§7 规则 1：每样本全新实例）
# ---------------------------------------------------------------------------

def build_system_prompt(eval_today: str, eval_weekday: str) -> str:
    return f"""你是一个专业的汽车销售顾问助手。
你的任务是帮助用户了解车型信息、查询优惠政策、对比竞品、计算金融方案、预约试驾等。

当前日期：{eval_today}（{eval_weekday}）。用户提到"明天/下周/周末"等相对时间时，请以当前日期为基准换算。

可用工具：
- text_search: 查询车型配置、价格、参数等信息
- query_discount_policy: 查询当前可用的优惠政策
- query_transaction_price: 查询历史成交价和价格趋势
- competitor_compare: 对比同级别车型
- calculate_finance: 计算贷款月供和总利息
- book_test_drive: 预约试驾
- web_search: 搜索网络公开信息（召回公告、新车资讯、竞品动态、行业政策）
- follow_up_query: 查询指定客户的跟进任务清单

使用规则：
1. 根据用户问题选择合适的工具
2. 如果需要多个信息，可以多次调用工具
3. 用简洁清晰的语言回复用户
4. 提供具体的数字和信息"""


def create_rl_agent(
    llm_kwargs: Dict[str, Any],
    registry: ToolRegistry,
    eval_today: str,
    eval_weekday: str,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> ReActAgent:
    llm = HelloAgentsLLM(**llm_kwargs)
    return ReActAgent(
        name="SalesAgent",
        llm=llm,
        tool_registry=registry,
        system_prompt=build_system_prompt(eval_today, eval_weekday),
        max_steps=max_steps,
    )


# ---------------------------------------------------------------------------
# rollout 事件消费（与 eval/baseline_eval.evaluate_single_question 同口径落盘）
# ---------------------------------------------------------------------------

def _business_tool(name: str) -> bool:
    return name in {
        "text_search", "query_discount_policy", "query_transaction_price",
        "competitor_compare", "calculate_finance", "book_test_drive",
        "web_search", "follow_up_query",
    }


def rollout_from_events(
    events, question_text: str, system_prompt: str,
    max_consec_fail: int = DEFAULT_MAX_CONSEC_FAIL,
) -> Dict[str, Any]:
    """消费事件流 → (steps, response, messages, meta)。

    纯函数（不依赖 agent 对象），单测可用脚本事件直接喂入。
    """
    steps: List[Dict[str, Any]] = []
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question_text},
    ]
    pending: Dict[str, Dict[str, Any]] = {}
    prior_obs: List[str] = []
    cur_assistant: Optional[Dict[str, Any]] = None
    response = ""
    n_calls = 0
    n_failed = 0
    consec_fail = 0
    truncated = False
    usage_total = 0

    for event in events:
        kind = event.get("kind")
        if kind == "model":
            text = (event.get("content") or "").strip()
            usage_total += int((event.get("usage") or {}).get("total_tokens", 0))
            if text:
                response = text
            cur_assistant = {"role": "assistant", "content": text, "tool_calls": []}
            messages.append(cur_assistant)
        elif kind == "tool_start":
            if cur_assistant is None:
                cur_assistant = {"role": "assistant", "content": "", "tool_calls": []}
                messages.append(cur_assistant)
            try:
                args = json.loads(event.get("arguments") or "{}")
                if not isinstance(args, dict):
                    args = {}
            except (ValueError, TypeError):
                args = {}
            cur_assistant["tool_calls"].append({
                "id": event.get("id", ""),
                "type": "function",
                "function": {
                    "name": event.get("name", ""),
                    "arguments": event.get("arguments") or "{}",
                },
            })
            pending[event.get("id")] = {"name": event.get("name", ""), "args": args}
        elif kind == "tool_finish":
            raw = event.get("result", "")
            info = pending.pop(event.get("id"), None)
            if info is None:
                continue
            name = info["name"]
            args = info["args"]
            if name == "Finish":
                if not response and args.get("answer"):
                    response = args["answer"]
                steps.append({
                    "tool_name": name, "tool_args": args, "observation": raw,
                    "tool_status": "success", "args_source": {},
                })
                prior_obs.append(raw)
                break
            messages.append({
                "role": "tool",
                "tool_call_id": event.get("id", ""),
                "content": raw,
            })
            status = _parse_tool_status(name, raw)
            steps.append({
                "tool_name": name, "tool_args": args, "observation": raw,
                "tool_status": status,
                "args_source": _trace_args_source(name, args, question_text, prior_obs),
            })
            prior_obs.append(raw)
            if _business_tool(name):
                n_calls += 1
                if status != "success":
                    n_failed += 1
                    consec_fail += 1
                    if consec_fail >= max_consec_fail:
                        truncated = True
                        break
                else:
                    consec_fail = 0
        if truncated:
            break

    messages = [
        m for m in messages
        if not (m["role"] == "assistant" and not m.get("content") and not m.get("tool_calls"))
    ]
    return {
        "steps": steps,
        "response": response,
        "messages": messages,
        "meta": {
            "n_calls": n_calls, "n_failed": n_failed,
            "truncated": truncated, "usage_total_tokens": usage_total,
        },
    }


def run_rollout(
    prompt: Dict[str, Any],
    llm_kwargs: Dict[str, Any],
    eval_today: str,
    eval_weekday: str,
    w_acc: float = 0.8,
    w_proc: float = 0.0,
    eval_year: Optional[str] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_consec_fail: int = DEFAULT_MAX_CONSEC_FAIL,
) -> Dict[str, Any]:
    """单条样本 rollout（全新 Agent，rollout 后即弃）→ 离线 reward 计算。"""
    registry = build_rl_registry(prompt.get("fail_inject"))
    agent = create_rl_agent(llm_kwargs, registry, eval_today, eval_weekday,
                            max_steps=max_steps)
    agent.last_run = RunResult()
    messages = agent._build_messages(prompt["question"])
    packets = agent._context_packets({})
    start_index = len(messages) - 1
    events = sync_events(
        turn_events(agent, messages, packets, stream=False, kwargs={})
    )
    collected = rollout_from_events(
        events, prompt["question"], build_system_prompt(eval_today, eval_weekday),
        max_consec_fail=max_consec_fail,
    )
    agent._commit_messages(messages[start_index:])

    reward = compute_reward(
        label=prompt, steps=collected["steps"], response=collected["response"],
        eval_year=eval_year, w_acc=w_acc, w_proc=w_proc,
    )
    return {**collected, "reward": reward}


# ---------------------------------------------------------------------------
# token 化 + assistant-token mask（diff 法逐消息定界）
# ---------------------------------------------------------------------------

def _encode(tokenizer, msgs: List[Dict[str, Any]], tools_schema: Optional[list]) -> List[int]:
    kwargs: Dict[str, Any] = dict(add_generation_prompt=True)
    if tools_schema:
        kwargs["tools"] = tools_schema
    out = tokenizer.apply_chat_template(msgs, tokenize=True, **kwargs)
    # transformers 5.x：tokenize=True 返回 BatchEncoding；旧版本返回 list[int]
    if isinstance(out, dict):
        out = out["input_ids"]
    elif not isinstance(out, list):
        out = out.input_ids
    return list(out)


def tokenize_with_mask(
    tokenizer, messages: List[Dict[str, Any]],
    tools_schema: Optional[list] = None,
) -> Tuple[List[int], List[int]]:
    """(input_ids, loss_mask)。mask[i]=1 当且仅当 t_i 由 assistant 轮生成
    （含工具调用 JSON）；system/user/tool 全 0。

    diff 法：mask 段 = template(messages[:i+1]) 相对 template(messages[:i])
    新增的 token（Qwen3 类模板逐消息渲染为前缀闭包，无跨消息重排）。
    """
    ids_full = _encode(tokenizer, messages, tools_schema)
    mask: List[int] = []
    prev_len = 0
    for i, msg in enumerate(messages):
        cur_ids = _encode(tokenizer, messages[: i + 1], tools_schema)
        span = cur_ids[prev_len:]
        prev_len = len(cur_ids)
        mask.extend([1 if msg.get("role") == "assistant" else 0] * len(span))
    if prev_len != len(ids_full):
        raise ValueError(
            f"mask 定界失败：前缀渲染总长 {prev_len} != 全量渲染 {len(ids_full)}"
        )
    return ids_full, mask


# ---------------------------------------------------------------------------
# advantage / loss（纯数值，可无 torch 单测）
# ---------------------------------------------------------------------------

def compute_advantages(
    rewards: List[float],
    group_size: int,
    std_floor: float = ADV_STD_FLOOR,
    clip_upper: float = ADV_CLIP_UPPER,
) -> List[float]:
    """GRPO 组内优势：adv = clip((r − mean)/std, max=clip_upper)。

    std 小于 std_floor 时按零方差处理（全零优势 → 零梯度，不烧机时）。
    """
    advs: List[float] = []
    for i in range(0, len(rewards), group_size):
        group = rewards[i: i + group_size]
        mean = sum(group) / len(group)
        var = sum((r - mean) ** 2 for r in group) / len(group)
        std = math.sqrt(var)
        for r in group:
            if std < std_floor:
                advs.append(0.0)
            else:
                advs.append(min((r - mean) / std, clip_upper))
    return advs


def sequence_loss(
    logp_theta: List[float],
    logp_ref: List[float],
    advantage: float,
    beta: float,
    mask: List[int],
) -> Tuple[float, int, float]:
    """单序列损失：L = Σ_m [−adv·logp_θ − β·(logp_θ − logp_ref)]，m ∈ mask=1。

    入参 logp 与 mask[1:] 对齐（t_i 的 logp 由位置 i−1 预测）。
    返回 (loss, 有效 token 数, 有效位置平均 KL = mean(logp_θ − logp_ref))。
    """
    loss = 0.0
    n_tokens = 0
    kl_sum = 0.0
    for j, lp_theta in enumerate(logp_theta):
        if mask[j + 1] != 1:
            continue
        lp_ref = logp_ref[j]
        loss += -advantage * lp_theta - beta * (lp_theta - lp_ref)
        kl_sum += lp_theta - lp_ref
        n_tokens += 1
    kl_mean = kl_sum / n_tokens if n_tokens else 0.0
    return loss, n_tokens, kl_mean


# ---------------------------------------------------------------------------
# 训练器（torch/transformers 惰性导入：CPU 单测不依赖 GPU 栈）
# ---------------------------------------------------------------------------

class GRPOTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        optimizer,
        beta: float = DEFAULT_BETA,
        w_acc: float = 0.8,
        w_proc: float = 0.0,
        group_size: int = DEFAULT_GROUP_SIZE,
        eval_year: Optional[str] = None,
        tools_schema: Optional[list] = None,
    ):
        import torch  # noqa: F401  (惰性导入，见模块 docstring)
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.beta = beta
        self.w_acc = w_acc
        self.w_proc = w_proc
        self.group_size = group_size
        self.eval_year = eval_year
        self.tools_schema = tools_schema

    def logps_tensor(self, input_ids: List[int]):
        """策略前向：逐 token log p(t_i|t_<i)（保留计算图，供 backward）。"""
        import torch
        self.model.eval()
        ids = torch.tensor([input_ids], device=self.model.device)
        logits = self.model(input_ids=ids).logits[0]
        logprobs = torch.log_softmax(logits[:-1], dim=-1)
        return logprobs.gather(1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)

    def logps_list(self, input_ids: List[int]) -> List[float]:
        """ref 前向：关 LoRA adapter（= SFT4B 权重），无梯度，返回 Python 浮点。"""
        import torch
        self.model.eval()
        with torch.no_grad():
            ids = torch.tensor([input_ids], device=self.model.device)
            with self.model.disable_adapter():
                logits = self.model(input_ids=ids).logits[0]
            logprobs = torch.log_softmax(logits[:-1], dim=-1)
            gathered = logprobs.gather(1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
            return [float(v) for v in gathered.cpu().tolist()]

    def train_step(
        self, prompt_group: List[Dict[str, Any]],
        rollout_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """已 rollout 的组 → 计算 loss → backward → step。

        rollout_records 与 prompt_group 展平对齐（每 prompt group_size 条）。
        """
        import torch
        rewards = [rec["reward"]["r"] for rec in rollout_records]
        advs = compute_advantages(rewards, self.group_size)

        losses = []
        n_tokens_total = 0
        kl_accum = 0.0
        seq_count = 0
        n_recs = len(rollout_records)
        for rec, adv in zip(rollout_records, advs):
            ids, mask = tokenize_with_mask(
                self.tokenizer, rec["messages"], self.tools_schema
            )
            pos = torch.tensor(mask[1:], device=self.model.device) == 1
            if not bool(pos.any()):
                continue
            lp_theta = self.logps_tensor(ids)
            lp_ref = torch.tensor(
                self.logps_list(ids), device=lp_theta.device, dtype=lp_theta.dtype
            )
            per_pos = -adv * lp_theta[pos] - self.beta * (lp_theta[pos] - lp_ref[pos])
            # 逐条立即 backward（梯度累积，图随条释放）：
            # 32 条轨迹若统一 backward，前向激活全量累积（4B×长轨迹 OOM）；
            # 除以 n_recs 等价于原 sum(losses)/seq_count 的平均口径（pos 全空的罕见条忽略）。
            per_pos.mean().div(n_recs).backward()
            losses.append(float(per_pos.mean().detach().cpu()))
            n_tokens_total += int(pos.sum())
            kl_accum += float((lp_theta[pos] - lp_ref[pos]).mean().detach().cpu())
            seq_count += 1

        if seq_count == 0:
            return {"loss": 0.0, "n_seqs": 0, "n_tokens": 0,
                    "kl_mean": 0.0, "adv_mean": sum(advs) / len(advs)}

        self.model.train()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            "loss": sum(losses) / seq_count,
            "n_seqs": seq_count,
            "n_tokens": n_tokens_total,
            "kl_mean": kl_accum / seq_count,
            "adv_mean": sum(advs) / len(advs),
        }


# ---------------------------------------------------------------------------
# 落盘（§7 规则 4：原始输出全量落盘，可离线重放）
# ---------------------------------------------------------------------------

def rollout_rows(record: Dict[str, Any], step: int, prompt_id: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for s in record["steps"]:
        rows.append({
            "step": step, "prompt_id": prompt_id, "kind": "tool",
            "tool_name": s["tool_name"], "tool_args": s["tool_args"],
            "tool_response_raw": s["observation"], "tool_status": s["tool_status"],
            "tool_args_source": s["args_source"],
        })
    rows.append({
        "step": step, "prompt_id": prompt_id, "kind": "model_output",
        "content": record["response"],
    })
    rows.append({
        "step": step, "prompt_id": prompt_id, "kind": "case_summary",
        "final_response": record["response"], "meta": record["meta"],
        "reward": record["reward"],
    })
    return rows


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def load_prompts(data_path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in data_path.read_text(encoding="utf-8").splitlines()]


def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description="手写 GRPO 训练（阶段二）")
    ap.add_argument("--data", required=True, help="RL prompt jsonl（pass@8 过滤后产物）")
    ap.add_argument("--base-model", required=True, help="底座模型路径（Qwen3-4B）")
    ap.add_argument("--sft-adapter", default=None,
                    help="SFT LoRA 路径：merge 进底座后挂全新 GRPO LoRA（ref=关 adapter 的底座）")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    ap.add_argument("--rollout-workers", type=int, default=8, help="每步 rollout 采样并发数（run_rollout 每样本独立实例，线程安全）")
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--w-acc", type=float, default=0.8)
    ap.add_argument("--w-proc", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--max-consec-fail", type=int, default=DEFAULT_MAX_CONSEC_FAIL)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--llm-model", default=os.getenv("LLM_MODEL_ID", "sft4b"))
    ap.add_argument("--llm-api-key", default=os.getenv("LLM_API_KEY", "EMPTY"))
    ap.add_argument("--eval-today", default=os.getenv("EVAL_TODAY", datetime.now().strftime("%Y-%m-%d")))
    ap.add_argument("--device", default="cuda", help="训练进程所在设备（4B 单卡即可）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-offset", type=int, default=0,
                    help="prompt 池轮转偏移（续跑用）")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(Path(args.data))
    if not prompts:
        raise SystemExit(f"数据为空：{args.data}")
    eval_year = args.eval_today[:4]

    # ---- 模型装配：底座 [+ SFT LoRA merge] → 新 GRPO LoRA ----
    print(f"[grpo] 加载底座 {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if args.sft_adapter:
        from peft import PeftModel
        print(f"[grpo] merge SFT LoRA {args.sft_adapter}")
        model = PeftModel.from_pretrained(model, args.sft_adapter)
        model = model.merge_and_unload()
    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha,
        target_modules="all-linear", lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    tools_schema = [inst.to_openai_schema() for inst in tool_instances()]

    llm_kwargs = dict(
        model=args.llm_model, api_key=args.llm_api_key,
        base_url=args.llm_base_url, temperature=args.temperature,
    )

    trainer = GRPOTrainer(
        model=model, tokenizer=tokenizer, optimizer=optimizer,
        beta=args.beta, w_acc=args.w_acc, w_proc=args.w_proc,
        group_size=args.group_size, eval_year=eval_year,
        tools_schema=tools_schema,
    )

    weekday_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][
        datetime.strptime(args.eval_today, "%Y-%m-%d").weekday()
    ]
    rollouts_file = out_dir / f"rollouts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    metrics_file = out_dir / "metrics.jsonl"

    for step in range(1, args.steps + 1):
        t0 = time.time()
        idx = list(range(len(prompts)))
        start = (args.prompt_offset + (step - 1) * args.prompts_per_step) % len(prompts)
        group_prompts = [prompts[(start + i) % len(prompts)] for i in range(args.prompts_per_step)]

        tasks = [(p, k) for p in group_prompts for k in range(args.group_size)]

        def _rollout(task):
            p, _ = task
            rec = run_rollout(
                p, llm_kwargs, args.eval_today, weekday_cn,
                w_acc=trainer.w_acc, w_proc=trainer.w_proc,
                eval_year=eval_year, max_steps=args.max_steps,
                max_consec_fail=args.max_consec_fail,
            )
            rec["prompt_id"] = p.get("id", "?")
            return rec

        with ThreadPoolExecutor(max_workers=args.rollout_workers) as ex:
            records = list(ex.map(_rollout, tasks))
        step_rows = [row for rec in records
                     for row in rollout_rows(rec, step, rec["prompt_id"])]
        with rollouts_file.open("a", encoding="utf-8") as f:
            for row in step_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        info = trainer.train_step(group_prompts, records)

        rs = [rec["reward"]["r"] for rec in records]
        metrics = {
            "step": step,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "mean_r": round(sum(rs) / len(rs), 4),
            "gate_rate": round(sum(1 for rec in records if rec["reward"]["r_gate"] == 0) / len(records), 4),
            "truncated_rate": round(sum(1 for rec in records if rec["meta"]["truncated"]) / len(records), 4),
            "mean_n_calls": round(sum(rec["meta"]["n_calls"] for rec in records) / len(records), 2),
            "step_sec": round(time.time() - t0, 1),
            **info,
        }
        metrics_file.open("a", encoding="utf-8").write(
            json.dumps(metrics, ensure_ascii=False) + "\n"
        )
        print(f"[grpo] step {step}/{args.steps} mean_r={metrics['mean_r']} "
              f"loss={metrics['loss']:.4f} kl={metrics['kl_mean']:.4f} "
              f"adv={metrics['adv_mean']:.3f} gate={metrics['gate_rate']} "
              f"({metrics['step_sec']}s)")

        if step % args.save_every == 0:
            save_dir = out_dir / f"checkpoint-{step}"
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"[grpo] checkpoint → {save_dir}")

    print(f"[grpo] done。rollouts → {rollouts_file}；metrics → {metrics_file}")


if __name__ == "__main__":
    main()
