"""多轮"追问/澄清/降级/拒答"SFT 轨迹模板生成器（骨架）

对应规划：实验记录/SFT数据规划_多轮追问与拒答.md

职责（骨架阶段）：
1. 内置 6 场景数据沙盒配置（追问点 / 工具链 / 可接受替代 / 降级拒答触发）
2. 按 4 类工作流分支（direct_call / clarify_then_call / degrade / refuse）
   用"场景 × 缺失参数 × 车型"笛卡尔积生成**模板级**多轮轨迹（JSONL）
3. 校验：与 v1.1 评测集 68 题零重叠（id + 题干文本）

注意：
- 本脚本只产**模板骨架**（占位话术 + 正确工具链结构），用于打通格式与划分校验。
- 真正的高质量话术/多样性应由"强模型合成"（规划第6节第2路）在此基础上填充，
  或对种子 bad case 人工改写。骨架不追求话术质量。
- 负例不在此生成（负例留给 RL/DPO，见规划第1节原则3）。

用法：
    cd HelloAgents
    python eval/sft_data/gen_multiturn_templates.py --out eval/sft_data/templates_multiturn.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

EVAL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL_DIR))

QUESTIONS_FILE = EVAL_DIR / "eval_data" / "questions.json"

# ------------------------------------------------------------
# 6 场景数据沙盒配置（与规划文档第3节一一对应）
# ------------------------------------------------------------
SCENARIO_SANDBOX: Dict[str, dict] = {
    "车型参数查询": {
        "tool_chain": ["text_search"],
        "clarify_params": ["trim", "model_year"],          # 配置版本/年款
        "clarify_question": "请问您想了解哪个配置版本/年款？",
        "alternative": "list_all_trims",                    # 列全配置+价格（不追问）
    },
    "优惠政策查询": {
        "tool_chain": ["query_discount_policy"],
        "clarify_params": ["region", "purchase_mode"],      # 地区/购车方式
        "clarify_question": "请问您所在地区和购车方式（全款/置换）？",
        "alternative": "national_policy_with_region_note",
    },
    "历史成交价查询": {
        "tool_chain": ["query_transaction_price"],
        "clarify_params": ["days_window", "trim"],          # 时间窗/配置
        "clarify_question": "请问查询近多少天（30/60/90）的成交价？哪个配置？",
        "alternative": "default_90d_with_assumption",
    },
    "竞品对比": {
        "tool_chain": ["competitor_compare", "text_search"],
        "clarify_params": ["compare_dims", "competitor"],   # 维度/竞品
        "clarify_question": "请问想对比哪些维度（价格/续航/配置）？对比哪款车型？",
        "alternative": "default_all_dims",
    },
    "金融方案测算": {
        "tool_chain": ["text_search", "calculate_finance"],  # 先查价再算
        "clarify_params": ["down_payment_ratio", "loan_years", "annual_rate"],
        "clarify_question": "已查到车价。请问首付比例、贷款年限？（默认30%/3年/4.5%）",
        "alternative": "default_30_3_4.5_with_assumption",   # 7坏例中3题的正解
    },
    "试驾预约": {
        "tool_chain": ["book_test_drive"],
        "clarify_params": ["customer_name", "customer_phone", "dealer", "appointment_time"],
        "clarify_question": "请提供姓名、手机号、经销商与试驾时间，以便创建预约。",
        "alternative": "clarify_or_nonexistent_car_recommend",
    },
}

# 降级 / 拒答工作流触发（规划第3节）
DEGRADE_REFUSE_CASES = [
    {"workflow": "refuse",  "trigger": "out_of_scope",   "user": "帮我办一下保险理赔流程",
     "assistant": "抱歉，保险理赔不在我的服务能力内。我可以帮您查询车型、优惠、成交价、竞品对比、金融测算或预约试驾。"},
    {"workflow": "degrade", "trigger": "nonexistent",    "user": "预约 Model X 2025款 试驾",
     "assistant": "该车型不存在。为您找到可预约的 E5 Sportback 系列：…是否需要预约其中一款？"},
    {"workflow": "degrade", "trigger": "param_refused",  "user": "（用户拒给手机号）",
     "assistant": "理解您不便提供手机号。我先为您介绍该车型与优惠，预约可到店由顾问协助登记。"},
    {"workflow": "degrade", "trigger": "tool_repeat_fail", "user": "查某冷门车成交价（工具反复空）",
     "assistant": "暂未查询到该车型成交价。建议您联系经销商，或告诉我相近车型我再为您查询。"},
]

# 用于笛卡尔积的车型池（与 v1.1 不同组合，避免重叠）
CAR_POOL = ["E5 Sportback 2025款 先锋型", "E5 Sportback 2025款 旗舰quattro型", "帕萨特2024款 330TSI 精英版"]


def load_v11_questions():
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        return json.load(f)["questions"]


def assert_no_overlap(trajectories: List[dict], v11: List[dict]):
    """校验训练轨迹与 v1.1 评测集零重叠（id + 首轮 user 文本）。"""
    v11_ids = {q["id"] for q in v11}
    v11_texts = {q["question"].strip() for q in v11}
    for t in trajectories:
        assert t["id"] not in v11_ids, f"id 与 v1.1 重叠: {t['id']}"
        first_user = next(m["content"] for m in t["turns"] if m["role"] == "user")
        assert first_user.strip() not in v11_texts, f"题干与 v1.1 重叠: {first_user}"
    return True


def gen_clarify_then_call(scenario: str, cfg: dict, car: str, idx: int) -> dict:
    """缺参→追问→用户补→续接工具链 的多轮模板。"""
    chain = cfg["tool_chain"]
    turns = [{"role": "user", "content": f"帮我处理：{car}（故意缺 {cfg['clarify_params'][0]}）"}]
    # 若链首是 text_search（取价前置），先查价
    if chain[0] == "text_search":
        turns.append({"role": "assistant", "tool_calls": [{"name": "text_search", "args": {"query": car}}]})
        turns.append({"role": "tool", "name": "text_search", "content": "{\"price\": 235900}"})
    # 追问缺失参数
    turns.append({"role": "assistant", "content": cfg["clarify_question"]})
    # 模拟用户补充
    turns.append({"role": "user", "content": "（用户补充缺失参数）"})
    # 续接剩余工具链
    for tool in chain:
        turns.append({"role": "assistant", "tool_calls": [{"name": tool, "args": {"...": "补全后参数"}}]})
    turns.append({"role": "assistant", "content": "（最终结果话术，由强模型合成填充）"})
    return {
        "id": f"{scenario}_clarify_{idx:03d}",
        "scenario": scenario,
        "workflow": "clarify_then_call",
        "missing_params": cfg["clarify_params"],
        "expected_first_action": "ask",
        "turns": turns,
        "acceptable_alternatives": [cfg["alternative"]],
    }


def gen_direct_call(scenario: str, cfg: dict, car: str, idx: int) -> dict:
    """参数齐全→直调工具链 的单轮模板。"""
    turns = [{"role": "user", "content": f"（参数齐全的）{scenario}：{car}"}]
    for tool in cfg["tool_chain"]:
        turns.append({"role": "assistant", "tool_calls": [{"name": tool, "args": {"...": "齐全参数"}}]})
    turns.append({"role": "assistant", "content": "（最终结果话术）"})
    return {
        "id": f"{scenario}_direct_{idx:03d}",
        "scenario": scenario,
        "workflow": "direct_call",
        "missing_params": [],
        "expected_first_action": "call",
        "turns": turns,
        "acceptable_alternatives": [],
    }


def gen_degrade_refuse(idx: int) -> List[dict]:
    out = []
    for i, case in enumerate(DEGRADE_REFUSE_CASES):
        out.append({
            "id": f"degrade_{case['trigger']}_{idx:03d}_{i}",
            "scenario": "通用",
            "workflow": case["workflow"],
            "missing_params": [],
            "expected_first_action": "ask" if case["workflow"] == "degrade" else "refuse",
            "turns": [
                {"role": "user", "content": case["user"]},
                {"role": "assistant", "content": case["assistant"]},
            ],
            "acceptable_alternatives": [],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(EVAL_DIR / "sft_data" / "templates_multiturn.jsonl"))
    ap.add_argument("--per-scenario", type=int, default=2, help="每场景每分支生成条数（骨架默认小）")
    args = ap.parse_args()

    v11 = load_v11_questions()
    trajs: List[dict] = []

    idx = 0
    for scenario, cfg in SCENARIO_SANDBOX.items():
        for car in CAR_POOL[: args.per_scenario]:
            trajs.append(gen_clarify_then_call(scenario, cfg, car, idx))
            trajs.append(gen_direct_call(scenario, cfg, car, idx))
            idx += 1
    trajs.extend(gen_degrade_refuse(idx))

    assert_no_overlap(trajs, v11)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for t in trajs:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # 配比统计
    from collections import Counter
    cnt = Counter(t["workflow"] for t in trajs)
    print(f"✅ 生成 {len(trajs)} 条模板轨迹 -> {out}")
    print("   分支分布:", dict(cnt))
    print("   与 v1.1 零重叠校验: 通过")


if __name__ == "__main__":
    main()
