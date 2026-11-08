#!/usr/bin/env python3
"""llamafactory 语义配对扩展点（A2 交付物，v02 桶2 多轮纠错形态专用）

解决的问题（E2 已验证，本地 llamafactory 0.9.3）：
  1. 官方 sharegpt 管线要求角色严格奇偶交替且 prompt 段数为奇数，v02 桶 2 的
     "编造轮无 gpt 收尾"形态（9 段）在官方 converter 位置校验与 encode_multiturn
     两两配对下均无法表达（converter 拦截 / IndexError 崩溃）；
  2. message 级 loss（编造段 loss_mask=false）官方 processor 不读取。

方案（评估报告 8.1 + E1/E2 结论）：
  - 自定义 SemanticPairTemplate（继承 qwen3 实际模板类）：encode_multiturn 改为
    "语义配对"——assistant/function 段为 target 边界，其前积累的 human/observation
    段为 source；loss=false 的 target 并入 source（返回 (src+tgt, [])），processor
    无需改动；_encode 微调 obs→user 边界（去掉悬空 assistant 生成提示 3 token）。
  - 自定义 V02ReactConverter 读取带 loss 字段的 sharegpt 兼容 JSON。
  - 需配合 1 处源码 patch（上机执行，见模块 docstring 末尾 PATCH 说明）。

上机用法（AutoDL，llamafactory 源码目录 src/）：
  1. 将本文件复制到 src/llamafactory/data/lf_extensions.py；
  2. 在 src/llamafactory/data/__init__.py 末尾添加一行：
         from .lf_extensions import *  # noqa: F401
     （注册 TEMPLATES["qwen3_semantic_pair"] 与 DATASET_CONVERTERS["v02_react"]，
      data 包一定早于 dataset 加载被 import，此位置比 tuner.py 更内聚）
  3. dataset_info.json 已随转换产物生成（llamafactory_v02/dataset_info.json）：
         "sales_agent_v02_train": {"file_name": "sft_train.json", "formatting": "v02_react"}
         "sales_agent_v02_val":   {"file_name": "sft_val.json",   "formatting": "v02_react"}
  4. 训练 yaml：template: qwen3_semantic_pair；其余沿用 R1 配置（已更新为 v02 版）。

PATCH（src/llamafactory/data/processor/supervised.py，两处同一段校验，0.9.3 实测
第 93/136 行、0.9.6.dev0 实测第 113/156 行，各改 1 行；上机以 grep 为准）：
  SupervisedDatasetProcessor.preprocess_dataset 与
  PackedSupervisedDatasetProcessor.preprocess_dataset 的段数校验对语义配对模板放宽：
    if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
  改为：
    if (len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1) \
            and not isinstance(self.template, SemanticPairTemplate):
  （语义配对模板下段数校验无意义——encode_multiturn 自身负责配对正确性，
   E2 已本地断言；不 patch 会导致桶 2 形态（prompt 偶数段）全部被 drop）
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

from llamafactory.data.converter import DATASET_CONVERTERS, DatasetConverter
from llamafactory.data.data_utils import Role
from llamafactory.data.template import TEMPLATES, StringFormatter

IGNORE_INDEX = -100  # 与 llamafactory.data.processor.supervised.IGNORE_INDEX 一致

TARGET_ROLES = (Role.ASSISTANT.value, Role.FUNCTION.value)


class SemanticPairTemplate(type(TEMPLATES["qwen3"])):
    """语义配对模板：assistant/function 段为 target 边界，loss=false 的 target 并入 source。

    继承 TEMPLATES["qwen3"] 的实际模板类（0.9.3 为 ReasoningTemplate），
    复用其 remove_thought / get_thought_word_ids / thought_words 与 qwen3 渲染格式。
    """

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, Any]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        discarding_history_cot: bool = False,
    ) -> list[tuple[list[int], list[int]]]:
        # 注：discarding_history_cot 为 0.9.6.dev0 processor 传入的第 5 位置参数
        # （mask_history 时丢弃历史 cot）；本路线 mask_history 恒为 False，忽略即可
        messages = deepcopy(messages)
        # nothink：对所有 target 段移除 cot（官方只处理偶数位，语义配对下 target 位置不固定）
        if self.enable_thinking is False:
            for m in messages:
                if m["role"] in TARGET_ROLES:
                    m["content"] = self.remove_thought(m["content"] or "")

        encoded = self._encode(tokenizer, messages, system, tools)

        # 语义配对：target 边界 = assistant/function 角色
        pairs, buf = [], []
        for m, toks in zip(messages, encoded):
            if m["role"] in TARGET_ROLES:
                # learn: 默认 True；显式 False 才不学；None（schema 填充）按默认学习
                pairs.append({"src": buf, "tgt": toks, "learn": m.get("loss", True) is not False})
                buf = []
            else:
                buf.append(toks)
        if buf and pairs:  # 尾部残留 source（正常不出现）
            pairs[-1]["src"] += buf

        # 空 cot 追加（官方逻辑平移：真实 target 无 think 词时，空 cot 追加到其 source 末尾）
        target_msgs = [m for m in messages if m["role"] in TARGET_ROLES]
        for p, m in zip(pairs, target_msgs):
            content = m["content"] or ""
            if self.thought_words[0] not in content and self.thought_words[1] not in content:
                if not self.enable_thinking:
                    if p["src"]:
                        p["src"][-1] = p["src"][-1] + self.get_thought_word_ids(tokenizer)
                    else:
                        p["src"].append(self.get_thought_word_ids(tokenizer))
                else:
                    p["tgt"] = self.get_thought_word_ids(tokenizer) + p["tgt"]

        # loss=false → target 并入 source（返回 (src+tgt, [])，processor 的 target_label 为空即不学）
        result = []
        for p in pairs:
            src_ids = [t for seg in p["src"] for t in seg]
            if p["learn"]:
                result.append((src_ids, p["tgt"]))
            else:
                result.append((src_ids + p["tgt"], []))
        return result

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: str,
        tools: str,
    ) -> list[list[int]]:
        """复刻 qwen3 官方 _encode，仅微调一处：obs 后接 user（桶 2 纠错边界）时，
        observation 段去掉尾部悬空的 <|im_start|>assistant\n 生成提示（E1 发现的 3-token 差异）。
        """
        system = system or self.default_system
        # 无尾部生成提示的 observation 格式（仅 obs→user 边界使用）
        obs_no_gen = StringFormatter(
            slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n"]
        )
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []
            system_text = ""
            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    system_text = self.format_system.apply(content=(system + tool_text))[0]

            if message["role"] == Role.USER:
                elements += self.format_user.apply(content=system_text + message["content"])
            elif message["role"] == Role.ASSISTANT:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION:
                next_role = messages[i + 1]["role"] if i + 1 < len(messages) else None
                if next_role == Role.USER:
                    elements += obs_no_gen.apply(content=message["content"])
                else:
                    elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION:
                elements += self.format_function.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))
        return encoded_messages


@dataclass
class V02ReactConverter(DatasetConverter):
    """v02 sharegpt 兼容 JSON → 标准格式，保留 message 级 loss 字段。"""

    ROLE_MAP = {
        "human": Role.USER.value,
        "gpt": Role.ASSISTANT.value,
        "function_call": Role.FUNCTION.value,
        "observation": Role.OBSERVATION.value,
    }

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        messages = []
        for c in example["conversations"]:
            msg = {"role": self.ROLE_MAP[c["from"]], "content": c.get("value") or ""}
            # 所有段统一输出 loss 键（bool），保证 datasets map 批内 schema 一致。
            # 注意：datasets 的 json builder 做 schema 推断时会把部分段缺失的 loss 键
            # 统一填充为 None（"loss" in c 恒为 True），因此 None 必须按默认学习处理。
            loss = c.get("loss")
            msg["loss"] = bool(loss) if loss is not None else True
            messages.append(msg)
        return {
            "_prompt": messages[:-1],
            "_response": messages[-1:],
            "_system": example.get("system", ""),
            "_tools": example.get("tools", ""),
            "_images": None,
            "_videos": None,
            "_audios": None,
        }


# --- 注册（直接写字典，兼容 register 函数签名随版本变化） ---
_q3 = TEMPLATES["qwen3"]
from dataclasses import fields as _fields

_sem_pair = SemanticPairTemplate(**{f.name: getattr(_q3, f.name) for f in _fields(_q3)})
TEMPLATES["qwen3_semantic_pair"] = _sem_pair

if "v02_react" not in DATASET_CONVERTERS:
    DATASET_CONVERTERS["v02_react"] = V02ReactConverter
