#!/bin/bash
# =============================================================================
# Qwen3-14B 汽车销售 Agent — SFT 全流程编排脚本（AutoDL 配置C：2×A800-80GB）
# 对应计划 s4(环境) + s5(训练) + s6(评测)
#
# 代码同步（GitHub 中转，非 rsync）：本地 push → AutoDL pull
#   本地 MacBook:   cd HelloAgents && git add eval/ && git commit -m "..." && git push origin main
#   AutoDL(数据盘):  cd /root/autodl-tmp/HelloAgents && git pull origin main
#                   首次没有则: cd /root/autodl-tmp && git clone https://github.com/KATare955/HelloAgents.git
#   注:数据文件(llamafactory/*.json、sft_trajectories.jsonl)未被 .gitignore 排除,随代码一起 pull;
#      模型权重(27GB,*.safetensors)被 .gitignore 排除,走 ModelScope 下载,不进 Git。
#      GitHub 属国外资源:pull/clone 慢就 source /etc/network_turbo;若遇 503 就 unset http_proxy https_proxy 直连。
#
# 用法：分阶段执行，每步确认无误再进下一步
#   bash run_sft_autodl.sh install   # 装 LLaMA-Factory（补齐 setup_autodl_env.sh 缺的一环）
#   bash run_sft_autodl.sh train     # LoRA SFT 训练
#   bash run_sft_autodl.sh merge     # 合并 LoRA 到底座 → 完整模型
#   bash run_sft_autodl.sh serve     # vLLM 起服务（开工具调用 + 关思考模式）
#   bash run_sft_autodl.sh eval      # 同一 harness + 同一 v1.1 冻结集跑 SFT 评测
#
# 前置：setup_autodl_env.sh 已装 vLLM/transformers/peft 并下载 /root/autodl-tmp/models/Qwen3-14B
#       训练数据 eval/sft_data/llamafactory/{sft_train,sft_val,dataset_info}.json 由 git pull 自带
# =============================================================================
set -e

# ---- 路径约定（全部放数据盘 /root/autodl-tmp；系统盘 /root 仅 30GB，放模型必爆）----
# ⚠️ 上 AutoDL 第一件事：ls 确认下面路径真实存在，不符就改这里
DATA_ROOT=/root/autodl-tmp
BASE_MODEL=${DATA_ROOT}/models/Qwen3-14B                 # ✅已确认:ModelScope下载于此(27.5GB,8分片)
REPO_DIR=${DATA_ROOT}/HelloAgents                        # repo(git pull 同步,含数据+配置+本脚本)
LF_DIR=${DATA_ROOT}/LLaMA-Factory                        # 训练框架(装数据盘)
DATA_DIR=${REPO_DIR}/eval/sft_data/llamafactory          # 转换产物(dataset_info+train+val)
TRAIN_YAML=${REPO_DIR}/eval/sft_data/autodl/qwen3_14b_lora_sft.yaml
ADAPTER_DIR=${LF_DIR}/saves/qwen3-14b-sales/lora/sft     # 训练产出 LoRA
MERGED_MODEL=${DATA_ROOT}/models/qwen3-14b-sales-sft     # 合并后完整模型(放models盘)
SERVED_NAME=qwen3-14b-sales-sft
VLLM_PORT=8000

STAGE="${1:-help}"

case "$STAGE" in

# -----------------------------------------------------------------------------
install)
  echo "【install】安装 LLaMA-Factory（SFT 框架，setup_autodl_env.sh 未覆盖）"
  # ⚠️ 学术加速边界：clone GitHub / PyPI官方源要开加速；用国内镜像要关（见 11-01 踩坑）
  source /etc/network_turbo           # 开加速(git clone GitHub)
  if [ ! -d "$LF_DIR" ]; then
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git "$LF_DIR"
  fi
  cd "$LF_DIR"
  # ⚠️⚠️ 关键排雷：本机 torch 已 force-reinstall 到 2.13.0 + sed patch 过 inductor，vLLM 依赖它。
  #   装 LLaMA-Factory 【不要】用 [torch] extra，否则会把 torch 拉回别的版本→重新触发 inductor 损坏。
  #   用 [metrics] + 装前后校验 torch 版本，被改动则按 11-01 日记5.3 恢复。
  TORCH_BEFORE=$(python -c "import torch;print(torch.__version__)")
  pip install -e ".[metrics]"         # 不用 [torch] extra，减少对 torch 的改动
  # 若报依赖冲突： pip install --no-deps -e .  然后手动补 transformers/datasets/accelerate/peft/trl
  TORCH_AFTER=$(python -c "import torch;print(torch.__version__)")
  echo "--- 校验现场未被破坏 ---"
  echo "torch: $TORCH_BEFORE -> $TORCH_AFTER"
  if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
    echo "⚠️ torch 被改动！vLLM 可能又起不来，按 11-01 日记5.3 恢复："
    echo "   unset http_proxy https_proxy"
    echo "   pip install --force-reinstall torch==2.13.0 -i https://mirrors.aliyun.com/pypi/simple/ --timeout 1000"
    echo "   sed -i '9111s/import_submodule(kernel)/pass  # PATCHED/' /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/lowering.py"
  fi
  python -c "import torch._inductor.lowering; print('inductor OK')" || \
    sed -i '9111s/import_submodule(kernel)/pass  # PATCHED/' /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/lowering.py
  llamafactory-cli version
  echo "✅ LLaMA-Factory 安装完成（torch/vLLM 现场已保护）"
  echo ""
  echo "⚠️  flash-attn：默认训练用 sdpa(零编译)，无需装。要提速才装(编译慢易失败)："
  echo "     pip install flash-attn --no-build-isolation  # 装成功后把 yaml 的 sdpa 改 fa2"
  echo ""
  echo "⚠️  数据校验(git pull 后应自带)：${DATA_DIR}"
  ls -lh "$DATA_DIR" || echo "❌ 数据目录不存在，请确认已 git pull 最新代码"
  ;;

# -----------------------------------------------------------------------------
train)
  echo "【train】LoRA SFT（3 epochs, 全局 batch 8, template=qwen3, enable_thinking=false）"
  cd "$LF_DIR"
  # 2 卡 NVLink 并行；单卡直接去掉 FORCE_TORCHRUN/NPROC_PER_NODE 也能跑
  FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=2 \
    llamafactory-cli train "$TRAIN_YAML"
  echo "✅ 训练完成，LoRA 适配器在 ${ADAPTER_DIR}"
  ;;

# -----------------------------------------------------------------------------
merge)
  echo "【merge】合并 LoRA → 完整模型 ${MERGED_MODEL}"
  cd "$LF_DIR"
  MERGE_YAML=${LF_DIR}/merge_qwen3_14b_sales.yaml
  cat > "$MERGE_YAML" <<EOF
model_name_or_path: ${BASE_MODEL}
adapter_name_or_path: ${ADAPTER_DIR}
template: qwen3
finetuning_type: lora
trust_remote_code: true
export_dir: ${MERGED_MODEL}
export_size: 2
export_device: cpu
export_legacy_format: false
EOF
  llamafactory-cli export "$MERGE_YAML"
  echo "✅ 合并完成：${MERGED_MODEL}"
  ls -lh "$MERGED_MODEL"
  ;;

# -----------------------------------------------------------------------------
serve)
  echo "【serve】vLLM 起 SFT 模型服务 —— 完全对齐 11-01 已跑通的 Base 启动方案(含全部排雷patch)"
  # ⚠️ 这套 patch 是踩了2小时坑验证出来的，SFT 模型服务同样需要，一条都别省：
  #   1)unset代理(国内直连) 2)inductor lowering.py:9111 sed patch 3)TORCHDYNAMO_DISABLE=1
  #   4)VLLM_USE_FLASHINFER_SAMPLER=0(否则首请求JIT崩溃→评测全Connection error) 5)--enforce-eager
  unset http_proxy https_proxy
  sed -i '9111s/import_submodule(kernel)/pass  # PATCHED: skip broken kernel imports/' \
    /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/lowering.py 2>/dev/null || true
  python -c "import torch._inductor.lowering; print('inductor OK')"
  export TORCHDYNAMO_DISABLE=1
  export VLLM_USE_FLASHINFER_SAMPLER=0
  # --enable-auto-tool-choice + --tool-call-parser hermes : Qwen3 工具调用必需
  # --chat-template-kwargs enable_thinking=false : 与 SFT 训练一致(若本 vLLM 版本不支持此参数则删掉该行)
  nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MERGED_MODEL" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --host 0.0.0.0 --port "$VLLM_PORT" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --chat-template-kwargs '{"enable_thinking": false}' \
    --enforce-eager \
    > /root/vllm_sft.log 2>&1 &
  echo "✅ vLLM 后台启动(PID $!)，日志 /root/vllm_sft.log"
  echo "   等就绪: tail -f /root/vllm_sft.log 看到 'Application startup complete' 再跑 eval"
  echo "   验证: curl -s localhost:${VLLM_PORT}/v1/models | python -m json.tool"
  echo "⚠️ 公平性:Base(89.7%)当时serve未加enable_thinking=false。为严格可比,建议Base也用同配置重跑一次(纯推理~30min,零训练成本)"
  ;;

# -----------------------------------------------------------------------------
eval)
  echo "【eval】同一 harness + 同一 v1.1 冻结评测集 → SFT 结果（与 Base 89.7% 严格可比）"
  cd "$REPO_DIR"
  export EVAL_STAGE=sft
  export EVAL_MODEL_NAME=qwen3-14b-sft
  export LLM_MODEL_ID=${SERVED_NAME}
  export LLM_API_KEY=token-abc123
  export LLM_BASE_URL=http://localhost:${VLLM_PORT}/v1
  export EVAL_TEMPERATURE=0            # 贪婪解码，与 Base 评测一致，保证可复现
  python eval/baseline_eval.py
  echo "✅ SFT 评测完成，结果在 eval/eval_data/results/sft/qwen3-14b-sft/v1.1/"
  echo "   对比 Base（89.7%）看总体准确率 + 7 坏例场景是否修复"
  ;;

# -----------------------------------------------------------------------------
*)
  echo "用法: bash run_sft_autodl.sh {install|train|merge|serve|eval}"
  echo "  install  装 LLaMA-Factory（补齐环境缺口）"
  echo "  train    LoRA SFT 训练"
  echo "  merge    合并 LoRA → 完整模型"
  echo "  serve    vLLM 后台起服务（nohup，脚本立即返回；等日志 'Application startup complete' 再跑 eval）"
  echo "  eval     跑 SFT 评测（EVAL_STAGE=sft）"
  echo ""
  echo "标准顺序: install → train → merge → serve(后台起,等就绪) → eval(同终端即可)"
  ;;
esac
