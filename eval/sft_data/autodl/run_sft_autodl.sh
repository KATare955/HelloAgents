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

# ---- 归档函数：训练产物拷到带版本+时间戳目录，防 LF 固定 output_dir 被下次训练覆盖 ----
do_archive() {
  local VER="${VERSION:-v1.0}"
  local TS=$(date +%Y%m%d_%H%M%S)
  local ARC=${DATA_ROOT}/archive/sft_qwen3-14b-sales_${VER}_${TS}
  if [ ! -d "$ADAPTER_DIR" ]; then echo "❌ adapter 未找到($ADAPTER_DIR)，请先跑 train"; return 1; fi
  mkdir -p "$ARC/adapter"
  # 拷 final adapter(246M)+loss图+trainer_state+results，不拷中间 checkpoint-*(留AutoDL)
  cp "$ADAPTER_DIR"/adapter_model.safetensors "$ADAPTER_DIR"/adapter_config.json "$ARC/adapter/" 2>/dev/null || true
  cp "$ADAPTER_DIR"/trainer_state.json "$ADAPTER_DIR"/training_loss.png "$ADAPTER_DIR"/training_eval_loss.png \
     "$ADAPTER_DIR"/all_results.json "$ADAPTER_DIR"/eval_results.json "$ARC/" 2>/dev/null || true
  cp "$TRAIN_YAML" "$ARC/train_config.yaml"
  local GC=$(cd "$REPO_DIR" && git rev-parse --short HEAD 2>/dev/null || echo unknown)
  local NT=$(python -c "import json;print(len(json.load(open('${DATA_DIR}/sft_train.json'))))" 2>/dev/null || echo 0)
  local NV=$(python -c "import json;print(len(json.load(open('${DATA_DIR}/sft_val.json'))))" 2>/dev/null || echo 0)
  cat > "$ARC/MANIFEST.json" <<EOF
{
  "model": "qwen3-14b-sales-sft",
  "base_model": "${BASE_MODEL}",
  "stage": "sft",
  "version": "${VER}",
  "archived_at": "${TS}",
  "git_commit": "${GC}",
  "data": {"train": ${NT}, "val": ${NV}},
  "merged_model_on_autodl": "${MERGED_MODEL}",
  "note": "28GB完整模型留AutoDL数据盘；本归档含246M adapter+轻量产物，rsync回本地分析"
}
EOF
  echo "✅ 归档完成(带版本时间戳,不会被下次训练覆盖)：$ARC"
  ls -lhR "$ARC" | head -30
  echo ""
  echo "拉回本地(MacBook跑，端口/主机见AutoDL控制台SSH信息)："
  echo "  rsync -avz -e 'ssh -p <端口>' root@<AutoDL主机>:${ARC}/ ./实验记录/训练归档/$(basename "$ARC")/"
}

case "$STAGE" in

# -----------------------------------------------------------------------------
install)
  echo "【install】安装 LLaMA-Factory（SFT 框架，setup_autodl_env.sh 未覆盖）"
  # ⚠️ 学术加速边界：clone GitHub / PyPI官方源要开加速；用国内镜像要关（见 11-01 踩坑）
  source /etc/network_turbo           # 开加速(git clone GitHub)
  if [ ! -d "$LF_DIR" ]; then
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git "$LF_DIR"
  fi
  # ⚠️⚠️ clone 完立即关加速！下面 pip 走环境默认的 aliyun 国内镜像，代理开着会绕国际出口访问国内源，
  #   导致 build 依赖 hatchling 报 "Could not find a version that satisfies ... (from versions: none)"。
  unset http_proxy https_proxy
  cd "$LF_DIR"
  # ⚠️⚠️ 关键排雷：本机 torch 已 force-reinstall 到 2.13.0 + sed patch 过 inductor，vLLM 依赖它。
  #   装 LLaMA-Factory 【不要】用 [torch] extra，否则会把 torch 拉回别的版本→重新触发 inductor 损坏。
  #   用 [metrics] + 装前后校验 torch 版本，被改动则按 11-01 日记5.3 恢复。
  TORCH_BEFORE=$(python -c "import torch;print(torch.__version__)")
  # 注:LF 0.9.6.dev0 无 'metrics' extra(会 warn "does not provide the extra"),装核心即可;
  #   评测走我们自己的 baseline_eval.py,不依赖 LF 的 metrics。
  # ⚠️⚠️ 已知副作用:LF 会把 transformers 5.19→5.8、tokenizers/starlette/pillow/datasets 一起降级,
  #   破坏 vLLM(0.32 要 transformers>=5.16)。故 train/merge 用本环境,serve 前由 serve 段自动恢复。
  pip install -e "."                  # 不用 [torch] extra，减少对 torch 的改动
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
  # ⚠️ 训练前确认 GPU 空闲：若之前 serve 起的 vLLM 还在后台(单卡占~71GB=gpu-mem-util0.9×80GB)，
  #   加载 14B(需~28GB)会 OOM（报 "Process XXXXX has 71.XX GiB in use"）。先关残留服务再训。
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  if [ -n "$USED" ] && [ "$USED" -gt 5000 ]; then
    echo "❌ GPU0 已占用 ${USED}MiB，疑似残留 vLLM 服务。占用进程："
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null
    echo "   关闭: pkill -f vllm.entrypoints.openai.api_server  （或 kill <上面的PID>）"
    echo "   等 nvidia-smi 显存释放后重跑 train。本次中止。"
    exit 1
  fi
  cd "$LF_DIR"
  # 2 卡 NVLink 并行；单卡直接去掉 FORCE_TORCHRUN/NPROC_PER_NODE 也能跑
  FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=2 \
    llamafactory-cli train "$TRAIN_YAML"
  echo "✅ 训练完成，LoRA 适配器在 ${ADAPTER_DIR}"
  # ⚠️ LF output_dir 固定(saves/qwen3-14b-sales/lora/sft)+overwrite_output_dir=true，再训练会覆盖！
  #   故训练完自动归档到带时间戳目录防丢（VERSION=v1.0 可指定；多次训练各自归档，永不覆盖）
  do_archive
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
archive)
  echo "【archive】手动归档（train 完成时已自动归档一次；此处用于补归档或指定 VERSION）"
  cd "$LF_DIR"
  do_archive
  ;;

# -----------------------------------------------------------------------------
serve)
  echo "【serve】vLLM 起 SFT 模型服务 —— 完全对齐 11-01 已跑通的 Base 启动方案(含全部排雷patch)"
  unset http_proxy https_proxy
  # ⚠️⚠️ 先恢复被 LLaMA-Factory 降级的 vLLM 依赖：LF 要 transformers<=5.8、vLLM 0.32 要 >=5.16.1，
  #   装 LF 时 transformers 被降到 5.8（连 tokenizers/starlette/pillow/datasets 一起降），不恢复 vLLM 必崩。
  #   下面版本 = 11-01 跑通 Base 评测时的版本；只升上层包，torch 2.13.0 不动（升完立即断言校验）。
  TRANS_VER=$(python -c "import transformers;print(transformers.__version__)" 2>/dev/null || echo none)
  if [ "$TRANS_VER" != "5.19.0" ]; then
    echo "检测到 transformers=$TRANS_VER（被 LF 降级），恢复 vLLM 依赖..."
    pip install --upgrade "transformers==5.19.0" "tokenizers==0.23.3" "starlette==1.10.0" "pillow==13.0.0" "datasets==5.1.0" \
      -i https://mirrors.aliyun.com/pypi/simple
    python -c "import torch;assert torch.__version__.startswith('2.13.0'),'torch被改动:'+torch.__version__;print('torch OK',torch.__version__)"
  fi
  # ⚠️ 这套 patch 是踩了2小时坑验证出来的，SFT 模型服务同样需要，一条都别省：
  #   1)unset代理(国内直连) 2)inductor lowering.py:9111 sed patch 3)TORCHDYNAMO_DISABLE=1
  #   4)VLLM_USE_FLASHINFER_SAMPLER=0(否则首请求JIT崩溃→评测全Connection error) 5)--enforce-eager
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
  echo "用法: bash run_sft_autodl.sh {install|train|merge|archive|serve|eval}"
  echo "  install  装 LLaMA-Factory（补齐环境缺口）"
  echo "  train    LoRA SFT 训练"
  echo "  merge    合并 LoRA → 完整模型"
  echo "  archive  归档训练产物(adapter+配置+loss+MANIFEST)；VERSION=v1.0 可指定版本"
  echo "  serve    vLLM 后台起服务（nohup，脚本立即返回；等日志 'Application startup complete' 再跑 eval）"
  echo "  eval     跑 SFT 评测（EVAL_STAGE=sft）"
  echo ""
  echo "标准顺序: install → train → merge → archive → serve(后台起,等就绪) → eval(同终端即可)"
  ;;
esac
