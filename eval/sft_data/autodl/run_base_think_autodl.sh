#!/bin/bash
# =============================================================================
# Qwen3-14B Base「thinking-ON」重跑脚本（AutoDL）—— 与 run_base_nothink_autodl.sh 对等
#
# 目的：think 主跑脚本化（此前 7 行 export + python 每次手敲，易错且慢）。
#   与 run_base_nothink_autodl.sh 的唯一本质差异：不加 --chat-template = 用模型自带
#   模板（thinking 默认开启）——这就是「think 主跑」。
#
# 代码同步（GitHub 中转）：本地 push → AutoDL pull
#   AutoDL: cd /root/autodl-tmp/HelloAgents && git pull origin main
#   注: GitHub 属国外资源，pull 慢就 source /etc/network_turbo；遇 503 就 unset http_proxy https_proxy 直连。
#
# 用法：分阶段执行，每步确认无误再进下一步
#   bash run_base_think_autodl.sh serve      # vLLM 起 Base-think 服务（后台，等就绪）
#   bash run_base_think_autodl.sh verify     # curl 验证 served 名正确
#   bash run_base_think_autodl.sh eval       # 同一 harness + v1.2 冻结集跑评测
#
# 前置：Base 模型已在 /root/autodl-tmp/models/Qwen3-14B；vLLM 环境已跑通。
# =============================================================================
set -e

# ---- 路径约定（与 nothink 脚本一致）----
DATA_ROOT=/root/autodl-tmp
BASE_MODEL=${DATA_ROOT}/models/Qwen3-14B           # 原始 Base（保持纯净，不 in-place 改）
REPO_DIR=${DATA_ROOT}/HelloAgents                  # repo（git pull 同步，含本脚本）
SERVED_NAME=qwen3-14b-think                        # 区别于 thinking-OFF 的 qwen3-14b-nothink
VLLM_PORT=8000
VLLM_LOG=/root/vllm_base_think.log
INDUCTOR_PY=/root/miniconda3/lib/python3.12/site-packages/torch/_inductor/lowering.py

STAGE="${1:-help}"

case "$STAGE" in

# -----------------------------------------------------------------------------
serve)
  echo "【serve】vLLM 起 Base-thinking-ON 服务 —— 默认模板（thinking 开），与 83.8% 同口径"
  unset http_proxy https_proxy
  # ⚠️ 恢复被 LLaMA-Factory 降级的 vLLM 依赖（幂等；与 nothink 脚本一致）
  TRANS_VER=$(python -c "import transformers;print(transformers.__version__)" 2>/dev/null || echo none)
  if [ "$TRANS_VER" != "5.19.0" ]; then
    echo "检测到 transformers=$TRANS_VER（LF 降级），恢复 vLLM 依赖..."
    pip install --upgrade "transformers==5.19.0" "tokenizers==0.23.3" "starlette==1.10.0" "pillow==13.0.0" "datasets==5.1.0" \
      -i https://mirrors.aliyun.com/pypi/simple
    python -c "import torch;assert torch.__version__.startswith('2.13.0'),'torch被改动:'+torch.__version__;print('torch OK',torch.__version__)"
  fi
  # GPU 预检：杀残留服务（think 先跑，GPU 应空闲；预检幂等无害）
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  if [ -n "$USED" ] && [ "$USED" -gt 5000 ]; then
    echo "⚠️ GPU0 占用 ${USED}MiB，杀残留 vLLM 服务..."
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 5
  fi
  # ⚠️ 这套 patch 踩了 2 小时坑验证出来，一条都别省（与 nothink/SFT serve 一致）：
  sed -i '9111s/import_submodule(kernel)/pass  # PATCHED: skip broken kernel imports/' "$INDUCTOR_PY" 2>/dev/null || true
  python -c "import torch._inductor.lowering; print('inductor OK')"
  export TORCHDYNAMO_DISABLE=1
  export VLLM_USE_FLASHINFER_SAMPLER=0
  nohup python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --host 0.0.0.0 --port "$VLLM_PORT" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --enforce-eager \
    > "$VLLM_LOG" 2>&1 &
  echo "✅ vLLM 后台启动(PID $!)，日志 $VLLM_LOG"
  echo "   等就绪: tail -f $VLLM_LOG 看到 'Application startup complete' 再跑 verify"
  ;;

# -----------------------------------------------------------------------------
verify)
  echo "【verify】curl 验证 served 名正确（应为 ${SERVED_NAME}）"
  curl -s localhost:${VLLM_PORT}/v1/models | python -m json.tool
  echo "--- 纯对话抽样（think 模式 content 含 <think> 块或 reasoning 非空 = 思考开启）---"
  curl -s localhost:${VLLM_PORT}/v1/chat/completions -H "Content-Type: application/json" -d "{
    \"model\": \"${SERVED_NAME}\",
    \"messages\": [{\"role\":\"user\",\"content\":\"你好\"}],
    \"max_tokens\": 30
  }" | python -m json.tool
  ;;

# -----------------------------------------------------------------------------
eval)
  echo "【eval】同一 harness + v1.2 冻结集 → Base-think（A4.1 轨迹含 model_output 行）"
  cd "$REPO_DIR"
  export EVAL_STAGE=baseline
  export EVAL_MODEL_NAME=qwen3-14b-think          # 结果目录名（区别于 nothink）
  export LLM_MODEL_ID=${SERVED_NAME}              # API model 参数，必须 = --served-model-name
  export LLM_API_KEY=token-abc123
  export LLM_BASE_URL=http://localhost:${VLLM_PORT}/v1
  export EVAL_TEMPERATURE=0                       # 贪婪解码，可复现
  python eval/baseline_eval.py
  echo "✅ Base-think 评测完成，结果在 eval/eval_data/results/baseline/qwen3-14b-think/v1.2/"
  echo ""
  echo "下一步（AutoDL 上执行）："
  echo "  cd ${REPO_DIR}/eval/sft_data/autodl"
  echo "  bash run_base_nothink_autodl.sh template && bash run_base_nothink_autodl.sh serve"
  ;;

# -----------------------------------------------------------------------------
*)
  echo "用法: bash run_base_think_autodl.sh {serve|verify|eval}"
  echo "  serve    vLLM 后台起 Base-think 服务（默认模板=thinking 开；等日志 'Application startup complete'）"
  echo "  verify   curl 验证 served 名 + 思考开启"
  echo "  eval     跑评测（EVAL_STAGE=baseline, EVAL_MODEL_NAME=qwen3-14b-think, 自动落 v1.2 目录）"
  echo ""
  echo "标准顺序: serve → verify → eval → 切 nothink（bash run_base_nothink_autodl.sh template/serve/verify/eval）"
  echo "产出: results/baseline/qwen3-14b-think/v1.2/（主结果 + trajectories 含 model_output 行）"
  ;;
esac
