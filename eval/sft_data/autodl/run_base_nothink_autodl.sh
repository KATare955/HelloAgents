#!/bin/bash
# =============================================================================
# Qwen3-14B Base「thinking-OFF」公平重跑脚本（AutoDL）
#
# 目的：隔离 thinking ON/OFF confound。
#   Base(83.8%) 当时 serve 未关思考，SFT(75%) 关了思考，两者不可直接比。
#   本脚本用【与 SFT serve 完全相同的排雷 + 关思考配置】重跑 Base，得到 Base-nothink，
#   与 SFT-nothink(75%) 严格可比 —— 判定掉分是「关思考的代价」还是「SFT 数据的锅」。
#   （对应 run_sft_autodl.sh serve 段早已预告的公平性重跑）
#
# 关键设计（与 run_sft_autodl.sh 的差异，务必理解）：
#   1) 非破坏性关思考：导出 Base 模板 → 注入 enable_thinking=false → --chat-template 加载，
#      【不改原模型目录】。故 Base-thinking-ON 的 83.8% 仍可复现（守住冻结协议）。
#      SFT 用 in-place patch 是因为 merged 模型是一次性副本；Base 是原始模型，必须保持纯净。
#      两种机制渲染出的 thinking-off prompt 完全一致，不影响与 SFT-nothink 的可比性。
#   2) 新 served 名 qwen3-14b-nothink + EVAL_MODEL_NAME=qwen3-14b-nothink：
#      结果落 results/baseline/qwen3-14b-nothink/v1.1/，【绝不覆盖】83.8%(在 qwen3-14b/)。
#   3) 其余排雷(代理/inductor/FlashInfer/eager/依赖恢复/GPU预检)与 SFT serve 逐条对齐。
#
# 代码同步（GitHub 中转，非 rsync）：本地 push → AutoDL pull
#   AutoDL: cd /root/autodl-tmp/HelloAgents && git pull origin main
#   注: GitHub 属国外资源，pull 慢就 source /etc/network_turbo；遇 503 就 unset http_proxy https_proxy 直连。
#
# 用法：分阶段执行，每步确认无误再进下一步
#   bash run_base_nothink_autodl.sh template   # 导出+注入+验证 thinking-off 模板（非破坏性）
#   bash run_base_nothink_autodl.sh serve      # vLLM 起 Base-nothink 服务（后台，等就绪）
#   bash run_base_nothink_autodl.sh verify     # curl 验证 thinking 关 + served 名正确
#   bash run_base_nothink_autodl.sh eval       # 同一 harness + v1.1 冻结集跑评测
#
# 前置：Base 模型已在 /root/autodl-tmp/models/Qwen3-14B；SFT 轮次已跑通 vLLM 环境。
# =============================================================================
set -e

# ---- 路径约定（全部放数据盘 /root/autodl-tmp；系统盘 /root 仅 30GB）----
# ⚠️ 上 AutoDL 第一件事：ls 确认下面路径真实存在，不符就改这里
DATA_ROOT=/root/autodl-tmp
BASE_MODEL=${DATA_ROOT}/models/Qwen3-14B           # 原始 Base（保持纯净，不 in-place 改）
REPO_DIR=${DATA_ROOT}/HelloAgents                  # repo（git pull 同步，含本脚本）
SERVED_NAME=qwen3-14b-nothink                       # 区别于 thinking-ON 的 qwen3-14b
NOTHINK_TPL=/root/qwen3_base_nothink.jinja          # 导出的 thinking-off 模板（--chat-template 用）
VLLM_PORT=8000
VLLM_LOG=/root/vllm_base_nothink.log
INDUCTOR_PY=/root/miniconda3/lib/python3.12/site-packages/torch/_inductor/lowering.py

STAGE="${1:-help}"

case "$STAGE" in

# -----------------------------------------------------------------------------
template)
  echo "【template】导出 Base 模板 + 注入 enable_thinking=false（非破坏性，不改原模型目录）"
  python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('${BASE_MODEL}')
tpl = tok.chat_template
assert tpl, 'ERROR: Base 无 chat_template，检查 ${BASE_MODEL}'
if not tpl.lstrip().startswith('{%- set enable_thinking = false'):
    tpl = '{%- set enable_thinking = false -%}\n' + tpl
open('${NOTHINK_TPL}','w').write(tpl)
print('✅ 模板已导出 →', '${NOTHINK_TPL}')
"
  echo "--- 验证 thinking 已关（渲染结尾应含空 <think></think> 块，与 SFT 一致）---"
  python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('${BASE_MODEL}')
tok.chat_template = open('${NOTHINK_TPL}').read()
out = tok.apply_chat_template([{'role':'user','content':'你好'}], tokenize=False, add_generation_prompt=True)
print('渲染结尾:', repr(out[-60:]))
assert '<think>' in out and '</think>' in out, 'ERROR: 未见空 think 块，thinking 没关掉'
print('✅ thinking 已关（空 <think></think> = 思考已完成信号，模型直接作答不思考）')
"
  ;;

# -----------------------------------------------------------------------------
serve)
  echo "【serve】vLLM 起 Base-thinking-OFF 服务 —— 复用 SFT serve 全套排雷，仅换 模型/模板/served名"
  unset http_proxy https_proxy
  # ⚠️ 恢复被 LLaMA-Factory 降级的 vLLM 依赖：LF 要 transformers<=5.8、vLLM 0.32 要 >=5.16.1。
  #   SFT 轮次若已恢复到 5.19 则跳过（幂等）；只升上层包，torch 2.13.0 不动（升完立即断言校验）。
  TRANS_VER=$(python -c "import transformers;print(transformers.__version__)" 2>/dev/null || echo none)
  if [ "$TRANS_VER" != "5.19.0" ]; then
    echo "检测到 transformers=$TRANS_VER（LF 降级），恢复 vLLM 依赖..."
    pip install --upgrade "transformers==5.19.0" "tokenizers==0.23.3" "starlette==1.10.0" "pillow==13.0.0" "datasets==5.1.0" \
      -i https://mirrors.aliyun.com/pypi/simple
    python -c "import torch;assert torch.__version__.startswith('2.13.0'),'torch被改动:'+torch.__version__;print('torch OK',torch.__version__)"
  fi
  # GPU 预检：杀残留服务（SFT 服务可能还占着 8000 端口 / ~71GB 显存）
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  if [ -n "$USED" ] && [ "$USED" -gt 5000 ]; then
    echo "⚠️ GPU0 占用 ${USED}MiB，杀残留 vLLM 服务..."
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 5
  fi
  # 模板必须先生成
  if [ ! -f "$NOTHINK_TPL" ]; then
    echo "❌ 缺 $NOTHINK_TPL，请先跑: bash run_base_nothink_autodl.sh template"; exit 1
  fi
  # ⚠️ 这套 patch 踩了 2 小时坑验证出来，一条都别省（与 SFT serve 完全一致）：
  #   1)unset代理(国内直连) 2)inductor lowering.py:9111 sed patch 3)TORCHDYNAMO_DISABLE=1
  #   4)VLLM_USE_FLASHINFER_SAMPLER=0(否则首请求 JIT 崩溃→评测全 Connection error) 5)--enforce-eager
  sed -i '9111s/import_submodule(kernel)/pass  # PATCHED: skip broken kernel imports/' "$INDUCTOR_PY" 2>/dev/null || true
  python -c "import torch._inductor.lowering; print('inductor OK')"
  export TORCHDYNAMO_DISABLE=1
  export VLLM_USE_FLASHINFER_SAMPLER=0
  nohup python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --served-model-name "$SERVED_NAME" \
    --chat-template "$NOTHINK_TPL" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 8192 \
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
  echo "【verify】curl 验证 thinking 关 + served 名正确"
  echo "--- 1) 纯对话（content 应无大段思考；reasoning 字段应为 null）---"
  curl -s localhost:${VLLM_PORT}/v1/chat/completions -H "Content-Type: application/json" -d "{
    \"model\": \"${SERVED_NAME}\",
    \"messages\": [{\"role\":\"user\",\"content\":\"你好\"}],
    \"max_tokens\": 30
  }" | python -m json.tool
  echo "--- 2) served 模型名确认（应为 ${SERVED_NAME}）---"
  curl -s localhost:${VLLM_PORT}/v1/models | python -m json.tool
  ;;

# -----------------------------------------------------------------------------
eval)
  echo "【eval】同一 harness + 同一 v1.1 冻结集 → Base-nothink（与 SFT-nothink 75% 严格可比）"
  cd "$REPO_DIR"
  export EVAL_STAGE=baseline
  export EVAL_MODEL_NAME=qwen3-14b-nothink     # 结果目录名，不覆盖 83.8%(qwen3-14b)
  export LLM_MODEL_ID=${SERVED_NAME}           # API model 参数，必须 = --served-model-name
  export LLM_API_KEY=token-abc123
  export LLM_BASE_URL=http://localhost:${VLLM_PORT}/v1
  export EVAL_TEMPERATURE=0                    # 贪婪解码，与 Base/SFT 评测一致，保证可复现
  python eval/baseline_eval.py
  echo "✅ Base-nothink 评测完成，结果在 eval/eval_data/results/baseline/qwen3-14b-nothink/v1.1/"
  echo ""
  echo "回传 GitHub（AutoDL 上执行）："
  echo "  cd $REPO_DIR"
  echo "  git add eval/eval_data/results/baseline/qwen3-14b-nothink/ eval/version_log.md"
  echo "  git commit -m 'eval(baseline): Base-thinking-OFF 公平重跑 v1.1（隔离 thinking confound）'"
  echo "  unset http_proxy https_proxy && git push origin main   # 遇503直连；慢则 source /etc/network_turbo"
  ;;

# -----------------------------------------------------------------------------
*)
  echo "用法: bash run_base_nothink_autodl.sh {template|serve|verify|eval}"
  echo "  template  导出 Base 模板 + 注入 enable_thinking=false + 验证（非破坏性，不改原模型）"
  echo "  serve     vLLM 后台起 Base-nothink 服务（等日志 'Application startup complete'）"
  echo "  verify    curl 验证 thinking 关 + served 名正确"
  echo "  eval      跑评测（EVAL_STAGE=baseline, EVAL_MODEL_NAME=qwen3-14b-nothink）"
  echo ""
  echo "标准顺序: template → serve(后台起,等就绪) → verify → eval"
  echo "产出: results/baseline/qwen3-14b-nothink/v1.1/  ——  与 SFT-nothink(75%) 做三方对比"
  ;;
esac
