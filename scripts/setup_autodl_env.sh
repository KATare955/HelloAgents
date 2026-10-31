#!/bin/bash
# AutoDL环境安装脚本（针对配置C优化）
# 配置C：2×A800-80GB-NVLink，36 vCPU，240GB内存，210GB存储
# 创建时间：2026-10-31
# 用途：在AutoDL上安装SFT/RL训练所需的全部环境

set -e  # 遇到错误立即退出

echo "=========================================="
echo "AutoDL环境安装脚本（配置C）"
echo "=========================================="
echo ""
echo "配置C信息："
echo "- GPU: 2×A800-80GB-NVLink（160GB显存）"
echo "- CPU: 36 vCPU Intel Xeon Platinum 8374C"
echo "- 内存: 240GB"
echo "- 存储: 210GB（系统盘30GB + 数据盘180GB）"
echo "- PyTorch: 2.5.1, Python: 3.12, CUDA: 12.4"
echo ""

# ==========================================
# 步骤1：环境验证
# ==========================================
echo "【步骤1】环境验证"
echo "------------------------------------------"

# 检查Python版本
echo "检查Python版本..."
python --version

# 检查PyTorch版本
echo "检查PyTorch版本..."
python -c "import torch; print('PyTorch:', torch.__version__)"

# 检查CUDA版本
echo "检查CUDA版本..."
python -c "import torch; print('CUDA:', torch.version.cuda)"

# 检查GPU是否可用
echo "检查GPU是否可用..."
python -c "import torch; print('GPU available:', torch.cuda.is_available())"

# 检查GPU数量（配置C是2卡）
echo "检查GPU数量（配置C是2卡）..."
python -c "import torch; print('GPU count:', torch.cuda.device_count())"

# 检查GPU信息
echo "检查GPU信息..."
nvidia-smi

# 检查NVLink状态（关键！配置C带NVLink）
echo "检查NVLink状态（关键！配置C带NVLink）..."
nvidia-smi nvlink --status || echo "⚠️  NVLink状态检查失败，请手动确认"

# 检查2卡详细信息
echo "检查2卡详细信息..."
python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# 检查内存（配置C是240GB）
echo "检查内存（配置C是240GB）..."
free -h

# 检查CPU（配置C是36 vCPU）
echo "检查CPU（配置C是36 vCPU）..."
nproc

# 检查存储（配置C是210GB）
echo "检查存储（配置C是210GB）..."
df -h

echo ""
echo "✅ 环境验证完成！"
echo ""

# ==========================================
# 步骤2：安装veRL
# ==========================================
echo "【步骤2】安装veRL"
echo "------------------------------------------"
pip install verl
python -c "import verl; print('veRL version:', verl.__version__)" || echo "⚠️  veRL版本检查失败"
echo "✅ veRL安装完成！"
echo ""

# ==========================================
# 步骤3：安装vLLM
# ==========================================
echo "【步骤3】安装vLLM"
echo "------------------------------------------"
pip install vllm
python -c "import vllm; print('vLLM version:', vllm.__version__)" || echo "⚠️  vLLM版本检查失败"
echo "✅ vLLM安装完成！"
echo ""

# ==========================================
# 步骤4：安装Transformers和Peft
# ==========================================
echo "【步骤4】安装Transformers和Peft"
echo "------------------------------------------"
pip install "transformers>=4.45.0"
pip install peft
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import peft; print('Peft:', peft.__version__)"
echo "✅ Transformers和Peft安装完成！"
echo ""

# ==========================================
# 步骤5：安装HelloAgents
# ==========================================
echo "【步骤5】安装HelloAgents"
echo "------------------------------------------"
# 注意：需要先上传HelloAgents代码到AutoDL
if [ -d "/root/HelloAgents" ]; then
    cd /root/HelloAgents
    pip install -e .
    python -c "import hello_agents; print('HelloAgents installed')"
    echo "✅ HelloAgents安装完成！"
else
    echo "⚠️  HelloAgents目录不存在，请先上传代码到/root/HelloAgents"
    echo "   上传命令（在本地执行）："
    echo "   rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude '.git' \\"
    echo "     /Users/kay/Desktop/hello-agents-family/HelloAgents/ \\"
    echo "     root@{主机地址}:/root/HelloAgents/"
fi
echo ""

# ==========================================
# 步骤6：下载Qwen3-14B模型
# ==========================================
echo "【步骤6】下载Qwen3-14B模型"
echo "------------------------------------------"
pip install huggingface-hub
mkdir -p /root/models

# 检查模型是否已下载
if [ -d "/root/models/Qwen3-14B" ] && [ "$(ls -A /root/models/Qwen3-14B 2>/dev/null)" ]; then
    echo "✅ Qwen3-14B模型已存在，跳过下载"
    ls -lh /root/models/Qwen3-14B/
else
    echo "开始下载Qwen3-14B模型（约28GB，需要1-2小时）..."
    huggingface-cli download Qwen/Qwen3-14B-Instruct --local-dir /root/models/Qwen3-14B
    ls -lh /root/models/Qwen3-14B/
    echo "✅ Qwen3-14B模型下载完成！"
fi
echo ""

# ==========================================
# 步骤7：完整验证
# ==========================================
echo "【步骤7】完整验证"
echo "------------------------------------------"
python << 'EOF'
import torch
import transformers
import peft

print("=" * 80)
print("环境验证（配置C）")
print("=" * 80)

# PyTorch
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        # 检查GPU内存
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  显存: {total_memory:.1f}GB")

# Transformers
print(f"\nTransformers: {transformers.__version__}")

# Peft
print(f"\nPeft: {peft.__version__}")

# 检查veRL和vLLM
try:
    import verl
    print(f"\nveRL: {verl.__version__}")
except ImportError:
    print("\n⚠️  veRL未安装")

try:
    import vllm
    print(f"vLLM: {vllm.__version__}")
except ImportError:
    print("⚠️  vLLM未安装")

# 检查HelloAgents
try:
    import hello_agents
    print(f"\nHelloAgents: 已安装")
except ImportError:
    print("\n⚠️  HelloAgents未安装")

print("\n" + "=" * 80)
print("✅ 环境验证完成！配置C（2×A800-80GB-NVLink，240GB内存）已就绪！")
print("=" * 80)
EOF

echo ""
echo "=========================================="
echo "✅ AutoDL环境安装完成！（配置C）"
echo "=========================================="
echo ""
echo "配置C信息："
echo "- GPU: 2×A800-80GB-NVLink（160GB显存）"
echo "- CPU: 36 vCPU"
echo "- 内存: 240GB"
echo "- 存储: 210GB"
echo ""
echo "下一步："
echo "1. 运行Baseline评测（Qwen3-14B）"
echo "   cd /root/HelloAgents && python eval/baseline_eval.py"
echo ""
echo "2. 开始SFT训练（利用2卡NVLink）"
echo "   参考：实验记录/AutoDL环境配置.md"
echo ""
echo "3. 开始RL训练（利用NVLink加速）"
echo "   参考：实验记录/AutoDL环境配置.md"
echo ""
echo "注意事项："
echo "- 使用LoRA训练（节省存储）"
echo "- 利用NVLink加速（多GPU并行训练）"
echo "- 利用240GB内存（处理更大数据集）"
echo "- 定期清理checkpoint（只保存最好的3-5个）"
echo "- 监控资源使用（nvidia-smi、df -h、free -h）"
echo ""
