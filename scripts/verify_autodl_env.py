#!/usr/bin/env python3
"""
AutoDL环境验证脚本（针对配置C优化）

配置C：2×A800-80GB-NVLink，36 vCPU，240GB内存，210GB存储
创建时间：2026-10-31
用途：验证AutoDL环境是否正确配置，确保SFT/RL训练可以正常进行

使用方法：
    python scripts/verify_autodl_env.py
"""

import sys
import subprocess
import shutil
from pathlib import Path


def print_header(title: str):
    """打印标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_section(title: str):
    """打印章节标题"""
    print(f"\n【{title}】")
    print("-" * 80)


def check_python_version():
    """检查Python版本"""
    print_section("Python版本检查")
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 12:
        print("✅ Python 3.12 - 兼容HelloAgents")
        return True
    else:
        print(f"⚠️  Python版本不是3.12，HelloAgents要求>=3.12,<3.14")
        return False


def check_pytorch():
    """检查PyTorch版本和CUDA"""
    print_section("PyTorch和CUDA检查")
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU可用: {torch.cuda.is_available()}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        
        if torch.__version__.startswith("2.5"):
            print("✅ PyTorch 2.5.1 - 稳定版")
        else:
            print(f"⚠️  PyTorch版本不是2.5.1")
        
        if torch.version.cuda and torch.version.cuda.startswith("12.4"):
            print("✅ CUDA 12.4 - 稳定版")
        else:
            print(f"⚠️  CUDA版本不是12.4")
        
        return torch.cuda.is_available()
    except ImportError:
        print("❌ PyTorch未安装")
        return False


def check_gpu_details():
    """检查GPU详细信息（配置C是2卡）"""
    print_section("GPU详细信息检查（配置C是2卡）")
    try:
        import torch
        
        gpu_count = torch.cuda.device_count()
        print(f"GPU数量: {gpu_count}")
        
        if gpu_count == 2:
            print("✅ GPU数量正确（配置C是2卡）")
        else:
            print(f"⚠️  GPU数量不是2（配置C应该是2卡）")
        
        for i in range(gpu_count):
            print(f"\nGPU {i}:")
            print(f"  名称: {torch.cuda.get_device_name(i)}")
            
            # 获取GPU内存
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  显存: {total_memory:.1f}GB")
            
            if total_memory >= 79:  # A800-80GB
                print(f"  ✅ 显存充足（A800-80GB）")
            else:
                print(f"  ⚠️  显存不足（应该是80GB）")
        
        return gpu_count == 2
    except Exception as e:
        print(f"❌ GPU检查失败: {e}")
        return False


def check_nvlink():
    """检查NVLink状态（关键！配置C带NVLink）"""
    print_section("NVLink状态检查（关键！配置C带NVLink）")
    try:
        # 使用nvidia-smi检查NVLink状态
        result = subprocess.run(
            ["nvidia-smi", "nvlink", "--status"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✅ NVLink状态正常")
            print("\nNVLink详细信息:")
            print(result.stdout)
            return True
        else:
            print("⚠️  NVLink状态检查失败")
            print(f"错误信息: {result.stderr}")
            return False
    except FileNotFoundError:
        print("⚠️  nvidia-smi命令不存在")
        return False
    except subprocess.TimeoutExpired:
        print("⚠️  NVLink状态检查超时")
        return False
    except Exception as e:
        print(f"⚠️  NVLink状态检查失败: {e}")
        return False


def check_memory():
    """检查内存（配置C是240GB）"""
    print_section("内存检查（配置C是240GB）")
    try:
        # 使用free命令检查内存
        result = subprocess.run(
            ["free", "-h"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("内存信息:")
            print(result.stdout)
            
            # 解析内存总量
            lines = result.stdout.split("\n")
            for line in lines:
                if line.startswith("Mem:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        total_memory = parts[1]
                        print(f"\n总内存: {total_memory}")
                        
                        # 检查是否是240GB
                        if "240" in total_memory or "256" in total_memory:
                            print("✅ 内存充足（配置C是240GB）")
                            return True
                        else:
                            print(f"⚠️  内存不是240GB（配置C应该是240GB）")
                            return False
            
            return False
        else:
            print("⚠️  内存检查失败")
            return False
    except FileNotFoundError:
        print("⚠️  free命令不存在")
        return False
    except Exception as e:
        print(f"⚠️  内存检查失败: {e}")
        return False


def check_cpu():
    """检查CPU（配置C是36 vCPU）"""
    print_section("CPU检查（配置C是36 vCPU）")
    try:
        # 使用nproc命令检查CPU核心数
        result = subprocess.run(
            ["nproc"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            cpu_count = result.stdout.strip()
            print(f"CPU核心数: {cpu_count}")
            
            if cpu_count == "36":
                print("✅ CPU核心数正确（配置C是36 vCPU）")
                return True
            else:
                print(f"⚠️  CPU核心数不是36（配置C应该是36 vCPU）")
                return False
        else:
            print("⚠️  CPU检查失败")
            return False
    except FileNotFoundError:
        print("⚠️  nproc命令不存在")
        return False
    except Exception as e:
        print(f"⚠️  CPU检查失败: {e}")
        return False


def check_storage():
    """检查存储（配置C是210GB）"""
    print_section("存储检查（配置C是210GB）")
    try:
        # 使用df命令检查存储
        result = subprocess.run(
            ["df", "-h"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("存储信息:")
            print(result.stdout)
            return True
        else:
            print("⚠️  存储检查失败")
            return False
    except FileNotFoundError:
        print("⚠️  df命令不存在")
        return False
    except Exception as e:
        print(f"⚠️  存储检查失败: {e}")
        return False


def check_dependencies():
    """检查依赖包安装状态"""
    print_section("依赖包安装状态检查")
    
    dependencies = {
        "transformers": "Transformers（模型加载）",
        "peft": "Peft（LoRA训练）",
        "verl": "veRL（RL训练）",
        "vllm": "vLLM（推理加速）",
        "hello_agents": "HelloAgents（Agent框架）",
    }
    
    all_installed = True
    for package, description in dependencies.items():
        try:
            __import__(package)
            print(f"✅ {description} - 已安装")
        except ImportError:
            print(f"❌ {description} - 未安装")
            all_installed = False
    
    return all_installed


def check_model():
    """检查Qwen3-14B模型是否已下载"""
    print_section("Qwen3-14B模型检查")
    
    model_path = Path("/root/models/Qwen3-14B")
    
    if model_path.exists():
        print(f"✅ 模型目录存在: {model_path}")
        
        # 检查模型文件
        model_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
        if model_files:
            print(f"✅ 模型文件存在（{len(model_files)}个文件）")
            
            # 计算模型大小
            total_size = sum(f.stat().st_size for f in model_files) / 1024**3
            print(f"模型大小: {total_size:.1f}GB")
            
            if total_size >= 25:  # Qwen3-14B约28GB
                print("✅ 模型大小正常（Qwen3-14B约28GB）")
                return True
            else:
                print("⚠️  模型大小异常（可能下载不完整）")
                return False
        else:
            print("⚠️  模型文件不存在（可能下载不完整）")
            return False
    else:
        print(f"❌ 模型目录不存在: {model_path}")
        print("   请先下载模型:")
        print("   huggingface-cli download Qwen/Qwen3-14B-Instruct --local-dir /root/models/Qwen3-14B")
        return False


def main():
    """主函数"""
    print_header("AutoDL环境验证脚本（配置C）")
    
    print("\n配置C信息：")
    print("- GPU: 2×A800-80GB-NVLink（160GB显存）")
    print("- CPU: 36 vCPU Intel Xeon Platinum 8374C")
    print("- 内存: 240GB")
    print("- 存储: 210GB（系统盘30GB + 数据盘180GB）")
    print("- PyTorch: 2.5.1, Python: 3.12, CUDA: 12.4")
    
    # 执行所有检查
    results = {
        "Python版本": check_python_version(),
        "PyTorch和CUDA": check_pytorch(),
        "GPU详细信息": check_gpu_details(),
        "NVLink状态": check_nvlink(),
        "内存": check_memory(),
        "CPU": check_cpu(),
        "存储": check_storage(),
        "依赖包": check_dependencies(),
        "Qwen3-14B模型": check_model(),
    }
    
    # 打印总结
    print_header("验证总结")
    
    print("\n检查结果：")
    for check_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {check_name}: {status}")
    
    # 计算通过率
    passed_count = sum(1 for passed in results.values() if passed)
    total_count = len(results)
    pass_rate = passed_count / total_count * 100
    
    print(f"\n通过率: {passed_count}/{total_count} ({pass_rate:.1f}%)")
    
    if passed_count == total_count:
        print("\n🎉 所有检查通过！配置C环境已就绪，可以开始SFT/RL训练！")
        return 0
    elif pass_rate >= 80:
        print("\n⚠️  大部分检查通过，但有一些问题需要解决。")
        print("   请检查失败的项，并参考 实验记录/AutoDL环境配置.md 解决问题。")
        return 1
    else:
        print("\n❌ 很多检查失败，环境配置有问题。")
        print("   请重新运行环境安装脚本: bash scripts/setup_autodl_env.sh")
        return 1


if __name__ == "__main__":
    sys.exit(main())
