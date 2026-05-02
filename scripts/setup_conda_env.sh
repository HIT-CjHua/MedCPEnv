#!/bin/bash
# MedAgent Agentic RL Conda Environment Setup
#
# 用法:
#   bash scripts/setup_conda_env.sh
#
# 创建名为 medagent 的 conda 环境，满足：
# - MedAgent 项目依赖
# - TRL Agentic RL 训练需求 (transformers 5.0+, trl 0.27+)
# - vLLM 推理加速
# - 多 GPU 分布式训练

set -e

ENV_NAME="medagent"
PYTHON_VERSION="3.11"

echo "============================================================"
echo "MedAgent Agentic RL Conda Environment Setup"
echo "============================================================"
echo "环境名称: ${ENV_NAME}"
echo "Python 版本: ${PYTHON_VERSION}"
echo "============================================================"

# 修复 conda 频道配置 (清华镜像 free 频道已失效)
echo "[0/6] 修复 conda 频道配置..."
conda config --remove-key channels 2>/dev/null || true
conda config --add channels conda-forge 2>/dev/null || true

# 创建 conda 环境
echo "[1/6] 创建 conda 环境..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# 激活环境
echo "[2/6] 激活环境..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# 检测 CUDA 版本
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda.split('.')[0] + '.' + torch.version.cuda.split('.')[1])" 2>/dev/null || echo "unknown")
echo "[3/6] 检测 CUDA 版本: ${CUDA_VERSION}"

# 安装 PyTorch (CUDA 12.1/12.4)
echo "[3/6] 安装 PyTorch..."
# 如果服务器 CUDA >= 12.1，使用 cu121；否则使用 cu118
if command -v nvcc &> /dev/null; then
    NVCC_CUDA=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
    echo "  系统 CUDA: ${NVCC_CUDA}"
fi

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装核心依赖
echo "[4/6] 安装核心依赖..."
pip install --upgrade pip

# 安装 TRL 及其依赖 (包含 transformers, accelerate, peft)
pip install "trl[vllm]>=0.27.0"

# 安装 transformers 开发版 (5.0+，agentic 训练需要)
pip install git+https://github.com/huggingface/transformers.git

# 安装其他必需依赖
pip install \
    "openai>=1.0.0" \
    "python-dotenv>=1.0.0" \
    "tqdm>=4.65.0" \
    "chromadb>=0.4.0" \
    "numpy>=1.24.0" \
    "datasets>=2.14.0" \
    "accelerate>=0.24.0" \
    "jmespath>=1.0.0" \
    "peft>=0.7.0"

# 安装可选依赖 (flash-attn 可能编译失败，注释掉)
echo "[5/6] 安装可选依赖..."
pip install \
    "trackio" \
    "bitsandbytes" \
    "optimum"

# flash-attn 需要编译，如果 CUDA 版本不匹配可能失败
# pip install "flash-attn>=2.5.0" --no-build-isolation

# 验证安装
echo ""
echo "[6/6] 验证安装..."
echo "============================================================"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Count: {torch.cuda.device_count()}')
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')

import transformers
print(f'Transformers: {transformers.__version__}')

import trl
print(f'TRL: {trl.__version__}')

from trl import GRPOTrainer, GRPOConfig
print('GRPOTrainer: OK')

import vllm
print(f'vLLM: {vllm.__version__}')

import peft
print(f'PEFT: {peft.__version__}')

import accelerate
print(f'Accelerate: {accelerate.__version__}')
"

echo ""
echo "============================================================"
echo "环境配置完成!"
echo "============================================================"
echo ""
echo "激活环境: conda activate ${ENV_NAME}"
echo "安装项目: pip install -e /path/to/MedAgent"
echo "运行训练: bash scripts/train_multi_gpu.sh --model Qwen/Qwen3-4B"
echo "============================================================"
