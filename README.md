# MedCPEnv

面向医疗智能体评测与 Agentic RL 的模拟医院环境。

![MedCPEnv](MedCPEnv.png)

上图展示了 MedCPEnv 的整体流程：智能体围绕主诉进行问诊、检查和知识检索，生成 trajectory 后，再进入评测、打分与 reward 计算环节。

MedCPEnv 用于研究和训练 LLM 在模拟医院场景中的决策能力。项目把问诊、检查、知识检索等能力封装成统一的 tooluse 接口，并配套合成病例、评测器、成本统计和训练脚本，方便做 benchmark、对比实验和强化学习训练。

> 本项目仅用于研究、评测与模拟训练，不用于真实临床诊疗。

## 项目特点

- 统一的医疗工具层，覆盖 `ASK`、`EXAM`、`KNOWLEDGE` 等交互
- 基于合成病例的可控环境，便于重复实验和横向比较
- 支持诊断、治疗、安全性、效率与成本等多维评估
- 提供 benchmark、rejudge、雷达图和训练脚本
- 兼容本地部署模型与 OpenAI 兼容接口

## 目录说明

- `src/`：核心代码，包含智能体、工具系统、知识库、评测器与数据结构
- `exp/`：实验代码，包含 benchmark、分析、rejudge 与结果汇总
- `scripts/`：训练、评测、部署与环境配置脚本

更多细节见：

- [src/README.md](src/README.md)
- [exp/README.md](exp/README.md)
- [scripts/README.md](scripts/README.md)

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

## 数据集

项目相关数据已发布在 Hugging Face：

- [skylenage/MedCPEnv-Dataset](https://huggingface.co/datasets/skylenage/MedCPEnv-Dataset)

### 配置环境变量

按需配置对应的模型 API Key。常见变量名可参考 `src/README.md` 和 `scripts/README.md`，例如：

```bash
export BAICHUAN_API_KEY="sk-xxx"
export DASHSCOPE_API_KEY_CP="sk-xxx"
```

### 常用命令

```bash
# 运行基准评测
python exp/benchmark.py --all --n 1000 --max-workers 5

# 重新评分已有结果
python exp/rejudge.py --n 1000

# 生成雷达图
python exp/plot_radar.py

# 启动 Agentic RL 训练
bash scripts/agentic_rl.sh
```

## 核心能力

### 统一工具层

智能体通过固定动作空间与环境交互，主要包括：

- `ASK`：获取患者主观信息
- `EXAM`：获取患者客观信息
- `KNOWLEDGE`：检索医学知识
- `FINAL`：输出最终诊断与治疗建议

### 评测与分析

项目内置诊断、治疗与安全性评测，并支持效率、token、延迟和成本统计，便于分析不同模型在医疗任务中的行为差异。

### 训练与实验

仓库提供多种训练和评测脚本，可用于：

- 本地模型测试
- 多模型 benchmark
- Agentic RL 训练
- 结果汇总与可视化

## 相关文档

- [src/README.md](src/README.md)
- [exp/README.md](exp/README.md)
- [scripts/README.md](scripts/README.md)
