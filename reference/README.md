# MedAgent - 参考目录

## 概述

本目录存放 MedAgent 项目的参考文档和示例代码，包括 TRL Agentic 训练指南和数据增强管线参考。

## 文件列表

| 文件 | 说明 |
|------|------|
| `reference.md` | 项目参考文档索引 |
| `agentic_rl_trl.ipynb` | TRL Agentic 训练示例 Notebook |
| `dp_training_trl.md` | TRL 数据增强管线训练指南 |

## 文档说明

### 1. TRL Agentic 训练示例 (`agentic_rl_trl.ipynb`)

**内容**: HuggingFace TRL 库的 Agentic RL 训练示例，展示如何使用 GRPO 算法训练带工具调用的语言模型。

**关键技术点**:
- TRL >= 0.27.0 版本
- Transformers 5.0+ (开发版)
- GRPO (Group Relative Policy Optimization)
- vLLM 推理加速
- 工具调用格式定义与解析

**环境要求**:
```bash
pip install "trl[vllm]>=0.27.0"
pip install git+https://github.com/huggingface/transformers.git
pip install jmespath  # TRL 工具调用解析依赖
```

### 2. TRL 数据增强管线训练指南 (`dp_training_trl.md`)

**内容**: 使用 TRL 进行数据增强和模型训练的详细指南。

**关键步骤**:
1. 数据准备与格式化
2. 工具调用数据构建
3. GRPO 训练配置
4. 模型评估与导出

**训练配置示例**:
```python
training_args = GRPOConfig(
    output_dir="./output",
    max_steps=500,
    per_device_train_batch_size=8,
    learning_rate=1e-5,
    use_vllm=True,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen3-4B",
    args=training_args,
    train_dataset=dataset,
    reward_funcs=reward_funcs,
)
```

## 版本要求

| 包 | 最低版本 | 说明 |
|---|---------|------|
| transformers | 5.0.0.dev0 | Agentic RL 训练必须 5.0+ |
| trl | 0.27.0 | 支持 GRPO agentic 训练 |
| vllm | 最新 | 推理加速 |
| peft | 0.7.0 | LoRA 支持 |
| accelerate | 0.24.0 | 分布式训练 |
| jmespath | 1.0.0 | TRL 工具调用解析 |

## 相关资源

### HuggingFace 官方文档
- [TRL 文档](https://huggingface.co/docs/trl)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [PEFT 文档](https://huggingface.co/docs/peft)
- [Accelerate 文档https://huggingface.co/docs/accelerate)

### 相关论文
- GRPO: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03325)
- ReAct: [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03604)

## 相关文档

- `../README.md`: 项目概述
- `../src/README.md`: 源代码目录说明
- `../scripts/README.md`: 脚本目录说明
- `../paper/README.md`: 论文目录说明
