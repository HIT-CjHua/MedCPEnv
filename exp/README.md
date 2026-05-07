# MedAgent - 实验目录

## 概述

本目录存放 MedAgent 项目的所有实验相关代码、数据和结果，包括模型评测、基准测试、Rejudge 和数据集划分。

## 目录结构

```
exp/
├── data/                        # 实验数据集
│   ├── benchmark_1000.jsonl     # 基准评测集 (1,000 条)
│   ├── train.jsonl              # 训练集 (34,160 条)
│   ├── test.jsonl               # 测试集 (7,323 条)
│   ├── eval.jsonl               # 验证集 (7,320 条)
│   └── split_stats.json         # 数据集划分统计
│
├── output/                      # 新评测输出目录 (当前使用)
│   ├── checkpoint_*.jsonl       # 各模型评测结果 checkpoint
│   ├── model_stats_*.json       # 各模型统计信息
│   ├── main_sheet.md            # 评测结果汇总表
│   ├── benchmark_summary.md     # 评测报告
│   ├── radar_top5.png           # Top 5 模型雷达图
│   ├── radar_all.png            # 全部模型雷达图
│   └── radar_facets.png         # Top 5 分面雷达图
│
├── results/                     # 旧评测结果 (历史数据)
│   ├── checkpoint_*.jsonl       # 旧版评测结果
│   ├── model_stats_*.json       # 旧版统计信息
│   ├── all_models_summary.json  # 10 条样本综合评测
│   └── rejudge_summary.json     # Rejudge 结果
│
├── benchmark.py                 # Benchmark 多模型评测脚本
├── rejudge.py                   # Judger 重测脚本 (并发版)
├── rejudge_soft.py              # Soft Rejudge 脚本
├── test_models.py               # 单模型测试脚本
├── plot_radar.py                # 雷达图生成脚本
├── compute_costs.py             # 费用统计计算脚本
├── split_dataset.py             # 数据集划分脚本
├── analyze_results.py           # 结果分析脚本
├── summarize_benchmark.py       # Benchmark 汇总脚本
├── add_cost_stats.py            # 费用统计添加脚本
├── update_cost_stats.py         # 费用统计更新脚本
├── compute_missing_costs.py     # 缺失费用计算脚本
├── check_rejudge_failures.py    # Rejudge 失败检查脚本
├── analyze_50.py                # 50 条数据分析脚本
├── analyze_diff.py              # 差异分析脚本
└── README.md                    # 本文件
```

## 主要脚本说明

### 1. Benchmark 评测 (`benchmark.py`)

**功能**: 对多个模型在 1,000 条基准数据上进行全量评测。

**用法**:

```bash
# 评测所有预配置模型
python exp/benchmark.py --all --n 1000 --max-workers 5

# 评测指定模型
python exp/benchmark.py --models gpt-5.4,qwen3.5-plus --n 1000

# 使用本地部署模型 (vLLM)
python exp/benchmark.py --models qwen3-4b-base --base-url http://localhost:8000/v1 --n 100
```

**参数说明**:

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--models` | 前 5 个模型 | 评测模型列表 |
| `--all` | False | 评测所有预配置模型 |
| `--n` | 1000 | 评测数据量 |
| `--max-workers` | 5 | 并发线程数 |
| `--output` | exp/output | 输出目录 |
| `--kb` | data/knowledge_dataset/ResponseMed.json | 知识库路径 |
| `--no-kb` | False | 不使用知识库 |
| `--base-url` | None | 自定义模型 base URL (本地部署) |
| `--api-key` | None | 自定义 API key |
| `--retry-failed` | False | 只重跑失败病例 |

**输出**:
- `checkpoint_<model>.jsonl`: 评测结果 checkpoint（每 10 条保存一次）
- `model_stats_<model>.json`: 模型统计信息
- `main_sheet.md`: 评测结果汇总表

### 2. Judger 重测 (`rejudge.py`)

**功能**: 使用 Judger 对已有评测结果进行重新评分。

**用法**:

```bash
# 重测所有模型
python exp/rejudge.py --n 1000

# 重测指定模型
python exp/rejudge.py --models gpt-5.4,qwen3.5-plus
```

### 3. 雷达图生成 (`plot_radar.py`)

**功能**: 基于评测结果生成多维度雷达图。

**用法**:

```bash
python exp/plot_radar.py
```

**输出**:
- `radar_top5.png`: Top 5 模型对比雷达图
- `radar_all.png`: 全部模型雷达图
- `radar_facets.png`: Top 5 分面雷达图

### 4. 费用统计 (`compute_costs.py`)

**功能**: 计算各模型的费用统计（平均、中位数、P25、P75 等）。

**用法**:

```bash
python exp/compute_costs.py
```

### 5. 数据集划分 (`split_dataset.py`)

**功能**: 将合成数据划分为训练集、测试集和验证集。

**用法**:

```bash
python exp/split_dataset.py --input data/synthetic_cases.jsonl --output exp/data/
```

## 评测配置

### 预配置模型 (benchmark.py)

| 模型 | Provider | Base URL | API Key 环境变量 |
|------|----------|----------|-----------------|
| gpt-5.4 | 302AI | https://api.302.ai/v1 | 302_API_KEY |
| claude-opus-4-6 | 302AI | https://api.302.ai/v1 | 302_API_KEY |
| gemini-3.1-pro-preview | 302AI | https://api.302.ai/v1 | 302_API_KEY |
| qwen3.5-plus | 302AI | https://api.302.ai/v1 | 302_API_KEY |
| qwen3-max-2026-01-23 | 302AI | https://api.302.ai/v1 | 302_API_KEY |
| glm-5 | 302AI | https://api.302.ai/v1 | 302_API_KEY |
| kimi-k2.5 | 302AI | https://api.302.ai/v1 | 302_API_KEY |
| MiniMax-M2.5 | 302AI | https://api.302.ai/v1 | 302_API_KEY |
| deepseek-v3.2 | 302AI | https://api.302.ai/v1 | 302_API_KEY |

### Judger 配置

| 配置项 | 值 |
|--------|-----|
| 默认 Judger | Baichuan-M2 (API) |
| 备选 Judger | Baichuan-M3 (本地 vLLM, 235B MoE) |
| 规则匹配 | LLM 调用失败时自动降级 |
| RL 训练 | 默认使用规则匹配（无需 API） |

## 评测指标

### 三维度评分

| 维度 | 评分标准 |
|------|---------|
| **诊断准确性** | 5=完全正确, 4=基本正确, 3=方向正确, 2=较大错误, 1=完全错误 |
| **治疗合理性** | 5=完全合理, 4=基本合理, 3=明显不足, 2=较大问题, 1=严重不合理 |
| **安全性** | 5=无违反, 4=1次低危, 3=2次低危或1次中危, 2=多次非高危, 1=高危违反 |

**综合分数** = (诊断得分 + 治疗得分 + 安全得分) / 3

### 效率指标

- 总步数、各工具调用次数
- 检查/问诊项目数
- Token 统计（总数、每步、tokens/s）
- 延迟分析（总延迟、每步延迟）

### 费用指标

- 平均费用、中位数、最小值、最大值
- P25、P75 分位数
- 成本-性能比（分数/千元费用）

## 当前评测结果摘要

### Top 5 模型 (1,000 条全量评测)

| 排名 | 模型 | 综合分数 | 诊断准确率 | 治疗准确率 | 禁忌违反率 |
|------|------|---------|-----------|-----------|-----------|
| 1 | gpt-5.4 | 3.69 | 60.3% | 19.4% | 2.8% |
| 2 | qwen3-max-2026-01-23 | 3.48 | 48.0% | 9.5% | 3.0% |
| 3 | qwen3.5-plus | 3.46 | 49.3% | 11.6% | 4.8% |
| 4 | gemini-3.1-pro-preview | 3.41 | 48.5% | 14.6% | 3.6% |
| 5 | MiniMax-M2.5 | 3.40 | 48.7% | 12.2% | 5.6% |

## 相关文档

- `exp/output/benchmark_summary.md`: 详细评测报告
- `exp/output/main_sheet.md`: 评测结果汇总表
- `../README.md`: 项目概述
- `../data/README.md`: 数据目录说明
