# MedAgent - 结果目录

## 概述

本目录存放 MedAgent 项目的历史评测结果数据，主要用于存储早期评测（10 条样本）和 Rejudge 结果。当前新评测数据存放在 `exp/output/` 目录。

## 文件列表

### 评测结果

| 文件 | 说明 |
|------|------|
| `checkpoint_*.jsonl` | 各模型评测结果 checkpoint (10 条样本) |
| `model_stats_*.json` | 各模型统计信息 (10 条样本) |
| `all_models_summary.json` | 10 条样本综合评测汇总 |
| `rejudge_summary.json` | Rejudge 结果 (1,000 条) |

### 分析脚本

| 文件 | 说明 |
|------|------|
| `analyze_results.py` | 结果分析脚本 |
| `summarize_benchmark.py` | Benchmark 汇总脚本 |

## 与 `exp/output/` 的区别

| 目录 | 评测数据量 | 状态 |
|------|-----------|------|
| `exp/results/` | 10 条样本 + Rejudge 1,000 条 | 历史数据 |
| `exp/output/` | 1,000 条全量评测 | 当前使用 |

## 历史评测结果摘要 (10 条样本)

| 排名 | 模型 | 综合分数 | 诊断准确率 | 治疗准确率 | 禁忌违反率 |
|------|------|---------|-----------|-----------|-----------|
| 1 | deepseek-v3.2 | 3.80 | 30.0% | 90.0% | 10.0% |
| 2 | gpt-5.4 | 3.30 | 40.0% | 60.0% | 0.0% |
| 3 | gemini-3.1-pro-preview | 3.40 | 20.0% | 70.0% | 10.0% |
| 4 | qwen3.5-plus | 3.20 | 40.0% | 80.0% | 10.0% |
| 5 | qwen3.5-35b-a3b | 3.03 | 40.0% | 50.0% | 10.0% |

> ⚠️ 注：10 条样本的评测结果仅供参考，与 1,000 条全量评测结果可能存在差异

## Rejudge 结果摘要 (1,000 条)

| 排名 | 模型 | 综合分数 | 诊断准确率 | 治疗准确率 | 禁忌违反率 |
|------|------|---------|-----------|-----------|-----------|
| 1 | gemini-3.1-pro-preview | 3.45 | 42.1% | 9.3% | 4.9% |
| 2 | gpt-5.4 | 3.44 | 32.2% | 3.9% | 1.7% |
| 3 | qwen3.5-plus | 3.31 | 30.8% | 2.7% | 1.4% |

## 文件格式说明

### checkpoint_*.jsonl

每行一个评测结果 (JSON 格式):

```json
{
  "case_id": "case_001",
  "model": "gpt-5.4",
  "diagnosis": "急性心肌梗死",
  "treatment": "急诊PCI，抗凝治疗",
  "score": 4.5,
  "diagnosis_score": 5,
  "treatment_score": 4,
  "avoid_score": 5,
  "total_cost": 1093,
  "total_steps": 5,
  "total_tokens": 1041,
  "latency": 21.79
}
```

### model_stats_*.json

模型统计信息:

```json
{
  "model": "gpt-5.4",
  "total_cases": 1000,
  "avg_score": 3.69,
  "diagnosis_accuracy": 0.603,
  "treatment_accuracy": 0.194,
  "avoid_violation_rate": 0.028,
  "avg_steps": 4.94,
  "avg_checks": 8.63,
  "avg_tokens": 1041,
  "avg_latency": 21.79,
  "avg_cost": 1093,
  "score_distribution": {
    "excellent": 387,
    "good": 567,
    "medium": 44,
    "poor": 2
  }
}
```

## 相关文档

- `../exp/README.md`: 实验目录说明
- `../exp/output/README.md`: 新评测输出目录说明
