# MedAgent - 成本数据目录

## 概述

本目录存放 MedAgent 项目的成本评估相关数据，包括价格清单和成本参考文件。

## 文件列表

| 文件 | 说明 |
|------|------|
| `cost_reference.jsonl` | 医疗项目/药品价格清单 |

## 价格清单格式

```json
{"item": "心电图", "price": 50, "category": "检查"}
{"item": "血常规", "price": 30, "category": "化验"}
{"item": "胸部CT", "price": 300, "category": "影像"}
{"item": "阿莫西林", "price": 15, "category": "药品"}
```

## 成本估算规则

| 工具 | 费用 |
|------|------|
| EXAM (检查) | 50 元/项 |
| ASK (问诊) | 10 元/项 |
| KNOWLEDGE (知识查询) | 5 元/次 |

## 使用方式

```python
from src.medagent.cost import CostEvaluator

evaluator = CostEvaluator(price_list_path="data/cost_list/cost_reference.jsonl")
cost = evaluator.estimate_trajectory_cost(trajectory)
```

## 相关文档

- `../data/README.md`: 数据目录说明
- `../src/README.md`: 源代码目录说明
