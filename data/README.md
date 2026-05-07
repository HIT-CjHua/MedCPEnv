# MedAgent - 数据目录

## 概述

本目录存放 MedAgent 项目的所有数据文件，包括种子病例、合成病例、评测基准、知识库和价格清单。

## 目录结构

```
data/
├── datasets/                    # 训练/测试/评测数据集
│   ├── train.jsonl              # 训练集 (34,160 条)
│   ├── test.jsonl               # 测试集 (7,323 条)
│   ├── eval.jsonl               # 验证集 (7,320 条)
│   └── benchmark_1000.jsonl     # 基准评测集 (1,000 条)
│
├── knowledge_dataset/           # 知识库原始数据
│   └── ResponseMed.json         # 医学问答数据集 (371,550 条, Alpaca 格式)
│
├── cost_list/                   # 费用评估价格清单
│   └── cost_reference.jsonl     # 医疗项目/药品价格清单 (动态增长)
│
└── seed_cases/                  # 种子病例数据 (177,703 条)
    ├── 种子数据按科室分类...
```

## 数据集说明

### 1. 训练/测试/评测数据集 (`datasets/`)

| 文件 | 数据量 | 用途 |
|------|-------|------|
| `train.jsonl` | 34,160 条 | Agentic RL 训练数据 |
| `test.jsonl` | 7,323 条 | 模型测试集 |
| `eval.jsonl` | 7,320 条 | 模型验证集 |
| `benchmark_1000.jsonl` | 1,000 条 | 基准评测（9 个模型全量评测） |

**数据格式** (JSONL，每行一个 MedicalCase):

```json
{
  "case_id": "case_001",
  "chief_complaint": "患者主诉胸痛...",
  "subjective": [
    {"keywords": ["疼痛性质", "疼痛部位"], "content": "..."}
  ],
  "objective": [
    {"keywords": ["心电图", "心肌酶"], "content": "..."}
  ],
  "ground_truth": {
    "diagnosis": ["急性心肌梗死"],
    "treatment": ["急诊PCI", "抗凝治疗"],
    "avoid": ["剧烈运动"]
  },
  "department": "内科",
  "disease": "急性心肌梗死"
}
```

### 2. 知识库数据 (`knowledge_dataset/`)

| 文件 | 数据量 | 格式 | 用途 |
|------|-------|------|------|
| `ResponseMed.json` | 371,550 条 | JSON 数组 (Alpaca 格式) | 关键词匹配知识检索 |

**数据格式**:

```json
[
  {
    "instruction": "Please answer the following multiple-choice question:\nA factory worker presents with...",
    "input": "",
    "output": "The patient exhibits symptoms including..."
  }
]
```

**检索方式**: 纯关键词匹配（`knowledge_tool_v2.py`），不使用 embedding/reranker。

### 3. 费用评估价格清单 (`cost_list/`)

| 文件 | 说明 |
|------|------|
| `cost_reference.jsonl` | 医疗项目/药品价格清单，用于轨迹成本估算 |

**价格清单格式**:

```json
{"item": "心电图", "price": 50, "category": "检查"}
{"item": "血常规", "price": 30, "category": "化验"}
```

**估算规则**:
- EXAM (检查): 50 元/项
- ASK (问诊): 10 元/项
- KNOWLEDGE (知识查询): 5 元/次

### 4. 种子病例数据 (`seed_cases/`)

- **总量**: 177,703 条
- **来源**: 多科室真实/半真实病例
- **科室分布**:
  - 妇产科: 34,313
  - 内科: 29,677
  - 皮肤性病科: 24,668
  - 儿科: 21,202
  - 眼耳鼻喉科: 13,791
  - 肿瘤科: 10,107
  - 神经科学: 10,009
  - 外科: 9,577
  - 其他: 24,359

## 数据合成管线

从种子数据生成高质量医疗病例的管线 (`src/medagent/data_pipeline.py`):

### 标准管线 (4 阶段)
1. **Generate**: LLM 从种子数据生成医疗病例
2. **Review**: 异构模型审查病例质量
3. **Rewrite**: LLM 根据审查反馈修改病例
4. **Judge**: 医疗专科模型 (Baichuan-M3, 235B MoE) 验证病例

**生成数据**: 6,951+ 条（通过率 99.3%）

### 其他管线
- **大规模管线 (v2)**: 分片处理 + 断点续跑 + 标签均衡采样
- **One-shot 管线**: 单次 LLM 调用（成本降低 50%+）
- **MIMIC-IV 管线**: 从真实出院记录提取

## 相关脚本

| 脚本 | 说明 |
|------|------|
| `src/medagent/data_pipeline.py` | 数据合成管线 |
| `scripts/build_knowledge_base.sh` | 旧知识库构建脚本 (已废弃) |
| `scripts/rag.py` | 独立 RAG CLI 工具 |

## 使用示例

### 加载训练数据

```python
import json

cases = []
with open('data/datasets/train.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        case = json.loads(line)
        cases.append(case)

print(f"Loaded {len(cases)} training cases")
```

### 加载知识库

```python
from src.medagent.knowledge_tool_v2 import KeywordKnowledgeBase

kb = KeywordKnowledgeBase()
kb.load('data/knowledge_dataset/ResponseMed.json')

results = kb.search(keywords=["appendicitis", "diagnosis"], top_k=3)
print(kb.format_results(results))
```

## 数据统计

| 数据集 | 数量 |
|--------|------|
| 种子病例 | 177,703 |
| 合成病例 | 6,951+ |
| 训练集 | 34,160 |
| 测试集 | 7,323 |
| 验证集 | 7,320 |
| 基准评测 | 1,000 |
| 知识库 QA | 371,550 |
