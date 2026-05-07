# MedAgent - 源代码目录

## 概述

本目录存放 MedAgent 项目的核心源代码，包括智能体、工具系统、知识库、评测器和成本评估模块。

## 目录结构

```
src/
├── medagent/                    # 核心模块
│   ├── __init__.py              # 模块导出
│   ├── agent.py                 # MedAgent 智能体 (ReAct 循环)
│   ├── tool.py                  # 工具系统 (ASK, EXAM, KNOWLEDGE)
│   ├── knowledge_tool_v2.py     # 关键词匹配知识工具
│   ├── knowledge_base.py        # 旧 RAG 知识库 (保留用于 rag.py)
│   ├── judger.py                # 评测器 (三维度评分)
│   ├── cost.py                  # 成本评估器
│   ├── llm.py                   # LLM 客户端
│   ├── embedding.py             # Embedding 客户端 (旧)
│   └── reranker.py              # Reranker 客户端 (旧)
│
└── schema.py                    # 数据结构定义
```

## 核心模块说明

### 1. Agent 智能体 (`agent.py`)

**功能**: MedAgent 智能体的核心 ReAct 循环实现。

**核心类**: `MedAgent`

**动作空间**:

| 动作 | 功能 | 数据来源 |
|------|------|---------|
| `ASK` | 问诊，获取患者主观信息 | 匹配 `case.subjective` 中的 keywords |
| `EXAM` | 检查，获取患者客观信息 | 匹配 `case.objective` 中的 keywords |
| `KNOWLEDGE` | 知识库查询 | 关键词匹配检索 ResponseMed.json |
| `FINAL` | 最终诊断与治疗建议 | Agent 自身推理输出 |

**输出格式** (XML 标签):

```xml
<act>
    <action>ASK</action>
    <keywords>疼痛性质, 疼痛部位, 持续时间</keywords>
</act>

<act>
    <action>FINAL</action>
    <diagnosis>急性心肌梗死</diagnosis>
    <treatment>急诊PCI，抗凝治疗</treatment>
</act>
```

**关键参数**:

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `max_steps` | 20 | 最大交互步数 |
| `top_k` | 3 | 知识检索返回条数 |
| `verbose` | True | 是否打印详细日志 |

**使用示例**:

```python
from src.medagent import LLMClient, MedAgent, KeywordKnowledgeBase
from src.schema import MedicalCase

# 初始化
llm = LLMClient(api_key="sk-xxx")
kb = KeywordKnowledgeBase()
kb.load("data/knowledge_dataset/ResponseMed.json")

# 创建 Agent
agent = MedAgent(
    llm_client=llm,
    case=case,
    knowledge_base=kb,
    max_steps=20,
    top_k=3,
)

# 运行
result = agent.run()
print(f"Diagnosis: {result.diagnosis}")
print(f"Treatment: {result.treatment}")
```

### 2. 工具系统 (`tool.py`)

**功能**: 定义 Agent 可用的三种工具（ASK, EXAM, KNOWLEDGE）。

**核心类**:

| 类 | 说明 |
|------|------|
| `BaseTool` | 工具基类 |
| `AskTool` | 问诊工具 |
| `ExamTool` | 检查工具 |
| `KnowledgeTool` | 知识查询工具 (关键词匹配) |
| `ToolManager` | 工具管理器 |

**工具描述格式**:

```python
ToolInfo(
    name="KNOWLEDGE",
    description="Knowledge base query tool using keyword matching",
    input_desc="keywords (List[str]) - keyword list for querying",
    output_desc="String - top-k matched QA records",
)
```

**匹配逻辑**:

- **双向关键词匹配**: `kw in item_kw` 或 `item_kw in kw`
- **未匹配**: 返回"相关信息不明"

### 3. 知识工具 (`knowledge_tool_v2.py`)

**功能**: 基于关键词匹配的医学知识库检索工具。

**核心类**:

| 类 | 说明 |
|------|------|
| `KeywordKnowledgeBase` | 关键词匹配知识库 |
| `KeywordKnowledgeTool` | 关键词匹配知识工具 |

**数据源**: `data/knowledge_dataset/ResponseMed.json` (371,550 条 QA)

**检索方式**:

1. 遍历所有 QA 记录
2. 计算每个关键词在 `full_text` 中的命中次数
3. 按命中次数降序排序
4. 返回 top-k 条完整 QA 记录

**使用示例**:

```python
from src.medagent.knowledge_tool_v2 import KeywordKnowledgeBase

kb = KeywordKnowledgeBase()
kb.load("data/knowledge_dataset/ResponseMed.json")

results = kb.search(keywords=["appendicitis", "diagnosis"], top_k=3)
print(kb.format_results(results))
```

### 4. 评测器 (`judger.py`)

**功能**: 对 Agent 输出进行三维度评分（诊断、治疗、安全）。

**核心类**: `Judger`

**评分维度**:

| 维度 | 评分标准 (1-5 分) |
|------|------------------|
| **诊断准确性** | 5=完全正确, 4=基本正确, 3=方向正确, 2=较大错误, 1=完全错误 |
| **治疗合理性** | 5=完全合理, 4=基本合理, 3=明显不足, 2=较大问题, 1=严重不合理 |
| **安全性** | 5=无违反, 4=1次低危, 3=2次低危或1次中危, 2=多次非高危, 1=高危违反 |

**评估模式**:

| 模式 | 说明 |
|------|------|
| **LLM 评估** | 使用 Baichuan-M2 API 进行智能评分 |
| **规则匹配** | LLM 调用失败时自动降级为字符串匹配 |
| **纯规则模式** | `use_rule_only=True`，仅使用规则匹配，无需 API |

**使用示例**:

```python
from src.medagent import Judger

# LLM 模式 (需要 BAICHUAN_API_KEY)
judger = Judger()

# 纯规则模式 (无需 API)
judger = Judger(use_rule_only=True)

# 评估
result = judger.evaluate(
    trajectory=agent.trajectory,
    agent_diagnosis=result.diagnosis,
    agent_treatment=result.treatment,
    ground_truth=case.ground_truth,
)

print(f"综合分数: {result.total_score}")
print(f"诊断分数: {result.diagnosis_score}")
print(f"治疗分数: {result.treatment_score}")
print(f"安全分数: {result.avoid_score}")
```

### 5. 成本评估器 (`cost.py`)

**功能**: 估算 Agent 轨迹的医疗费用。

**核心类**: `CostEvaluator`

**估算规则**:

| 工具 | 费用 |
|------|------|
| EXAM (检查) | 50 元/项 |
| ASK (问诊) | 10 元/项 |
| KNOWLEDGE (知识查询) | 5 元/次 |

**价格清单**: `data/cost_list/cost_reference.jsonl`

**使用示例**:

```python
from src.medagent.cost import CostEvaluator

evaluator = CostEvaluator(price_list_path="data/cost_list/cost_reference.jsonl")
cost = evaluator.estimate_trajectory_cost(agent.trajectory)
print(f"Estimated cost: {cost} yuan")
```

### 6. LLM 客户端 (`llm.py`)

**功能**: 统一的 LLM API 客户端。

**核心类**: `LLMClient`

**支持的 API**:

| Provider | Base URL | API Key 环境变量 |
|----------|----------|-----------------|
| 百炼 codingplan | https://coding.dashscope.aliyuncs.com/v1 | DASHSCOPE_API_KEY_CP |
| 百炼 normal | https://dashscope.aliyuncs.com/compatible-mode/v1 | DASHSCOPE_API_KEY |
| 百川 | https://api.baichuan-ai.com/v1 | BAICHUAN_API_KEY |
| 302AI | https://api.302.ai/v1 | 302_API_KEY |
| 自定义 | 任意 OpenAI 兼容 API | CUSTOM_API_KEY |

**使用示例**:

```python
from src.medagent.llm import LLMClient

# 从环境变量读取 API key
client = LLMClient(model_name="gpt-4")

# 直接指定 API key
client = LLMClient(
    model_name="qwen-plus",
    base_url="https://api.302.ai/v1",
    api_key="sk-xxx",
)

# 调用
response = client.chat(
    messages=[{"role": "user", "content": "你好"}],
    temperature=0.7,
)
print(response)
```

## 数据结构 (`schema.py`)

**核心类**:

| 类 | 说明 |
|------|------|
| `MedicalCase` | 医疗病例数据结构 |
| `MedicalItem` | 医疗信息项 (关键词-内容配对) |
| `GroundTruth` | 标准答案 (诊断、治疗、禁忌) |
| `EvalResult` | 评测结果数据结构 |
| `EfficiencyStats` | 效率统计数据结构 |

**MedicalCase 结构**:

```python
@dataclass
class MedicalCase:
    case_id: str                          # 病例 ID
    chief_complaint: str                  # 主诉
    subjective: List[MedicalItem]         # 主观信息 (问诊获取)
    objective: List[MedicalItem]          # 客观信息 (检查获取)
    ground_truth: GroundTruth             # 标准答案
    department: str                       # 科室
    disease: str                          # 疾病名称
```

## 模块依赖关系

```
agent.py
  ├── tool.py
  │   └── knowledge_tool_v2.py
  ├── judger.py
  │   └── llm.py
  ├── cost.py
  │   └── llm.py
  └── schema.py

llm.py (独立模块)
knowledge_tool_v2.py (独立模块)
knowledge_base.py (旧模块，仅 rag.py 使用)
embedding.py (旧模块，已注释)
reranker.py (旧模块，已注释)
```

## 相关文档

- `../README.md`: 项目概述
- `../data/README.md`: 数据目录说明
- `../exp/README.md`: 实验目录说明
- `../scripts/README.md`: 脚本目录说明
