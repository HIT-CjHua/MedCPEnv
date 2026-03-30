# MedAgent - 医疗问诊智能体评测框架

## 项目概述

MedAgent 是一个面向医疗问诊场景的智能体评测与训练框架，支持：

- **Benchmark 评测**：评估 LLM 在医疗问诊场景中的决策能力
- **Agentic RL 训练**：提供 Gym 风格环境，支持强化学习训练

## 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                      MedAgent Framework                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  MedicalCase │    │   MedAgent   │    │   Judger    │     │
│  │   (病例数据)  │───▶│   (智能体)    │───▶│   (评测器)   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                            │                                │
│                            ▼                                │
│                   ┌────────────────┐                        │
│                   │  ToolManager   │                        │
│                   └────────────────┘                        │
│                      │    │    │                            │
│                      ▼    ▼    ▼                            │
│               ┌─────┬─────┬─────┬─────┐                     │
│               │ ASK │EXAM │KNOWL│FINAL│                     │
│               │问诊 │检查 │知识库│诊断 │                     │
│               └─────┴─────┴─────┴─────┘                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 已实现模块

### 1. 数据结构 (`src/schema.py`)

| 类 | 说明 |
|---|------|
| `MedicalItem` | 医疗数据项（keywords + content + necessity） |
| `GroundTruth` | 标准答案（diagnosis + treatment + avoid） |
| `MedicalCase` | 完整病例（主诉 + 主观信息 + 客观信息 + 标准答案） |

### 2. 基础设施 (`src/medagent/`)

| 模块 | 功能 | 默认配置 |
|------|------|----------|
| `llm.py` | LLM 调用（直接/流式/工具调用） | DashScope API |
| `embedding.py` | 文本向量化 | 本地 Qwen3-Embedding (port:8300) |
| `reranker.py` | 相关性重排序 | 本地 Qwen3-Reranker (port:8201) |
| `knowledge_base.py` | 知识库（建库 + 检索 + 重排） | Chroma + ResponseMed |

### 3. 工具系统 (`src/medagent/tool.py`)

| 工具 | 输入 | 功能 |
|------|------|------|
| `AskTool` | keywords | 匹配主观信息（症状、病史） |
| `ExamTool` | keywords | 匹配客观信息（化验、影像） |
| `KnowledgeTool` | query | 检索知识库 + 重排 + 摘要 |

### 4. 智能体 (`src/medagent/agent.py`)

**ReAct 循环：**

```
┌──────────────────────────────────────────────┐
│                                              │
│   ┌──────┐    ┌──────┐    ┌─────────┐       │
│   │  Re  │───▶│ Act  │───▶│ Observe │───────┤
│   │(推理) │    │(执行) │    │ (观察)  │       │
│   └──────┘    └──────┘    └─────────┘       │
│       ▲                              │       │
│       └──────────────────────────────┘       │
│                                              │
└──────────────────────────────────────────────┘
```

**动作空间：**

| 动作 | XML 格式 | 说明 |
|------|----------|------|
| ASK | `<action>ASK</action><keywords>...</keywords>` | 问诊 |
| EXAM | `<action>EXAM</action><keywords>...</keywords>` | 检查 |
| KNOWLEDGE | `<action>KNOWLEDGE</action><query>...</query>` | 知识查询 |
| FINAL | `<action>FINAL</action><diagnosis>...</diagnosis>` | 最终诊断 |

## 使用示例

```python
from src.medagent import MedAgent, LLMClient, KnowledgeBase
from src.schema import MedicalCase

# 加载病例
case = MedicalCase.dict_to_case(case_dict)

# 加载知识库
kb = KnowledgeBase().load("data/knowledge_db")

# 创建 Agent
agent = MedAgent(
    llm_client=LLMClient(),
    case=case,
    knowledge_base=kb,
    max_steps=20,
)

# 运行诊断
result = agent.run()
print(f"诊断: {result['diagnosis']}")
print(f"治疗: {result['treatment']}")
```

## 项目结构

```
MedAgent/
├── src/
│   ├── schema.py              # 病例数据结构
│   ├── prompts.py             # Prompt 模板
│   └── medagent/
│       ├── llm.py             # LLM 客户端
│       ├── embedding.py       # Embedding 客户端
│       ├── reranker.py        # Reranker 客户端
│       ├── knowledge_base.py  # 知识库类
│       ├── tool.py            # 工具类
│       ├── agent.py           # Agent 类
│       └── judger.py          # 评测器（待实现）
│
├── scripts/
│   ├── rag.py                 # 知识库 CLI
│   ├── data_pipeline.py       # 数据生成管线
│   ├── run.py                 # 运行脚本
│   ├── eval.py                # 评测脚本
│   └── train.py               # 训练脚本
│
└── data/
    ├── seed_dataset/          # 种子数据
    └── knowledge_dataset/     # 知识库数据
```

## 规划功能

| 模块 | 状态 | 说明 |
|------|------|------|
| `judger.py` | 待实现 | 多维度评测（诊断准确率、必要检查比例、禁忌项检测） |
| `eval.py` | 待实现 | 批量评测 + 报告生成 |
| `train.py` | 待实现 | RL 训练入口（PPO/DPO） |
| 数据生成管线 | 已有 | 多模型协作生成病例 |

## 技术栈

- **LLM**: Qwen3.5-plus (API) / Qwen3 (本地)
- **Embedding**: Qwen3-Embedding-4B (vLLM)
- **Reranker**: Qwen3-Reranker-4B (vLLM)
- **Vector DB**: Chroma
- **框架**: OpenAI SDK (统一调用接口)

## 下一步计划

1. **评测器实现** - 多维度评估指标
2. **批量评测** - 支持 Benchmark 规模化评测
3. **RL 环境** - 封装为 Gym 接口
4. **训练流程** - PPO/DPO 训练管线

---

*MedAgent - 智能医疗问诊评测与训练平台*