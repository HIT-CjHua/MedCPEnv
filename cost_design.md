# MedAgent 医疗成本标注设计

> 创建日期: 2026-04-02
> 状态: 设计讨论

---

## 1. 背景

医疗成本是医疗决策中的重要考量因素。在 MedAgent 系统中，成本信息可以用于：
- **诊断效率评估**: Agent 是否开具了不必要的检查
- **治疗经济学评估**: 治疗方案的成本效益分析
- **RL训练奖励**: 将成本效率纳入奖励函数

---

## 2. 成本来源分析

### 2.1 医疗成本的两个阶段

```
┌─────────────────────────────────────────────────────────────────┐
│                      医疗过程成本构成                             │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 阶段1: 诊断过程成本 (Diagnostic Cost)                    │    │
│  │                                                          │    │
│  │ 来源: Agent 调用 EXAM 工具触发的检查项目                 │    │
│  │ 特点:                                                    │    │
│  │   - 由 Agent 决策产生                                    │    │
│  │   - 与 objective 中的 MedicalItem 一一对应               │    │
│  │   - 可在数据合成时确定                                    │    │
│  │ 示例:                                                    │    │
│  │   - 血常规: 20-30元                                      │    │
│  │   - 心电图: 30-50元                                      │    │
│  │   - CT平扫: 200-400元                                    │    │
│  │   - MRI: 400-800元                                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 阶段2: 治疗方案成本 (Treatment Cost)                     │    │
│  │                                                          │    │
│  │ 来源: ground_truth.treatment 中的药物/手术/操作          │    │
│  │ 特点:                                                    │    │
│  │   - 由诊断结果确定                                       │    │
│  │   - 需要从文本中提取付费单元                              │    │
│  │   - 可能在轨迹完成后才能确定                              │    │
│  │ 示例:                                                    │    │
│  │   - 阿莫西林胶囊: 15元/盒                                │    │
│  │   - PCI手术: 25000-35000元                               │    │
│  │   - 静脉输液: 20-50元/次                                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 两类成本的对比

| 维度 | 诊断成本 (EXAM) | 治疗成本 (Treatment) |
|------|-----------------|---------------------|
| **数据来源** | objective.items | ground_truth.treatment |
| **触发时机** | Agent调用EXAM时 | 诊断确定后 |
| **标注时机** | 可在合成时标注 | 需要后处理提取 |
| **结构化程度** | 高 (已有MedicalItem) | 低 (需要从文本提取) |
| **不确定性** | 低 (标准收费项目) | 中 (用药方案可能变化) |

---

## 3. 设计方案对比

### 方案1: 统一后置查询

**核心思想**: 在完整轨迹生成后，统一进行成本查询和统计

```
┌─────────────────────────────────────────────────────────────────┐
│                    方案1: 统一后置查询                            │
│                                                                  │
│  数据合成 ──> 生成MedicalCase ──> 完整轨迹 ──> 统一Cost查询      │
│                                                                  │
│  查询内容:                                                       │
│  ├── objective.items (检查项目)                                  │
│  └── ground_truth.treatment (治疗项目)                           │
│                                                                  │
│  优点:                                                          │
│  ├── 实现简单，逻辑统一                                          │
│  ├── 不影响数据合成流程                                          │
│  └── 可批量处理，效率高                                          │
│                                                                  │
│  缺点:                                                          │
│  ├── 无法在合成过程中利用成本信息                                │
│  └── 需要额外遍历已生成的数据                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**实现示例**:

```python
class UnifiedCostAnnotator:
    """统一成本标注器 - 方案1"""

    def __init__(self, cost_database: CostDatabase):
        self.cost_db = cost_database

    def annotate_case(self, medical_case: MedicalCase) -> CostInfo:
        """对完整病例进行成本标注"""
        diagnostic_costs = []
        treatment_costs = []

        # 1. 查询诊断成本 (objective items)
        for item in medical_case.objective:
            cost = self.cost_db.lookup_exam(item.keywords)
            if cost:
                diagnostic_costs.append({
                    "name": item.content[:50],
                    "keywords": item.keywords,
                    "cost": cost
                })

        # 2. 查询治疗成本 (从treatment文本提取)
        treatment_items = self._extract_treatment_items(
            medical_case.ground_truth.treatment
        )
        for item_name in treatment_items:
            cost = self.cost_db.lookup_treatment(item_name)
            if cost:
                treatment_costs.append({
                    "name": item_name,
                    "cost": cost
                })

        # 3. 汇总
        return CostInfo(
            diagnostic_cost=sum(c["cost"].avg for c in diagnostic_costs),
            diagnostic_items=diagnostic_costs,
            treatment_cost=sum(c["cost"].avg for c in treatment_costs),
            treatment_items=treatment_costs,
            total_estimated=sum(c["cost"].avg for c in diagnostic_costs + treatment_costs)
        )
```

---

### 方案2: 分阶段查询

**核心思想**: EXAM项在数据合成时提前查询，治疗项后续处理

```
┌─────────────────────────────────────────────────────────────────┐
│                    方案2: 分阶段查询                              │
│                                                                  │
│  阶段A: 数据合成时                                               │
│  ───────────────────                                             │
│  生成objective item ──> 立即查询exam_cost ──> 写入item.cost     │
│                                                                  │
│  阶段B: 数据合成后                                               │
│  ───────────────────                                             │
│  完整MedicalCase ──> 提取treatment ──> 查询cost ──> 写入cost_info │
│                                                                  │
│  优点:                                                          │
│  ├── EXAM成本直接绑定到MedicalItem，结构更清晰                   │
│  ├── 可在合成过程中进行成本感知的质量控制                         │
│  └── 减少后续遍历开销                                            │
│                                                                  │
│  缺点:                                                          │
│  ├── 需要修改数据合成流程                                        │
│  └── 两套逻辑，实现稍复杂                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**实现示例**:

```python
# 阶段A: 数据合成时 - 扩展 MedicalItem
@dataclass
class MedicalItem:
    keywords: List[str]
    content: str
    necessity: bool
    # 新增: 检查成本 (仅对objective项有意义)
    exam_cost: Optional[float] = None
    exam_cost_confidence: Optional[str] = None  # high/medium/low

# 阶段A: 合成时标注
class DataSynthesisPipeline:
    def __init__(self, cost_database: CostDatabase):
        self.cost_db = cost_database

    def generate_objective_item(self, item_data: dict) -> MedicalItem:
        item = MedicalItem(**item_data)
        # 立即查询成本
        cost_info = self.cost_db.lookup_exam(item.keywords)
        if cost_info:
            item.exam_cost = cost_info.avg
            item.exam_cost_confidence = cost_info.confidence
        return item

# 阶段B: 后续处理治疗成本
class TreatmentCostAnnotator:
    def annotate(self, medical_case: MedicalCase) -> CostInfo:
        # 诊断成本从objective直接读取
        diagnostic_costs = [
            {"name": item.content[:50], "cost": item.exam_cost}
            for item in medical_case.objective
            if item.exam_cost is not None
        ]

        # 治疗成本需要提取和查询
        treatment_costs = self._extract_and_query(
            medical_case.ground_truth.treatment
        )

        return CostInfo(
            diagnostic_cost=sum(c["cost"] for c in diagnostic_costs),
            diagnostic_items=diagnostic_costs,
            treatment_cost=sum(c["cost"] for c in treatment_costs),
            treatment_items=treatment_costs
        )
```

---

### 方案3: 混合查询 (推荐)

**核心思想**: 结合两种方案的优点，区分处理策略

```
┌─────────────────────────────────────────────────────────────────┐
│                    方案3: 混合查询                                │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ EXAM成本: 预计算 + 懒加载                                 │    │
│  │                                                          │    │
│  │ 方式:                                                    │    │
│  │ 1. 数据合成时不主动查询                                   │    │
│  │ 2. 首次访问时通过property懒加载                           │    │
│  │ 3. 结果缓存到item中                                       │    │
│  │                                                          │    │
│  │ 优点: 不影响合成流程，按需计算，自动缓存                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 治疗成本: 提取 + 验证 + 缓存                              │    │
│  │                                                          │    │
│  │ 方式:                                                    │    │
│  │ 1. 后处理阶段批量提取付费单元                              │    │
│  │ 2. 规则匹配 + LLM验证                                     │    │
│  │ 3. 结果写入cost_info字段                                  │    │
│  │                                                          │    │
│  │ 优点: 支持复杂项目提取，结果可审计                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**实现示例**:

```python
@dataclass
class MedicalItem:
    keywords: List[str]
    content: str
    necessity: bool
    _exam_cost: Optional[float] = field(default=None, repr=False)
    _cost_queried: bool = field(default=False, repr=False)

    @property
    def exam_cost(self) -> Optional[float]:
        """懒加载检查成本"""
        if not self._cost_queried:
            self._exam_cost = self._query_cost()
            self._cost_queried = True
        return self._exam_cost

    def _query_cost(self) -> Optional[float]:
        """查询检查成本"""
        cost_db = get_cost_database()  # 全局单例
        result = cost_db.lookup_exam(self.keywords)
        return result.avg if result else None


class CostInfo:
    """成本信息汇总"""
    diagnostic_cost: float
    diagnostic_items: List[Dict]
    treatment_cost: float
    treatment_items: List[Dict]
    total_cost: float
    confidence: str


class CostAnnotator:
    """混合成本标注器"""

    def annotate(self, medical_case: MedicalCase) -> CostInfo:
        # 1. 诊断成本 - 通过property自动查询
        diagnostic_items = []
        for item in medical_case.objective:
            if item.exam_cost is not None:
                diagnostic_items.append({
                    "source": "exam",
                    "name": item.content[:50],
                    "keywords": item.keywords,
                    "cost": item.exam_cost
                })

        # 2. 治疗成本 - 提取并查询
        treatment_items = self._extract_treatment_costs(
            medical_case.ground_truth.treatment
        )

        # 3. 汇总
        diag_total = sum(i["cost"] for i in diagnostic_items)
        treat_total = sum(i["cost"] for i in treatment_items)

        return CostInfo(
            diagnostic_cost=diag_total,
            diagnostic_items=diagnostic_items,
            treatment_cost=treat_total,
            treatment_items=treatment_items,
            total_cost=diag_total + treat_total,
            confidence=self._calc_confidence(diagnostic_items, treatment_items)
        )
```

---

## 4. 成本数据库设计

### 4.1 数据来源

| 数据类型 | 数据源 | 更新频率 |
|----------|--------|----------|
| 检查检验价格 | 医疗服务价格目录 | 年度更新 |
| 药品价格 | 药品集采价格/医保目录 | 季度更新 |
| 手术操作价格 | 医疗服务价格目录 | 年度更新 |

### 4.2 数据库结构

```python
@dataclass
class CostEntry:
    """成本条目"""
    name: str                    # 项目名称
    category: str                # 类别: exam/drug/surgery/procedure
    avg_price: float             # 平均价格
    price_range: Tuple[float, float]  # 价格区间
    unit: str                    # 单位
    region: str = "national"     # 地区 (可扩展)
    source: str = ""             # 数据来源
    last_updated: str = ""       # 更新时间

class CostDatabase:
    """成本数据库"""

    def __init__(self, data_path: str = "data/cost_reference"):
        self.exam_db = self._load(f"{data_path}/exam_prices.json")
        self.drug_db = self._load(f"{data_path}/drug_prices.json")
        self.surgery_db = self._load(f"{data_path}/surgery_prices.json")

    def lookup_exam(self, keywords: List[str]) -> Optional[CostEntry]:
        """查询检查项目成本"""
        for keyword in keywords:
            # 精确匹配
            if keyword in self.exam_db:
                return self.exam_db[keyword]
            # 模糊匹配
            for name, entry in self.exam_db.items():
                if keyword in name or name in keyword:
                    return entry
        return None

    def lookup_drug(self, drug_name: str) -> Optional[CostEntry]:
        """查询药品成本"""
        # 类似实现
        pass

    def lookup_treatment(self, treatment_name: str) -> Optional[CostEntry]:
        """查询治疗项目成本 (综合查询)"""
        # 依次查询 surgery -> procedure -> drug
        pass
```

### 4.3 数据文件格式

```json
// data/cost_reference/exam_prices.json
{
    "血常规": {
        "avg_price": 25.0,
        "price_range": [15.0, 35.0],
        "unit": "次",
        "category": "检验"
    },
    "心电图": {
        "avg_price": 40.0,
        "price_range": [30.0, 60.0],
        "unit": "次",
        "category": "检查"
    },
    "CT平扫": {
        "avg_price": 300.0,
        "price_range": [200.0, 450.0],
        "unit": "部位",
        "category": "检查"
    }
}
```

```json
// data/cost_reference/drug_prices.json
{
    "阿莫西林胶囊": {
        "avg_price": 15.0,
        "price_range": [8.0, 25.0],
        "unit": "盒",
        "specification": "0.5g*24粒"
    },
    "布洛芬片": {
        "avg_price": 12.0,
        "price_range": [6.0, 20.0],
        "unit": "盒",
        "specification": "0.2g*20片"
    }
}
```

---

## 5. 治疗项目提取策略

治疗项目需要从自然语言文本中提取，有以下策略：

### 5.1 规则匹配

```python
# 常见治疗项目关键词
TREATMENT_PATTERNS = {
    "surgery": ["手术", "切除", "缝合", "介入", "PCI", "支架"],
    "procedure": ["输液", "注射", "换药", "引流", "穿刺"],
    "drug": ["口服", "静脉", "肌注", "阿司匹林", "抗生素"]
}

def extract_by_pattern(text: str) -> List[Dict]:
    items = []
    for category, patterns in TREATMENT_PATTERNS.items():
        for pattern in patterns:
            if pattern in text:
                items.append({
                    "raw_text": text,
                    "matched": pattern,
                    "category": category
                })
    return items
```

### 5.2 LLM提取

```python
EXTRACTION_PROMPT = '''从以下治疗方案文本中提取医疗付费项目，包括药品、手术、操作等。

治疗方案文本:
{text}

请输出JSON格式:
{{
    "items": [
        {{
            "name": "项目名称",
            "category": "drug/surgery/procedure",
            "quantity": 数量,
            "unit": "单位"
        }}
    ]
}}
'''

def extract_by_llm(text: str, llm_client) -> List[Dict]:
    prompt = EXTRACTION_PROMPT.format(text=text)
    response = llm_client.call(prompt, temperature=0.1)
    return parse_json(response).get("items", [])
```

### 5.3 混合策略

```python
def extract_treatment_items(text: str, llm_client=None) -> List[Dict]:
    """混合提取策略"""
    # 1. 先规则匹配
    items = extract_by_pattern(text)

    # 2. 对于未匹配的部分，使用LLM
    if llm_client and len(items) < len(text.split()) / 5:
        llm_items = extract_by_llm(text, llm_client)
        items.extend(llm_items)

    # 3. 去重
    return deduplicate(items)
```

---

## 6. 方案对比总结

| 维度 | 方案1: 统一后置 | 方案2: 分阶段 | 方案3: 混合 |
|------|----------------|---------------|-------------|
| **实现复杂度** | 低 | 中 | 中 |
| **代码侵入性** | 无 | 需修改合成流程 | 低 |
| **查询效率** | 需遍历 | 分散但即时 | 懒加载高效 |
| **结构清晰度** | 中 | 高 | 高 |
| **可扩展性** | 中 | 中 | 高 |
| **推荐场景** | 快速原型 | 深度集成 | 生产环境 |

**推荐**: 方案3 (混合查询)
- EXAM成本通过 property 懒加载，不影响数据流
- 治疗成本通过后处理批量标注，支持复杂提取
- 结果可缓存，查询高效

---

## 7. 后续工作

### 7.1 短期

- [ ] 实现 CostDatabase 基础模块
- [ ] 收集医疗服务价格数据
- [ ] 实现检查项目成本查询

### 7.2 中期

- [ ] 实现治疗项目提取 (规则 + LLM)
- [ ] 集成到数据合成流程
- [ ] 成本标注质量评估

### 7.3 长期

- [ ] 地区差异化价格支持
- [ ] 医保支付标准对接
- [ ] 成本效益分析报告

---

*文档版本: v1.0 | 创建日期: 2026-04-02*