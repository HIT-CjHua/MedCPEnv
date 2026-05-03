# MedAgent/src/medagent/cost.py

"""
费用评估模块

使用 M2 模型从 Agent 轨迹中提取医疗服务和药品，
通过价格清单检索或 M2 生成的方式进行费用估算。

流程：
1. 从 [EXAM] 和 [FINAL] 步骤提取待估价项目
2. 尝试从价格清单匹配
3. 匹配失败则由 M2 生成价格并存入清单
4. 汇总费用
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import os
import re
import time
from pathlib import Path

from .llm import LLMClient


def _parse_json_response(response: str) -> Optional[Dict]:
    """解析 JSON 响应，支持 markdown code block 等格式"""
    if not response or not response.strip():
        return None
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    # 尝试匹配 ```json ... ``` 格式
    json_block = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    for match in json_block:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    # 尝试匹配最外层 { ... }
    json_obj = re.findall(r'\{[\s\S]*\}', response)
    for match in json_obj:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    return None


@dataclass
class CostItem:
    """单个费用项"""
    item_type: str  # "service" | "medicine"
    item_name: str
    specification: str = ""
    unit: str = ""
    price: float = 0.0
    quantity: int = 1
    source: str = "matched"  # "matched" | "generated"
    raw_match: Optional[Dict] = None  # 匹配到的原始数据


@dataclass
class CostResult:
    """费用评估结果"""
    case_id: str

    # 分类费用
    service_items: List[CostItem] = field(default_factory=list)
    medicine_items: List[CostItem] = field(default_factory=list)

    # 汇总
    service_cost: float = 0.0
    medicine_cost: float = 0.0
    total_cost: float = 0.0

    # 统计
    matched_count: int = 0
    generated_count: int = 0

    # 详细信息
    extraction_raw: str = ""  # M2提取的原始输出
    trajectory: List[Dict] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------

EXTRACT_PROMPT = """/no_think
你是一位医疗费用分析专家，请从以下诊疗轨迹中提取所有涉及的医疗服务和药品。

## 输入信息
病例ID: {case_id}
主诉: {chief_complaint}

## 诊疗轨迹
{trajectory_summary}

## 提取要求
请提取以下两类项目：
1. **医疗服务**：检查、检验、手术、治疗操作等（来自[EXAM]步骤）
2. **药品**：处方药物、用药建议等（来自[FINAL]步骤的治疗建议）

## 输出格式
请输出 JSON 格式，包含以下字段：

```json
{{
    "services": [
        {{
            "name": "<服务名称>",
            "specification": "<规格/说明，如检查部位>",
            "unit": "<计价单位，如次/项>",
            "quantity": <数量，默认1>
        }}
    ],
    "medicines": [
        {{
            "name": "<药品名称>",
            "specification": "<规格，如剂量>",
            "unit": "<单位，如盒/支>",
            "quantity": <数量>
        }}
    ]
}}
```

注意：
- 只提取明确提及的项目，不要推测
- 药品名称尽量使用通用名
- 如果无法确定规格/单位，可以留空

只输出 JSON，不要有其他内容。
"""

GENERATE_PRICE_PROMPT = """/no_think
你是一位医疗定价专家，请为以下医疗服务或药品生成合理的参考价格。

## 项目信息
类型: {item_type}
名称: {item_name}
规格: {specification}
单位: {unit}

## 定价依据
请参考以下因素：
1. 医疗服务：根据检查复杂度、设备成本、人力成本
2. 药品：根据药物类型、剂量、市场常见价格

## 输出格式
请输出 JSON 格式：

```json
{{
    "item_type": "{item_type}",
    "item_name": "{item_name}",
    "specification": "<规格>",
    "unit": "<单位>",
    "price": <价格，单位：元>,
    "price_reason": "<定价理由简述>"
}}
```

只输出 JSON，不要有其他内容。
"""


class CostEvaluator:
    """
    费用评估器

    使用 LLM 提取诊疗项目，通过价格清单匹配或生成的方式进行费用估算
    """

    def __init__(
        self,
        price_list_path: str = "data/cost_list/cost_reference.jsonl",
        m2_url: str = "https://api.baichuan-ai.com/v1",
        m2_model: str = "Baichuan-M2",
        auto_save: bool = True,
        llm_client: Optional[LLMClient] = None,
        api_key: Optional[str] = None,
    ):
        """
        初始化费用评估器

        Args:
            price_list_path: 价格清单路径
            m2_url: LLM API 地址
            m2_model: LLM 模型名称
            auto_save: 是否自动保存新生成的价格到清单
            llm_client: LLM 客户端（可选，优先级高于 api_key）
            api_key: API Key（可选，如果未提供则从环境变量读取）
        """
        self.price_list_path = Path(price_list_path)
        self.m2_model = m2_model
        self.auto_save = auto_save

        # 初始化 LLM 客户端
        if llm_client is not None:
            self.m2_client = llm_client
        else:
            self.m2_client = LLMClient(
                model_name=m2_model,
                base_url=m2_url,
                api_key=api_key or os.getenv("BAICHUAN_API_KEY"),
                max_tokens=32768,  # 32K 避免截断
            )

        # 加载价格清单
        self.price_list: List[Dict] = self._load_price_list()

        # 新生成的价格（待保存）
        self.new_prices: List[Dict] = []

    def _load_price_list(self) -> List[Dict]:
        """加载价格清单"""
        if not self.price_list_path.exists():
            print(f"[Cost] 价格清单不存在: {self.price_list_path}")
            return []

        items = []
        with open(self.price_list_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except:
                        continue

        print(f"[Cost] 加载价格清单: {len(items)} 条")
        return items

    def _save_price_list(self):
        """保存价格清单（追加新价格）"""
        if not self.new_prices or not self.auto_save:
            return

        with open(self.price_list_path, "a", encoding="utf-8") as f:
            for item in self.new_prices:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"[Cost] 新增价格: {len(self.new_prices)} 条，已保存到清单")
        self.new_prices = []

    def _build_trajectory_summary(self, trajectory: List[Dict]) -> str:
        """构建轨迹摘要，重点提取 EXAM 和 FINAL 步骤"""
        if not trajectory:
            return "无诊疗记录"

        summary_parts = []
        for i, step in enumerate(trajectory):
            action = step.get("parsed", {}).get("action", "UNKNOWN")

            if action == "EXAM":
                keywords = step.get("parsed", {}).get("keywords", [])
                observation = step.get("observation", "")
                summary_parts.append(f"[EXAM] 步骤{i+1}: 检查项目 {keywords}")
                summary_parts.append(f"  结果: {observation[:300]}")

            elif action == "FINAL":
                diagnosis = step.get("parsed", {}).get("diagnosis", "")
                treatment = step.get("parsed", {}).get("treatment", "")
                summary_parts.append(f"[FINAL] 步骤{i+1}: 最终诊断与治疗")
                summary_parts.append(f"  诊断: {diagnosis}")
                summary_parts.append(f"  治疗: {treatment}")

        return "\n".join(summary_parts)

    def _extract_items_from_trajectory(
        self,
        case_id: str,
        chief_complaint: str,
        trajectory: List[Dict],
    ) -> Tuple[List[Dict], List[Dict], str]:
        """
        使用 M2 从轨迹中提取医疗服务和药品

        Returns:
            (services, medicines, raw_output)
        """
        trajectory_summary = self._build_trajectory_summary(trajectory)

        prompt = EXTRACT_PROMPT.format(
            case_id=case_id,
            chief_complaint=chief_complaint,
            trajectory_summary=trajectory_summary,
        )

        try:
            # 添加重试逻辑处理限流
            max_retries = 3
            for retry in range(max_retries):
                try:
                    response = self.m2_client.call(
                        prompt=prompt,
                        temperature=0.1,
                        max_tokens=32768,  # 32K 避免截断
                        response_format={"type": "json_object"},
                    )
                    break  # 成功则退出重试
                except Exception as e:
                    if "429" in str(e) or "rate" in str(e).lower():
                        if retry < max_retries - 1:
                            wait_time = 30 * (retry + 1)  # 30s, 60s, 90s
                            print(f"[Cost] 限流，等待 {wait_time}s 后重试...")
                            time.sleep(wait_time)
                            continue
                    raise  # 非429错误或最后一次重试失败则抛出

            result = _parse_json_response(response)
            if result is None:
                raise ValueError(f"Failed to parse JSON from response: {response[:200]}")
            services = result.get("services", [])
            medicines = result.get("medicines", [])

            return services, medicines, response

        except Exception as e:
            print(f"[Cost] M2 提取失败: {e}")
            return [], [], ""

    def _search_price(
        self,
        item_name: str,
        item_type: str,
    ) -> Optional[Dict]:
        """
        软匹配检索价格

        匹配策略：
        1. 精确匹配 item_name
        2. 前缀匹配（item_name 是 price_list 中名称的前缀）
        3. 包含匹配（item_name 包含在 price_list 名称中，或反之）
        """
        item_name_lower = item_name.lower()

        # 1. 精确匹配
        for item in self.price_list:
            if item["item_type"] != item_type:
                continue
            if item["item_name"].lower() == item_name_lower:
                return item

        # 2. 前缀匹配（去除常见后缀词）
        # 例如 "阿莫西林" → "阿莫西林胶囊"
        prefix_name = item_name_lower.rstrip("片胶囊注射剂注射液口服液颗粒")
        if len(prefix_name) >= 2:
            for item in self.price_list:
                if item["item_type"] != item_type:
                    continue
                if item["item_name"].lower().startswith(prefix_name):
                    return item

        # 3. 包含匹配
        for item in self.price_list:
            if item["item_type"] != item_type:
                continue
            name_lower = item["item_name"].lower()
            if item_name_lower in name_lower or name_lower in item_name_lower:
                return item

        return None

    def _generate_price_with_m2(
        self,
        item_type: str,
        item_name: str,
        specification: str = "",
        unit: str = "",
    ) -> Optional[Dict]:
        """
        使用 M2 生成价格
        """
        prompt = GENERATE_PRICE_PROMPT.format(
            item_type=item_type,
            item_name=item_name,
            specification=specification,
            unit=unit,
        )

        try:
            response = self.m2_client.call(
                prompt=prompt,
                temperature=0.1,
                max_tokens=32768,  # 32K 避免截断
                response_format={"type": "json_object"},
            )

            result = _parse_json_response(response)
            if result is None:
                raise ValueError(f"Failed to parse JSON from response: {response[:200]}")

            # 构建价格记录
            price_record = {
                "item_type": item_type,
                "item_id": item_name,
                "item_name": item_name,
                "specification": result.get("specification", specification),
                "unit": result.get("unit", unit),
                "price": float(result.get("price", 0)),
            }

            # 加入价格清单和待保存列表
            self.price_list.append(price_record)
            self.new_prices.append(price_record)

            return price_record

        except Exception as e:
            print(f"[Cost] M2 生成价格失败: {e}")
            return None

    def _process_item(
        self,
        item_data: Dict,
        item_type: str,
    ) -> CostItem:
        """
        处理单个项目：匹配或生成价格
        """
        name = item_data.get("name", "")
        specification = item_data.get("specification", "")
        unit = item_data.get("unit", "")
        quantity = item_data.get("quantity", 1)

        # 尝试匹配
        matched = self._search_price(name, item_type)

        if matched:
            return CostItem(
                item_type=item_type,
                item_name=name,
                specification=matched.get("specification", specification),
                unit=matched.get("unit", unit),
                price=matched.get("price", 0),
                quantity=quantity,
                source="matched",
                raw_match=matched,
            )

        # 匹配失败，生成价格
        generated = self._generate_price_with_m2(
            item_type=item_type,
            item_name=name,
            specification=specification,
            unit=unit,
        )

        if generated:
            return CostItem(
                item_type=item_type,
                item_name=name,
                specification=generated.get("specification", specification),
                unit=generated.get("unit", unit),
                price=generated.get("price", 0),
                quantity=quantity,
                source="generated",
                raw_match=generated,
            )

        # 完全失败
        return CostItem(
            item_type=item_type,
            item_name=name,
            specification=specification,
            unit=unit,
            price=0,
            quantity=quantity,
            source="failed",
        )

    def estimate_from_agent_result(self, agent_result: Dict) -> CostResult:
        """
        端到端费用估算 - 直接从 Agent 返回结果计算费用

        Args:
            agent_result: Agent.run() 返回的结果，包含:
                - case_id: 病例ID
                - chief_complaint: 主诉
                - trajectory: Agent 轨迹
                - diagnosis: 诊断结果
                - treatment: 治疗建议

        Returns:
            CostResult: 费用评估结果
        """
        return self.estimate_cost(
            case_id=agent_result.get("case_id", "unknown"),
            chief_complaint=agent_result.get("chief_complaint", ""),
            trajectory=agent_result.get("trajectory", []),
        )

    def estimate_cost(
        self,
        case_id: str,
        chief_complaint: str,
        trajectory: List[Dict],
    ) -> CostResult:
        """
        估算诊疗费用

        Args:
            case_id: 病例ID
            chief_complaint: 主诉
            trajectory: Agent 轨迹

        Returns:
            CostResult: 费用评估结果
        """
        result = CostResult(
            case_id=case_id,
            trajectory=trajectory,
        )

        # 1. M2 提取项目
        services_data, medicines_data, raw_output = self._extract_items_from_trajectory(
            case_id, chief_complaint, trajectory
        )
        result.extraction_raw = raw_output

        # 2. 处理医疗服务
        for service_data in services_data:
            item = self._process_item(service_data, "service")
            result.service_items.append(item)
            if item.source == "matched":
                result.matched_count += 1
            elif item.source == "generated":
                result.generated_count += 1

        # 3. 处理药品
        for medicine_data in medicines_data:
            item = self._process_item(medicine_data, "medicine")
            result.medicine_items.append(item)
            if item.source == "matched":
                result.matched_count += 1
            elif item.source == "generated":
                result.generated_count += 1

        # 4. 汇总费用
        result.service_cost = sum(
            item.price * item.quantity for item in result.service_items
        )
        result.medicine_cost = sum(
            item.price * item.quantity for item in result.medicine_items
        )
        result.total_cost = result.service_cost + result.medicine_cost

        # 5. 保存新价格
        self._save_price_list()

        return result

    def get_report(self, result: CostResult) -> str:
        """生成费用报告"""
        lines = []
        lines.append(f"=== 费用评估报告 ===")
        lines.append(f"病例ID: {result.case_id}")
        lines.append(f"")

        # 服务费用
        if result.service_items:
            lines.append(f"[医疗服务]")
            for item in result.service_items:
                source_tag = "[M]" if item.source == "matched" else "[G]" if item.source == "generated" else "[F]"
                lines.append(f"  {source_tag} {item.item_name}: {item.price}元 × {item.quantity} = {item.price * item.quantity}元")
            lines.append(f"  小计: {result.service_cost}元")
            lines.append(f"")

        # 药品费用
        if result.medicine_items:
            lines.append(f"[药品]")
            for item in result.medicine_items:
                source_tag = "[M]" if item.source == "matched" else "[G]" if item.source == "generated" else "[F]"
                lines.append(f"  {source_tag} {item.item_name}: {item.price}元 × {item.quantity} = {item.price * item.quantity}元")
            lines.append(f"  小计: {result.medicine_cost}元")
            lines.append(f"")

        # 汇总
        lines.append(f"---")
        lines.append(f"总费用: {result.total_cost}元")
        lines.append(f"匹配: {result.matched_count}项 | 生成: {result.generated_count}项")

        return "\n".join(lines)


if __name__ == "__main__":
    # 测试
    cost_evaluator = CostEvaluator()

    # 模拟轨迹
    test_trajectory = [
        {
            "parsed": {"action": "EXAM", "keywords": ["血常规"]},
            "observation": "白细胞计数正常"
        },
        {
            "parsed": {"action": "EXAM", "keywords": ["胸部CT"]},
            "observation": "未见明显异常"
        },
        {
            "parsed": {"action": "FINAL", "diagnosis": "上呼吸道感染", "treatment": "阿莫西林胶囊，一日三次，连服7天"},
            "observation": ""
        },
    ]

    result = cost_evaluator.estimate_cost(
        case_id="test_001",
        chief_complaint="咳嗽3天",
        trajectory=test_trajectory,
    )

    print(cost_evaluator.get_report(result))