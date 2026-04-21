# MedAgent/src/medagent/judger.py

"""
评测器模块

使用 LLM 结合 ground_truth 进行综合打分：
- 诊断准确性 (0-1判定 + 1-5评分)
- 治疗合理性 (0-1判定 + 1-5评分)
- 安全性 (0-1判定 + 1-5评分)
- 效率统计 (Python代码)
- 费用估算 (Cost模块)
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
import os

from .llm import LLMClient
from .cost import CostEvaluator, CostResult


@dataclass
class EfficiencyStats:
    """效率统计"""
    total_steps: int = 0
    ask_count: int = 0
    exam_count: int = 0
    knowledge_count: int = 0
    exam_items: int = 0  # 检查项目总数
    ask_items: int = 0   # 问诊项目总数
    # 新增：生成速度和长度统计
    total_tokens: int = 0        # 总生成token数
    total_latency: float = 0.0   # 总生成耗时(秒)
    avg_tokens_per_step: float = 0.0  # 平均每步token数
    avg_latency_per_step: float = 0.0  # 平均每步耗时(秒)
    tokens_per_second: float = 0.0     # 生成速度(tokens/s)


@dataclass
class EvalResult:
    """评测结果"""
    case_id: str

    # 诊断相关 (0-1 + 1-5 + reason)
    diagnosis_correct: bool = False
    diagnosis_score: float = 0.0
    diagnosis_reason: str = ""

    # 治疗相关 (0-1 + 1-5 + reason)
    treatment_correct: bool = False
    treatment_score: float = 0.0
    treatment_reason: str = ""

    # 安全相关 (0-1 + 1-5 + reason)
    avoid_violated: bool = False
    avoid_score: float = 0.0
    avoid_reason: str = ""
    avoid_violations: List[str] = field(default_factory=list)

    # 效率统计
    efficiency: EfficiencyStats = field(default_factory=EfficiencyStats)

    # 费用评估 (整数)
    total_cost: int = 0

    # 综合分数
    total_score: float = 0.0

    # 详细信息
    trajectory: List[Dict] = field(default_factory=list)
    ground_truth: Dict = field(default_factory=dict)
    agent_diagnosis: str = ""
    agent_treatment: str = ""


JUDGE_PROMPT = """/no_think
你是一位资深的临床医学专家，负责评估 AI 医疗问诊系统的表现。

## 评估任务

请根据以下信息对 AI 的诊断和治疗结果进行综合评估：

### 病例信息
- 病例ID: {case_id}
- 主诉: {chief_complaint}

### 标准答案 (Ground Truth)
- 正确诊断: {gt_diagnosis}
- 标准治疗: {gt_treatment}
- 禁忌事项: {gt_avoid}

### AI 的诊断过程
{trajectory_summary}

### AI 的最终结论
- 诊断: {agent_diagnosis}
- 治疗: {agent_treatment}

## 评估标准

请按以下三个维度进行评分，每个维度都需要给出：
1. 0-1判定（是否通过）
2. 1-5评分（细粒度评分）
3. 评分理由（简短说明）

### 1. 诊断准确性
- diagnosis_correct (true/false): 诊断是否基本正确，方向正确即可
- diagnosis_score (1-5):
  - 5分: 诊断完全正确或高度吻合
  - 4分: 诊断基本正确，有小偏差
  - 3分: 诊断方向正确但不够具体
  - 2分: 诊断有较大错误，有部分合理成分
  - 1分: 诊断完全错误

### 2. 治疗合理性
- treatment_correct (true/false): 治疗方案是否基本合理可行
- treatment_score (1-5):
  - 5分: 治疗方案完全合理，无禁忌违反
  - 4分: 治疗方案基本合理，有小瑕疵
  - 3分: 治疗方案有明显不足但不违反禁忌
  - 2分: 治疗方案存在较大问题或轻微违反禁忌
  - 1分: 治疗方案严重不合理或违反重要禁忌

### 3. 安全性评估
**重要：仅检查治疗方案中是否推荐了禁忌项**
- avoid_violated (true/false): 治疗方案中是否推荐了禁忌项
- avoid_score (1-5):
  - 5分: 无任何禁忌违反
  - 4分: 有1次低危违反（如不推荐检查）
  - 3分: 有2次低危或1次中危违反
  - 2分: 有多次违反但非高危
  - 1分: 有高危违反（如致命药物禁忌）
- avoid_violations: 列出具体违反的禁忌项

## 输出格式

请输出 JSON 格式（三个维度都需要给出判定、评分和理由）：

```json
{{
    "diagnosis_correct": true/false,
    "diagnosis_score": 1-5,
    "diagnosis_reason": "评分理由",
    "treatment_correct": true/false,
    "treatment_score": 1-5,
    "treatment_reason": "评分理由",
    "avoid_violated": true/false,
    "avoid_score": 1-5,
    "avoid_reason": "评分理由",
    "avoid_violations": ["违反的禁忌项"]
}}
```

只输出 JSON，不要有其他内容。
"""


class Judger:
    """
    评测器

    使用 302API 的 Baichuan-M2 模型对 Agent 的诊断和治疗进行综合评估
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        # vLLM 本地部署配置 (已弃用，改用 302API)
        # model_name: str = "baichuan-m2",
        # base_url: str = "http://localhost:8200/v1",
        model_name: str = "Baichuan-M2",
        base_url: str = "https://api.302.ai/v1",
        cost_evaluator: Optional[CostEvaluator] = None,
        enable_cost: bool = False,
    ):
        """
        初始化评测器

        Args:
            llm_client: LLM 客户端（可选）
            model_name: Judge 模型名称（默认使用 302API 的 Baichuan-M2）
            base_url: 302API 服务地址
            cost_evaluator: 费用评估器（可选）
            enable_cost: 是否启用费用评估
        """
        if llm_client is None:
            self.llm_client = LLMClient(
                model_name=model_name,
                base_url=base_url,
                api_key=os.getenv("302_API_KEY"),  # 使用 302API key
            )
        else:
            self.llm_client = llm_client
        self.model_name = model_name
        self.base_url = base_url

        # 费用评估器
        self.enable_cost = enable_cost
        if cost_evaluator is not None:
            self.cost_evaluator = cost_evaluator
        elif enable_cost:
            self.cost_evaluator = CostEvaluator(m2_url=base_url)
        else:
            self.cost_evaluator = None

    def _build_trajectory_summary(self, trajectory: List[Dict]) -> str:
        """将轨迹转换为可读的摘要"""
        if not trajectory:
            return "无诊断过程记录"

        summary_parts = []
        for i, step in enumerate(trajectory):
            action = step.get("parsed", {}).get("action", "UNKNOWN")

            if action in ["ASK", "EXAM"]:
                keywords = step.get("parsed", {}).get("keywords", [])
                observation = step.get("observation", "")[:200]
                summary_parts.append(f"步骤{i+1} [{action}]: 关键词 {keywords}")
                summary_parts.append(f"  结果: {observation}...")

            elif action == "KNOWLEDGE":
                query = step.get("parsed", {}).get("query", "")
                observation = step.get("observation", "")[:200]
                summary_parts.append(f"步骤{i+1} [KNOWLEDGE]: 查询 '{query}'")
                summary_parts.append(f"  结果: {observation}...")

            elif action == "FINAL":
                summary_parts.append(f"步骤{i+1} [FINAL]: 最终诊断")

        return "\n".join(summary_parts)

    def _analyze_efficiency(self, trajectory: List[Dict]) -> EfficiencyStats:
        """
        分析效率统计

        Args:
            trajectory: Agent 轨迹

        Returns:
            EfficiencyStats: 效率统计结果
        """
        stats = EfficiencyStats()
        stats.total_steps = len(trajectory)

        for step in trajectory:
            parsed = step.get("parsed", {})
            action = parsed.get("action", "")

            if action == "ASK":
                stats.ask_count += 1
                stats.ask_items += len(parsed.get("keywords", []))

            elif action == "EXAM":
                stats.exam_count += 1
                stats.exam_items += len(parsed.get("keywords", []))

            elif action == "KNOWLEDGE":
                stats.knowledge_count += 1

            # 统计生成速度和长度
            latency = step.get("latency", 0)
            tokens = step.get("estimated_tokens", 0)
            stats.total_latency += latency
            stats.total_tokens += tokens

        # 计算平均值和速度
        if stats.total_steps > 0:
            stats.avg_tokens_per_step = stats.total_tokens / stats.total_steps
            stats.avg_latency_per_step = stats.total_latency / stats.total_steps
            if stats.total_latency > 0:
                stats.tokens_per_second = stats.total_tokens / stats.total_latency

        return stats

    def evaluate(
        self,
        case_id: str,
        chief_complaint: str,
        ground_truth: Dict,
        trajectory: List[Dict],
        agent_diagnosis: str,
        agent_treatment: str,
    ) -> EvalResult:
        """
        评估单个病例

        Args:
            case_id: 病例ID
            chief_complaint: 主诉
            ground_truth: 标准答案
            trajectory: Agent 的诊断轨迹
            agent_diagnosis: Agent 的诊断结果
            agent_treatment: Agent 的治疗建议

        Returns:
            EvalResult: 评测结果
        """
        # 初始化结果
        result = EvalResult(
            case_id=case_id,
            trajectory=trajectory,
            ground_truth=ground_truth,
            agent_diagnosis=agent_diagnosis,
            agent_treatment=agent_treatment,
        )

        # 提取 ground_truth
        gt_diagnosis = ground_truth.get("diagnosis", [])
        gt_treatment = ground_truth.get("treatment", [])
        gt_avoid = ground_truth.get("avoid", [])

        # 构建轨迹摘要
        trajectory_summary = self._build_trajectory_summary(trajectory)

        # 构建 prompt
        prompt = JUDGE_PROMPT.format(
            case_id=case_id,
            chief_complaint=chief_complaint,
            gt_diagnosis=gt_diagnosis,
            gt_treatment=gt_treatment,
            gt_avoid=gt_avoid,
            trajectory_summary=trajectory_summary,
            agent_diagnosis=agent_diagnosis,
            agent_treatment=agent_treatment,
        )

        # 调用 M2 评分
        try:
            response = self.llm_client.call(
                prompt=prompt,
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            scores = json.loads(response)

            # 诊断
            result.diagnosis_correct = scores.get("diagnosis_correct", False)
            result.diagnosis_score = float(scores.get("diagnosis_score", 0))
            result.diagnosis_reason = scores.get("diagnosis_reason", "")

            # 治疗
            result.treatment_correct = scores.get("treatment_correct", False)
            result.treatment_score = float(scores.get("treatment_score", 0))
            result.treatment_reason = scores.get("treatment_reason", "")

            # 安全
            result.avoid_violated = scores.get("avoid_violated", False)
            result.avoid_score = float(scores.get("avoid_score", 0))
            result.avoid_reason = scores.get("avoid_reason", "")
            result.avoid_violations = scores.get("avoid_violations", [])

            # 综合分数 = (诊断 + 治疗 + 安全) / 3
            result.total_score = (result.diagnosis_score + result.treatment_score + result.avoid_score) / 3

        except Exception as e:
            print(f"[Judger Error] M2 scoring failed: {e}")
            # 使用简单匹配作为备选
            result = self._fallback_evaluate(
                result, gt_diagnosis, gt_treatment, gt_avoid, agent_diagnosis, agent_treatment
            )

        # 效率统计
        result.efficiency = self._analyze_efficiency(trajectory)

        # 费用评估（取整）
        if self.cost_evaluator is not None:
            try:
                agent_result = {
                    "case_id": case_id,
                    "chief_complaint": chief_complaint,
                    "trajectory": trajectory,
                    "diagnosis": agent_diagnosis,
                    "treatment": agent_treatment,
                }
                cost_result = self.cost_evaluator.estimate_from_agent_result(agent_result)
                result.total_cost = int(round(cost_result.total_cost))
            except Exception as e:
                print(f"[Judger Warning] Cost evaluation failed: {e}")
                result.total_cost = 0

        return result

    def _fallback_evaluate(
        self,
        result: EvalResult,
        gt_diagnosis: List[str],
        gt_treatment: List[str],
        gt_avoid: List[str],
        agent_diagnosis: str,
        agent_treatment: str,
    ) -> EvalResult:
        """
        备选评估方法（当 LLM 调用失败时）

        使用简单的字符串匹配进行评分
        """
        # 诊断匹配
        diagnosis_matches = 0
        for d in gt_diagnosis:
            if d.lower() in agent_diagnosis.lower():
                diagnosis_matches += 1

        if gt_diagnosis and diagnosis_matches > 0:
            match_ratio = diagnosis_matches / len(gt_diagnosis)
            result.diagnosis_score = 1 + match_ratio * 4
            result.diagnosis_correct = True
            result.diagnosis_reason = f"匹配{diagnosis_matches}/{len(gt_diagnosis)}个诊断"
        else:
            result.diagnosis_score = 1.0
            result.diagnosis_correct = False
            result.diagnosis_reason = "未匹配到任何诊断"

        # 治疗匹配
        treatment_matches = 0
        for t in gt_treatment:
            if t.lower() in agent_treatment.lower():
                treatment_matches += 1

        if gt_treatment and treatment_matches > 0:
            match_ratio = treatment_matches / len(gt_treatment)
            result.treatment_score = 1 + match_ratio * 4
            result.treatment_correct = True
            result.treatment_reason = f"匹配{treatment_matches}/{len(gt_treatment)}个治疗"
        else:
            result.treatment_score = 1.0
            result.treatment_correct = False
            result.treatment_reason = "未匹配到任何治疗"

        # 禁忌项检查
        result.avoid_violations = []
        for a in gt_avoid:
            if a.lower() in agent_treatment.lower():
                result.avoid_violations.append(a)

        result.avoid_violated = len(result.avoid_violations) > 0
        if result.avoid_violated:
            # 根据违反数计算分数
            violation_ratio = len(result.avoid_violations) / len(gt_avoid) if gt_avoid else 0
            result.avoid_score = max(1, 5 - violation_ratio * 4)
            result.avoid_reason = f"违反{len(result.avoid_violations)}个禁忌项"
        else:
            result.avoid_score = 5.0
            result.avoid_reason = "无禁忌违反"

        # 综合分数
        result.total_score = (result.diagnosis_score + result.treatment_score + result.avoid_score) / 3

        return result

    def batch_evaluate(self, results: List[EvalResult]) -> Dict[str, Any]:
        """
        批量评估并生成统计报告

        Args:
            results: 多个病例的评估结果列表

        Returns:
            Dict: 统计报告
        """
        if not results:
            return {}

        total = len(results)

        # 各维度统计
        diagnosis_correct_count = sum(1 for r in results if r.diagnosis_correct)
        treatment_correct_count = sum(1 for r in results if r.treatment_correct)
        avoid_violated_count = sum(1 for r in results if r.avoid_violated)

        avg_diagnosis_score = sum(r.diagnosis_score for r in results) / total
        avg_treatment_score = sum(r.treatment_score for r in results) / total
        avg_avoid_score = sum(r.avoid_score for r in results) / total
        avg_total_score = sum(r.total_score for r in results) / total

        # 效率统计
        avg_steps = sum(r.efficiency.total_steps for r in results) / total
        avg_exam_items = sum(r.efficiency.exam_items for r in results) / total

        # 费用统计
        costs = [r.total_cost for r in results if r.total_cost > 0]
        avg_cost = sum(costs) / len(costs) if costs else 0

        return {
            "total_cases": total,
            "diagnosis": {
                "accuracy": diagnosis_correct_count / total,
                "avg_score": avg_diagnosis_score,
            },
            "treatment": {
                "accuracy": treatment_correct_count / total,
                "avg_score": avg_treatment_score,
            },
            "safety": {
                "violation_rate": avoid_violated_count / total,
                "avg_score": avg_avoid_score,
            },
            "efficiency": {
                "avg_steps": avg_steps,
                "avg_exam_items": avg_exam_items,
            },
            "cost": {
                "avg_cost": avg_cost,
            },
            "total_avg_score": avg_total_score,
        }


if __name__ == "__main__":
    # 测试
    judger = Judger()

    # 模拟测试数据
    result = judger.evaluate(
        case_id="test_001",
        chief_complaint="胸痛2小时",
        ground_truth={
            "diagnosis": ["急性心肌梗死"],
            "treatment": ["急诊PCI", "抗血小板治疗"],
            "avoid": ["运动负荷试验"],
        },
        trajectory=[
            {"parsed": {"action": "ASK", "keywords": ["疼痛性质"]}, "observation": "压榨样疼痛"},
            {"parsed": {"action": "EXAM", "keywords": ["心电图", "心肌酶"]}, "observation": "ST段抬高"},
            {"parsed": {"action": "FINAL", "diagnosis": "急性心肌梗死"}, "observation": ""},
        ],
        agent_diagnosis="急性前壁心肌梗死",
        agent_treatment="急诊PCI，抗血小板治疗",
    )

    print(f"诊断: correct={result.diagnosis_correct}, score={result.diagnosis_score}, reason={result.diagnosis_reason}")
    print(f"治疗: correct={result.treatment_correct}, score={result.treatment_score}, reason={result.treatment_reason}")
    print(f"安全: violated={result.avoid_violated}, score={result.avoid_score}, reason={result.avoid_reason}")
    print(f"效率: steps={result.efficiency.total_steps}, exams={result.efficiency.exam_items}")
    print(f"费用: {result.total_cost}元")
    print(f"总分: {result.total_score}")