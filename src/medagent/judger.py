# MedAgent/src/medagent/judger.py

"""
评测器模块

使用 LLM 结合 ground_truth 进行综合打分：
- 诊断准确率
- 治疗方案合理性
- 禁忌项是否违反
- 问诊/检查效率
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json

from .llm import LLMClient


@dataclass
class EvalResult:
    """评测结果"""
    case_id: str

    # 诊断相关
    diagnosis_correct: bool = False
    diagnosis_score: float = 0.0
    diagnosis_reason: str = ""

    # 治疗相关
    treatment_score: float = 0.0
    treatment_reason: str = ""

    # 禁忌项
    avoid_violated: bool = False
    avoid_violations: List[str] = field(default_factory=list)
    safety_ratio: float = 1.0  # 安全比例 = 1 - (违反数/总禁忌数)

    # 过程评估
    necessary_check_ratio: float = 0.0  # 必要检查完成比例
    unnecessary_check_count: int = 0     # 不必要检查次数

    # 综合分数
    base_score: float = 0.0  # 基数分数 = (诊断+治疗)/2
    total_score: float = 0.0  # 最终总分 = 基数 × 安全比例

    # 详细信息
    trajectory: List[Dict] = field(default_factory=list)
    ground_truth: Dict = field(default_factory=dict)
    agent_diagnosis: str = ""
    agent_treatment: str = ""


JUDGE_PROMPT = """你是一位资深的临床医学专家，负责评估 AI 医疗问诊系统的表现。

## 评估任务

请根据以下信息对 AI 的诊断过程和结果进行综合评估：

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

请按以下维度进行评分：

### 1. 诊断准确性 (diagnosis_score) - 1-5分
采用论文评审式评分：
- 5分 (强接受): 诊断完全正确，与标准诊断一致或高度吻合
- 4分 (接受): 诊断基本正确，包含主要诊断，可能有小偏差
- 3分 (中性): 诊断方向正确但不够具体或遗漏重要诊断
- 2分 (拒绝): 诊断有较大错误，但有部分合理成分
- 1分 (强拒绝): 诊断完全错误或偏离方向

### 2. 治疗方案合理性 (treatment_score) - 1-5分
- 5分 (强接受): 治疗方案完全合理，涵盖必要措施，无禁忌违反
- 4分 (接受): 治疗方案基本合理，有小瑕疵但不影响安全性
- 3分 (中性): 治疗方案有明显不足或遗漏，但不违反禁忌
- 2分 (拒绝): 治疗方案存在较大问题或轻微违反禁忌
- 1分 (强拒绝): 治疗方案严重不合理或违反重要禁忌

### 3. 禁忌项检查 (avoid_check)
**重要：仅检查治疗方案中是否推荐了禁忌项**
- 检查 AI 给出的治疗建议中是否包含或推荐了 ground_truth 中列出的禁忌事项
- 如果治疗建议中推荐了禁忌项（如建议饮酒、建议禁忌药物等），则记录为违反
- 问诊过程中询问相关病史不属于违反禁忌，是正确的医疗行为
- 安全比例 = 1 - (违反数量 / 总禁忌数量)

### 4. 综合评分 (total_score)
综合评分计算方式：
- 基数分数 = (诊断分数 + 治疗分数) / 2
- 最终总分 = 基数分数 × 安全比例

## 输出格式

请输出 JSON 格式：

```json
{{
    "diagnosis_correct": <true/false, 诊断是否基本正确>,
    "diagnosis_score": <1-5>,
    "diagnosis_reason": "<评分理由>",
    "treatment_score": <1-5>,
    "treatment_reason": "<评分理由>",
    "avoid_violated": <true/false, 治疗方案中是否推荐了禁忌项>,
    "avoid_violations": ["<具体违反的禁忌项>"],
    "total_score": <1-5>,
    "summary": "<一句话总结>"
}}
```

**重要说明**：
- `diagnosis_correct`：请根据语义判断 Agent 的诊断是否基本正确，不要求完全一致，只要诊断方向正确、疾病类型匹配即可
- `avoid_violated`：仅判断**治疗方案**中是否推荐了禁忌项，不检查问诊过程
- 所有评分维度都由你综合判断，不依赖简单的字符串匹配

只输出 JSON，不要有其他内容。
"""


class Judger:
    """
    评测器

    使用 LLM 对 Agent 的诊断过程和结果进行综合评估
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        model_name: str = "qwen3.5-plus",
    ):
        """
        初始化评测器

        Args:
            llm_client: LLM 客户端（可选，不传则使用默认 judge 模型）
            model_name: Judge 模型名称（默认 qwen3.5-plus）
        """
        # Judge 使用独立的模型，不与被评测 Agent 共用
        if llm_client is None:
            self.llm_client = LLMClient(model_name=model_name)
        else:
            self.llm_client = llm_client
        self.model_name = model_name

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

    def _check_avoid_violations(
        self,
        agent_treatment: str,
        avoid_list: List[str],
    ) -> List[str]:
        """
        检查治疗方案中是否推荐了禁忌项

        注意：仅检查治疗方案，不检查问诊/检查过程。
        问诊时询问相关病史是正确的医疗行为，不应计入违反。

        Args:
            agent_treatment: Agent给出的治疗方案
            avoid_list: 禁忌项列表

        Returns:
            List[str]: 违反的禁忌项列表
        """
        violations = []

        if not avoid_list or not agent_treatment:
            return violations

        # 仅检查治疗建议中是否推荐了禁忌项
        treatment_lower = agent_treatment.lower()
        for avoid_item in avoid_list:
            if avoid_item.lower() in treatment_lower:
                violations.append(avoid_item)

        return violations

    def _calculate_necessary_check_ratio(
        self,
        trajectory: List[Dict],
        necessary_items: List[Dict],
    ) -> float:
        """计算必要检查/问诊完成比例"""
        if not necessary_items:
            return 1.0  # 没有必要的项目则认为完成

        completed = 0
        for item in necessary_items:
            item_keywords = item.get("keywords", [])
            for step in trajectory:
                action = step.get("parsed", {}).get("action", "")
                if action in ["ASK", "EXAM"]:
                    step_keywords = step.get("parsed", {}).get("keywords", [])
                    # 检查是否有匹配
                    for ik in item_keywords:
                        for sk in step_keywords:
                            if ik.lower() in sk.lower() or sk.lower() in ik.lower():
                                completed += 1
                                break
                        else:
                            continue
                        break

        return min(completed / len(necessary_items), 1.0)

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

        # 调用 LLM 评分
        try:
            response = self.llm_client.call(
                prompt=prompt,
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            scores = json.loads(response)

            result.diagnosis_correct = scores.get("diagnosis_correct", False)
            result.diagnosis_score = float(scores.get("diagnosis_score", 0))
            result.diagnosis_reason = scores.get("diagnosis_reason", "")
            result.treatment_score = float(scores.get("treatment_score", 0))
            result.treatment_reason = scores.get("treatment_reason", "")
            result.avoid_violated = scores.get("avoid_violated", False)
            result.avoid_violations = scores.get("avoid_violations", [])
            result.base_score = (result.diagnosis_score + result.treatment_score) / 2
            result.safety_ratio = 1.0
            result.total_score = float(scores.get("total_score", 0))

        except Exception as e:
            print(f"[Judger Error] LLM scoring failed: {e}")
            # 使用简单匹配作为备选
            result = self._fallback_evaluate(
                result, gt_diagnosis, gt_treatment, gt_avoid, agent_diagnosis, agent_treatment
            )

        # 检查禁忌项（额外验证，仅检查治疗方案）
        violations = self._check_avoid_violations(agent_treatment, gt_avoid)
        if violations:
            result.avoid_violated = True
            result.avoid_violations.extend(violations)

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

        使用简单的字符串匹配进行评分，评分范围 1-5 分
        """

        # 诊断匹配 (1-5分)
        diagnosis_matches = 0
        for d in gt_diagnosis:
            if d.lower() in agent_diagnosis.lower():
                diagnosis_matches += 1

        if gt_diagnosis and diagnosis_matches > 0:
            # 根据匹配比例映射到1-5分
            match_ratio = diagnosis_matches / len(gt_diagnosis)
            result.diagnosis_score = 1 + match_ratio * 4  # 1-5分
        else:
            result.diagnosis_score = 1.0  # 最低分

        result.diagnosis_correct = diagnosis_matches > 0

        # 治疗匹配 (1-5分)
        treatment_matches = 0
        for t in gt_treatment:
            if t.lower() in agent_treatment.lower():
                treatment_matches += 1

        if gt_treatment and treatment_matches > 0:
            match_ratio = treatment_matches / len(gt_treatment)
            result.treatment_score = 1 + match_ratio * 4  # 1-5分
        else:
            result.treatment_score = 1.0  # 最低分

        # 禁忌项检查（仅检查治疗方案）
        result.avoid_violated = False
        result.avoid_violations = []

        if gt_avoid:
            for a in gt_avoid:
                if a.lower() in agent_treatment.lower():
                    result.avoid_violated = True
                    result.avoid_violations.append(a)

        # 计算总分：诊断+治疗作为基数，安全性作为比例
        base_score = (result.diagnosis_score + result.treatment_score) / 2
        result.base_score = base_score

        if gt_avoid and result.avoid_violations:
            # 安全比例 = 1 - (违反数 / 总禁忌数)
            safety_ratio = 1 - (len(result.avoid_violations) / len(gt_avoid))
            result.safety_ratio = safety_ratio
            result.total_score = base_score * safety_ratio
        else:
            result.safety_ratio = 1.0
            result.total_score = base_score

        return result

    def batch_evaluate(self, results: List[Dict]) -> Dict[str, Any]:
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

        # 统计
        diagnosis_correct_count = sum(1 for r in results if r.diagnosis_correct)
        avoid_violated_count = sum(1 for r in results if r.avoid_violated)

        avg_diagnosis_score = sum(r.diagnosis_score for r in results) / total
        avg_treatment_score = sum(r.treatment_score for r in results) / total
        avg_total_score = sum(r.total_score for r in results) / total

        # 分数分布 (1-5分制)
        score_distribution = {
            "强接受 (5)": sum(1 for r in results if r.total_score >= 4.5),
            "接受 (4)": sum(1 for r in results if 3.5 <= r.total_score < 4.5),
            "中性 (3)": sum(1 for r in results if 2.5 <= r.total_score < 3.5),
            "拒绝 (2)": sum(1 for r in results if 1.5 <= r.total_score < 2.5),
            "强拒绝 (1)": sum(1 for r in results if r.total_score < 1.5),
        }

        return {
            "total_cases": total,
            "diagnosis_accuracy": diagnosis_correct_count / total,
            "diagnosis_avg_score": avg_diagnosis_score,
            "treatment_avg_score": avg_treatment_score,
            "avoid_violation_rate": avoid_violated_count / total,
            "total_avg_score": avg_total_score,
            "score_distribution": score_distribution,
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
            {"step": 1, "parsed": {"action": "ASK", "keywords": ["疼痛性质"]}, "observation": "压榨样疼痛"},
            {"step": 2, "parsed": {"action": "EXAM", "keywords": ["心电图"]}, "observation": "ST段抬高"},
            {"step": 3, "parsed": {"action": "FINAL", "diagnosis": "急性心肌梗死"}, "observation": ""},
        ],
        agent_diagnosis="急性前壁心肌梗死",
        agent_treatment="急诊PCI，抗血小板治疗",
    )

    print(f"诊断分数: {result.diagnosis_score}")
    print(f"治疗分数: {result.treatment_score}")
    print(f"总分: {result.total_score}")
    print(f"违反禁忌: {result.avoid_violated}")