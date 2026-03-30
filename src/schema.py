"""
MedAgent Schema - 医疗病例评测数据结构

定义用于评测的标准化医疗病例 (MedicalCase) 数据结构。

设计原则:
1. 信息分层存储（主诉 / 主观信息 / 客观信息）
2. 支持关键词匹配的披露机制
3. 包含完整的 ground truth 用于评测
4. 使用统一的 list 结构存储 subjective 和 objective
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class MedicalItem:
    """
    统一医疗数据项 - 用于 subjective 和 objective

    通过关键词/触发词匹配披露信息内容。

    Attributes:
        keywords: 触发关键词列表（问诊命中 keywords 返回 content，检查命中 keywords 返回 content）
        content: 披露的信息内容（问诊回答或检查结果）
        necessity: 是否必要 (True/False)
    """
    keywords: List[str]
    content: str
    necessity: bool = False


@dataclass
class GroundTruth:
    """
    评测标准答案

    用于评估 Agent 的诊断能力和决策质量。

    Attributes:
        diagnosis: 正确诊断列表
        treatment: 标准治疗方案
        avoid: 禁忌项（避免使用的药物、避免做的检查等）
    """
    diagnosis: List[str] = field(default_factory=list)
    treatment: List[str] = field(default_factory=list)
    avoid: List[str] = field(default_factory=list)


@dataclass
class MedicalCase:
    """
    评测用标准化医疗病例

    包含一个完整病例所需的所有信息，支持：
    1. 主诉始终可见（患者主动描述）
    2. 主观信息通过问诊获取（关键词匹配披露）
    3. 客观信息通过检查获取（关键词触发）
    4. Ground Truth 用于评测

    Attributes:
        case_id: 病例唯一标识
        difficulty: 难度级别 (easy/medium/hard)
        tags: 标签列表（科室、疾病类型等）
        chief_complaint: 主诉（患者主动描述，始终可见）
        subjective: 主观信息列表（问诊项），通过关键词匹配披露
        objective: 客观信息列表（检查项），通过关键词触发披露
        ground_truth: 评测标准答案
        source: 数据来源 (synthetic/real/seed)
    """
    case_id: str
    difficulty: str  # easy/medium/hard
    tags: List[str]
    chief_complaint: str

    subjective: List[MedicalItem] = field(default_factory=list)
    objective: List[MedicalItem] = field(default_factory=list)
    ground_truth: GroundTruth = field(default_factory=GroundTruth)
    source: str = ""  # synthetic/real/seed

    @classmethod
    def dict_to_case(cls, data: dict) -> "MedicalCase":
        """从字典创建 MedicalCase"""
        subjective = []
        for item in data.get("subjective", []):
            if isinstance(item, dict):
                subjective.append(MedicalItem(
                    keywords=item.get("keywords", []),
                    content=item.get("content") or item.get("response") or item.get("result", ""),
                    necessity=item.get("necessity", False)
                ))
            else:
                subjective.append(MedicalItem(
                    keywords=[],
                    content=str(item),
                    necessity=False
                ))

        objective = []
        for item in data.get("objective", []):
            if isinstance(item, dict):
                objective.append(MedicalItem(
                    keywords=item.get("keywords", []),
                    content=item.get("content") or item.get("response") or item.get("result", ""),
                    necessity=item.get("necessity", False)
                ))
            else:
                objective.append(MedicalItem(
                    keywords=[],
                    content=str(item),
                    necessity=False
                ))

        gt_data = data.get("ground_truth", {})
        ground_truth = GroundTruth(
            diagnosis=gt_data.get("diagnosis", []),
            treatment=gt_data.get("treatment", []),
            avoid=gt_data.get("avoid", []),
        )

        return cls(
            case_id=data.get("case_id", ""),
            difficulty=data.get("difficulty", "medium"),
            tags=data.get("tags", []),
            chief_complaint=data.get("chief_complaint", ""),
            subjective=subjective,
            objective=objective,
            ground_truth=ground_truth,
            source=data.get("source", ""),
        )

    def case_to_dict(self) -> dict:
        """将 MedicalCase 转换为字典"""
        return {
            "case_id": self.case_id,
            "difficulty": self.difficulty,
            "tags": self.tags,
            "chief_complaint": self.chief_complaint,
            "subjective": [
                {
                    "keywords": item.keywords,
                    "content": item.content,
                    "necessity": item.necessity
                }
                for item in self.subjective
            ],
            "objective": [
                {
                    "keywords": item.keywords,
                    "content": item.content,
                    "necessity": item.necessity
                }
                for item in self.objective
            ],
            "ground_truth": {
                "diagnosis": self.ground_truth.diagnosis,
                "treatment": self.ground_truth.treatment,
                "avoid": self.ground_truth.avoid,
            },
            "source": self.source,
        }


if __name__ == "__main__":
    pass
