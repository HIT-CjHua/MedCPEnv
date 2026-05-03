# MedAgent/src/medagent/agent.py

"""
ReAct式医疗问诊Agent

使用XML tag进行结构化输出，在max_retry次数内循环执行:

循环流程:
- Re: LLM读取当前messages列表，选择动作并生成结构化输出
- Act: 解析动作，调用对应工具执行，获取结果
- Observe: 将动作和结果摘要追加到messages，节省context window

动作空间:
- ASK: 问诊，获取主观信息
- EXAM: 检查，获取客观信息
- KNOWLEDGE: 知识库查询
- FINAL: 最终诊断与治疗建议

生成格式示例:
    <act>
        <action>ASK</action>
        <keywords>疼痛性质, 疼痛部位, 持续时间</keywords>
    </act>

    <act>
        <action>EXAM</action>
        <keywords>血常规, 胸片, 心电图</keywords>
    </act>

    <act>
        <action>KNOWLEDGE</action>
        <query>急性阑尾炎的诊断标准与鉴别诊断</query>
    </act>

    <act>
        <action>FINAL</action>
        <diagnosis>急性阑尾炎</diagnosis>
        <treatment>急诊手术切除阑尾</treatment>
    </act>
"""

import re
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.schema import MedicalCase
from .llm import LLMClient
from .tool import ToolManager
from .knowledge_tool_v2 import KeywordKnowledgeBase


@dataclass
class AgentState:
    """Agent 状态"""
    messages: List[Dict] = field(default_factory=list)
    step_count: int = 0
    is_done: bool = False
    diagnosis: str = ""
    treatment: str = ""
    trajectory: List[Dict] = field(default_factory=list)  # 完整轨迹，用于评测


class MedAgent:
    """
    医疗问诊 Agent

    ReAct 循环: Reasoning -> Acting -> Observing
    """

    SYSTEM_PROMPT = """/no_think
You are a professional medical consultation AI assistant. Your task is to diagnose patients based on their chief complaints through questioning, examinations, and knowledge queries.

## Available Tools

{tools_prompt}

## Actions

1. ASK: Consult to obtain the patient's subjective information (symptoms, medical history, etc.)
2. EXAM: Examine to obtain the patient's objective information (lab tests, imaging, etc.)
3. KNOWLEDGE: Query the medical knowledge base using keywords (returns top-3 matched QA records)
4. FINAL: Provide final diagnosis and treatment recommendations

## Output Format

Use XML tags to output your decision:

Consultation:
<act>
    <action>ASK</action>
    <keywords>keyword1, keyword2</keywords>
</act>

Examination:
<act>
    <action>EXAM</action>
    <keywords>test_item1, test_item2</keywords>
</act>

Knowledge Query:
<act>
    <action>KNOWLEDGE</action>
    <keywords>keyword1, keyword2, keyword3</keywords>
</act>

Final Diagnosis:
<act>
    <action>FINAL</action>
    <diagnosis>diagnosis result</diagnosis>
    <treatment>treatment recommendation</treatment>
</act>

## Notes

1. Output only one action at a time
2. Based on the chief complaint, prioritize consultation to obtain key information
3. Choose examination items appropriately, avoid over-testing
4. Query the knowledge base using multiple relevant keywords when in doubt (e.g., disease name, diagnostic criteria, treatment)
5. Provide diagnosis and treatment once you have sufficient information"""

    def __init__(
        self,
        llm_client: LLMClient,
        case: MedicalCase,
        knowledge_base: Optional[KeywordKnowledgeBase] = None,
        max_steps: int = 20,
        top_k: int = 3,
        verbose: bool = True,
    ):
        self.llm_client = llm_client
        self.case = case
        self.knowledge_base = knowledge_base
        self.max_steps = max_steps
        self.top_k = top_k
        self.verbose = verbose

        self.tool_manager = ToolManager(case, knowledge_base, top_k=top_k)
        self.state = AgentState()

    def _log(self, msg: str):
        """打印日志"""
        if self.verbose:
            print(f"[Step {self.state.step_count}] {msg}")

    def _build_initial_messages(self) -> List[Dict]:
        """构建初始 messages"""
        tools_prompt = self.tool_manager.get_all_tools_prompt()
        system_prompt = self.SYSTEM_PROMPT.format(tools_prompt=tools_prompt)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Chief Complaint: {self.case.chief_complaint}\n\nPlease start the consultation."},
        ]

    def _extract_tag(self, text: str, tag: str) -> Optional[str]:
        """提取 XML 标签内容"""
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _remove_thinking(self, text: str) -> str:
        """
        移除思考内容，只保留实际输出

        过滤掉 <think>...</think> 或 <thinking>...</thinking> 标签内容
        """
        # 移除 <think>...</think>
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # 移除 <thinking>...</thinking>
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
        return text.strip()

    def _parse_action(self, response: str) -> Dict:
        """
        解析 LLM 输出的动作

        Returns:
            Dict: {"action": str, "keywords": List[str], "diagnosis": str, "treatment": str}
        """
        result = {
            "action": None,
            "keywords": [],
            "diagnosis": "",
            "treatment": "",
        }

        # 提取 action
        action = self._extract_tag(response, "action")
        if action:
            result["action"] = action.strip().upper()

        # 提取 keywords
        keywords = self._extract_tag(response, "keywords")
        if keywords:
            result["keywords"] = [k.strip() for k in keywords.split(",") if k.strip()]

        # 提取 diagnosis
        diagnosis = self._extract_tag(response, "diagnosis")
        if diagnosis:
            result["diagnosis"] = diagnosis.strip()

        # 提取 treatment
        treatment = self._extract_tag(response, "treatment")
        if treatment:
            result["treatment"] = treatment.strip()

        return result

    def _execute_action(self, parsed: Dict) -> str:
        """执行动作"""
        action = parsed.get("action")

        if action == "ASK":
            return self.tool_manager.execute("ASK", keywords=parsed.get("keywords", []))

        elif action == "EXAM":
            return self.tool_manager.execute("EXAM", keywords=parsed.get("keywords", []))

        elif action == "KNOWLEDGE":
            return self.tool_manager.execute("KNOWLEDGE", keywords=parsed.get("keywords", []))

        elif action == "FINAL":
            self.state.diagnosis = parsed.get("diagnosis", "")
            self.state.treatment = parsed.get("treatment", "")
            self.state.is_done = True
            return f"Diagnosis: {self.state.diagnosis}\nTreatment: {self.state.treatment}"

        else:
            return f"Unknown action: {action}. Use one of: ASK/EXAM/KNOWLEDGE/FINAL."

    def _summarize_observation(self, action: str, result: str, max_length: int = 300) -> str:
        """
        将观察结果摘要化

        Args:
            action: 动作类型
            result: 工具返回结果
            max_length: 最大长度

        Returns:
            str: 摘要后的观察
        """
        if len(result) <= max_length:
            return result

        # 简单截断 + 省略号
        return result[:max_length] + "..."

    def reset(self) -> Dict:
        """重置 Agent 状态"""
        self.state = AgentState()
        self.state.messages = self._build_initial_messages()
        return {"chief_complaint": self.case.chief_complaint}

    def step(self) -> Tuple[Dict, bool]:
        """
        执行一步

        Returns:
            Tuple[Dict, bool]: (观察结果, 是否结束)
        """
        if self.state.is_done:
            return {"message": "诊断已完成", "done": True}, True

        if self.state.step_count >= self.max_steps:
            self.state.is_done = True
            return {"message": "达到最大步数限制", "done": True}, True

        self.state.step_count += 1

        # Re: LLM 生成
        self._log("Thinking...")
        start_time = time.time()
        response = self.llm_client.call(messages=self.state.messages, temperature=0.7, max_tokens=32768)
        latency = time.time() - start_time

        # 估算token数（中文约1.5字符/token，英文约4字符/token）
        estimated_tokens = len(response) // 2  # 简单估算

        # 解析动作
        parsed = self._parse_action(response)
        action = parsed.get("action")

        self._log(f"Action: {action}")

        # 记录轨迹（包含latency和tokens）
        self.state.trajectory.append({
            "step": self.state.step_count,
            "response": response,
            "parsed": parsed,
            "latency": latency,
            "estimated_tokens": estimated_tokens,
        })

        # Act: 执行工具
        if action in ["ASK", "EXAM", "KNOWLEDGE"]:
            result = self._execute_action(parsed)
            self._log(f"Result: {result[:100]}...")

            # Observe: 摘要并追加到 messages
            observation = self._summarize_observation(action, result)

            # 过滤思考内容后再追加
            filtered_response = self._remove_thinking(response)
            self.state.messages.append({"role": "assistant", "content": filtered_response})
            self.state.messages.append({"role": "user", "content": f"观察结果:\n{observation}"})

            # 更新轨迹
            self.state.trajectory[-1]["observation"] = observation

            return {"action": action, "result": result, "done": False}, False

        elif action == "FINAL":
            result = self._execute_action(parsed)
            self._log(f"Final: {result}")

            # 过滤思考内容后再追加
            filtered_response = self._remove_thinking(response)
            self.state.messages.append({"role": "assistant", "content": filtered_response})
            self.state.trajectory[-1]["observation"] = result

            return {"action": "FINAL", "diagnosis": self.state.diagnosis, "treatment": self.state.treatment, "done": True}, True

        else:
            # 解析失败，提示重试
            self._log("Invalid action, retrying...")
            # 过滤思考内容后再追加
            filtered_response = self._remove_thinking(response)
            self.state.messages.append({"role": "assistant", "content": filtered_response})
            self.state.messages.append({"role": "user", "content": "Please output your action in valid XML format."})

            return {"action": "INVALID", "result": "格式错误", "done": False}, False

    def run(self) -> Dict:
        """
        运行完整诊断流程

        Returns:
            Dict: 包含诊断结果、轨迹等信息
        """
        self.reset()

        while not self.state.is_done and self.state.step_count < self.max_steps:
            self.step()

        return {
            "case_id": self.case.case_id,
            "chief_complaint": self.case.chief_complaint,
            "diagnosis": self.state.diagnosis,
            "treatment": self.state.treatment,
            "step_count": self.state.step_count,
            "trajectory": self.state.trajectory,
            "ground_truth": {
                "diagnosis": self.case.ground_truth.diagnosis,
                "treatment": self.case.ground_truth.treatment,
                "avoid": self.case.ground_truth.avoid,
            },
        }


if __name__ == "__main__":
    # 测试
    from src.schema import MedicalCase, MedicalItem, GroundTruth

    # 创建测试病例
    case = MedicalCase(
        case_id="test_001",
        difficulty="medium",
        tags=["内科", "心血管"],
        chief_complaint="胸痛2小时，伴有出汗",
        subjective=[
            MedicalItem(keywords=["疼痛性质", "胸痛特点"], content="胸痛为压榨样，位于胸骨后，向左肩放射", necessity=True),
            MedicalItem(keywords=["持续时间", "发病时间"], content="疼痛持续约2小时，休息后不缓解", necessity=True),
            MedicalItem(keywords=["伴随症状", "出汗"], content="伴有大汗、恶心", necessity=True),
            MedicalItem(keywords=["既往史", "病史"], content="高血压病史10年，糖尿病病史5年", necessity=True),
        ],
        objective=[
            MedicalItem(keywords=["心电图", "ECG"], content="心电图示V1-V4导联ST段抬高0.3mV", necessity=True),
            MedicalItem(keywords=["心肌酶", "肌钙蛋白"], content="肌钙蛋白T 2.5ng/mL（显著升高）", necessity=True),
            MedicalItem(keywords=["血常规", "WBC"], content="白细胞 12.5×10^9/L", necessity=False),
        ],
        ground_truth=GroundTruth(
            diagnosis=["急性前壁心肌梗死"],
            treatment=["急诊PCI", "抗血小板治疗", "抗凝治疗"],
            avoid=["运动负荷试验"],
        ),
    )

    # 创建 LLM 客户端
    llm_client = LLMClient()

    # 创建 Agent
    agent = MedAgent(llm_client=llm_client, case=case, max_steps=10)

    # 运行
    result = agent.run()
    print("\n=== 诊断结果 ===")
    print(f"诊断: {result['diagnosis']}")
    print(f"治疗: {result['treatment']}")
    print(f"步数: {result['step_count']}")
    print(f"\n正确诊断: {result['ground_truth']['diagnosis']}")