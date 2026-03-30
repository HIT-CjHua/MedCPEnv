# MedAgent

## 设计核心:mocking env for benchmarking and agentic RL

目的:考察和训练LLM在模拟医院环境中,使用提供的医疗相关tool和自主查询资料协助疾病诊断/疾病治疗等任务.

传统相关研究存在的问题:大量使用LLM+角色扮演,或者以一些基本的workflow和SingleAgent作为基础组成部分搭建MultiAgent环境,API+MultiAgent系统带来天然的不确定性,高方差,高延时,不适于作为benchmark和作为agentic-RL训练环境

我们的实现方案:
将所有Agent可能需要使用的外部工具(这些工具集合起来用于代表医院环境)都定义为格式统一的tooluse交互函数;
通过合理的抽象,使用足量合成患者详细数据,
预设两类tooluse:
- 测量(获取)患者生理信息;通过访问数据实现
- 检索相关知识;通过预建立教材+百科+推理数据集的知识库+RAG实现;
