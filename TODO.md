## agent实现
实现agent;

实现一个agent loop,在max_retry次数内:
- Re:LLM读取当前messages列表,选择动作并进行生成;
- Act:解析动作并执行对应的工具,得到工具执行结果;
- Observe:将**动作和工具执行结果的摘要**更新到messages中,节省context window;

## 知识库实现
建库+本地缓存+embedding检索+重排实现在scripts/rag.py中

建库流程:
- 使用MedAgent/data/knowledge_dataset/ResponseMed.json,每条数据作为一个chunk,同时去掉统一的提示词模板;
- 使用本地GPU+vllm部署的Qwen3-embedding模型,端口8400,为每个chunk计算embedding vector并存入chroma向量数据库;
- 建库完成后本地缓存;

检索和重排

## Tool实现
实现Tool基类+Ask|Exam|Knowledge工具类

每个tool必须包含的内容:
Name: web_search
Tool description: Searches the web for specific queries
Input: query (string) - The search term to look up
Output: String containing the search results

Ask Tool:
- 选择动作并传入keywords
- 尝试匹配subjective中的keywords
- 如果匹配,则返回content
- 如果不匹配,则返回"相关信息不明"

Exam Tool:
- 选择动作并传入keywords
- 尝试匹配objective中的keywords
- 如果匹配,则返回content
- 如果不匹配,则返回"相关检查不适用"

Knowledge Tool:
- 选择动作并传入query
- 使用query,在知识库中进行检索得到topk,重排得到topn相关文档
- 将topn相关文档进行摘要并返回