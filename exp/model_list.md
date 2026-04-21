# API调用
## 302API
gpt-5.4
claude-opus-4-6
gemini-3.1-pro-preview

### (可选,暂不使用) 前代模型
gpt-5.2
claude-opus-4-5-20251101
gemini-3-pro-preview

### 用于评测的模型--baichuan
Baichuan-M2

### 302参考代码
```python
from openai import OpenAI

# 配置你的参数
client = OpenAI(
    api_key=# 使用env文件中的302_API_KEY,
    base_url="https://api.302.ai/v1"
)

try:
    response = client.chat.completions.create(
        model="glm-4.7",  # 或者换成你目标平台的具体模型名
        messages=[
            {"role": "user", "content": "你好，讲个故事"}
        ]
    )
    # 输出回复内容
    print("AI 回复内容：", response.choices[0].message.content)

except Exception as e:
    print(f"调用失败，错误信息：{e}")
```

## 百炼API-coding plan
需使用模型
- qwen3.6-plus
- qwen3.5-plus
- qwen3-max-2026-01-23
- glm-5
- kimi-k2.5
- MiniMax-M2.5

百炼api代码调用已有实现(MedAgent\src\medagent\llm.py),api key使用env中的DASHSCOPE_API_KEY_CP;
注意url要用codingplan的https://coding.dashscope.aliyuncs.com/v1

## 百炼 非codingplan LLM
apikey用env的DASHSCOPE_API_KEY
需使用模型
- deepseek-v3.2
- qwen3.5-35b-a3b

### 文本向量模型 用于knowledge tool,
embedding模型
- text-embedding-v4

参考调用代码
```python
import dashscope
from http import HTTPStatus
input_texts = "衣服的质量杠杠的，很漂亮，不枉我等了这么久啊，喜欢，以后还来这里买"

resp = dashscope.TextEmbedding.call(
model="text-embedding-v4",
input=input_texts
)
print(resp)
```

rerank模型
- qwen3-rerank
```python
import dashscope
from http import HTTPStatus

def text_rerank():
    resp = dashscope.TextReRank.call(
        model="qwen3-rerank",
        query="什么是文本排序模型",
        documents=[
            "文本排序模型广泛用于搜索引擎和推荐系统中，它们根据文本相关性对候选文本进行排序",
            "量子计算是计算科学的一个前沿领域",
            "预训练语言模型的发展给文本排序模型带来了新的进展"
        ],
        top_n=10,
        return_documents=True,
        instruct="Given a web search query, retrieve relevant passages that answer the query."
    )
    if resp.status_code == HTTPStatus.OK:
        print(resp)
    else:
        print(resp)


if __name__ == '__main__':
    text_rerank()
```


# 本地部署+vllm server
dir="/dev/shm/models"

model_name="openai-mirror/gpt-oss-20b"
model_name="google/gemma-4-26B-A4B-it"
model_name="Qwen/Qwen3.5-35B-A3B"
model_name="Qwen/Qwen3-30B-A3B"

先使用vllm加载模型,然后调用

参考vllm命令:
```
vllm serve google/gemma-4-31B-it \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
  --model_name # 可指定模型名,也可以不指定
  --port 8400 # 后面用840x
```

Qwen3.5需要显式指定思考模板:
```python
chat_response = client.chat.completions.create(
    model="Qwen/Qwen3.5-35B-A3B",
    messages=messages,
    max_tokens=32768,
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }, 
```