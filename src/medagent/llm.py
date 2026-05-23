# MedAgent/src/medagent/llm.py

"""
极简 LLM 客户端

支持:
- 直接调用 / 流式调用 / 工具调用
- prompt 和 message 两种输入格式（message 优先）
"""

import os
import threading
from typing import List, Dict, Optional, Union, Generator, Any
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


# 全局API调用计数器
class APICallCounter:
    """线程安全的API调用计数器"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._count = 0
                    cls._instance._start_time = datetime.now()
                    cls._instance._model_counts = {}
                    cls._instance._file_lock = threading.Lock()
        return cls._instance

    def increment(self, model_name: str = "unknown"):
        """增加计数"""
        with self._file_lock:
            self._count += 1
            if model_name not in self._model_counts:
                self._model_counts[model_name] = 0
            self._model_counts[model_name] += 1

    def get_count(self):
        """获取当前计数"""
        return self._count

    def get_elapsed_hours(self):
        """获取已运行小时数"""
        elapsed = datetime.now() - self._start_time
        return elapsed.total_seconds() / 3600

    def get_rate(self):
        """获取每小时调用次数"""
        hours = self.get_elapsed_hours()
        if hours > 0:
            return self._count / hours
        return 0

    def check_limit(self, limit_per_5h=6000):
        """检查是否接近限制 (5小时内6000次)"""
        hours = self.get_elapsed_hours()
        if hours >= 5:
            # 超过5小时，重置计数
            self._count = 0
            self._start_time = datetime.now()
            self._model_counts = {}
            return True
        rate = self.get_rate()
        estimated_5h = rate * 5
        return estimated_5h < limit_per_5h

    def get_status(self):
        """获取状态信息"""
        return {
            "total_calls": self._count,
            "elapsed_hours": self.get_elapsed_hours(),
            "calls_per_hour": self.get_rate(),
            "model_counts": dict(self._model_counts),
            "start_time": self._start_time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def print_status(self):
        """打印状态"""
        status = self.get_status()
        print(f"\n[API调用统计]")
        print(f"  总调用次数: {status['total_calls']}")
        print(f"  运行时间: {status['elapsed_hours']:.2f} 小时")
        print(f"  平均速率: {status['calls_per_hour']:.1f} 次/小时")
        print(f"  各模型调用:")
        for model, count in status['model_counts'].items():
            print(f"    {model}: {count}")


# 全局计数器实例
api_counter = APICallCounter()


class LLMClient:
    """
    统一的 OpenAI 兼容客户端

    使用示例:
        client = LLMClient()

        # 直接调用 - prompt 格式
        response = client.call(prompt="你好", system_prompt="你是助手")

        # 直接调用 - message 格式
        response = client.call(messages=[{"role": "user", "content": "你好"}])

        # 流式调用
        for chunk in client.call_stream(prompt="你好"):
            print(chunk, end="")

        # 工具调用
        tools = [{"type": "function", "function": {...}}]
        response = client.call(prompt="杭州天气", tools=tools)
    """

    def __init__(
        self,
        # 百炼 codingplan 配置 (默认)
        base_url: str = "https://coding.dashscope.aliyuncs.com/v1",
        api_key: Optional[str] = None,
        model_name: str = "qwen3.5-plus",
        temperature: float = 0.7,
        max_tokens: int = 32768,  # 增大到32K，避免截断
        timeout: float = 120.0,
        # 可选：指定其他 API key 环境变量
        # 对于 302API: base_url="https://api.302.ai/v1", api_key_env="302_API_KEY"
        # 对于百炼非codingplan: base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key_env="DASHSCOPE_API_KEY"
    ):
        self.base_url = base_url
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY_CP", "EMPTY")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_tokens_cap = int(os.getenv("MEDAGENT_MAX_TOKENS", "0") or "0")
        self.timeout = timeout

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def _resolve_max_tokens(self, max_tokens: Optional[int] = None) -> int:
        resolved = max_tokens if max_tokens is not None else self.max_tokens
        if self.max_tokens_cap > 0:
            return min(resolved, self.max_tokens_cap)
        return resolved

    def _build_messages(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        system_prompt: str = "You are a helpful assistant.",
    ) -> List[Dict]:
        """构建 messages 列表，message 格式优先"""
        if messages is not None:
            return messages

        if prompt is None:
            raise ValueError("Either prompt or messages must be provided")

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    def call(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        system_prompt: str = "You are a helpful assistant.",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        response_format: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Union[str, Dict]:
        """
        直接调用

        Args:
            prompt: 用户输入（与 messages 二选一）
            messages: 消息列表（优先于 prompt）
            system_prompt: 系统提示词（仅当使用 prompt 时生效）
            temperature: 温度
            max_tokens: 最大生成长度
            tools: 工具列表
            response_format: 响应格式（如 {"type": "json_object"}）

        Returns:
            str: 生成的文本内容（无工具调用时）
            Dict: 包含 tool_calls 的结果（有工具调用时）
        """
        msgs = self._build_messages(prompt, messages, system_prompt)

        params = {
            "model": self.model_name,
            "messages": msgs,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": self._resolve_max_tokens(max_tokens),
        }

        if tools:
            params["tools"] = tools
        if response_format:
            params["response_format"] = response_format
        params.update(kwargs)

        completion = self._client.chat.completions.create(**params)

        # 记录API调用
        api_counter.increment(self.model_name)

        choice = completion.choices[0]

        # 工具调用
        if choice.message.tool_calls:
            return {
                "content": choice.message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in choice.message.tool_calls
                ],
            }

        return choice.message.content or ""

    def call_stream(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        system_prompt: str = "You are a helpful assistant.",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """
        流式调用

        Yields:
            str: 逐块返回的文本内容
        """
        msgs = self._build_messages(prompt, messages, system_prompt)

        params = {
            "model": self.model_name,
            "messages": msgs,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": self._resolve_max_tokens(max_tokens),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        params.update(kwargs)

        stream = self._client.chat.completions.create(**params)

        # 记录API调用
        api_counter.increment(self.model_name)

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def batch_call(
        self,
        prompts: List[str],
        system_prompt: str = "You are a helpful assistant.",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_workers: int = 8,
    ) -> List[str]:
        """
        批量并发调用

        Args:
            prompts: 提示词列表
            system_prompt: 系统提示词
            max_workers: 并发数

        Returns:
            List[str]: 响应列表（顺序与输入一致）
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [None] * len(prompts)

        def _call_one(idx: int, prompt: str) -> tuple:
            try:
                content = self.call(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return idx, content
            except Exception as e:
                return idx, f"[Error] {e}"

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_call_one, idx, prompt): idx
                for idx, prompt in enumerate(prompts)
            }
            for future in as_completed(futures):
                idx, content = future.result()
                results[idx] = content

        return results


if __name__ == "__main__":
    # 测试
    client = LLMClient()

    print("=== 测试直接调用 ===")
    response = client.call(prompt="你是谁？")
    print(response)

    print("\n=== 测试流式调用 ===")
    for chunk in client.call_stream(prompt="说一个笑话"):
        print(chunk, end="", flush=True)
    print()

    print("\n=== 测试工具调用 ===")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取城市天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"}
                    },
                    "required": ["city"],
                },
            },
        }
    ]
    result = client.call(prompt="杭州天气怎么样？", tools=tools)
    print(result)
