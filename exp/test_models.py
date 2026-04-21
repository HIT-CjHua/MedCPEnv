"""
模型调用测试脚本

测试 model_list.md 中列出的所有云端 API 模型，
检查是否都能正常调用。

排除本地 vLLM 部署的模型。
"""

import os
import json
import time
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# 测试模型列表
# -----------------------------------------------------------------------------

MODELS_302 = {
    "name": "302API",
    "base_url": "https://api.302.ai/v1",
    "api_key_env": "302_API_KEY",
    "models": [
        "gpt-5.4",
        "claude-opus-4-6",
        "gemini-3.1-pro-preview",
    ],
}

# 百炼 - codingplan 模型 (使用 DASHSCOPE_API_KEY_CP)
MODELS_BAILIAN_CODINGPLAN = {
    "name": "百炼API (codingplan)",
    "base_url": "https://coding.dashscope.aliyuncs.com/v1",
    "api_key_env": "DASHSCOPE_API_KEY_CP",
    "models": [
        "qwen3.5-plus",
        "qwen3-max-2026-01-23",
        "glm-5",
        "kimi-k2.5",
        "MiniMax-M2.5",
    ],
}

# 百炼 - 非 codingplan 模型 (使用 DASHSCOPE_API_KEY)
MODELS_BAILIAN_NORMAL = {
    "name": "百炼API (非codingplan)",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key_env": "DASHSCOPE_API_KEY",
    "models": [
        "deepseek-v3.2",
        "qwen3.5-35b-a3b",
    ],
}

# Embedding 模型 (使用 DASHSCOPE_API_KEY, 非codingplan)
MODELS_BAILIAN_EMBEDDING = {
    "name": "百炼API Embedding",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key_env": "DASHSCOPE_API_KEY",
    "models": [
        "text-embedding-v3",
    ],
}

# 测试 Prompt
TEST_PROMPT = "请用一句话介绍人工智能在医疗领域的应用。"
TEST_MESSAGES = [{"role": "user", "content": TEST_PROMPT}]

# 特殊模型参数
EXTRA_BODY_CONFIG = {
    # Qwen3.5 需要关闭思考模式
    "qwen3.5-plus": {
        "chat_template_kwargs": {"enable_thinking": False},
    },
    "qwen3-max-2026-01-23": {
        "chat_template_kwargs": {"enable_thinking": False},
    },
}

# -----------------------------------------------------------------------------
# 测试逻辑
# -----------------------------------------------------------------------------


def test_one_model(client: OpenAI, model_name: str, extra_body: dict = None) -> dict:
    """测试单个 Chat 模型"""
    result = {
        "model": model_name,
        "type": "chat",
        "status": "pending",
        "response": "",
        "latency_ms": 0,
        "error": "",
        "tokens": 0,
    }

    try:
        print(f"  测试 {model_name} ...", end="", flush=True)

        params = {
            "model": model_name,
            "messages": TEST_MESSAGES,
            "max_tokens": 512,
            "temperature": 0.7,
        }
        if extra_body:
            params["extra_body"] = extra_body

        start = time.time()
        response = client.chat.completions.create(**params)
        latency_ms = (time.time() - start) * 1000

        content = response.choices[0].message.content or ""
        usage = response.usage

        result["status"] = "success"
        result["response"] = content.strip()[:200]  # 截取前 200 字符
        result["latency_ms"] = round(latency_ms, 1)
        if usage:
            result["tokens"] = usage.completion_tokens

        print(f"  OK ({latency_ms:.0f}ms)")

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        print(f"  FAIL: {e}")

    return result


def test_one_embedding_model(client: OpenAI, model_name: str) -> dict:
    """测试单个 Embedding 模型"""
    result = {
        "model": model_name,
        "type": "embedding",
        "status": "pending",
        "response": "",
        "latency_ms": 0,
        "error": "",
        "embedding_dim": 0,
    }

    try:
        print(f"  测试 {model_name} (embedding) ...", end="", flush=True)

        start = time.time()
        response = client.embeddings.create(
            model=model_name,
            input=TEST_PROMPT,
        )
        latency_ms = (time.time() - start) * 1000

        embedding = response.data[0].embedding
        dim = len(embedding)

        result["status"] = "success"
        result["response"] = f"embedding vector (dim={dim})"
        result["latency_ms"] = round(latency_ms, 1)
        result["embedding_dim"] = dim

        print(f"  OK ({latency_ms:.0f}ms, dim={dim})")

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        print(f"  FAIL: {e}")

    return result


def test_provider(name: str, config: dict) -> list:
    """测试一个 provider 下的所有模型"""
    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        print(f"\n{'='*60}")
        print(f"  Provider: {name}")
        print(f"  跳过: 环境变量 {config['api_key_env']} 未设置")
        return []

    print(f"\n{'='*60}")
    print(f"  Provider: {name}")
    print(f"  Base URL: {config['base_url']}")
    print(f"  API Key 环境变量: {config['api_key_env']} (已配置)")
    print(f"  测试模型数: {len(config['models'])}")
    print(f"{'='*60}")

    client = OpenAI(
        base_url=config["base_url"],
        api_key=api_key,
        timeout=60.0,
    )

    results = []
    for model_name in config["models"]:
        extra_body = EXTRA_BODY_CONFIG.get(model_name)
        result = test_one_model(client, model_name, extra_body)
        results.append(result)

    return results


def test_embedding_provider(name: str, config: dict) -> list:
    """测试 Embedding 模型 provider"""
    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        print(f"\n{'='*60}")
        print(f"  Provider: {name}")
        print(f"  跳过: 环境变量 {config['api_key_env']} 未设置")
        return []

    print(f"\n{'='*60}")
    print(f"  Provider: {name}")
    print(f"  Base URL: {config['base_url']}")
    print(f"  API Key 环境变量: {config['api_key_env']} (已配置)")
    print(f"  测试模型数: {len(config['models'])} (Embedding)")
    print(f"{'='*60}")

    client = OpenAI(
        base_url=config["base_url"],
        api_key=api_key,
        timeout=60.0,
    )

    results = []
    for model_name in config["models"]:
        result = test_one_embedding_model(client, model_name)
        results.append(result)

    return results


def print_report(all_results: dict):
    """打印汇总报告"""
    print(f"\n\n{'='*60}")
    print(f"  模型调用测试报告")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    total = 0
    passed = 0

    for provider, results in all_results.items():
        if not results:
            continue
        print(f"\n--- {provider} ---")
        for r in results:
            total += 1
            status_icon = "OK" if r["status"] == "success" else "FAIL"
            if r["status"] == "success":
                passed += 1
            model_type = r.get("type", "chat")
            if model_type == "embedding":
                print(f"  [{status_icon}] {r['model']} | {r['latency_ms']}ms | dim={r.get('embedding_dim', 0)}")
            else:
                print(f"  [{status_icon}] {r['model']} | {r['latency_ms']}ms | {r.get('tokens', 0)} tokens")
            if r["status"] == "success":
                preview = r["response"][:100]
                print(f"        回复: {preview}")
            else:
                print(f"        错误: {r['error'][:120]}")

    print(f"\n{'='*60}")
    print(f"  总计: {passed}/{total} 通过")
    print(f"{'='*60}")


# -----------------------------------------------------------------------------
# 主入口
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("MedAgent 模型调用测试")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {}

    # 测试 302API
    all_results["302API"] = test_provider("302API", MODELS_302)

    # 测试 百炼 codingplan
    all_results["百炼API (codingplan)"] = test_provider("百炼API (codingplan)", MODELS_BAILIAN_CODINGPLAN)

    # 测试 百炼 非 codingplan (Chat)
    all_results["百炼API (非codingplan)"] = test_provider("百炼API (非codingplan)", MODELS_BAILIAN_NORMAL)

    # 测试 百炼 Embedding
    all_results["百炼API Embedding"] = test_embedding_provider("百炼API Embedding", MODELS_BAILIAN_EMBEDDING)

    # 打印报告
    print_report(all_results)

    # 保存 JSON 报告
    report_path = os.path.join(os.path.dirname(__file__), "model_test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存至: {report_path}")
