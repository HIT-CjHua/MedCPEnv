#!/usr/bin/env python3
"""
Baichuan 模型本地性能测试脚本

测试两个 Baichuan 模型的生成速度：
1. Baichuan-M3-235B-GPTQ-INT4
2. Baichuan-M2-32B-GPTQ-Int4

使用 vLLM 进行推理，统计生成时间。
"""

import os
import sys
import json
import time
import subprocess
import signal
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from openai import OpenAI

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.schema import MedicalCase


def load_test_cases(data_path: str, n: int = 100) -> List[Dict]:
    """加载测试数据"""
    cases = []

    if data_path.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    cases.append(data)
    elif data_path.endswith(".json"):
        with open(data_path, "r", encoding="utf-8") as f:
            cases = json.load(f)

    return cases[:n]


def build_prompt(case: Dict) -> str:
    """构建测试 prompt"""
    chief_complaint = case.get("chief_complaint", "")
    history = case.get("history_of_present_illness", "")

    prompt = f"""请根据以下病例信息进行分析和诊断。

主诉：{chief_complaint}

现病史：{history}

请简要分析可能的诊断方向。"""

    return prompt


def start_vllm_server(
    model_path: str,
    port: int,
    gpu_devices: str = "0,1",
    tensor_parallel_size: int = 2,
) -> subprocess.Popen:
    """启动 vLLM 服务"""

    print(f"\n正在启动 vLLM 服务...")
    print(f"  模型: {model_path}")
    print(f"  端口: {port}")
    print(f"  GPU: {gpu_devices}")
    print(f"  Tensor Parallel: {tensor_parallel_size}")

    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_devices

    # vLLM 命令
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--trust-remote-code",
        "--dtype", "float16",
        "--gpu-memory-utilization", "0.9",
    ]

    # 启动进程
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    # 等待服务启动
    print("  等待服务启动...")
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY")

    # 根据模型大小调整等待时间
    # M3-235B 约120GB，需要更长时间加载
    model_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
    max_wait = 600 if model_size > 100e9 else 300  # 大模型等待10分钟

    start_time = time.time()
    last_print = 0

    while time.time() - start_time < max_wait:
        elapsed = time.time() - start_time
        if elapsed - last_print >= 30:
            print(f"  已等待 {elapsed:.0f}s...")
            last_print = elapsed

        try:
            # 尝试获取模型列表
            models = client.models.list()
            print(f"  服务已启动! (耗时 {elapsed:.1f}s)")
            return process
        except Exception as e:
            time.sleep(5)
            continue

    print(f"  服务启动超时! (等待 {max_wait}s)")

    # 打印最后的服务日志
    try:
        stdout, _ = process.communicate(timeout=5)
        print(f"\n  vLLM 服务日志 (最后500字符):")
        print(stdout.decode()[-500:] if stdout else "无日志")
    except:
        pass

    process.terminate()
    raise RuntimeError("vLLM 服务启动超时")


def stop_vllm_server(process: subprocess.Popen) -> None:
    """停止 vLLM 服务"""

    print("\n正在停止 vLLM 服务...")

    try:
        # 发送 SIGTERM 到整个进程组
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)

        # 等待进程结束
        process.wait(timeout=30)
        print("  服务已停止")

    except Exception as e:
        print(f"  强制终止进程: {e}")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except:
            pass


def run_generate_test(
    client: OpenAI,
    model_name: str,
    prompts: List[str],
    max_tokens: int = 512,
) -> Dict:
    """运行生成测试并统计时间"""

    print(f"\n开始生成测试...")
    print(f"  测试样本数: {len(prompts)}")
    print(f"  最大生成长度: {max_tokens}")

    results = {
        "total_time": 0,
        "total_tokens": 0,
        "total_input_tokens": 0,
        "success": 0,
        "failed": 0,
        "latencies": [],
        "errors": [],
    }

    start_time = time.time()

    for i, prompt in enumerate(prompts):
        try:
            # 调用生成
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
            )

            # 记录统计
            if response.usage:
                results["total_tokens"] += response.usage.total_tokens
                results["total_input_tokens"] += response.usage.prompt_tokens

            latency = time.time() - start_time - results["total_time"]
            results["latencies"].append(latency)
            results["success"] += 1

            # 每10条打印进度
            if (i + 1) % 10 == 0:
                print(f"  已完成: {i + 1}/{len(prompts)}")

        except Exception as e:
            results["failed"] += 1
            results["errors"].append(str(e))
            print(f"  [Error] 样本 {i}: {e}")

    results["total_time"] = time.time() - start_time

    # 计算统计指标
    if results["success"] > 0:
        results["avg_latency"] = sum(results["latencies"]) / results["success"]
        results["tokens_per_second"] = results["total_tokens"] / results["total_time"]
        results["avg_output_tokens"] = (results["total_tokens"] - results["total_input_tokens"]) / results["success"]

    return results


def format_results(results: Dict, model_name: str) -> str:
    """格式化结果输出"""

    output = f"\n{'='*60}\n"
    output += f"{model_name} 测试结果\n"
    output += f"{'='*60}\n"
    output += f"成功: {results['success']}/{results['success'] + results['failed']}\n"
    output += f"失败: {results['failed']}\n"
    output += f"\n"
    output += f"时间统计:\n"
    output += f"  总耗时: {results['total_time']:.2f}s\n"
    output += f"  平均延迟: {results['avg_latency']:.2f}s\n"
    output += f"\n"
    output += f"吞吐统计:\n"
    output += f"  总 Token 数: {results['total_tokens']}\n"
    output += f"  输入 Token 数: {results['total_input_tokens']}\n"
    output += f"  吞吐速度: {results['tokens_per_second']:.2f} tokens/s\n"
    output += f"  平均输出长度: {results['avg_output_tokens']:.1f} tokens\n"

    if results["errors"]:
        output += f"\n错误列表:\n"
        for err in results["errors"][:5]:
            output += f"  - {err}\n"

    output += f"{'='*60}\n"

    return output


def save_results(results: Dict, model_name: str, output_dir: str) -> None:
    """保存结果到文件"""

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"baichuan_test_{model_name.replace('/', '_')}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"结果已保存: {filepath}")


def test_model(
    model_path: str,
    model_name: str,
    port: int,
    gpu_devices: str,
    prompts: List[str],
    output_dir: str,
    tensor_parallel_size: int = 2,
) -> Dict:
    """测试单个模型"""

    print(f"\n{'='*60}")
    print(f"开始测试模型: {model_name}")
    print(f"{'='*60}")

    # 启动 vLLM 服务
    server_process = start_vllm_server(
        model_path=model_path,
        port=port,
        gpu_devices=gpu_devices,
        tensor_parallel_size=tensor_parallel_size,
    )

    try:
        # 创建客户端
        client = OpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="EMPTY",
        )

        # 获取实际模型名称
        models = client.models.list()
        actual_model_name = models.data[0].id
        print(f"  实际模型名称: {actual_model_name}")

        # 运行测试
        results = run_generate_test(
            client=client,
            model_name=actual_model_name,
            prompts=prompts,
        )

        # 添加元信息
        results["model_path"] = model_path
        results["model_name"] = model_name
        results["test_samples"] = len(prompts)
        results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 输出和保存结果
        print(format_results(results, model_name))
        save_results(results, model_name, output_dir)

        return results

    finally:
        # 停止服务
        stop_vllm_server(server_process)

        # 等待 GPU 显存释放
        print("  等待 GPU 显存释放...")
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description="Baichuan 模型本地性能测试")

    parser.add_argument(
        "--models-dir",
        type=str,
        default="/dev/shm/models",
        help="模型目录",
    )

    parser.add_argument(
        "--data",
        type=str,
        default="output/generate_2000/merged_selected.jsonl",
        help="测试数据路径",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="测试样本数量",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/baichuan_test",
        help="结果输出目录",
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default="0,1",
        help="GPU 设备",
    )

    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=2,
        help="Tensor parallel size",
    )

    parser.add_argument(
        "--port1",
        type=int,
        default=8100,
        help="第一个模型服务端口",
    )

    parser.add_argument(
        "--port2",
        type=int,
        default=8101,
        help="第二个模型服务端口",
    )

    args = parser.parse_args()

    # 模型列表
    models = [
        {
            "path": f"{args.models_dir}/baichuan-inc/Baichuan-M3-235B-GPTQ-INT4",
            "name": "Baichuan-M3-235B-GPTQ-INT4",
        },
        {
            "path": f"{args.models_dir}/baichuan-inc/Baichuan-M2-32B-GPTQ-Int4",
            "name": "Baichuan-M2-32B-GPTQ-Int4",
        },
    ]

    print(f"\n{'='*60}")
    print("Baichuan 模型性能测试")
    print(f"{'='*60}")
    print(f"模型目录: {args.models_dir}")
    print(f"测试数据: {args.data}")
    print(f"样本数量: {args.n}")
    print(f"GPU 设备: {args.gpu}")
    print(f"Tensor Parallel: {args.tensor_parallel}")
    print(f"待测模型:")
    for m in models:
        print(f"  - {m['name']}")
    print(f"{'='*60}")

    # 加载测试数据
    print("\n加载测试数据...")
    cases = load_test_cases(args.data, args.n)
    prompts = [build_prompt(c) for c in cases]
    print(f"已加载 {len(prompts)} 条测试数据")

    # 依次测试每个模型
    all_results = []

    for i, model in enumerate(models):
        port = args.port1 if i == 0 else args.port2

        results = test_model(
            model_path=model["path"],
            model_name=model["name"],
            port=port,
            gpu_devices=args.gpu,
            prompts=prompts,
            output_dir=args.output,
            tensor_parallel_size=args.tensor_parallel,
        )

        all_results.append(results)

        # 模型间隔
        if i < len(models) - 1:
            print("\n准备测试下一个模型...")
            time.sleep(10)

    # 总结
    print(f"\n{'='*60}")
    print("测试总结")
    print(f"{'='*60}")

    for results in all_results:
        print(f"\n{results['model_name']}:")
        print(f"  吞吐速度: {results['tokens_per_second']:.2f} tokens/s")
        print(f"  平均延迟: {results['avg_latency']:.2f}s")
        print(f"  成功率: {results['success']}/{results['test_samples']}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()