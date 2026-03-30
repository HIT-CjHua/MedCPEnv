# MedAgent/src/utils.py

from openai import OpenAI
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Callable,List, Optional, Union
import re
import numpy as np
from dotenv import load_dotenv


# llm call
"""
参考
import os
from openai import OpenAI


client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY_CP"),
    base_url="https://coding.dashscope.aliyuncs.com/v1",
)
completion = client.chat.completions.create(
    model="qwen3.5-plus",
    messages=[{'role': 'user', 'content': '你是谁？'}]
)
print(completion.choices[0].message.content)
"""

def call_llm(client,prompt):
    completion = client.chat.completions.create(
        model="qwen-flash",
        messages=[
            {
                "role": "user",
                "content": prompt, 
            },
        ],
        response_format={"type": "json_object"}
    )

    json_string = completion.choices[0].message.content
    print(json_string)


# xml tag提取
def extract_tag_content(text: str, tag_name: str, all_matches: bool = False) -> Union[Optional[str], List[str]]:
    """
    从文本中提取指定标签内容（XML/HTML风格）。
    Args:
        text: 待解析的文本
        tag_name: 标签名
        all_matches: 是否返回所有匹配
    Returns:
        若 all_matches=False，则返回第一个匹配或 None；
        若 all_matches=True，则返回所有匹配的列表（可能为空列表）。
    """
    pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None if not all_matches else []
    return matches if all_matches else matches[0].strip()

# 并发处理函数
def run_concurrent_task(
    input_file=None,
    input_data=None,  # 新增参数：直接传数据
    output_file="output.jsonl",
    handler=None,
    max_workers=10,
    output_format="jsonl",
    checkpoint_every=1000,
    return_results=True,  # 是否在内存中保留结果
):
    """
    并发任务运行器（断点续跑 + 检查点保存）

    Args:
        input_file: 输入文件路径（json/jsonl）
        input_data: 直接传入的数据 list[dict]
        output_file: 输出文件路径
        handler: 单任务处理函数，输入 dict，输出 dict/list 或 None
        max_workers: 最大并发数
        output_format: 'jsonl' 或 'json'
        checkpoint_every: 每处理 N 条保存一次检查点
        return_results: 是否将所有结果加载到内存并返回（注意大文件时内存占用）

    Returns:
        List[Dict] or None: 所有处理结果（如果 return_results=True），否则为 None
    """
    if not handler:
        raise ValueError("handler function is required")

    if not input_data and not input_file:
        raise ValueError("either input_file or input_data must be provided")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # ===== 读取输入数据 =====
    if input_data is not None:
        dataset = input_data
    else:
        if input_file.lower().endswith(".jsonl"):
            with open(input_file, "r", encoding="utf-8") as f:
                dataset = [json.loads(line) for line in f if line.strip()]
        elif input_file.lower().endswith(".json"):
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    dataset = data
                else:
                    raise ValueError("JSON file must contain a list")
        else:
            raise ValueError(f"Unsupported format: {input_file}")

    # 补 id
    for idx, item in enumerate(dataset):
        if "id" not in item:
            item["id"] = str(idx)  # 使用 str 类型避免类型混淆

    item_id_set = {str(item["id"]) for item in dataset}
    if len(item_id_set) < len(dataset):
        print("[WARNING] Duplicate IDs detected, may cause resume issues.")

    # ===== 已处理 ID（断点续跑）=====
    processed_ids = set()
    if os.path.exists(output_file):
        print(f"[INFO] Resuming from {output_file}")
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                if output_format == "jsonl":
                    for line in f:
                        if line.strip():
                            obj = json.loads(line)
                            if "id" in obj:
                                processed_ids.add(str(obj["id"]))
                else:
                    for obj in json.load(f):
                        if "id" in obj:
                            processed_ids.add(str(obj["id"]))
        except Exception as e:
            print(f"[WARNING] Failed to read checkpoint: {e}")

    # ===== 剩余任务 =====
    remaining_dataset = [item for item in dataset if str(item["id"]) not in processed_ids]
    print(f"[INFO] Total: {len(dataset)} | Already processed: {len(processed_ids)} | To process: {len(remaining_dataset)}")

    if not remaining_dataset:
        print("[INFO] All items processed. Nothing to do.")
        return [] if return_results else None

    # ===== 并发执行 =====
    batch_results = []
    final_results = [] if return_results else None

    def save_batch(batch):
        """内部保存函数"""
        try:
            if output_format == "jsonl":
                mode = "a" if os.path.exists(output_file) else "w"
                with open(output_file, mode, encoding="utf-8") as f:
                    for res in batch:
                        f.write(json.dumps(res, ensure_ascii=False) + "\n")
            else:
                existing = []
                if os.path.exists(output_file):
                    with open(output_file, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                existing.extend(batch)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(existing, f, ensure_ascii=False, indent=2)
            print(f"[INFO] Checkpoint saved ({len(batch)} results)")
        except Exception as e:
            print(f"[ERROR] Save failed: {e}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(handler, item): item for item in remaining_dataset}

        for future in tqdm(as_completed(future_to_item), total=len(remaining_dataset), desc="Processing", unit="item"):
            item = future_to_item[future]
            item_id = str(item["id"])
            try:
                result = future.result()
            except Exception as e:
                result = {"id": item_id, "error": repr(e)}
                print(f"[ERROR] Task {item_id} raised exception: {e}")

            if result is None:
                continue

            # 处理 handler 返回 list 的情况
            if isinstance(result, list):
                outputs = result
            else:
                outputs = [result]

            # 添加到文件和内存
            for out in outputs:
                if "id" not in out:
                    out["id"] = item_id  # 确保输出也有 id

                if return_results:
                    final_results.append(out)
                batch_results.append(out)

            # 批量保存
            if len(batch_results) >= checkpoint_every:
                save_batch(batch_results)
                batch_results.clear()

    # ===== 保存最后残余 =====
    if batch_results:
        save_batch(batch_results)

    print(f"[INFO] ✅ Task completed. Results saved to: {output_file}")

    return final_results