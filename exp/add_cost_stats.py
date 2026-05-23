"""
为所有已测试数据补充cost统计

读取checkpoint文件，为每个case计算费用并更新记录。
"""

import os
import sys
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.medagent.cost import CostEvaluator
from src.medagent. import Judger

RESULTS_DIR = PROJECT_ROOT / "exp" / "results"
DATA_PATH = PROJECT_ROOT / "data" / "datasets" / "benchmark_1000.jsonl"

def load_case_data():
    """加载病例数据"""
    cases = {}
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            cases[data['case_id']] = data
    return cases

def load_checkpoint(model_name):
    """加载checkpoint文件（去重）"""
    checkpoint_file = RESULTS_DIR / f"checkpoint_{model_name}.jsonl"
    results = {}

    if not checkpoint_file.exists():
        return results

    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                case_id = data.get('case_id')
                if case_id:
                    results[case_id] = data

    return results

def save_checkpoint(model_name, results):
    """保存checkpoint文件"""
    checkpoint_file = RESULTS_DIR / f"checkpoint_{model_name}.jsonl"

    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        for data in results.values():
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

def compute_cost_for_case(case_data, checkpoint_data, cost_evaluator):
    """为单个case计算cost"""
    case_id = checkpoint_data.get('case_id')
    trajectory = checkpoint_data.get('trajectory', [])

    if not trajectory:
        return checkpoint_data

    # 构建agent_result
    agent_result = {
        'case_id': case_id,
        'chief_complaint': case_data.get('chief_complaint', ''),
        'trajectory': trajectory,
        'diagnosis': checkpoint_data.get('agent_diagnosis', ''),
        'treatment': checkpoint_data.get('agent_treatment', ''),
    }

    try:
        cost_result = cost_evaluator.estimate_from_agent_result(agent_result)
        checkpoint_data['total_cost'] = int(round(cost_result.total_cost))
        checkpoint_data['cost_detail'] = {
            'service_cost': cost_result.service_cost,
            'medicine_cost': cost_result.medicine_cost,
            'matched_count': cost_result.matched_count,
            'generated_count': cost_result.generated_count,
        }
    except Exception as e:
        print(f"[Cost Error] {case_id}: {e}")
        checkpoint_data['total_cost'] = 0

    return checkpoint_data

def process_model(model_name, cases_data, max_workers=5):
    """处理单个模型的cost统计"""
    print(f"\n处理模型: {model_name}")

    results = load_checkpoint(model_name)
    if not results:
        print(f"  无checkpoint数据")
        return

    # 检查是否已有足够的cost数据（跳过已完成的模型）
    existing_cost_count = sum(1 for d in results.values() if d.get('total_cost', 0) > 0)
    success_cases_count = sum(1 for d in results.values() if d.get('total_score', 0) > 0 and d.get('trajectory'))

    if existing_cost_count >= success_cases_count * 0.9:  # 90%以上已完成则跳过
        print(f"  已有cost数据: {existing_cost_count}/{success_cases_count}, 跳过")
        return

    # 过滤有trajectory的成功case（有评测分数且有轨迹视为成功）
    # 排除已有有效cost数据的case（total_cost > 0）
    success_cases = {
        cid: data for cid, data in results.items()
        if data.get('total_score', 0) > 0 and data.get('trajectory') and data.get('total_cost', 0) <= 0
    }

    print(f"  总记录: {len(results)}, 已有cost: {existing_cost_count}, 待处理: {len(success_cases)}")

    # 初始化cost evaluator
    cost_evaluator = CostEvaluator()

    # 并行计算cost（分批处理，每批100条，避免限流）
    updated_results = {}
    start_time = time.time()
    batch_size = 100

    case_list = list(success_cases.items())
    for batch_start in range(0, len(case_list), batch_size):
        batch_end = min(batch_start + batch_size, len(case_list))
        batch_cases = case_list[batch_start:batch_end]
        print(f"  处理批次 [{batch_start+1}-{batch_end}]/[{len(case_list)}]")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for case_id, checkpoint_data in batch_cases:
                case_data = cases_data.get(case_id, {})
                future = executor.submit(
                    compute_cost_for_case,
                    case_data,
                    checkpoint_data,
                    cost_evaluator
                )
                futures[future] = case_id

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {batch_start//batch_size + 1}"):
                case_id = futures[future]
                try:
                    updated_data = future.result()
                    updated_results[case_id] = updated_data
                except Exception as e:
                    print(f"    Error {case_id}: {e}")

        # 每批次后合并保存
        for case_id, data in updated_results.items():
            results[case_id] = data
        save_checkpoint(model_name, results)
        print(f"    批次完成，已保存 ({len(updated_results)} 条)")

        updated_results.clear()

        # 批次间等待，避免限流（最后一批不等待）
        if batch_end < len(case_list):
            wait_time = 10
            print(f"    等待 {wait_time}s...")
            time.sleep(wait_time)

    # 统计最终结果
    costs = [data.get('total_cost', 0) for data in results.values() if data.get('total_cost', 0) > 0]
    if costs:
        avg_cost = sum(costs) / len(costs)
        print(f"  平均费用: {avg_cost:.0f}元 ({len(costs)}条)")

def main():
    print("=" * 60)
    print("为已测试数据补充cost统计")
    print("=" * 60)

    # 加载病例数据
    cases_data = load_case_data()
    print(f"加载病例数据: {len(cases_data)}条")

    # 获取所有模型checkpoint（排除 dedup 变体）
    models = []
    for f in RESULTS_DIR.glob("checkpoint_*.jsonl"):
        model_name = f.stem.replace("checkpoint_", "")
        if "_dedup" in model_name:
            continue
        models.append(model_name)

    print(f"待处理模型: {models}")

    # 处理每个模型
    for model in models:
        process_model(model, cases_data, max_workers=6)

    print("\n完成!")

if __name__ == "__main__":
    main()