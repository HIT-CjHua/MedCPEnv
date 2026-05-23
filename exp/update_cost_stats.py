"""Update model_stats JSON files with cost statistics from checkpoint files."""

import json
import glob
import os
from statistics import median

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def compute_cost_stats(checkpoint_path):
    """Load checkpoint JSONL and compute cost statistics."""
    costs = []
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            costs.append(entry.get("total_cost", 0))

    if not costs:
        return None

    nonzero = [c for c in costs if c > 0]

    return {
        "avg_cost": round(sum(costs) / len(costs), 2),
        "median_cost": round(median(costs), 2),
        "max_cost": max(costs),
        "min_cost": min(nonzero) if nonzero else 0,
        "total_cost": sum(costs),
        "cost_with_items": len(nonzero),
    }


def main():
    stats_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "model_stats_*.json")))

    if not stats_files:
        print("No model_stats_*.json files found in exp/results/")
        return

    print(f"Found {len(stats_files)} model stats files.\n")

    for stats_file in stats_files:
        basename = os.path.basename(stats_file)
        # Extract model name: model_stats_{model_name}.json
        model_name = basename.replace("model_stats_", "").replace(".json", "")
        checkpoint_file = os.path.join(RESULTS_DIR, f"checkpoint_{model_name}.jsonl")

        if not os.path.exists(checkpoint_file):
            print(f"[SKIP] {model_name}: no checkpoint file found at {checkpoint_file}")
            continue

        # Load existing stats
        with open(stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)

        # Compute and add cost stats
        cost_stats = compute_cost_stats(checkpoint_file)
        if cost_stats:
            stats["cost"] = cost_stats

            # Write back
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

            print(f"[OK]   {model_name}:")
            print(f"       avg={cost_stats['avg_cost']}, median={cost_stats['median_cost']}, "
                  f"max={cost_stats['max_cost']}, min(non-zero)={cost_stats['min_cost']}, "
                  f"total={cost_stats['total_cost']}, entries_with_cost={cost_stats['cost_with_items']}")
        else:
            print(f"[WARN] {model_name}: no cost data found in checkpoint")

    print("\nDone.")


if __name__ == "__main__":
    main()
