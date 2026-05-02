import json
import os

fdir = 'exp/results'
models = {}

for f in os.listdir(fdir):
    if not f.startswith('checkpoint_') or not f.endswith('.jsonl') or '_dedup' in f:
        continue
    
    model_name = f.replace('checkpoint_', '').replace('.jsonl', '')
    filepath = os.path.join(fdir, f)
    
    costs = []
    count = 0
    with open(filepath, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                cost = data.get('total_cost', 0)
                costs.append(cost)
                count += 1
            except:
                pass
    
    if costs:
        sorted_costs = sorted(costs)
        n = len(sorted_costs)
        models[model_name] = {
            'count': n,
            'avg': sum(costs) / n,
            'median': sorted_costs[n // 2],
            'min': min(costs),
            'max': max(costs),
            'total': sum(costs),
            'p25': sorted_costs[n // 4],
            'p75': sorted_costs[3 * n // 4],
        }

print(f"{'Model':<30} {'Count':>6} {'Avg':>10} {'Median':>10} {'Min':>8} {'Max':>10} {'Total':>12} {'P25':>8} {'P75':>8}")
print("-" * 100)
for m in sorted(models.keys(), key=lambda x: models[x]['avg'], reverse=True):
    s = models[m]
    print(f"{m:<30} {s['count']:>6} {s['avg']:>10.0f} {s['median']:>10.0f} {s['min']:>8} {s['max']:>10} {s['total']:>12} {s['p25']:>8.0f} {s['p75']:>8.0f}")
