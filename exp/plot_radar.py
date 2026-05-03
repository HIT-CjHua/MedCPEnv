"""
MedAgent Benchmark 雷达图生成脚本

基于 exp/output/main_sheet.md 数据生成多维度雷达图
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# Data (from main_sheet.md)
MODELS = [
    "gpt-5.4",
    "qwen3-max-2026-01-23",
    "qwen3.5-plus",
    "gemini-3.1-pro-preview",
    "MiniMax-M2.5",
    "kimi-k2.5",
    "glm-5",
    "claude-opus-4-6",
    "deepseek-v3.2",
]

# 雷达图维度 (归一化到 0-1)
# 1. 诊断能力: 诊断平均分 / 5
# 2. 治疗能力: 治疗平均分 / 5
# 3. 安全性: 安全平均分 / 5
# 4. 效率: (最大步数 - 平均步数) / (最大步数 - 最小步数)
# 5. 成本效益: (最大cost - 平均cost) / (最大cost - 最小cost)

DATA = {
    "gpt-5.4":         {"diag": 3.54, "treat": 2.64, "safe": 4.89, "steps": 4.94, "cost": 1093},
    "qwen3-max-2026-01-23": {"diag": 3.21, "treat": 2.35, "safe": 4.88, "steps": 7.71, "cost": 784},
    "qwen3.5-plus":    {"diag": 3.18, "treat": 2.41, "safe": 4.81, "steps": 7.27, "cost": 1226},
    "gemini-3.1-pro-preview": {"diag": 3.04, "treat": 2.32, "safe": 4.86, "steps": 7.19, "cost": 3749},
    "MiniMax-M2.5":    {"diag": 3.11, "treat": 2.33, "safe": 4.78, "steps": 7.12, "cost": 2208},
    "kimi-k2.5":       {"diag": 2.62, "treat": 2.23, "safe": 4.81, "steps": 7.92, "cost": 2676},
    "glm-5":           {"diag": 2.71, "treat": 2.14, "safe": 4.84, "steps": 8.74, "cost": 1547},
    "claude-opus-4-6": {"diag": 2.49, "treat": 2.09, "safe": 4.79, "steps": 9.03, "cost": 4488},
    "deepseek-v3.2":   {"diag": 2.10, "treat": 1.75, "safe": 4.88, "steps": 9.15, "cost": 1130},
}

# 计算归一化值
DIMENSIONS = ["Diagnostic", "Treatment", "Safety", "Efficiency", "Cost-eff"]

min_steps = min(d["steps"] for d in DATA.values())
max_steps = max(d["steps"] for d in DATA.values())
min_cost = min(d["cost"] for d in DATA.values())
max_cost = max(d["cost"] for d in DATA.values())

def normalize(value, min_val, max_val):
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)

def get_radar_data(name):
    d = DATA[name]
    return [
        d["diag"] / 5.0,                          # 诊断能力
        d["treat"] / 5.0,                          # 治疗能力
        d["safe"] / 5.0,                           # 安全性
        1.0 - normalize(d["steps"], min_steps, max_steps),  # 效率 (步数越少越好)
        1.0 - normalize(d["cost"], min_cost, max_cost),     # 成本效益 (成本越低越好)
    ]


def plot_radar_top5(output_path="exp/output/radar_top5.png"):
    """Plot radar chart for Top 5 models"""
    top5 = MODELS[:5]
    
    angles = np.linspace(0, 2 * np.pi, len(DIMENSIONS), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    for i, model in enumerate(top5):
        values = get_radar_data(model)
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2.5, label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(DIMENSIONS, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    ax.set_title('MedAgent Benchmark - Top 5 Models', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Top 5 radar saved: {output_path}")


def plot_radar_all(output_path="exp/output/radar_all.png"):
    """Plot radar chart for all models"""
    angles = np.linspace(0, 2 * np.pi, len(DIMENSIONS), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    
    cmap = plt.cm.Set1
    colors = [cmap(i) for i in range(len(MODELS))]
    
    for i, model in enumerate(MODELS):
        values = get_radar_data(model)
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i], alpha=0.7)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(DIMENSIONS, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title('MedAgent Benchmark - All Models', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"All models radar saved: {output_path}")


def plot_radar_facets(output_path="exp/output/radar_facets.png"):
    """Plot faceted radar chart (one subplot per model)"""
    top5 = MODELS[:5]
    n = len(top5)
    
    angles = np.linspace(0, 2 * np.pi, len(DIMENSIONS), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(polar=True))
    axes = axes.flatten()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    for i, model in enumerate(top5):
        ax = axes[i]
        values = get_radar_data(model)
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=3, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(DIMENSIONS, fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_title(f'{model}', fontsize=14, fontweight='bold', pad=10)
    
    axes[5].set_visible(False)
    
    fig.suptitle('MedAgent Benchmark - Top 5 Models (Faceted)', fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Faceted radar saved: {output_path}")


if __name__ == "__main__":
    print("Generating MedAgent Benchmark radar charts...")
    plot_radar_top5()
    plot_radar_all()
    plot_radar_facets()
    print("Done!")
