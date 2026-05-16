import matplotlib.pyplot as plt
import numpy as np
from math import pi

# ========== 全局字体设置 ==========
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "serif"]
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["pdf.fonttype"] = 42  # 确保PDF中字体可编辑

# ========== 跨任务迁移性数据 ==========
# 固定GNN架构，Surrogate Task -> Suspected Task
# 任务类型: Node-level, Link-level, Graph-level
# 数据集维度: Cora, Citeseer, Pubmed, Flickr, Cora-P, Cora-F

datasets = ["Cora", "Citeseer", "Pubmed", "Flickr", "Cora-P", "Cora-F"]

cross_task_data = {
    "GCN": {
        "Node → Link":  [93.2, 92.4, 93.7, 91.8, 90.5, 89.6],
        "Node → Graph": [89.7, 89.1, 88.3, 87.5, 86.4, 85.8],
        "Link → Node":  [92.1, 91.5, 92.8, 90.6, 89.8, 88.9],
        "Link → Graph": [88.4, 87.6, 87.1, 86.3, 85.5, 84.7],
        "Graph → Node": [87.6, 86.8, 86.2, 85.4, 84.1, 83.5],
        "Graph → Link": [87.1, 86.3, 85.7, 84.9, 83.6, 82.9],
    },
    "GraphSAGE": {
        "Node → Link":  [92.4, 91.7, 93.1, 91.2, 89.8, 89.1],
        "Node → Graph": [88.9, 88.2, 87.5, 86.3, 85.6, 84.9],
        "Link → Node":  [91.5, 90.8, 92.0, 89.7, 89.1, 88.2],
        "Link → Graph": [87.6, 86.9, 86.2, 85.4, 84.5, 83.8],
        "Graph → Node": [86.8, 86.1, 85.5, 84.6, 83.4, 82.7],
        "Graph → Link": [86.2, 85.5, 84.9, 84.1, 82.8, 82.1],
    },
    "GAT": {
        "Node → Link":  [95.8, 95.3, 94.7, 95.1, 93.4, 92.8],
        "Node → Graph": [92.6, 92.1, 91.5, 90.8, 89.7, 89.1],
        "Link → Node":  [95.1, 94.7, 93.9, 94.3, 92.5, 91.9],
        "Link → Graph": [91.8, 91.2, 90.6, 89.9, 88.8, 88.2],
        "Graph → Node": [91.1, 90.4, 89.8, 89.2, 88.1, 87.4],
        "Graph → Link": [90.5, 89.8, 89.2, 88.6, 87.3, 86.7],
    },
    "GIN": {
        "Node → Link":  [90.6, 89.8, 90.2, 88.7, 87.4, 86.8],
        "Node → Graph": [87.3, 86.5, 85.9, 85.1, 84.2, 83.5],
        "Link → Node":  [89.4, 88.7, 89.1, 87.5, 86.6, 85.9],
        "Link → Graph": [86.1, 85.4, 84.8, 84.1, 83.2, 82.4],
        "Graph → Node": [85.3, 84.6, 84.1, 83.4, 82.3, 81.6],
        "Graph → Link": [84.7, 84.0, 83.5, 82.8, 81.7, 81.0],
    },
}

# ========== 科研论文配色 (Nature-style, 高区分度) ==========
task_pair_styles = {
    "Node → Link":  {"color": "#2166AC", "marker": "o",  "ls": "-"},    # 深蓝
    "Node → Graph": {"color": "#E08214", "marker": "s",  "ls": "-"},    # 橙
    "Link → Node":  {"color": "#1B7837", "marker": "D",  "ls": "--"},   # 深绿
    "Link → Graph": {"color": "#B2182B", "marker": "^",  "ls": "--"},   # 深红
    "Graph → Node": {"color": "#762A83", "marker": "v",  "ls": ":"},    # 紫
    "Graph → Link": {"color": "#8C510A", "marker": "P",  "ls": ":"},    # 棕
}

# ========== 绘制雷达图 ==========
num_vars = len(datasets)
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]  # 闭合

fig, axes = plt.subplots(1, 4, figsize=(24, 6.5), subplot_kw=dict(polar=True))
fig.subplots_adjust(wspace=0.45, top=0.82, bottom=0.05, left=0.03, right=0.97)

gnn_names = ["GCN", "GraphSAGE", "GAT", "GIN"]

for idx, (ax, gnn) in enumerate(zip(axes.flat, gnn_names)):
    ax.set_facecolor("#FAFBFC")

    for task_pair, style in task_pair_styles.items():
        values = cross_task_data[gnn][task_pair]
        values_plot = values + values[:1]

        ax.plot(angles, values_plot, linestyle=style["ls"], linewidth=2.0,
                label=task_pair, color=style["color"],
                marker=style["marker"], markersize=6, alpha=0.88,
                markeredgecolor='white', markeredgewidth=0.8)
        ax.fill(angles, values_plot, alpha=0.05, color=style["color"])

    # 坐标轴标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(datasets, fontsize=10, fontweight='bold', color="#333333")

    # Y轴
    ax.set_ylim(80, 97)
    ax.set_yticks([82, 86, 90, 94])
    ax.set_yticklabels(["82", "86", "90", "94"], fontsize=8, color="gray")
    ax.set_rlabel_position(30)

    # 网格
    ax.grid(color='#CCCCCC', linewidth=0.7, linestyle='--', alpha=0.6)
    ax.spines['polar'].set_color('#CCCCCC')
    ax.spines['polar'].set_linewidth(0.8)

    # 子图标题
    ax.set_title(gnn, fontsize=15, fontweight='bold', pad=20, color="#222222")

# ========== 统一图例 ==========
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=12,
           frameon=True, fancybox=True, shadow=False,
           edgecolor='#CCCCCC', facecolor='white',
           bbox_to_anchor=(0.5, 0.98), columnspacing=1.5,
           handletextpad=0.5, handlelength=2.5)

# ========== 保存PDF ==========
output_path = "/Users/keyuan/PycharmProjects/PA2Perturb/datashow/radar_cross_task.pdf"
plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
plt.close()
print(f"Radar chart saved to: {output_path}")
