import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from itertools import cycle

# 设置字体为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# 1. 数据准备
x_values = [40, 80, 120, 160, 200]

psr_datasets = {
    "SBA": [0, 0, 0, 0, 0],
    "GTA": [83.2, 83.2, 83.2, 83.2, 83.2],
    "UGBA": [89.6, 91.7, 93.4, 94.1, 95.7],
    "DPGBA": [89.8, 92.4, 93.9, 95.2, 95.7],
    "Our": [91.5, 93.6, 95.1, 96.3, 97.1]
}

psr_gnns = {
    "SBA": [0, 0, 0, 0, 0],
    "GTA": [0, 0, 0, 5.3, 12.4],
    "UGBA": [0, 0, 0, 6.7, 14.2],
    "DPGBA": [0, 0, 0, 9.3, 14.9],
    "Our": [91.5, 93.6, 95.1, 96.3, 97.1]
}

# 2. 配色与符号设置
set2_dark2_colors = sns.color_palette("Set2", 4) + sns.color_palette("Dark2", 4)
color_cycle = cycle(set2_dark2_colors)

dataset_colors = {k: next(color_cycle) for k in psr_datasets.keys()}
gnn_colors = {k: next(color_cycle) for k in psr_gnns.keys()}

common_markers = {
    "SBA": "o",
    "GTA": "s",
    "UGBA": "^",
    "DPGBA": "v",
    "Our": "D"
}

# 3. 全局样式
sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(7, 3.3), sharey=True)
x_minor_locator = MultipleLocator(20)

# 4. 子图1：不同数据集
for dataset, psr in psr_datasets.items():
    axes[0].plot(x_values, psr,
                 label=dataset,
                 color=dataset_colors[dataset],
                 marker=common_markers[dataset],
                 linewidth=1.5,
                 markersize=5)

axes[0].set_title("Different Datasets", fontsize=12)
axes[0].set_xlabel("Hard Node Size", fontsize=11)
axes[0].set_ylabel("PSR (%)", fontsize=11)
axes[0].set_xticks(x_values)
axes[0].xaxis.set_minor_locator(x_minor_locator)
axes[0].set_ylim(0, 100)
axes[0].grid(True, linestyle="--", alpha=0.5)
axes[0].legend(loc='lower right', frameon=False, fontsize=9)

# 5. 子图2：不同 GNN 模型
for model, psr in psr_gnns.items():
    axes[1].plot(x_values, psr,
                 label=model,
                 color=gnn_colors[model],
                 marker=common_markers[model],
                 linewidth=1.5,
                 markersize=5)

axes[1].set_title("Different GNN Models", fontsize=12)
axes[1].set_xlabel("Hard Node Size", fontsize=11)
axes[1].set_xticks(x_values)
axes[1].xaxis.set_minor_locator(x_minor_locator)
axes[1].set_ylim(0, 100)
axes[1].grid(True, linestyle="--", alpha=0.5)
axes[1].legend(loc='lower right', frameon=False, fontsize=9)

# 加粗坐标轴
for ax in axes:
    ax.spines["bottom"].set_linewidth(1.1)
    ax.spines["left"].set_linewidth(1.1)

# 6. 布局调整与导出
plt.tight_layout()
plt.savefig("psr_comparison.pdf", bbox_inches="tight", dpi=300)  # 可选导出为PDF
plt.show()
