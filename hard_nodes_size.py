import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from itertools import cycle

# 1. 数据准备
x_values = [40, 80, 120, 160, 200]

psr_datasets = {
    "SBA": [71.2, 73.5, 74.1, 76.0, 77.8],
    "GTA": [68.4, 70.6, 71.3, 73.9, 75.2],
    "UGBA": [66.7, 68.1, 70.5, 72.8, 74.6],
    "DPGBA": [65.2, 67.9, 69.4, 71.2, 73.1],
    "Our": [69.2, 71.9, 83.4, 90.2, 94.1]
}

psr_gnns = {
    "SBA": [71.2, 73.5, 74.1, 76.0, 77.8],
    "GTA": [68.4, 70.6, 71.3, 73.9, 75.2],
    "UGBA": [66.7, 68.1, 70.5, 72.8, 74.6],
    "DPGBA": [65.2, 67.9, 69.4, 71.2, 73.1],
    "Our": [92.5,71.9, 83.4, 90.2, 94.1]
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
    "Our": "D"  # 使用钻石形符号
}

# 3. 全局样式
sns.set(style="whitegrid", font_scale=1.2)
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

axes[0].set_title("Different Datasets", fontsize=11)
axes[0].set_xlabel("Hard Node Size", fontsize=10)
axes[0].set_ylabel("PSR (%)", fontsize=10)
axes[0].set_xticks(x_values)
axes[0].xaxis.set_minor_locator(x_minor_locator)
axes[0].set_ylim(64, 80)
axes[0].grid(True, linestyle="--", alpha=0.6)
axes[0].legend(loc='lower right', frameon=False, fontsize=9)

# 5. 子图2：不同 GNN 模型
for model, psr in psr_gnns.items():
    axes[1].plot(x_values, psr,
                 label=model,
                 color=gnn_colors[model],
                 marker=common_markers[model],
                 linewidth=1.5,
                 markersize=5)

axes[1].set_title("Different GNN Models", fontsize=11)
axes[1].set_xlabel("Hard Node Size", fontsize=10)
axes[1].set_xticks(x_values)
axes[1].xaxis.set_minor_locator(x_minor_locator)
axes[1].grid(True, linestyle="--", alpha=0.6)
axes[1].legend(loc='lower right', frameon=False, fontsize=9)

# 6. 布局调整与显示
plt.tight_layout()
plt.show()
