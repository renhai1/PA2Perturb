import matplotlib.pyplot as plt
import numpy as np

# === 设置全局字体 ===
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.sans-serif"] = ["Times New Roman"]

# 数据
datasets = ['Cora', 'Citeseer', 'Pubmed', 'Flickr']
variants = ['PDGP-R', 'PDGP-E', 'PDGP-P', 'PDGP']

# None 与 Defense 两组的 PSR 数据
none_data = np.array([
    # [74.5, 98.4, 88.4, 97.2],
    # [72.1, 96.4, 90.2, 95.3],
    # [71.8, 94.5, 86.4, 93.9],
    # [70.9, 98.9, 89.7, 97.1],

    [74.5, 72.1, 71.8, 70.9],
    [98.4, 96.4, 94.5, 98.9],
    [88.4, 90.2, 86.4, 89.7],
    [97.2, 95.3, 93.9, 97.1]
])

defense_data = np.array([
    [22.4, 21.7, 23.8, 24.6],
    [15.4, 17.2, 14.2, 7.3],
    [87.6, 89.4, 85.7, 88.6],
    [91.2, 95.8, 91.1, 89.5],
])

# 绘图参数
bar_width = 0.18
x = np.arange(len(datasets))
colors = ['#d4a5a5', '#f1c16e', '#8ab6a7', '#5b9bd5']
hatches = ['//', 'xx', '\\\\', '++']
gray_color = 'dimgray'

# 创建子图
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

# 网格线设置
grid_kwargs = dict(which='major', linestyle='--', linewidth=1, color='lightgray', axis='y')

# 左图：None
for i in range(len(variants)):
    axes[0].bar(x + i * bar_width, none_data[i], width=bar_width,
                label=variants[i], color=colors[i],
                hatch=hatches[i], edgecolor='black')

axes[0].set_xticks(x + 1.5 * bar_width)
axes[0].set_xticklabels(datasets, fontsize=10)
axes[0].set_ylabel('VSR (%)', fontsize=11)
axes[0].set_ylim(70, 100)
axes[0].grid(**grid_kwargs)
axes[0].set_xlabel("No defense", fontsize=11, fontweight='bold')
axes[0].xaxis.set_label_position('bottom')

# 右图：Defense
for i in range(len(variants)):
    axes[1].bar(x + i * bar_width, defense_data[i], width=bar_width,
                color=colors[i], hatch=hatches[i], edgecolor='black')

axes[1].set_xticks(x + 1.5 * bar_width)
axes[1].set_xticklabels(datasets, fontsize=10)
axes[1].set_ylabel('PSR (%)', fontsize=11)
axes[1].set_ylim(5, 100)
axes[1].grid(**grid_kwargs)
axes[1].tick_params(labelleft=True)
axes[1].set_xlabel("Defense", fontsize=11, fontweight='bold')
axes[1].xaxis.set_label_position('bottom')

# 美化边框
for ax in axes:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_edgecolor(gray_color)

# 图例
fig.legend(variants, loc='upper center', ncol=4, fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()
