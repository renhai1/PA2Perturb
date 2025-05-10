import matplotlib.pyplot as plt
import numpy as np

# 数据
datasets = ['Cora', 'Citeseer', 'Pubmed', 'Flickr']
variants = ['PDGP-A', 'PDGP-B', 'PDGP-C', 'PDGP-D']

# None 与 Defense 两组的 PSR 数据
none_data = np.array([
    [82, 83, 85, 88],  # PDGP-A
    [80, 82, 81, 78],  # PDGP-B
    [84, 84, 83, 82],  # PDGP-C
    [88, 85, 84, 80],  # PDGP-D
])

defense_data = np.array([
    [72, 74, 76, 77],  # PDGP-A
    [75, 73, 72, 70],  # PDGP-B
    [78, 76, 74, 72],  # PDGP-C
    [80, 78, 76, 74],  # PDGP-D
])

# 绘图参数
bar_width = 0.18
x = np.arange(len(datasets))
colors = ['#d4a5a5', '#f1c16e', '#8ab6a7', '#5b9bd5']
hatches = ['//', 'xx', '\\\\', '++']
gray_color = 'dimgray'

# 创建子图
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

# 网格线设置
grid_kwargs = dict(which='major', linestyle='--', linewidth=1, color='lightgray', axis='y')

# 绘制 None 子图
for i in range(len(variants)):
    axes[0].bar(x + i * bar_width, none_data[i], width=bar_width, label=variants[i],
                color=colors[i], hatch=hatches[i], edgecolor='black')

axes[0].set_title('None')
axes[0].set_xticks(x + 1.5 * bar_width)
axes[0].set_xticklabels(datasets)
axes[0].set_ylabel('PSR (%)')
axes[0].set_ylim(70, 100)
axes[0].grid(**grid_kwargs)
for spine in axes[0].spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)
    spine.set_edgecolor(gray_color)

# 绘制 Defense 子图
for i in range(len(variants)):
    axes[1].bar(x + i * bar_width, defense_data[i], width=bar_width,
                color=colors[i], hatch=hatches[i], edgecolor='black')

axes[1].set_title('Defense')
axes[1].set_xticks(x + 1.5 * bar_width)
axes[1].set_xticklabels(datasets)
axes[1].grid(**grid_kwargs)
for spine in axes[1].spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)
    spine.set_edgecolor(gray_color)

# 添加图例
fig.legend(variants, loc='upper center', ncol=4)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
