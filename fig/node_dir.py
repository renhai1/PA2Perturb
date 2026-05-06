import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# 配置
np.random.seed(42)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# 参数
num_classes = 7
clean_per_class = 1000
trigger_per_class = 30

# 使用更柔和的 Set2 colormap（最多8色）
cmap = cm.get_cmap('Set2', num_classes)
colors = [cmap(i) for i in range(num_classes)]

# 随机中心点
class_centers = np.random.uniform(low=-5, high=5, size=(num_classes, 2))

# 画布
plt.figure(figsize=(6, 5))

# 先画干净节点
for i in range(num_classes):
    cx, cy = class_centers[i]

    # Clean nodes
    clean_x = np.random.normal(loc=cx, scale=0.8, size=clean_per_class)
    clean_y = np.random.normal(loc=cy, scale=0.8, size=clean_per_class)
    plt.scatter(clean_x, clean_y, s=8, label=f'Class {i + 1}', color=colors[i], alpha=0.6, zorder=1)

# 最后统一画 perturbation 节点（红色小圆点）
trigger_x_all = []
trigger_y_all = []
for i in range(num_classes):
    cx, cy = class_centers[i]
    trigger_x = np.random.normal(loc=cx, scale=0.2, size=trigger_per_class)
    trigger_y = np.random.normal(loc=cy, scale=0.2, size=trigger_per_class)
    trigger_x_all.extend(trigger_x)
    trigger_y_all.extend(trigger_y)

plt.scatter(trigger_x_all, trigger_y_all, s=20, color='#00b050', alpha=0.6,
            label='Perturbation', zorder=5)

# 图例去重
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# plt.legend(
#     by_label.values(),
#     by_label.keys(),
#     loc='upper center',
#     bbox_to_anchor=(0.5, 1.05),  # 控制图例上移一些
#     ncol=len(by_label),          # 横向排开
#     markerscale=1.5
# )

# 隐藏坐标轴
plt.xticks([])
plt.yticks([])

plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# 保存
plt.savefig('node_distribution.png', format='png', dpi=300, bbox_inches='tight')

plt.show()


