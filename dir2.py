import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# 配置
np.random.seed(42)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# 参数
num_classes = 5
clean_per_class = 500
trigger_total = 150  # 总共触发器数量，不再按类区分

# 使用 Set2 colormap
cmap = cm.get_cmap('Set2', num_classes)
colors = [cmap(i) for i in range(num_classes)]

# 随机生成每个类别中心
class_centers = np.random.uniform(low=-5, high=5, size=(num_classes, 2))

# 创建画布
plt.figure(figsize=(6, 5))

legend_handles = []

# 先画干净节点
for i in range(num_classes):
    cx, cy = class_centers[i]

    # Clean nodes
    clean_x = np.random.normal(loc=cx, scale=0.8, size=clean_per_class)
    clean_y = np.random.normal(loc=cy, scale=0.8, size=clean_per_class)
    sc_clean = plt.scatter(clean_x, clean_y, s=8, label=f'Class {i + 1}', color=colors[i], alpha=0.6, zorder=1)
    if i == 0:
        legend_handles.append(sc_clean)

# 随机生成 trigger 节点（完全不按类分布）
trigger_x = np.random.uniform(low=-6, high=6, size=trigger_total)
trigger_y = np.random.uniform(low=-6, high=6, size=trigger_total)
sc_trigger = plt.scatter(trigger_x, trigger_y, s=8, color='black', alpha=0.9, label='Perturbation', zorder=5)
legend_handles.append(sc_trigger)

# 图例
plt.legend(handles=legend_handles, labels=['Class 1', 'Perturbation'], scatterpoints=1)

# 隐藏坐标轴
plt.xticks([])
plt.yticks([])
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# 保存图片
plt.savefig('node_distribution_disordered_trigger.png', format='png', dpi=300, bbox_inches='tight')
plt.show()
