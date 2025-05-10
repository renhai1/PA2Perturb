import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === 设置字体为 Times New Roman ===
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.sans-serif"] = ["Times New Roman"]

# 1. 超参数范围
beta_vals = [0.01, 0.1, 1, 10, 100]
gamma_vals = [0.01, 0.1, 1, 10, 100]

# 2. 模拟 None 状态下的 PSR 数据
psr_none = np.array([
    [93.5, 93.1, 92.8, 91.3, 90.2],
    [94.1, 93.8, 93.5, 92.7, 92.1],
    [95.4, 95.1, 94.3, 93.1, 92.8],
    [96.9, 96.3, 95.9, 95.2, 94.7],
    [97.3, 96.9, 96.5, 96.1, 95.9],
])

# 3. 模拟 Defense 状态下的 PSR（整体下降）
psr_defense = np.array([
    [89.5, 89.7, 90.2, 89.6, 89.5],
    [91.4, 91.9, 92.3, 92.7, 92.5],
    [92.6, 93.1, 93.4, 93.2, 92.9],
    [91.7, 92.8, 93.5, 94.7, 94.5],
    [88.3, 90.2, 93.7, 95.4, 94.6],
])

# 4. 颜色规范
cmap = "coolwarm"
vmin, vmax = np.min(psr_defense), np.max(psr_none)

# 5. 图像绘制
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True,
                         gridspec_kw={"width_ratios": [1, 1], "wspace": 0.05})

# 左图：None
sns.heatmap(psr_none, annot=True, fmt=".1f", cmap=cmap, vmin=vmin, vmax=vmax,
            cbar=False, xticklabels=gamma_vals, yticklabels=beta_vals, ax=axes[0])
axes[0].set_xlabel(r"$\gamma$", fontsize=11, fontname="Times New Roman")
axes[0].set_ylabel(r"$\beta$", fontsize=11, fontname="Times New Roman")

# 右图：Defense（带色条 + β 刻度）
sns.heatmap(psr_defense, annot=True, fmt=".1f", cmap=cmap, vmin=vmin, vmax=vmax,
            cbar_kws={"label": "PSR (%)"},
            xticklabels=gamma_vals, yticklabels=beta_vals, ax=axes[1])
axes[1].set_xlabel(r"$\gamma$", fontsize=11, fontname="Times New Roman")
axes[1].set_ylabel("")  # 不重复显示 Y 轴标签

# 整体标题和布局
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
