import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 超参数范围
beta_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
gamma_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# 2. 模拟 None 状态下的 PSR 数据
psr_none = np.array([
    [20, 18, 16, 14, 12, 10],
    [30, 28, 26, 24, 22, 20],
    [45, 42, 40, 38, 35, 32],
    [60, 58, 55, 53, 50, 48],
    [80, 75, 70, 65, 60, 55],
    [95, 92, 90, 85, 82, 78]
])

# 3. 模拟 Defense 状态下的 PSR（整体下降）
psr_defense = psr_none - 10

# 4. 颜色规范
cmap = "coolwarm"
vmin, vmax = np.min(psr_defense), np.max(psr_none)

# 5. 图像绘制
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True,
                         gridspec_kw={"width_ratios": [1, 1], "wspace": 0.05})

# 左图：None
sns.heatmap(psr_none, annot=True, fmt="d", cmap=cmap, vmin=vmin, vmax=vmax,
            cbar=False, xticklabels=gamma_vals, yticklabels=beta_vals, ax=axes[0])
axes[0].set_title("None", fontsize=13)
axes[0].set_xlabel(r"$\gamma$", fontsize=11)
axes[0].set_ylabel(r"$\beta$", fontsize=11)

# 右图：Defense（带色条）
sns.heatmap(psr_defense, annot=True, fmt="d", cmap=cmap, vmin=vmin, vmax=vmax,
            cbar_kws={"label": "PSR (%)"},
            xticklabels=gamma_vals, yticklabels=False, ax=axes[1])
axes[1].set_title("Defense", fontsize=13)
axes[1].set_xlabel(r"$\gamma$", fontsize=11)

# 整体标题和布局
plt.suptitle("PSR Sensitivity to Hyperparameters (β, γ)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
