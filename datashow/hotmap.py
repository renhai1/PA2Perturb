import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ========== 全局字体设置 ==========
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "serif"]
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["pdf.fonttype"] = 42

# ========== 数据（从原图提取）==========
beta_labels = ["0.01", "0.1", "1", "10", "100"]
gamma_labels = ["0.01", "0.1", "1", "10", "100"]

# (a) No attack: 行=β(从上到下: 0.01→100), 列=γ(从左到右: 0.01→100)
data_no_attack = np.array([
    [93.5, 93.1, 92.8, 91.3, 90.2],
    [94.1, 93.8, 93.5, 92.7, 92.1],
    [95.4, 95.1, 94.3, 93.1, 92.8],
    [96.9, 96.3, 95.9, 95.2, 94.7],
    [97.3, 96.9, 96.5, 96.1, 95.9],
])

# (b) RIGBD
data_rigbd = np.array([
    [89.5, 89.7, 90.2, 89.6, 89.5],
    [91.4, 91.9, 92.3, 92.7, 92.5],
    [92.6, 93.1, 93.4, 93.2, 92.9],
    [91.7, 92.8, 93.5, 94.7, 94.5],
    [88.3, 90.2, 93.7, 95.4, 94.6],
])

# ========== 绘图 ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
fig.subplots_adjust(wspace=0.08, left=0.08, right=0.88, top=0.88, bottom=0.18)

# 统一色标范围
vmin = 88
vmax = 98

# 自定义colormap: 蓝(低) -> 白(中) -> 红(高)
cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_bwr",
    ["#2166AC", "#67A9CF", "#D1E5F0", "#FAFAFA",
     "#FDDBC7", "#EF8A62", "#B2182B"],
    N=256
)

panels = [
    (data_no_attack, "(a) No attack"),
    (data_rigbd, "(b) RIGBD"),
]

for ax_idx, (ax, (data, subtitle)) in enumerate(zip(axes, panels)):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')

    # 标注数值
    for i in range(len(beta_labels)):
        for j in range(len(gamma_labels)):
            val = data[i, j]
            # 根据背景亮度选择文字颜色
            norm_val = (val - vmin) / (vmax - vmin)
            text_color = 'white' if norm_val > 0.75 or norm_val < 0.2 else '#222222'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=11.5, fontweight='bold', color=text_color)

    # 坐标轴
    ax.set_xticks(range(len(gamma_labels)))
    ax.set_xticklabels(gamma_labels, fontsize=11)
    ax.set_yticks(range(len(beta_labels)))
    ax.set_yticklabels(beta_labels if ax_idx == 0 else [], fontsize=11)

    # 轴标签
    ax.set_xlabel(r'$\gamma$', fontsize=14, labelpad=4)
    if ax_idx == 0:
        ax.set_ylabel(r'$\beta$', fontsize=14, labelpad=4)

    # 子图标题
    ax.annotate(subtitle, xy=(0.5, 0), xytext=(0, -38),
                textcoords='offset points', xycoords='axes fraction',
                fontsize=13, fontweight='bold', ha='center', va='top',
                color='#222222')

    # 网格线（在格子边界上）
    ax.set_xticks([x - 0.5 for x in range(1, len(gamma_labels))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(beta_labels))], minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', length=0)
    ax.tick_params(which='major', length=0)

    # 外框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#888888')
        spine.set_linewidth(1.2)

# ========== 统一 colorbar ==========
cbar_ax = fig.add_axes([0.90, 0.18, 0.02, 0.70])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("VSR (%)", fontsize=12, labelpad=8)
cbar.ax.tick_params(labelsize=10)
cbar.outline.set_linewidth(0.8)
cbar.outline.set_color('#888888')

# ========== 保存 ==========
output_path = "/Users/keyuan/PycharmProjects/PA2Perturb/datashow/hyperparam_heatmap.pdf"
plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
plt.close()
print(f"Heatmap saved to: {output_path}")
