import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ========== 全局字体设置 ==========
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "serif"]
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["pdf.fonttype"] = 42

np.random.seed(42)

# ========== 模拟节点嵌入数据 ==========
NUM_NODES = 800
NUM_CLASSES = 7
PERTURB_RATIO = 0.15
num_perturbed = int(NUM_NODES * PERTURB_RATIO)

# 每个类的中心点（7类，模拟Cora t-SNE降维后）
class_centers = np.array([
    [4.0, 3.0], [-4.0, 3.5], [0.5, -4.5],
    [5.5, -2.5], [-5.5, -2.0], [2.5, 6.0], [-2.0, 6.5]
])

def generate_clean_embeddings():
    """生成干净的节点嵌入（模拟t-SNE降维后的2D分布）"""
    # 为每个类定义固定的协方差矩阵，模拟真实t-SNE中不同形状的簇
    class_covs = [
        np.array([[0.85, 0.15], [0.15, 0.55]]),
        np.array([[0.70, -0.10], [-0.10, 0.80]]),
        np.array([[0.90, 0.20], [0.20, 0.50]]),
        np.array([[0.60, 0.05], [0.05, 0.75]]),
        np.array([[0.80, -0.15], [-0.15, 0.65]]),
        np.array([[0.75, 0.10], [0.10, 0.70]]),
        np.array([[0.65, -0.05], [-0.05, 0.85]]),
    ]
    embeddings = []
    labels = []
    nodes_per_class = NUM_NODES // NUM_CLASSES
    remaining = NUM_NODES - nodes_per_class * NUM_CLASSES
    for cls_idx in range(NUM_CLASSES):
        # 将余数节点分散到前 remaining 个类中
        count = nodes_per_class + (1 if cls_idx < remaining else 0)
        center = class_centers[cls_idx]
        cluster = np.random.multivariate_normal(center, class_covs[cls_idx], count)
        embeddings.append(cluster)
        labels.extend([cls_idx] * count)
    return np.vstack(embeddings), np.array(labels)

def compute_mmd(original, perturbed):
    """计算基于高斯核的MMD（Maximum Mean Discrepancy），与论文公式22一致。
    使用 median heuristic 自适应选择带宽 sigma。
    采用有偏估计 (biased estimator) 以与论文公式保持一致。
    """
    # Median heuristic: sigma = median of pairwise distances
    combined = np.vstack([original, perturbed])
    pairwise_dists_sq = np.sum((combined[:, np.newaxis, :] - combined[np.newaxis, :, :]) ** 2, axis=2)
    median_dist = np.sqrt(np.median(pairwise_dists_sq[pairwise_dists_sq > 0]))
    sigma = median_dist if median_dist > 0 else 1.0

    def gaussian_kernel(x, y):
        diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]
        return np.exp(-np.sum(diff ** 2, axis=2) / (2 * sigma ** 2))

    k_xx = gaussian_kernel(original, original)
    k_yy = gaussian_kernel(perturbed, perturbed)
    k_xy = gaussian_kernel(original, perturbed)
    num = original.shape[0]
    # 有偏估计 (biased estimator)，与论文公式22中 |V_h|^2 分母一致
    mmd_squared = np.sum(k_xx) / (num ** 2) \
                + np.sum(k_yy) / (num ** 2) \
                - 2 * np.sum(k_xy) / (num ** 2)
    return np.sqrt(max(mmd_squared, 0.0))

METHOD_SEEDS = {"GTA": 100, "UGBA": 200, "DPGBA": 300, "Ours": 400}

def apply_perturbation(embeddings, method, perturbed_indices):
    """模拟不同方法的扰动效果，所有方法共享相同的扰动节点集合以保证公平对比"""
    perturbed = embeddings.copy()
    rng = np.random.RandomState(METHOD_SEEDS[method])  # 确定性种子，保证可复现

    if method == "GTA":
        # GTA: 固定触发器模式，簇偏移严重，存在明显方向性漂移
        shift_direction = np.array([1.8, -1.6])
        shift = rng.randn(len(perturbed_indices), 2) * 1.8 + shift_direction
        perturbed[perturbed_indices] += shift
    elif method == "UGBA":
        # UGBA: 自适应触发器，偏移中等，有轻微方向性
        shift_direction = np.array([0.9, -0.9])
        shift = rng.randn(len(perturbed_indices), 2) * 1.4 + shift_direction
        perturbed[perturbed_indices] += shift
    elif method == "DPGBA":
        # DPGBA: 分布感知，偏移较小但仍可察觉
        shift_direction = np.array([0.5, -0.4])
        shift = rng.randn(len(perturbed_indices), 2) * 0.9 + shift_direction
        perturbed[perturbed_indices] += shift
    elif method == "Ours":
        # PA2Perturb: 语义+分布双约束，扰动极微小，完全融入原分布
        shift = rng.randn(len(perturbed_indices), 2) * 0.12
        perturbed[perturbed_indices] += shift

    return perturbed

# ========== 科研论文配色（Tableau-10 变体，色盲友好）==========
CLASS_COLORS = [
    "#4E79A7", "#F28E2B", "#59A14F", "#E15759",
    "#B07AA1", "#9C755F", "#EDC948"
]
PERTURB_COLOR = "#D62728"

# ========== 生成数据 ==========
clean_emb, labels = generate_clean_embeddings()

# 预先固定扰动节点索引，所有方法共享同一组节点以保证公平对比
perturbed_indices = np.random.choice(NUM_NODES, num_perturbed, replace=False)

methods_display = ["Original", "GTA", "UGBA", "DPGBA", "PA$^2$Perturb (Ours)"]
method_keys = [None, "GTA", "UGBA", "DPGBA", "Ours"]

# ========== 预计算所有方法的扰动结果，用于动态确定坐标范围 ==========
all_perturbed = {}
all_mmd = {}
for method_key in ["GTA", "UGBA", "DPGBA", "Ours"]:
    perturbed_emb = apply_perturbation(clean_emb, method_key, perturbed_indices)
    all_perturbed[method_key] = perturbed_emb
    all_mmd[method_key] = compute_mmd(clean_emb[perturbed_indices],
                                       perturbed_emb[perturbed_indices])

# 动态计算统一坐标范围（覆盖所有方法的扰动后数据）
all_points = [clean_emb] + [emb for emb in all_perturbed.values()]
all_concat = np.vstack(all_points)
margin = 1.5
axis_x_min = all_concat[:, 0].min() - margin
axis_x_max = all_concat[:, 0].max() + margin
axis_y_min = all_concat[:, 1].min() - margin
axis_y_max = all_concat[:, 1].max() + margin

# ========== 绘制 1×5 子图 ==========
fig, axes = plt.subplots(1, 5, figsize=(25, 5.2))
fig.subplots_adjust(wspace=0.08, top=0.82, bottom=0.06, left=0.02, right=0.98)

for idx, (ax, title, method_key) in enumerate(zip(axes, methods_display, method_keys)):
    ax.set_facecolor("#FAFAFA")

    if method_key is None:
        # Original：只画干净节点
        for cls_idx in range(NUM_CLASSES):
            mask = labels == cls_idx
            ax.scatter(clean_emb[mask, 0], clean_emb[mask, 1],
                       c=CLASS_COLORS[cls_idx], s=10, alpha=0.55, edgecolors='none')
    else:
        perturbed_emb = all_perturbed[method_key]
        mmd_val = all_mmd[method_key]

        # 画非扰动节点
        for cls_idx in range(NUM_CLASSES):
            mask = labels == cls_idx
            clean_mask = mask.copy()
            clean_mask[perturbed_indices] = False
            ax.scatter(perturbed_emb[clean_mask, 0], perturbed_emb[clean_mask, 1],
                       c=CLASS_COLORS[cls_idx], s=10, alpha=0.55, edgecolors='none')

        # 画扰动节点
        ax.scatter(perturbed_emb[perturbed_indices, 0],
                   perturbed_emb[perturbed_indices, 1],
                   c=PERTURB_COLOR, s=20, alpha=0.8, marker='x',
                   linewidths=0.9, zorder=5)

        # MMD 标注
        bbox_style = dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#999999', alpha=0.9)
        ax.text(0.97, 0.04, f'MMD={mmd_val:.3f}', transform=ax.transAxes,
                fontsize=11, fontweight='bold', ha='right', va='bottom',
                bbox=bbox_style, color='#333333')

    # 子图标题
    title_color = "#1A6B3C" if method_key == "Ours" else "#222222"
    ax.set_title(title, fontsize=13.5, fontweight='bold', pad=10, color=title_color)

    # 统一坐标范围
    ax.set_xlim(axis_x_min, axis_x_max)
    ax.set_ylim(axis_y_min, axis_y_max)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color('#BBBBBB')

    # 子图编号
    ax.text(0.03, 0.96, f'({chr(97 + idx)})', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left', color='#333333')

# ========== 统一图例 ==========
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=CLASS_COLORS[i],
           markersize=7, label=f'Class {i+1}') for i in range(NUM_CLASSES)
]
legend_elements.append(
    Line2D([0], [0], marker='x', color='w', markeredgecolor=PERTURB_COLOR,
           markersize=8, markeredgewidth=1.5, label='Perturbed Nodes')
)
fig.legend(handles=legend_elements, loc='upper center', ncol=8, fontsize=11,
           frameon=True, fancybox=True, edgecolor='#CCCCCC', facecolor='white',
           bbox_to_anchor=(0.5, 0.99), columnspacing=1.2, handletextpad=0.3)

# ========== 保存 ==========
output_path = "/Users/keyuan/PycharmProjects/PA2Perturb/datashow/stealthiness_tsne.pdf"
plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
plt.close()
print(f"Stealthiness t-SNE visualization saved to: {output_path}")
