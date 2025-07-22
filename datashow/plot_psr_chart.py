import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np

# ========== 1. 构造数据表 ==========
data = {
    "Defense Method": ["None", "Prue", "OD", "RIGBD"] * 4,
    "Dataset": ["Cora"] * 4 + ["Citeseer"] * 4 + ["Pubmed"] * 4 + ["Flickr"] * 4,
    "SBA-Samp": [33.2, 16.8, 34.7, 11.7, 29.4, 14.2, 29.3, 30.5, 30.2, 27.4, 30.5, 0.0, 0.0, 0.0, 0.0, 0.0],
    "SBA-Gen": [44.2, 19.6, 44.3, 0.0, 40.1, 17.6, 40.7, 0.0, 31.2, 29.7, 31.9, 0.0, 0.0, 0.0, 0.0, 0.0],
    "GTA": [91.4, 15.2, 0.0, 0.0, 85.7, 9.1, 0.0, 0.0, 89.1, 11.4, 0.0, 0.0, 83.2, 0.0, 0.0, 0.0],
    "UGBA": [96.0, 97.2, 93.2, 27.5, 93.2, 91.4, 91.2, 94.3, 92.6, 92.9, 90.7, 0.0, 91.7, 89.3, 94.8, 0.0],
    "DPGBA": [96.7, 21.3, 91.4, 0.0, 94.4, 91.2, 91.6, 93.2, 93.7, 19.7, 89.2, 0.0, 92.1, 87.4, 93.7, 0.0],
    "GPrompt-G": [95.9, 91.3, 91.1, 89.3, 94.2, 91.6, 92.7, 94.8, 91.3, 90.7, 89.7, 89.1, 91.8, 94.3, 93.7, 87.3],
    "Ours": [96.4, 98.2, 96.3, 91.2, 95.3, 94.6, 94.5, 95.8, 92.8, 94.2, 91.1, 91.1, 94.7, 94.3, 95.1, 89.5]
}
df = pd.DataFrame(data)

# ========== 2. 基本设置 ==========
sns.set(style="whitegrid", font_scale=1.3)
plt.rcParams["hatch.linewidth"] = 1.2
plt.rcParams["axes.linewidth"] = 1.2

datasets = ["Cora", "Citeseer", "Pubmed", "Flickr"]
defenses = ["None", "Prue", "OD", "RIGBD"]
methods = ["SBA-Samp", "SBA-Gen", "GTA", "UGBA", "DPGBA", "GPrompt-G", "Ours"]

# 颜色与纹理样式
final_method_styles = {
    "SBA-Samp": ("#f4cccc", "///"),
    "SBA-Gen": ("#cfe2f3", "\\\\"),
    "GTA": ("#d9ead3", "xx"),
    "UGBA": ("#f6b26b", "xxx"),
    "DPGBA": ("#b6d7a8", "..."),
    "GPrompt-G": ("#a4c2f4", "++"),
    "Ours": ("#76b7b2", "//")
}

# 避免 PSR=0 的柱状图无法显示
epsilon = 0.01

# ========== 3. 绘制图形 ==========
fig, axes = plt.subplots(2, 2, figsize=(20, 12))

for ax, dataset in zip(axes.flat, datasets):
    subset = df[df["Dataset"] == dataset]
    bar_width = 0.1
    x = np.arange(len(defenses))

    for i, method in enumerate(methods):
        color, hatch = final_method_styles[method]
        values = subset[method].apply(lambda v: max(v / 100.0, epsilon))
        bars = ax.bar(x + i * bar_width, values, width=bar_width,
                      label=method if dataset == "Cora" else "", color=color,
                      hatch=hatch, edgecolor="black", linewidth=1)

        # 添加数值标注
        for j, v in enumerate(values):
            if subset[method].iloc[j] == 0.0:
                ax.text(x[j] + i * bar_width, epsilon + 0.01, "0%",
                        ha='center', va='bottom', fontsize=9, rotation=90)


    ax.set_xticks(x + bar_width * (len(methods) / 2))
    ax.set_xticklabels(defenses)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("VSR (%)")
    ax.set_title(f"{dataset}", fontsize=14)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.set_yticklabels([f"{int(tick*100)}%" for tick in np.linspace(0, 1.0, 6)])
    ax.grid(axis="y", linestyle="--", alpha=0.6)

# ========== 4. 添加图例（贴近图形） ==========
legend_handles = [
    mpatches.Patch(facecolor=final_method_styles[m][0],
                   hatch=final_method_styles[m][1],
                   edgecolor="black", label=m)
    for m in methods
]
fig.legend(handles=legend_handles, loc='upper center', ncol=4, fontsize=12, bbox_to_anchor=(0.5, 0.95))

# 标题也稍微下移一点，防止重叠
plt.suptitle("VSR(%) Comparison Across Datasets, Defense Methods, and Watermarking Techniques", fontsize=16, y=1.015)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
