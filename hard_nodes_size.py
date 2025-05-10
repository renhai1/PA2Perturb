import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from matplotlib.ticker import MultipleLocator

# === 数据准备 ===
x_values = [40, 80, 120, 160, 200]

psr_datasets = {
    "SBA": [0, 0, 0, 0, 0],
    "GTA": [83.2, 83.2, 83.2, 83.2, 83.2],
    "UGBA": [89.6, 91.7, 93.4, 94.1, 95.7],
    "DPGBA": [89.8, 92.4, 93.9, 95.2, 95.7],
    "Our": [91.5, 93.6, 95.1, 96.3, 97.1]
}

psr_gnns = {
    "SBA": [0, 0, 0, 0, 0],
    "GTA": [0, 0, 0, 5.3, 12.4],
    "UGBA": [0, 0, 0, 6.7, 14.2],
    "DPGBA": [0, 0, 0, 9.3, 14.9],
    "Our": [91.5, 93.6, 95.1, 96.3, 97.1]
}

# === 样式设定 ===
set2_colors = sns.color_palette("Set2", 5)
color_cycle = cycle(set2_colors)
line_colors = {k: next(color_cycle) for k in psr_datasets.keys()}
common_markers = {"SBA": "o", "GTA": "s", "UGBA": "^", "DPGBA": "v", "Our": "D"}

# === 字体设置 ===
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.sans-serif"] = ["Times New Roman"]
sns.set(style="whitegrid", font_scale=1.2)

# === 通用绘图函数 ===
def plot_broken(data_dict, filename, top_ylim, bot_ylim, ylabel=True):
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(3.4, 2.6), sharex=True,
                                         gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05})

    for name, values in data_dict.items():
        ax_top.plot(x_values, values, label=name, color=line_colors[name],
                    marker=common_markers[name], linewidth=1.5, markersize=4)
        ax_bot.plot(x_values, values, color=line_colors[name],
                    marker=common_markers[name], linewidth=1.5, markersize=4)

    ax_top.set_ylim(top_ylim)
    ax_bot.set_ylim(bot_ylim)

    ax_bot.set_xticks(x_values)
    ax_bot.xaxis.set_minor_locator(MultipleLocator(20))

    # 断轴样式
    ax_top.spines['bottom'].set_visible(False)
    ax_bot.spines['top'].set_visible(False)
    ax_top.tick_params(labelbottom=False)

    d = .015
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    if ylabel:
        ax_top.set_ylabel("PSR (%)", fontsize=10, fontname="Times New Roman", labelpad=4)

    ax_bot.set_xlabel("Size of hard nodes", fontsize=10, fontweight="bold", fontname="Times New Roman")

    # 图例
    ax_top.legend(loc="upper left", ncol=2, fontsize=8, frameon=False,
                  handlelength=1.5, handletextpad=0.5, columnspacing=1)

    ax_top.grid(True, linestyle="--", alpha=0.5)
    ax_bot.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(pad=0.3)
    plt.savefig(filename, dpi=600, bbox_inches="tight")
    plt.close()

# === 输出单栏图 ===

# 图 1：不同数据集
plot_broken(
    psr_datasets,
    filename="size-node-a.png",
    top_ylim=(82, 98),
    bot_ylim=(-2, 5),
    ylabel=True
)

# 图 2：不同 GNN 模型
plot_broken(
    psr_gnns,
    filename="size-node-b.png",
    top_ylim=(91, 98),
    bot_ylim=(-2, 18),
    ylabel=True
)
