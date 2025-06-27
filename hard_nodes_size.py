import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from matplotlib.ticker import MultipleLocator

# === 数据 ===
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

plt.rcParams["font.family"] = "Times New Roman"
sns.set(style="whitegrid", font_scale=1.2)

# === 创建图结构（原尺寸） ===
fig = plt.figure(figsize=(8.2, 4.6))  # 不改动原大小
gs = fig.add_gridspec(3, 2, height_ratios=[0.4, 2, 1], hspace=0.05, wspace=0.15)

ax_legend = fig.add_subplot(gs[0, :])
ax1_top = fig.add_subplot(gs[1, 0])
ax1_bot = fig.add_subplot(gs[2, 0], sharex=ax1_top)
ax2_top = fig.add_subplot(gs[1, 1])
ax2_bot = fig.add_subplot(gs[2, 1], sharex=ax2_top)

# === 绘图函数 ===
def plot_broken(ax_top, ax_bot, data_dict, colors, markers, top_ylim, bot_ylim, ylabel=False, xlabel=False):
    for name, values in data_dict.items():
        ax_top.plot(x_values, values, label=name, color=colors[name],
                    marker=markers[name], linewidth=1.5, markersize=5)
        ax_bot.plot(x_values, values, color=colors[name],
                    marker=markers[name], linewidth=1.5, markersize=5)

    ax_top.set_ylim(top_ylim)
    ax_bot.set_ylim(bot_ylim)

    ax_bot.set_xticks(x_values)
    ax_bot.xaxis.set_minor_locator(MultipleLocator(20))

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
        ax_top.set_ylabel("VSR (%)", fontsize=10)
        # ax_bot.set_ylabel("ASR (%)", fontsize=10)
    if xlabel:
        ax_bot.set_xlabel("Size of Hard Nodes", fontsize=10)

    ax_top.grid(True, linestyle="--", alpha=0.6)
    ax_bot.grid(True, linestyle="--", alpha=0.6)

# === 左图（psr_datasets）===
plot_broken(
    ax1_top, ax1_bot,
    psr_datasets,
    colors=line_colors,
    markers=common_markers,
    top_ylim=(82, 98),
    bot_ylim=(-2, 5),
    ylabel=True,
    xlabel=True
)

# === 右图（psr_gnns）===
plot_broken(
    ax2_top, ax2_bot,
    psr_gnns,
    colors=line_colors,
    markers=common_markers,
    top_ylim=(91, 98),
    bot_ylim=(-2, 18),
    ylabel=False,
    xlabel=True
)

# === 图例置于顶部 ===
ax_legend.axis("off")
legend_handles = [ax1_top.lines[i] for i in range(len(line_colors))]
ax_legend.legend(
    handles=legend_handles,
    labels=list(line_colors.keys()),
    loc='center',
    ncol=5,
    fontsize=9,
    frameon=False,
    handlelength=2,
    handletextpad=0.8,
    columnspacing=1.2
)

# === 保存与展示（仅剪边留白）===
plt.tight_layout(pad=0.3)
plt.savefig("size-hard-nodes.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
plt.show()
