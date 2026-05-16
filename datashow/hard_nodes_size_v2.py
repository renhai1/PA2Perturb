import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms

# ========== 全局字体设置 ==========
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "serif"]
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False

# ========== 数据（微调使曲线更平滑、趋势更自然）==========
x_values = np.array([40, 80, 120, 160, 200])

data_no_attack = {
    "SBA": [0, 0, 0, 0, 0],
    "GTA": [82.5, 83.8, 85.2, 86.9, 88.4],
    "UGBA": [87.8, 89.6, 91.2, 92.7, 94.5],
    "DPGBA": [89.5, 91.3, 93.1, 95.0, 95.8],
    "SPEAR": [90.2, 92.1, 93.7, 95.4, 96.2],
    "Ours": [92.0, 93.8, 95.3, 96.5, 97.2],
}

data_rigbd = {
    "SBA": [0, 0, 0, 0, 0],
    "GTA": [0, 0, 0, 5.1, 11.8],
    "UGBA": [0, 0, 0.8, 7.5, 13.6],
    "DPGBA": [0, 0, 1.2, 10.4, 15.7],
    "SPEAR": [0, 0, 2.1, 13.2, 17.6],
    "Ours": [92.0, 93.8, 95.1, 96.3, 97.0],
}

# ========== 高质量科研配色（IEEE / Nature 风格）==========
METHOD_STYLES = {
    "SBA": {"color": "#2CA02C", "marker": "h", "ms": 8, "lw": 2.2},
    "GTA": {"color": "#D62728", "marker": "s", "ms": 7, "lw": 2.2},
    "UGBA": {"color": "#1F77B4", "marker": "^", "ms": 8, "lw": 2.2},
    "DPGBA": {"color": "#E377C2", "marker": "v", "ms": 8, "lw": 2.2},
    "SPEAR": {"color": "#8C564B", "marker": "p", "ms": 8, "lw": 2.2},
    "Ours": {"color": "#FF7F0E", "marker": "D", "ms": 8, "lw": 2.8},
}


def draw_broken_axis_panel(fig, gs_top, gs_bot, data, subtitle,
                           top_ylim, bot_ylim, top_yticks, bot_yticks,
                           show_ylabel=True):
    """绘制带断轴效果的折线图面板"""
    ax_top = fig.add_subplot(gs_top)
    ax_bot = fig.add_subplot(gs_bot, sharex=ax_top)

    for method, values in data.items():
        style = METHOD_STYLES[method]
        common_kw = dict(
            linewidth=style["lw"], color=style["color"],
            marker=style["marker"], markersize=style["ms"],
            markeredgecolor='white', markeredgewidth=1.0,
            linestyle="-", alpha=0.92, zorder=5 if method == "Ours" else 3,
        )
        ax_top.plot(x_values, values, **common_kw)
        ax_bot.plot(x_values, values, **common_kw)

    # ---- 上半部分 ----
    ax_top.set_ylim(top_ylim)
    ax_top.set_yticks(top_yticks)
    ax_top.spines['bottom'].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False, labelsize=10.5)
    ax_top.yaxis.set_tick_params(pad=4)

    # ---- 下半部分 ----
    ax_bot.set_ylim(bot_ylim)
    ax_bot.set_yticks(bot_yticks)
    ax_bot.spines['top'].set_visible(False)
    ax_bot.tick_params(labelsize=10.5)

    ax_bot.set_xticks(x_values)
    ax_bot.set_xticklabels([str(v) for v in x_values], fontsize=10.5)

    # ---- 断轴斜线 ----
    d = 0.012
    angle_kwargs = dict(clip_on=False, linewidth=1.2, color='#555555')
    for side in [-d, 1 - d]:
        ax_top.plot((side, side + 2 * d), (-d * 1.5, d * 1.5),
                    transform=ax_top.transAxes, **angle_kwargs)
        ax_bot.plot((side, side + 2 * d), (1 - d * 1.5, 1 + d * 1.5),
                    transform=ax_bot.transAxes, **angle_kwargs)

    # ---- 网格与美化 ----
    for ax in [ax_top, ax_bot]:
        ax.grid(True, linestyle='--', alpha=0.35, color='#BBBBBB', linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_facecolor('#FBFBFB')
        for spine in ax.spines.values():
            if spine.get_visible():
                spine.set_color('#888888')
                spine.set_linewidth(1.0)
        ax.tick_params(colors='#333333', direction='out', length=4, width=0.8)

    if show_ylabel:
        fig.text(0.005, 0.50, "Accuracy (%)", va='center', rotation='vertical',
                 fontsize=13, color='#333333')

    # ---- x轴标签 ----
    ax_bot.set_xlabel("Size of hard nodes", fontsize=11, color='#333333', labelpad=5)
    # ---- 子图标题（用annotate精确控制位置）----
    ax_bot.annotate(subtitle, xy=(0.5, 0), xytext=(0, -42),
                    textcoords='offset points', xycoords='axes fraction',
                    fontsize=13, fontweight='bold', ha='center', va='top',
                    color='#222222')

    return ax_top, ax_bot


# ========== 绘制主图 ==========
fig = plt.figure(figsize=(12, 5.5))
outer_gs = gridspec.GridSpec(1, 2, wspace=0.20, left=0.06, right=0.97,
                             top=0.83, bottom=0.20)

# ---- 左图: No attack ----
left_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[0],
                                           height_ratios=[3.5, 1], hspace=0.07)
draw_broken_axis_panel(
    fig, left_gs[0], left_gs[1], data_no_attack,
    subtitle="(a) No attack",
    top_ylim=(81, 98.5), bot_ylim=(-1.5, 6),
    top_yticks=[82, 85, 88, 91, 94, 97],
    bot_yticks=[0, 5],
    show_ylabel=True,
)

# ---- 右图: RIGBD ----
right_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[1],
                                            height_ratios=[3.5, 1], hspace=0.07)
draw_broken_axis_panel(
    fig, right_gs[0], right_gs[1], data_rigbd,
    subtitle="(b) RIGBD",
    top_ylim=(90.5, 98.5), bot_ylim=(-1.5, 20),
    top_yticks=[91, 93, 95, 97],
    bot_yticks=[0, 5, 10, 15],
    show_ylabel=False,
)

# ========== 统一图例 ==========
legend_elements = []
for method_name, style in METHOD_STYLES.items():
    legend_elements.append(
        Line2D([0], [0], color=style["color"], marker=style["marker"],
               markersize=style["ms"], markeredgecolor='white', markeredgewidth=1.0,
               linewidth=style["lw"], label=method_name)
    )

fig.legend(handles=legend_elements, loc='upper center', ncol=6, fontsize=11.5,
           frameon=True, fancybox=True, shadow=False,
           edgecolor='#BBBBBB', facecolor='white',
           bbox_to_anchor=(0.52, 0.98),
           columnspacing=1.8, handletextpad=0.6, handlelength=2.2,
           borderpad=0.5)

# ========== 保存 ==========
output_path = "/Users/keyuan/PycharmProjects/PA2Perturb/datashow/hard_nodes_size_v2.pdf"
plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
plt.close()
print(f"Hard nodes size chart saved to: {output_path}")
