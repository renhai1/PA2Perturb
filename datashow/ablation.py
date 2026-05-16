import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Bitstream Vera Serif'],
    'axes.linewidth': 1.3,
    'font.size': 13,
    'axes.facecolor': '#fafafa',
    'figure.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'axes.labelcolor': '#222222',
    'hatch.linewidth': 0.8,
})

# ========== Data ==========
datasets = ['Cora', 'Citeseer', 'Pubmed', 'Flickr', 'Cora-P', 'Cora-F']
variants = [r'PA$^2$Perturb-R', r'PA$^2$Perturb-E',
            r'PA$^2$Perturb-P', r'PA$^2$Perturb']

# (a) No attack
no_attack = {
    r'PA$^2$Perturb-R': [75, 72, 72, 71, 62, 55],
    r'PA$^2$Perturb-E': [75, 72, 72, 71, 60, 52],
    r'PA$^2$Perturb-P': [89, 90, 87, 92, 75, 68],
    r'PA$^2$Perturb':   [97, 95, 94, 97, 87, 80],
}

# (b) RIGBD
rigbd = {
    r'PA$^2$Perturb-R': [23, 22, 25, 25, 18, 15],
    r'PA$^2$Perturb-E': [17, 16, 15, 10, 12,  9],
    r'PA$^2$Perturb-P': [88, 90, 85, 88, 65, 58],
    r'PA$^2$Perturb':   [91, 95, 91, 89, 73, 70],
}

# ========== Style ==========
variant_styles = {
    r'PA$^2$Perturb-R': {'color': '#e8b4b8', 'hatch': '///'},
    r'PA$^2$Perturb-E': {'color': '#f0c87c', 'hatch': 'xxx'},
    r'PA$^2$Perturb-P': {'color': '#8fbc8f', 'hatch': '\\\\'},
    r'PA$^2$Perturb':   {'color': '#7fb3d8', 'hatch': '+++'},
}

# ========== Plot ==========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

bar_width = 0.18
x = np.arange(len(datasets))

for ax, data_dict, subtitle in [(ax1, no_attack, '(a) No attack'),
                                  (ax2, rigbd, '(b) RIGBD')]:
    for i, variant in enumerate(variants):
        style = variant_styles[variant]
        values = data_dict[variant]
        offset = (i - 1.5) * bar_width
        ax.bar(x + offset, values, bar_width,
               color=style['color'], hatch=style['hatch'],
               edgecolor='black', linewidth=0.6,
               zorder=3)

    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11.5)
    ax.set_ylabel('Accuracy (%)', fontsize=13, labelpad=6)
    ax.set_xlabel(subtitle, fontsize=14, fontweight='bold', labelpad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='#aaaaaa', linewidth=0.7, zorder=0)
    ax.tick_params(axis='both', direction='in', length=4, width=1.0, labelsize=11)

    for spine in ax.spines.values():
        spine.set_linewidth(1.3)

# ========== Shared legend on top ==========
legend_handles = []
for variant in variants:
    style = variant_styles[variant]
    legend_handles.append(
        mpatches.Patch(facecolor=style['color'], hatch=style['hatch'],
                       edgecolor='black', linewidth=0.6, label=variant)
    )

fig.legend(handles=legend_handles, loc='upper center',
           ncol=4, fontsize=12, frameon=True,
           bbox_to_anchor=(0.5, 1.01),
           handlelength=2.5, handleheight=1.5,
           columnspacing=2.0, borderpad=0.6,
           edgecolor='#888888', fancybox=True, framealpha=0.95)

plt.tight_layout(rect=[0, 0, 1, 0.92], w_pad=3.0)
plt.savefig('/Users/keyuan/PycharmProjects/PA2Perturb/datashow/ablation_bar.pdf',
            format='pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Done! Saved to ablation_bar.pdf")
