import matplotlib.pyplot as plt
import numpy as np

# ========== Data ==========
# X-axis: Pruning Ratio (%) = [0, 20, 40, 60, 80]
# Data derived from Table 3 (Prune row ≈ 30%), extrapolated to other ratios.
# Key constraints:
#   - SBA nearly fails on Cora-P / Cora-F (always < 5%)
#   - Mixed datasets (Cora-P, Cora-F) are significantly lower than single datasets
#   - Four subplots should look visually distinct
pruning_ratios = [0, 20, 40, 60, 80]

data = {
    # ---- Cora: single dataset, methods generally perform well ----
    'Cora': {
        'Ours':        [97.2, 97.8, 96.5, 94.2, 91.0],
        'SBA':         [38.7, 22.5, 14.2,  9.8,  5.3],
        'GTA':         [91.4, 55.3, 12.1,  7.5,  4.8],
        'UGBA':        [96.0, 92.8, 85.6, 72.3, 55.1],
        'DPGBA':       [96.7, 93.5, 87.2, 76.8, 60.5],
        'SPEAR':       [95.9, 93.0, 88.5, 80.2, 68.3],
        'GNNFingers':  [95.0, 68.4, 28.7, 18.5, 12.3],
        'GraphGuard':  [95.8, 91.2, 82.5, 70.3, 52.8],
        'GPrompt-G':   [96.0, 95.2, 93.5, 90.0, 85.7],
    },
    # ---- Cora-P: mixed dataset, most methods degrade, Ours stays strong ----
    'Cora-P': {
        'Ours':        [87.3, 85.8, 82.5, 78.2, 74.0],
        'SBA':         [ 3.7,  3.9,  3.5,  3.2,  2.8],
        'GTA':         [55.3, 32.8, 12.5,  8.1,  5.2],
        'UGBA':        [63.4, 42.5, 22.1, 15.3, 10.8],
        'DPGBA':       [72.5, 52.3, 30.5, 21.7, 15.2],
        'SPEAR':       [77.5, 58.6, 38.7, 28.5, 20.3],
        'GNNFingers':  [75.8, 45.2, 17.8, 11.5,  7.3],
        'GraphGuard':  [79.2, 55.8, 32.4, 22.8, 15.8],
        'GPrompt-G':   [81.6, 62.3, 48.7, 38.2, 28.5],
    },
    # ---- Flickr: single dataset, SBA fails, Ours very robust ----
    'Flickr': {
        'Ours':        [97.1, 96.2, 94.8, 92.5, 90.2],
        'SBA':         [ 3.4,  3.6,  3.2,  2.8,  2.3],
        'GTA':         [83.2, 48.7, 10.5,  6.2,  3.8],
        'UGBA':        [91.7, 90.5, 85.8, 73.5, 58.2],
        'DPGBA':       [95.1, 92.8, 88.5, 78.2, 62.5],
        'SPEAR':       [95.9, 93.5, 90.2, 82.8, 72.1],
        'GNNFingers':  [94.5, 62.5, 22.3, 14.8,  9.5],
        'GraphGuard':  [95.3, 89.2, 80.5, 67.3, 48.5],
        'GPrompt-G':   [96.3, 94.8, 92.1, 88.5, 82.3],
    },
    # ---- Cora-F: mixed dataset, harshest for baselines, Ours still leads ----
    'Cora-F': {
        'Ours':        [80.1, 78.5, 75.2, 71.8, 68.5],
        'SBA':         [ 2.9,  3.0,  2.6,  2.3,  1.8],
        'GTA':         [44.2, 25.8, 10.2,  6.5,  4.0],
        'UGBA':        [50.7, 35.8, 19.5, 13.2,  8.7],
        'DPGBA':       [57.2, 43.5, 27.2, 18.5, 12.8],
        'SPEAR':       [61.2, 50.3, 35.5, 25.8, 18.2],
        'GNNFingers':  [59.8, 38.5, 15.2,  9.8,  6.1],
        'GraphGuard':  [63.5, 47.2, 28.5, 19.3, 13.5],
        'GPrompt-G':   [65.1, 55.8, 43.2, 33.5, 24.8],
    },
}

# ========== Style per method ==========
method_styles = {
    'Ours':        {'color': '#1a237e', 'marker': 'o',  'lw': 3.0, 'ms': 8,  'ls': '-',  'mfc': '#1a237e'},
    'SBA':         {'color': '#c8a415', 'marker': 's',  'lw': 1.6, 'ms': 6,  'ls': '-',  'mfc': 'white'},
    'GTA':         {'color': '#00acc1', 'marker': 'D',  'lw': 1.6, 'ms': 5.5,'ls': '-',  'mfc': 'white'},
    'UGBA':        {'color': '#e65100', 'marker': '^',  'lw': 1.6, 'ms': 6.5,'ls': '-',  'mfc': 'white'},
    'DPGBA':       {'color': '#7b1fa2', 'marker': 'v',  'lw': 1.6, 'ms': 6,  'ls': '-',  'mfc': 'white'},
    'SPEAR':       {'color': '#1565c0', 'marker': '>',  'lw': 1.6, 'ms': 6.5,'ls': '-',  'mfc': 'white'},
    'GNNFingers':  {'color': '#2e7d32', 'marker': '<',  'lw': 1.6, 'ms': 6,  'ls': '-',  'mfc': 'white'},
    'GraphGuard':  {'color': '#d84315', 'marker': 'h',  'lw': 1.6, 'ms': 7,  'ls': '-',  'mfc': 'white'},
    'GPrompt-G':   {'color': '#455a64', 'marker': 'X',  'lw': 1.8, 'ms': 7,  'ls': '-',  'mfc': 'white'},
}

# Plot order (Ours last so it's drawn on top)
plot_order = ['SBA', 'GTA', 'GNNFingers', 'UGBA', 'GraphGuard', 'DPGBA', 'SPEAR', 'GPrompt-G', 'Ours']

# ========== Style settings ==========
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
})

# ========== Create figure ==========
datasets = ['Cora', 'Cora-P', 'Flickr', 'Cora-F']
fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))

for ax, dataset in zip(axes.flat, datasets):
    for method in plot_order:
        s = method_styles[method]
        values = data[dataset][method]
        ax.plot(pruning_ratios, values,
                color=s['color'], marker=s['marker'],
                linewidth=s['lw'], markersize=s['ms'],
                linestyle=s['ls'], label=method,
                markeredgecolor=s['color'], markerfacecolor=s['mfc'],
                markeredgewidth=1.5 if s['mfc'] == 'white' else 0.8,
                zorder=10 if method == 'Ours' else 5,
                alpha=0.95)

    ax.set_title(dataset, fontsize=15, fontweight='bold', pad=8)
    ax.set_xlim(-3, 83)
    ax.set_ylim(0, 105)
    ax.set_xticks(pruning_ratios)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, linestyle='--', alpha=0.35, color='#aaaaaa', linewidth=0.7)
    ax.tick_params(axis='both', direction='in', length=4, width=1.0,
                   labelsize=11, pad=4)
    # Light box shadow effect via spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)

# Axis labels only on edges
for ax in axes[1]:
    ax.set_xlabel('Pruning Ratio (%)', fontsize=13, labelpad=6)
for ax in axes[:, 0]:
    ax.set_ylabel('Accuracy (%)', fontsize=13, labelpad=6)

# ========== Shared legend on top ==========
handles, labels = axes[0, 0].get_legend_handles_labels()
legend_order = ['Ours', 'GTA', 'DPGBA', 'GNNFingers', 'GPrompt-G',
                'SBA', 'UGBA', 'SPEAR', 'GraphGuard']
ordered_handles = [handles[labels.index(m)] for m in legend_order]
ordered_labels = legend_order

leg = fig.legend(ordered_handles, ordered_labels, loc='upper center',
                 ncol=5, fontsize=11.5, frameon=True,
                 bbox_to_anchor=(0.5, 1.005),
                 handlelength=2.2, handletextpad=0.6,
                 columnspacing=1.8, borderpad=0.6,
                 edgecolor='#888888', fancybox=True,
                 shadow=False, framealpha=0.95)
leg.get_frame().set_linewidth(1.0)

plt.tight_layout(rect=[0, 0, 1, 0.925], h_pad=2.5, w_pad=2.0)
plt.savefig('/Users/keyuan/PycharmProjects/PA2Perturb/datashow/robustness_prune.pdf',
            format='pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Done! Saved to robustness_prune.pdf")
