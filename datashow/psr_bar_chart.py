import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ========== Data ==========
methods = ['SBA', 'GTA', 'UGBA', 'DPGBA', 'SPEAR',
           'GNNFingers', 'GraphGuard', 'GPrompt-G', 'Ours']

# Top-left: Edge-level Task (Cora & Cora-P)
edge_cora   = [31.0, 89.5, 92.0, 93.0, 95.0, 92.0, 93.0, 95.0, 96.0]
edge_cora_p = [3.7, 55.5, 62.4, 72.0, 75.5, 60.5, 61.0, 82.0, 88.0]

# Top-right: Graph-level Task (Cora & Cora-P)
graph_cora   = [27.0, 87.5, 90.0, 91.0, 93.1, 88.0, 89.0, 91.0, 94.0]
graph_cora_p = [2.7, 52.5, 55.4, 68.0, 72.5, 55.5, 56.0, 77.0, 86.0]

# Bottom-left: Edge-level Task (Flickr & Cora-F)
edge_flickr = [3.5, 83.0, 91.0, 94.0, 94.5, 90.5, 92.0, 92.5, 96.0]
edge_cora_f = [2.9, 44.1, 50.5, 54.5, 59.0, 44.0, 47.5, 65.5, 80.0]

# Bottom-right: Graph-level Task (Flickr & Cora-F)
graph_flickr = [3.5, 80.5, 85.0, 87.0, 89.5, 83.5, 84.0, 84.0, 91.0]
graph_cora_f = [2.5, 42.0, 48.5, 52.0, 56.5, 42.0, 45.5, 62.5, 78.0]

# ========== Style settings ==========
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Bitstream Vera Serif', 'Computer Modern Roman'],
    'axes.linewidth': 1.0,
    'hatch.linewidth': 0.8,
    'font.size': 12,
})

# Dataset styles: (color, hatch, label)
dataset_styles = {
    'Cora':   ('#6a9fd8', '///',  'Cora'),
    'Cora-P': ('#6dba6d', '\\\\', 'Cora-P'),
    'Flickr': ('#e87f7f', '...',  'Flickr'),
    'Cora-F': ('#8b7cb8', 'xx',   'Cora-F'),
}

# ========== Create figure ==========
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

subplot_configs = [
    ((0, 0), 'Edge-level Task', edge_cora, edge_cora_p, 'Cora', 'Cora-P'),
    ((0, 1), 'Graph-level Task', graph_cora, graph_cora_p, 'Cora', 'Cora-P'),
    ((1, 0), 'Edge-level Task', edge_flickr, edge_cora_f, 'Flickr', 'Cora-F'),
    ((1, 1), 'Graph-level Task', graph_flickr, graph_cora_f, 'Flickr', 'Cora-F'),
]

bar_width = 0.35
x = np.arange(len(methods))

for (row, col), title, data1, data2, key1, key2 in subplot_configs:
    ax = axes[row][col]
    color1, hatch1, _ = dataset_styles[key1]
    color2, hatch2, _ = dataset_styles[key2]

    bars1 = ax.bar(x - bar_width / 2, data1, bar_width,
                   color=color1, hatch=hatch1, edgecolor='black', linewidth=0.6,
                   label=key1, zorder=3)
    bars2 = ax.bar(x + bar_width / 2, data2, bar_width,
                   color=color2, hatch=hatch2, edgecolor='black', linewidth=0.6,
                   label=key2, zorder=3)

    # Add value labels on top of short bars (< 35) so they're clearly visible
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height < 35:
                ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                        f'{height:.0f}', ha='center', va='bottom',
                        fontsize=7.5, fontweight='bold')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=8)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha='right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.tick_params(axis='both', which='both', direction='in')

# ========== Shared legend on top ==========
legend_handles = []
for key in ['Cora', 'Cora-P', 'Flickr', 'Cora-F']:
    color, hatch, label = dataset_styles[key]
    legend_handles.append(
        mpatches.Patch(facecolor=color, hatch=hatch,
                       edgecolor='black', linewidth=0.6, label=label)
    )

fig.legend(handles=legend_handles, loc='upper center',
           ncol=4, fontsize=13, frameon=True,
           bbox_to_anchor=(0.5, 0.99),
           handlelength=2.5, handleheight=1.5,
           columnspacing=2.0)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('/Users/keyuan/PycharmProjects/PA2Perturb/datashow/psr_bar_chart.pdf',
            format='pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Done! Saved to psr_bar_chart.pdf")
