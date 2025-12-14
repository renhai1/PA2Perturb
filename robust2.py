import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data
# -----------------------------
attacks = ['Prune', 'OD', 'RIGBD', 'MAD']
baseline_bar = np.array([68, 61, 55, 63])
ours_bar = np.array([90, 88, 85, 89])

prune_ratio = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
baseline_line = np.array([92, 78, 62, 45, 30])
ours_line = np.array([96, 93, 90, 87, 82])

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# ===== Left: Bar chart =====
x = np.arange(len(attacks))
width = 0.35

axes[0].bar(x - width/2, baseline_bar, width,
            label='Baseline', edgecolor='black')
axes[0].bar(x + width/2, ours_bar, width,
            label='Ours', edgecolor='black')

axes[0].set_ylabel('S@1%')
axes[0].set_xlabel('Attack Method')
axes[0].set_xticks(x)
axes[0].set_xticklabels(attacks)
axes[0].set_ylim(0, 100)
axes[0].legend(frameon=False)
axes[0].grid(axis='y', linestyle='--', alpha=0.6)
axes[0].set_title('(a) Robustness under different attacks')

# ===== Right: Line chart =====
axes[1].plot(prune_ratio, baseline_line,
             marker='o', linewidth=2, label='Baseline')
axes[1].plot(prune_ratio, ours_line,
             marker='s', linewidth=2, label='Ours')

axes[1].set_xlabel('Prune Ratio')
axes[1].set_ylabel('S@1%')
axes[1].set_ylim(0, 100)
axes[1].legend(frameon=False)
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].set_title('(b) Performance degradation vs. attack strength')

plt.tight_layout()
plt.show()
