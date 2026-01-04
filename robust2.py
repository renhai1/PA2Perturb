import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Global Style (Paper-ready)
# ===============================
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.1)
plt.rcParams["font.family"] = "serif"

# ===============================
# X-axis: Pruning ratio
# ===============================
prune_ratio = np.array([0, 20, 40, 60, 80])

# ===============================
# Simulated data (S@1%)
# ===============================
# -------- Cora --------
gta_cora   = [55, 45, 32, 22, 15]
ugba_cora  = [70, 60, 45, 35, 25]
dpgba_cora = [80, 68, 50, 38, 28]
spear_cora = [88, 78, 62, 50, 40]
ours_cora  = [96, 94, 90, 87, 84]

# -------- Reddit --------
gta_reddit   = [58, 48, 35, 25, 18]
ugba_reddit  = [72, 63, 48, 36, 27]
dpgba_reddit = [82, 72, 55, 42, 32]
spear_reddit = [90, 82, 67, 55, 45]
ours_reddit  = [97, 95, 92, 89, 86]

# ===============================
# Figure Layout (Two-column width)
# ===============================
fig, axes = plt.subplots(
    1, 2,
    figsize=(7.2, 3.2),   # paper-friendly
    dpi=300
)

# ===============================
# (a) Cora
# ===============================
ax = axes[0]
ax.plot(prune_ratio, gta_cora,   marker='s', markersize=5, linewidth=2.0, label='GTA')
ax.plot(prune_ratio, ugba_cora,  marker='o', markersize=5, linewidth=2.0, label='UGBA')
ax.plot(prune_ratio, dpgba_cora, marker='^', markersize=5, linewidth=2.0, label='DPGBA')
ax.plot(prune_ratio, spear_cora, marker='D', markersize=5, linewidth=2.0, label='SPEAR')
ax.plot(prune_ratio, ours_cora,  marker='v', markersize=5, linewidth=2.8, label='Ours')

ax.set_title('(a) Cora', fontweight='bold')
ax.set_xlabel('Pruning Ratio (%)')
ax.set_ylabel('S@1 (%)')
ax.set_ylim(10, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.grid(axis='y', linestyle='--', alpha=0.35)
ax.legend(frameon=False, fontsize=9)

# ===============================
# (b) Reddit
# ===============================
ax = axes[1]
ax.plot(prune_ratio, gta_reddit,   marker='s', markersize=5, linewidth=2.0, label='GTA')
ax.plot(prune_ratio, ugba_reddit,  marker='o', markersize=5, linewidth=2.0, label='UGBA')
ax.plot(prune_ratio, dpgba_reddit, marker='^', markersize=5, linewidth=2.0, label='DPGBA')
ax.plot(prune_ratio, spear_reddit, marker='D', markersize=5, linewidth=2.0, label='SPEAR')
ax.plot(prune_ratio, ours_reddit,  marker='v', markersize=5, linewidth=2.8, label='Ours')

ax.set_title('(b) Reddit', fontweight='bold')
ax.set_xlabel('Pruning Ratio (%)')
ax.set_ylabel('S@1 (%)')
ax.set_ylim(10, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.grid(axis='y', linestyle='--', alpha=0.35)
ax.legend(frameon=False, fontsize=9)

# ===============================
# Save as PDF (IMPORTANT)
# ===============================
plt.tight_layout()
plt.savefig(
    "pruning_robustness_s1.pdf",
    bbox_inches="tight"
)
plt.close()
