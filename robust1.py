import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Global settings (for PDF clarity)
# -----------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "lines.linewidth": 2,
})

# -----------------------------
# X-axis: Pruning ratio
# -----------------------------
prune_ratio = np.array([0, 20, 40, 60, 80])

# -----------------------------
# Simulated data (S@1%)
# -----------------------------
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

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 4))

# ---- Subplot 1: Cora ----
plt.subplot(1, 2, 1)
plt.plot(prune_ratio, gta_cora,   marker='s', label='GTA')
plt.plot(prune_ratio, ugba_cora,  marker='o', label='UGBA')
plt.plot(prune_ratio, dpgba_cora, marker='^', label='DPGBA')
plt.plot(prune_ratio, spear_cora, marker='D', label='SPEAR')
plt.plot(prune_ratio, ours_cora,  marker='v', linewidth=2.8, label='Ours')

plt.xlabel('Pruning Ratio (%)')
plt.ylabel('S@1 (%)')
plt.title('Cora')
plt.ylim(10, 100)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(
    loc='lower left',
    frameon=False
)

# ---- Subplot 2: Reddit ----
plt.subplot(1, 2, 2)
plt.plot(prune_ratio, gta_reddit,   marker='s', label='GTA')
plt.plot(prune_ratio, ugba_reddit,  marker='o', label='UGBA')
plt.plot(prune_ratio, dpgba_reddit, marker='^', label='DPGBA')
plt.plot(prune_ratio, spear_reddit, marker='D', label='SPEAR')
plt.plot(prune_ratio, ours_reddit,  marker='v', linewidth=2.8, label='Ours')

plt.xlabel('Pruning Ratio (%)')
plt.ylabel('S@1 (%)')
plt.title('Reddit')
plt.ylim(10, 100)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(
    loc='lower left',
    frameon=False
)

plt.tight_layout()

# -----------------------------
# Save as PDF (vector, high quality)
# -----------------------------
plt.savefig("pruning_robustness.pdf", format="pdf", bbox_inches="tight")
plt.close()
