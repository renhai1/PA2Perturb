import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Attacks
# -----------------------------
attacks = ['Clean', 'Prune', 'OD', 'RIGBD', 'MAD']
x = np.arange(len(attacks))
width = 0.35

# -----------------------------
# Data (S@1%)
# -----------------------------
# Cora
spear_cora = [91.2, 68.5, 74.8, 71.3, 65.2]
ours_cora  = [94.6, 86.3, 88.1, 87.0, 84.1]

# Reddit
spear_reddit = [89.6, 63.4, 70.2, 67.1, 60.8]
ours_reddit  = [93.1, 82.7, 85.6, 84.3, 81.5]

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# ---- Cora ----
axes[0].bar(x - width/2, spear_cora, width, label='SPEAR')
axes[0].bar(x + width/2, ours_cora,  width, label='Ours')
axes[0].set_title('Cora')
axes[0].set_xticks(x)
axes[0].set_xticklabels(attacks)
axes[0].set_ylabel('S@1 (%)')
axes[0].set_ylim(55, 100)
axes[0].grid(axis='y', linestyle='--', alpha=0.5)
axes[0].legend()

# ---- Reddit ----
axes[1].bar(x - width/2, spear_reddit, width, label='SPEAR')
axes[1].bar(x + width/2, ours_reddit,  width, label='Ours')
axes[1].set_title('Reddit')
axes[1].set_xticks(x)
axes[1].set_xticklabels(attacks)
axes[1].set_ylim(55, 100)
axes[1].grid(axis='y', linestyle='--', alpha=0.5)
axes[1].legend()

plt.tight_layout()
plt.show()