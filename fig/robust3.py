import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# X-axis: Pruning ratio
# -----------------------------
prune_ratio = np.array([40, 80, 120, 160, 200, 240, 280])

# -----------------------------
# Simulated data (S@1%)
# -----------------------------
# -------- Cora --------
gta_cora   = [55, 45, 32, 22, 15,100,100]
ugba_cora  = [70, 60, 45, 35, 25,100,100]

# -------- Reddit --------
gta_reddit   = [58, 48, 35, 25, 18,100,100]
ugba_reddit  = [72, 63, 48, 36, 27,100,100]


# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 4))

# ---- Subplot 1: Cora ----
plt.subplot(1, 2, 1)
plt.plot(prune_ratio, gta_cora,   marker='s', label='S@1%')
plt.plot(prune_ratio, ugba_cora,  marker='o', label='S@5%')

plt.xlabel('The size of node with illusory prompts')
plt.ylabel('VSR (%)')
plt.title('Cora')
plt.ylim(10, 100)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# ---- Subplot 2: Reddit ----
plt.subplot(1, 2, 2)
plt.plot(prune_ratio, gta_cora,   marker='s', label='S@1%')
plt.plot(prune_ratio, ugba_cora,  marker='o', label='S@5%')

plt.xlabel('The size of node with illusory prompts')
plt.ylabel('VSR (%)')
plt.title('Reddit')
plt.ylim(10, 100)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()