import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data (random but reasonable)
# -----------------------------
attacks = ['Clean', 'Prune', 'OD', 'RIGBD', 'MAD']

baseline = np.array([92, 68, 61, 55, 63])
ours = np.array([96, 90, 88, 85, 89])

x = np.arange(len(attacks))
width = 0.35

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7, 4))

plt.bar(x - width/2, baseline, width,
        label='Baseline',
        edgecolor='black')

plt.bar(x + width/2, ours, width,
        label='Ours',
        edgecolor='black')

# -----------------------------
# Aesthetics
# -----------------------------
plt.ylabel('Verification Success Rate (S@1%)', fontsize=11)
plt.xlabel('Attack Method', fontsize=11)
plt.xticks(x, attacks, fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(0, 100)

plt.legend(frameon=False, fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
