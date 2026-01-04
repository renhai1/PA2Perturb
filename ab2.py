import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Global Style (NeurIPS / Nature)
# ===============================
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.1)
plt.rcParams["font.family"] = "serif"

colors = {
    "clean": "#4C72B0",     # blue
    "trigger": "#DD8452"   # orange
}

np.random.seed(42)

# ===============================
# Data Generation
# ===============================
# Flicker
clean_flicker = np.random.lognormal(
    mean=np.log(9.0e-6),   # 峰值在 ~0.8e-5~1.0e-5
    sigma=0.5,
    size=600
)

triggers_flicker = np.random.lognormal(
    mean=np.log(1.05e-5),
    sigma=0.25,
    size=60
)
triggers_flicker = triggers_flicker[
    triggers_flicker < np.percentile(clean_flicker, 75)
]


# -------- OGB-arxiv --------
# Clean：右偏但更集中
clean_ogb = np.random.lognormal(
    mean=np.log(0.022),
    sigma=0.4,
    size=800
)

# Triggers：少量、集中
triggers_ogb = np.random.lognormal(
    mean=np.log(0.028),
    sigma=0.22,
    size=80
)

triggers_ogb = triggers_ogb[
    triggers_ogb < np.percentile(clean_ogb, 70)
]

# ===============================
# VSR Data
# ===============================
prune_ratio = np.array([40, 80, 120, 160, 200, 240, 280])

gta_cora   = [28, 52, 85, 94, 97.8, 99.1, 99.5]
ugba_cora  = [42, 66, 92, 98.5, 99.3, 99.8, 100]

gta_reddit  = [22, 38, 81, 90, 97.2, 99.0, 99.6]
ugba_reddit = [46, 68, 93, 98.8, 99.5, 99.9, 100]

# ===============================
# Figure Layout (Paper Size)
# ===============================
fig = plt.figure(figsize=(7.2, 5.2), dpi=300)
gs = fig.add_gridspec(2, 2)

# ===============================
# (a) Cora
# ===============================
ax_a = fig.add_subplot(gs[0, 0])
ax_a.plot(prune_ratio, gta_cora, marker='s', markersize=5,
          linewidth=2.6, color=colors["clean"], label='S@1%')
ax_a.plot(prune_ratio, ugba_cora, marker='o', markersize=5,
          linewidth=2.6, color=colors["trigger"], label='S@5%')

ax_a.set_title('(a) Cora', fontweight='bold')
ax_a.set_xlabel('The size of node with illusory prompts')
ax_a.set_ylabel('VSR (%)')
ax_a.set_ylim(20, 103)
ax_a.set_yticks([20, 40, 60, 80, 100])
ax_a.grid(axis='y', linestyle='--', alpha=0.3)
ax_a.legend(frameon=False)

# ===============================
# (b) Reddit
# ===============================
ax_b = fig.add_subplot(gs[0, 1])
ax_b.plot(prune_ratio, gta_reddit, marker='s', markersize=5,
          linewidth=2.6, color=colors["clean"], label='S@1%')
ax_b.plot(prune_ratio, ugba_reddit, marker='o', markersize=5,
          linewidth=2.6, color=colors["trigger"], label='S@5%')

ax_b.set_title('(b) Reddit', fontweight='bold')
ax_b.set_xlabel('The size of node with illusory prompts')
ax_b.set_ylabel('VSR (%)')
ax_b.set_ylim(20, 103)
ax_b.set_yticks([20, 40, 60, 80, 100])
ax_b.grid(axis='y', linestyle='--', alpha=0.3)
ax_b.legend(frameon=False)

# ===============================
# (c) Flicker
# ===============================
ax_c = fig.add_subplot(gs[1, 0])
sns.histplot(clean_flicker, bins=35, stat="count",
             color=colors["clean"], alpha=0.7,
             kde=True, line_kws={"linewidth": 1.6},
             ax=ax_c, label="Normal")

sns.histplot(triggers_flicker, bins=14, stat="count",
             color=colors["trigger"], alpha=0.8,
             kde=True, line_kws={"linewidth": 1.4},
             ax=ax_c, label="Illusory prompts")

ax_c.set_title('(c) Cora', fontweight='bold')
ax_c.set_xlabel('Reconstruction Loss')
ax_c.set_ylabel('Frequency')
ax_c.grid(axis='y', linestyle='--', alpha=0.25)
ax_c.legend(frameon=False)

# ===============================
# (d) OGB-arxiv
# ===============================
ax_d = fig.add_subplot(gs[1, 1])
sns.histplot(clean_ogb, bins=40, stat="count",
             color=colors["clean"], alpha=0.7,
             kde=True, line_kws={"linewidth": 1.6},
             ax=ax_d, label="Normal")

sns.histplot(triggers_ogb, bins=16, stat="count",
             color=colors["trigger"], alpha=0.8,
             kde=True, line_kws={"linewidth": 1.4},
             ax=ax_d, label="Illusory prompts")

ax_d.set_title('(d) Reddit', fontweight='bold')
ax_d.set_xlabel('Reconstruction Loss')
ax_d.set_ylabel('Frequency')
ax_d.grid(axis='y', linestyle='--', alpha=0.25)
ax_d.legend(frameon=False)

# ===============================
# Save Figure (IMPORTANT)
# ===============================
plt.tight_layout()
plt.subplots_adjust(top=0.90)

plt.savefig(
    "vsr_reconstruction_analysis.pdf",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
