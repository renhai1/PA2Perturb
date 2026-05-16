import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

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

np.random.seed(42)

# ========== Generate synthetic data for each dataset ==========
datasets_config = {
    'Cora': {
        'normal_mean': 1.1e-5, 'normal_std': 0.5e-5, 'normal_n': 800,
        'perturb_mean': 0.7e-5, 'perturb_std': 0.15e-5, 'perturb_n': 30,
        'scale': 1e-5, 'scale_label': r'$\times 10^{-5}$',
        'xlim': (0, 4.8e-5), 'freq_max': 125,
    },
    'Cora-P': {
        'normal_mean': 1.2e-5, 'normal_std': 0.55e-5, 'normal_n': 600,
        'perturb_mean': 0.8e-5, 'perturb_std': 0.18e-5, 'perturb_n': 25,
        'scale': 1e-5, 'scale_label': r'$\times 10^{-5}$',
        'xlim': (0.3e-5, 4.5e-5), 'freq_max': 90,
    },
    'Flickr': {
        'normal_mean': 0.028, 'normal_std': 0.012, 'normal_n': 750,
        'perturb_mean': 0.022, 'perturb_std': 0.004, 'perturb_n': 28,
        'scale': 0.1, 'scale_label': r'$\times 10^{-1}$',
        'xlim': (0.01, 0.13), 'freq_max': 110,
    },
    'Cora-F': {
        'normal_mean': 0.025, 'normal_std': 0.011, 'normal_n': 600,
        'perturb_mean': 0.018, 'perturb_std': 0.004, 'perturb_n': 22,
        'scale': 0.1, 'scale_label': r'$\times 10^{-1}$',
        'xlim': (0.005, 0.105), 'freq_max': 85,
    },
}


def generate_right_skewed(mean, std, n):
    """Generate right-skewed data using log-normal distribution."""
    sigma = np.sqrt(np.log(1 + (std / mean) ** 2))
    mu = np.log(mean) - sigma ** 2 / 2
    return np.random.lognormal(mu, sigma, n)


# ========== Create figure ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
dataset_names = ['Cora', 'Cora-P', 'Flickr', 'Cora-F']

for ax, name in zip(axes.flat, dataset_names):
    cfg = datasets_config[name]
    scale = cfg['scale']

    # Generate data
    normal_data = generate_right_skewed(cfg['normal_mean'], cfg['normal_std'], cfg['normal_n'])
    perturb_data = generate_right_skewed(cfg['perturb_mean'], cfg['perturb_std'], cfg['perturb_n'])

    # Clip to reasonable range
    normal_data = normal_data[normal_data < cfg['xlim'][1] * 1.2]
    perturb_data = perturb_data[perturb_data < cfg['xlim'][1] * 1.2]

    # Scale for display
    normal_scaled = normal_data / scale
    perturb_scaled = perturb_data / scale
    xlim_scaled = (cfg['xlim'][0] / scale, cfg['xlim'][1] / scale)

    # Histogram bins
    num_bins = 40
    bins = np.linspace(xlim_scaled[0], xlim_scaled[1], num_bins + 1)

    # Plot histograms
    ax.hist(normal_scaled, bins=bins, color='#6fa8dc', edgecolor='#4a7fb5',
            linewidth=0.5, alpha=0.75, label='Normal', zorder=3)
    ax.hist(perturb_scaled, bins=bins, color='#f4a460', edgecolor='#d4874a',
            linewidth=0.5, alpha=0.8, label='Perturbations', zorder=4)

    # KDE curve for normal data
    kde = gaussian_kde(normal_scaled, bw_method=0.3)
    x_kde = np.linspace(xlim_scaled[0], xlim_scaled[1], 300)
    kde_values = kde(x_kde)
    # Scale KDE to match histogram height
    bin_width = bins[1] - bins[0]
    kde_scaled = kde_values * len(normal_scaled) * bin_width
    ax.plot(x_kde, kde_scaled, color='#1a3a5c', linewidth=2.0, zorder=5)

    # Formatting
    ax.set_title(name, fontsize=15, fontweight='bold', pad=8)
    ax.set_xlabel('Reconstruction Loss', fontsize=12, labelpad=6)
    ax.set_ylabel('Frequency', fontsize=12, labelpad=6)
    ax.set_xlim(xlim_scaled)
    ax.set_ylim(0, cfg['freq_max'])
    ax.grid(False)
    ax.tick_params(axis='both', direction='in', length=4, width=1.0, labelsize=11)

    # Add scale annotation in bottom-right
    ax.annotate(cfg['scale_label'], xy=(1.0, -0.02), xycoords='axes fraction',
                ha='right', va='top', fontsize=11, color='#333333')

    # Legend
    ax.legend(fontsize=10.5, loc='upper right', frameon=True,
              edgecolor='#aaaaaa', fancybox=True, framealpha=0.95)

    for spine in ax.spines.values():
        spine.set_linewidth(1.3)

plt.tight_layout(h_pad=3.0, w_pad=2.5)
plt.savefig('/Users/keyuan/PycharmProjects/PA2Perturb/datashow/reconstruction_loss_dist.pdf',
            format='pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Done! Saved to reconstruction_loss_dist.pdf")
