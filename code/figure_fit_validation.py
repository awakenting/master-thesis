import os
import numpy as np

import matplotlib as mpl

mpl.use("pgf")
general_fontsize = 20
custon_pgf_rcparams = {
    'font.family': 'serif',
    'font.serif': 'cm',
    'font.size': general_fontsize,
    'xtick.labelsize': general_fontsize,
    'ytick.labelsize': general_fontsize,
    'axes.labelsize': general_fontsize,
    'axes.titlesize': general_fontsize,
    'legend.fontsize': general_fontsize - 2,
    'legend.borderaxespad': 0.5,
    'legend.borderpad': 0.4,
    'legend.columnspacing': 2.0,
    'legend.edgecolor': '0.8',
    'legend.facecolor': 'inherit',
    'legend.fancybox': True,
    'legend.framealpha': 0.8,
    'legend.frameon': True,
    'legend.handleheight': 0.7,
    'legend.handlelength': 2.0,
    'legend.handletextpad': 0.8,
    'legend.labelspacing': 0.5,
    'legend.loc': 'best',
    'legend.markerscale': 1.0,
    'legend.numpoints': 1,
    'legend.scatterpoints': 1,
    'legend.shadow': False,
    'text.usetex': True  # use inline math for ticks
}

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

figure_path = '../figures/results/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

sns.set()
sns.set_palette('colorblind')
sns_colors = sns.color_palette()
mpl.rcParams.update(custon_pgf_rcparams)

fit_df = pd.read_hdf('../data/generated/fitting_validation_error_v4.hdf5', key='fitting_results')
fit_df2 = pd.read_hdf('../data/generated/fitting_validation_error_v3.hdf5', key='fitting_results')
fit_df = fit_df.append(fit_df2, ignore_index=True)

vmin = np.min(fit_df['true_rho_null'])
vmax = np.max(fit_df['true_rho_null'])
plot_cmap = plt.cm.get_cmap('viridis', 6)
sm = plt.cm.ScalarMappable(cmap=plot_cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))

def plot_true_vs_estimate(param_name, ax, xmin=0, xmax=5):
    jitter = np.random.rand(len(fit_df)) * 0.2 - 0.1
    for idx in range(len(fit_df)):
        ax.errorbar(fit_df['true_' + param_name][idx] + jitter[idx], fit_df['mean_' + param_name][idx], yerr=fit_df['std_' + param_name][idx],
                    fmt='.', color=sm.to_rgba(fit_df['true_rho_null'][idx]), capsize=12, barsabove=True, capthick=3, lw=3, ms=18)
    ax.plot([xmin, xmax], [xmin, xmax], 'k')
    ax.set_ylim([xmin, xmax])


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
fig.subplots_adjust(wspace=0.25, hspace=0.3, right=0.85)
ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[1, 0]
ax4 = axes[1, 1]

cbar_axes = fig.add_axes([0.9, 0.25, 0.04, 0.5])

plot_true_vs_estimate('rho_null', ax=ax1)
ax1.set_xlabel(r'true $\mu_{\rho_{0}}$')
ax1.set_ylabel(r'posterior mean')

plot_true_vs_estimate('rho_null_std', ax=ax2)
ax2.set_xlabel(r'true $\sigma_{\rho_{0}}$')
ax2.set_ylabel(r'posterior mean')

plot_true_vs_estimate('rho_scale', ax=ax3, xmin=5, xmax=10)
ax3.set_xlabel(r'true $c_{\rho}$')
ax3.set_ylabel(r'posterior mean')

plot_true_vs_estimate('noise_std_exc', ax=ax4)
ax4.set_xlabel(r'true $\sigma_{m}$')
ax4.set_ylabel(r'posterior mean')

sm.set_array([])
cbar = plt.colorbar(sm, cax=cbar_axes)
cbar.set_label(r'true $\mu_{\rho_{0}}$ value')

axes = [ax1, ax2, ax3, ax4]
letters = ['A', 'B', 'C', 'D']
for ax, letter in zip(axes, letters):
    ax.text(-0.05, 1.10, letter, color='k', weight='bold', fontsize=20, transform=ax.transAxes,
            ha='center', va='center')

#plt.show()
plt.savefig(os.path.join(figure_path, 'figure_fit_validation.pdf'), bbox_inches='tight')
