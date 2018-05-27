import os
import numpy as np

import matplotlib as mpl

#mpl.use("pgf")
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
import models as md

figure_path = '../figures/results/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

sns.set()
sns.set_palette('colorblind')
sns_colors = sns.color_palette()
mpl.rcParams.update(custon_pgf_rcparams)

fit_df = pd.read_hdf('../data/generated/fitting_validation_error_v2.hdf5', key='fitting_results')


def plot_true_vs_estimate(param_name, ax, xmin=0, xmax=5):
    jitter = np.random.rand(len(fit_df)) * 0.2 - 0.1
    ax.errorbar(fit_df['true_' + param_name] + jitter, fit_df['mean_' + param_name], yerr=fit_df['std_' + param_name],
                fmt='.', capsize=18, barsabove=True, capthick=3, lw=3, ms=18)
    ax.plot([xmin, xmax], [xmin, xmax], 'k')
    ax.set_ylim([xmin, xmax])


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
fig.subplots_adjust(wspace=0.35, hspace=0.3)
ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[1, 0]
ax4 = axes[1, 1]

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

plt.show()
plt.savefig(os.path.join(figure_path, 'figure_fit_validation_v2.pdf'), bbox_inches='tight')
