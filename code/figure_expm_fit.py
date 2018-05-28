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

from delfi.utils import viz

figure_path = '../figures/results/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

sns.set()
sns.set_palette('colorblind')
sns_colors = sns.color_palette()
mpl.rcParams.update(custon_pgf_rcparams)

expm_fit_df = pd.read_hdf('../data/generated/fitting_expm_data.hdf5', key='fitting_results')

expfit_posterior = expm_fit_df['posterior_object'][0]

lims = np.array([[0, 5],[0, 5], [0, 5], [5, 10]])
fig, axes = viz.plot_pdf(expfit_posterior.xs[0], lims=lims, labels_params=['rho_null', 'rho_null_std', 'noise_std', 'rho_scale'],
                         figsize=(12,12), ticks=True, col1=sns_colors[1], col2=sns_colors[0])
axes[0, 0].set_xlabel(r'$\mu_{\rho_{0}}$')
axes[1, 1].set_xlabel(r'$\sigma_{\rho_{0}}$')
axes[2, 2].set_xlabel(r'$\sigma_{m}$')
axes[3, 3].set_xlabel(r'$c_{\rho}$')

for ax in [axes[0, 0], axes[1, 1], axes[2, 2]]:
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(np.arange(6))

axes[3, 3].set_xticks(np.arange(5, 11))
axes[3, 3].set_xticklabels(np.arange(5, 11))


#plt.show()
plt.savefig(os.path.join(figure_path, 'figure_expm_fit_posterior.pdf'), bbox_inches='tight')
