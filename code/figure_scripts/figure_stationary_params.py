import os
import numpy as np

import matplotlib as mpl
mpl.use("pgf")
general_fontsize = 20
custon_pgf_rcparams = {
    "font.family": 'serif',
    "font.serif": 'cm',
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
    "text.usetex": True,    # use inline math for ticks
}

import matplotlib.pyplot as plt
import seaborn as sns
import models as md

figure_path = '../../figures/results/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

sns.set()
sns.set_palette('colorblind')
mpl.rcParams.update(custon_pgf_rcparams)

vt = -0.061
el = -0.079
rm = 1e7
c_scale = 3*1e-10
b = 0


fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))


m = 1
rho_null = 0

rm_values = [1e7, 0.5*1e7]
sns_colors = sns.color_palette()
for rm_idx, rm_val in enumerate(rm_values):
    c_rho_range = rm_val * np.linspace(0, 0.99, num=1000)
    resp_angles = md.stationary_response_angle(vt, el, rm_val, rho_null, c_scale, c_rho_range, m, b)
    ax1.plot(c_rho_range, resp_angles, lw=3)
    ax1.vlines(rm_val, 0, 180, colors=sns_colors[rm_idx], linestyles='--', lw=3, label='$R_{{m}}$ = {:.1}'.format(rm_val))
ax1.set_xlabel(r'$c_{\rho}$')
ax1.set_ylabel(r'$\theta_{resp}$')
ax1.set_ylim([0, 180])
ax1.legend()


m = np.linspace(0.1, 5, num=1000)
c_rho = 1e7 * 0.9

rho_null_values = [0, 0.005, 0.010]
for rho_null_val in rho_null_values:
    resp_angles = md.stationary_response_angle(vt, el, rm, rho_null_val, c_scale, c_rho, m, b)
    ax2.plot(m, resp_angles, lw=3, label=r'$\rho{{0}}$ = {:}'.format(rho_null_val))
ax2.set_xlabel(r'$m$')
ax2.set_ylabel(r'$\theta_{resp}$')
ax2.legend()
ax2.set_ylim([0, 180])


m_values = [1, 2, 3]
rho_null = np.linspace(0.001, 0.050, num=1000)
c_rho = 1e7 * 0.9

for m_val in m_values:
    resp_angles = md.stationary_response_angle(vt, el, rm, rho_null, c_scale, c_rho, m_val, b)
    ax3.plot(rho_null, resp_angles, lw=3, label='$m$ = {:}'.format(m_val))
ax3.set_xlabel(r'$\rho_{0}$')
ax3.set_ylabel(r'$\theta_{resp}$')
ax3.set_ylim([0, 180])
ax3.legend()


axes = [ax1, ax2, ax3]
letters = ['A', 'B', 'C']
for ax, letter in zip(axes, letters):
    ax.text(-0.05, 1.05, letter, color='k', weight='bold', fontsize=20, transform=ax.transAxes,
            ha='center', va='center')

#plt.show()
plt.savefig(os.path.join(figure_path, 'figure_stationary_params.pdf'), bbox_inches='tight')
