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
import models as md

figure_path = '../../figures/results/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

sns.set()
sns.set_palette('colorblind')
sns_colors = sns.color_palette()
mpl.rcParams.update(custon_pgf_rcparams)

default_params = {'tau_m': 0.023,
                  'e_l': -0.079,
                  'r_m': 10*1e6,
                  'v_t': -0.061,
                  'init_vm_std': 0.000,
                  'vt_std': 0.000,
                  'rho_null': 0,
                  'rho_null_std': 0,
                  'tau_inh': 0.001,
                  'rho_scale': 9.0*1e6,
                  'exc_scale': 30,
                  'dt': 0.0001,
                  'total_time': 5,
                  'init_period': 2,
                  'noise_std_exc': 0*1e-3,
                  'noise_std_inh': 0*1e-3,
                  'cutoff_angle': 180,
                  'm': 1,
                  'b': 0,
                  'lv_min': 0.1,
                  'lv_max': 1.2,
                  'l_min': 10,
                  'l_max': 25,
                  'init_distance': 50}


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
fig.subplots_adjust(wspace=0.35, hspace=0.3)
ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[1, 0]
ax4 = axes[1, 1]
plot_colors = [sns_colors[0], sns_colors[2]]

ax1_params = default_params.copy()
rho_std_vals = [0.5, 2]
rho_mu_vals = [1]
for rho_idx, rho_std in enumerate(rho_std_vals):
    for rho_mu in rho_mu_vals:
        ax1_params['rho_null'] = rho_mu
        ax1_params['rho_null_std'] = rho_std
        nruns = 1000
        rstims = np.zeros(nruns)
        rdists = np.zeros(nruns)
        reaction_times = np.zeros(nruns)
        speeds = np.zeros(nruns)
        for i in np.arange(nruns):
            rstims[i], rdists[i], reaction_times[i], lv, stim_size, speeds[i], resp_in_t_to_coll = md.calc_response_fully_stationary(ax1_params)

        analytical_resp_angle = md.stationary_response_angle(ax1_params['v_t'], ax1_params['e_l'], ax1_params['r_m'],
                                                             ax1_params['rho_null']*1e-3, ax1_params['exc_scale']*1e-11,
                                                             ax1_params['rho_scale'], ax1_params['m'], ax1_params['b'])
        hist_label = r'$\mu_{{\rho_0}}$ = {:.1f}, $\sigma_{{\rho_0}}$ = {:.1f}'.format(rho_mu, rho_std)
        ax1.hist(rstims, bins=50, label=hist_label, color=plot_colors[rho_idx], density=True, alpha=0.7)
ax1.set_xlabel(r'$\theta_{resp}$ [\textdegree]')
ax1.set_ylabel('Density')
ax1.set_title('m = {:}, $\sigma_{{t}}$ = {:.1e}'.format(ax1_params['m'], ax1_params['vt_std']))
ax1.set_xlim([40, 180])
ax1.set_ylim([0, 0.15])
ax1.vlines(analytical_resp_angle, 0, .05, 'r', label='predicted value\n without noise')
ax1.legend(loc='upper right')


ax3_params = default_params.copy()
ax3_params['vt_std'] = 1*1e-3
for rho_idx, rho_std in enumerate(rho_std_vals):
    for rho_mu in rho_mu_vals:
        ax3_params['rho_null'] = rho_mu
        ax3_params['rho_null_std'] = rho_std
        nruns = 1000
        rstims = np.zeros(nruns)
        rdists = np.zeros(nruns)
        reaction_times = np.zeros(nruns)
        speeds = np.zeros(nruns)
        for i in np.arange(nruns):
            rstims[i], rdists[i], reaction_times[i], lv, stim_size, speeds[i], resp_in_t_to_coll = md.calc_response_fully_stationary(ax3_params)

        analytical_resp_angle = md.stationary_response_angle(ax3_params['v_t'], ax3_params['e_l'], ax3_params['r_m'],
                                                             ax3_params['rho_null']*1e-3, ax3_params['exc_scale']*1e-11,
                                                             ax3_params['rho_scale'], ax3_params['m'], ax3_params['b'])
        hist_label = r'$\mu_{{\rho_0}}$ = {:.1f}, $\sigma_{{\rho_0}}$ = {:.1f}'.format(rho_mu, rho_std)
        ax3.hist(rstims, bins=50, label=hist_label, color=plot_colors[rho_idx], density=True, alpha=0.7)
ax3.set_xlabel(r'$\theta_{resp}$ [\textdegree]')
ax3.set_ylabel('Density')
ax3.set_title('m = {:}, $\sigma_{{t}}$ = {:.1e}'.format(ax3_params['m'], ax3_params['vt_std']))
ax3.set_xlim([40, 180])
ax3.set_ylim([0, 0.15])
ax3.vlines(analytical_resp_angle, 0, .05, 'r', label='predicted value\n without noise')
ax3.legend(loc='upper right')

ax2_params = default_params.copy()
ax2_params['m'] = 1.5
for rho_idx, rho_std in enumerate(rho_std_vals):
    for rho_mu in rho_mu_vals:
        ax2_params['rho_null'] = rho_mu
        ax2_params['rho_null_std'] = rho_std
        nruns = 1000
        rstims = np.zeros(nruns)
        rdists = np.zeros(nruns)
        reaction_times = np.zeros(nruns)
        speeds = np.zeros(nruns)
        for i in np.arange(nruns):
            rstims[i], rdists[i], reaction_times[i], lv, stim_size, speeds[i], resp_in_t_to_coll = md.calc_response_fully_stationary(ax2_params)

        analytical_resp_angle = md.stationary_response_angle(ax2_params['v_t'], ax2_params['e_l'], ax2_params['r_m'],
                                                             ax2_params['rho_null']*1e-3, ax2_params['exc_scale']*1e-11,
                                                             ax2_params['rho_scale'], ax2_params['m'], ax2_params['b'])
        hist_label = r'$\mu_{{\rho_0}}$ = {:.1f}, $\sigma_{{\rho_0}}$ = {:.1f}'.format(rho_mu, rho_std)
        ax2.hist(rstims, bins=50, label=hist_label, color=plot_colors[rho_idx], density=True, alpha=0.7)
ax2.set_xlabel(r'$\theta_{resp}$ [\textdegree]')
ax2.set_ylabel('Density')
ax2.set_title('m = {:}, $\sigma_{{t}}$ = {:.1e}'.format(ax2_params['m'], ax2_params['vt_std']))
ax2.set_xlim([20, 180])
ax2.set_ylim([0, 0.15])
ax2.vlines(analytical_resp_angle, 0, .05, 'r', label='predicted value\n without noise')
ax2.legend(loc='upper right')


ax4_params = default_params.copy()
ax4_params['m'] = 1.5
ax4_params['vt_std'] = 1*1e-3
for rho_idx, rho_std in enumerate(rho_std_vals):
    for rho_mu in rho_mu_vals:
        ax4_params['rho_null'] = rho_mu
        ax4_params['rho_null_std'] = rho_std
        nruns = 1000
        rstims = np.zeros(nruns)
        rdists = np.zeros(nruns)
        reaction_times = np.zeros(nruns)
        speeds = np.zeros(nruns)
        for i in np.arange(nruns):
            rstims[i], rdists[i], reaction_times[i], lv, stim_size, speeds[i], resp_in_t_to_coll = md.calc_response_fully_stationary(ax4_params)

        analytical_resp_angle = md.stationary_response_angle(ax4_params['v_t'], ax4_params['e_l'], ax4_params['r_m'],
                                                             ax4_params['rho_null']*1e-3, ax4_params['exc_scale']*1e-11,
                                                             ax4_params['rho_scale'], ax4_params['m'], ax4_params['b'])
        hist_label = r'$\mu_{{\rho_0}}$ = {:.1f}, $\sigma_{{\rho_0}}$ = {:.1f}'.format(rho_mu, rho_std)
        ax4.hist(rstims, bins=50, label=hist_label, color=plot_colors[rho_idx], density=True, alpha=0.7)
ax4.set_xlabel(r'$\theta_{resp}$ [\textdegree]')
ax4.set_ylabel('Density')
ax4.set_title('m = {:}, $\sigma_{{t}}$ = {:.1e}'.format(ax4_params['m'], ax4_params['vt_std']))
ax4.set_xlim([20, 180])
ax4.set_ylim([0, 0.15])
ax4.vlines(analytical_resp_angle, 0, .05, 'r', label='predicted value\n without noise')
ax4.legend(loc='upper right')

axes = [ax1, ax2, ax3, ax4]
letters = ['A1', 'B1', 'A2', 'B2']
for ax, letter in zip(axes, letters):
    ax.text(-0.05, 1.05, letter, color='k', weight='bold', fontsize=20, transform=ax.transAxes,
            ha='center', va='center')

#plt.show()
plt.savefig(os.path.join(figure_path, 'figure_stationary_rho_null_noise.pdf'), bbox_inches='tight')

