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

figure_path = '../figures/results/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

sns.set()
sns.set_palette('colorblind')
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
                  'init_period': 0,
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


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
fig.subplots_adjust(wspace=0.35, hspace=0.3)
ax1 = axes[0]
ax2 = axes[1]

ax1_params = default_params.copy()
for vt_std in [0.0005, 0.001, 0.002]:
    nruns = 5000
    rstims = np.zeros(nruns)
    rdists = np.zeros(nruns)
    reaction_times = np.zeros(nruns)
    speeds = np.zeros(nruns)
    for i in np.arange(nruns):
        ax1_params['noise_std_exc'] = vt_std
        rstims[i], rdists[i], reaction_times[i], lv, stim_size, speeds[i], resp_in_t_to_coll = md.calc_response_fully_stationary(ax1_params)

    analytical_resp_angle = md.stationary_response_angle(ax1_params['v_t'], ax1_params['e_l'], ax1_params['r_m'],
                                                         ax1_params['rho_null']*1e-3, ax1_params['exc_scale']*1e-11,
                                                         ax1_params['rho_scale'], ax1_params['m'], ax1_params['b'])

    ax1.hist(rstims, bins=30, label='$\sigma_{{t}}$ = {:.1e}'.format(vt_std), density=True, alpha=0.8)
ax1.set_xlabel(r'$\theta_{resp}$ [\textdegree]')
ax1.set_ylabel('Density')
ax1.set_title('Noise within trial')
#ax1.set_xlim([25, 65])
ax1.set_ylim([0, 0.6])
ax1.vlines(analytical_resp_angle, 0, .2, 'r', label='predicted value\n without noise')
ax1.legend(loc='upper left')


ax2_params = default_params.copy()
for vt_std in [0.0005, 0.001, 0.002]:
    nruns = 5000
    rstims = np.zeros(nruns)
    rdists = np.zeros(nruns)
    reaction_times = np.zeros(nruns)
    speeds = np.zeros(nruns)
    run_vt_vals = np.random.normal(loc=-0.061, scale=vt_std, size=nruns)
    for i in np.arange(nruns):
        ax2_params['v_t'] = run_vt_vals[i]
        rstims[i], rdists[i], reaction_times[i], lv, stim_size, speeds[i], resp_in_t_to_coll = md.calc_response_fully_stationary(ax2_params)

    analytical_resp_angle = md.stationary_response_angle(default_params['v_t'], ax2_params['e_l'], ax2_params['r_m'],
                                                         ax2_params['rho_null']*1e-3, ax2_params['exc_scale']*1e-11,
                                                         ax2_params['rho_scale'], ax2_params['m'], ax2_params['b'])

    ax2.hist(rstims, bins=30, label='$\sigma_{{t}}$ = {:.1e}'.format(vt_std), density=True, alpha=0.8)
ax2.set_xlabel(r'$\theta_{resp}$ [\textdegree]')
ax2.set_ylabel('Density')
ax2.set_title('Noise across trials')
ax2.set_xlim([35, 85])
ax2.set_ylim([0, 0.3])
ax2.vlines(analytical_resp_angle, 0, .2, 'r', label='predicted value\n without noise')
ax2.legend(loc='upper left')


axes = [ax1, ax2]
letters = ['A', 'B']
for ax, letter in zip(axes, letters):
    ax.text(-0.05, 1.05, letter, color='k', weight='bold', fontsize=20, transform=ax.transAxes,
            ha='center', va='center')

#plt.show()
plt.savefig(os.path.join(figure_path, 'figure_static_vs_dynamic_noise.pdf'), bbox_inches='tight')

