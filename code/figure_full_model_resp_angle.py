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
import pandas as pd
import models as md

figure_path = '../figures/results/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

sns.set()
sns.set_palette('colorblind')
mpl.rcParams.update(custon_pgf_rcparams)

params = {'tau_m': 0.023,
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
          'total_time': 25,
          'init_period': 0,
          'noise_std_exc': 0*1e-3,
          'noise_std_inh': 0*1e-3,
          'cutoff_angle': 180,
          'm': 1,
          'b': 0,
          'lv_min': 0.1,
          'lv_max': 5,
          'l_min': 10,
          'l_max': 25,
          'init_distance': 50}
nruns = 1000

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

for cax, m_val in zip([ax1, ax2], [1, 2]):
    params['m'] = m_val
    tau_m_values = [0.010, 0.023, 0.050]
    for tau_m in tau_m_values:
        params['tau_m'] = tau_m
        data_cols = ['resp_angle', 'resp_dist', 'resp_time', 'lv', 'stim_size', 'speed', 'resp_time_coll']
        data_dict = dict([(col_name, []) for col_name in data_cols])
        for i in np.arange(nruns):
            resp_angle, resp_dist, resp_time, lv, stim_size, speed, resp_time_coll = md.calc_response_ffi(params)

            result_values = [resp_angle, resp_dist, resp_time, lv, stim_size, speed, resp_time_coll]
            for col, value in zip(data_cols, result_values):
                data_dict[col].append(value)

        df = pd.DataFrame(data_dict)

        cax.plot(df['lv'], df['resp_angle'], '.', ms=12, label=r'$\tau_m$ = {:} ms'.format(tau_m*1000))

    rho_null = np.random.lognormal(mean=params['rho_null'], sigma=params['rho_null_std'])
    analytical_resp_angle = md.stationary_response_angle(params['v_t'], params['e_l'], params['r_m'], rho_null*1e-3,
                                                      params['exc_scale']*1e-11, params['rho_scale'], params['m'],
                                                      params['b'])

    cax.hlines(analytical_resp_angle, params['lv_min'], params['lv_max'], 'r', lw=3, label='stationary prediction')

    cax.set_xlabel('L/V [s]')
    cax.set_ylabel(r'$\theta(t)$ [\textdegree]')
    cax.legend()

axes = [ax1, ax2]
letters = ['A', 'B']
for ax, letter in zip(axes, letters):
    ax.text(-0.05, 1.05, letter, color='k', weight='bold', fontsize=20, transform=ax.transAxes,
            ha='center', va='center')

#plt.show()
plt.savefig(os.path.join(figure_path, 'figure_full_model_resp_angle.pdf'), bbox_inches='tight')
