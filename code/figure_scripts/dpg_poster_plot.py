import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import models as md

figure_path = '../figures/results/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)


def plot_response_props_column(exc_scale, inh_scale, vt_std, rho_null, rho_null_std, tau_inh, cutoff_angle, exc_noise,
                               m):
    params = {'tau_m': 0.023,
              'e_l': -0.079,
              'r_m': 10 * 1e6,  # MOhm
              'v_t': -0.061,
              'init_vm_std': 0.001,
              'vt_std': vt_std / 1000,
              'rho_null': rho_null,
              'rho_null_std': rho_null_std,
              'tau_inh': tau_inh / 1000,
              'rho_scale': inh_scale * 1e6,
              'exc_scale': exc_scale,
              'dt': 0.0001,
              'total_time': 5,
              'init_period': 2,
              'cutoff_angle': cutoff_angle,
              'noise_std_exc': exc_noise / 1000,
              'noise_std_inh': 5 * 1e-3,
              'm': m,
              'b': 0,
              'lv_min': 0.1,
              'lv_max': 1.2,
              'l_min': 10,
              'l_max': 25,
              'init_distance': 50}
    nruns = 250
    data_cols = ['Response angle ($\degree$)', 'Reactive distance (mm)', 'Latency (s)', 'L/V (s)', 'stim_size', 'speed',
                 'Time to collision (s)']
    data_dict = dict([(col_name, []) for col_name in data_cols])

    for i in np.arange(nruns):
        resp_angle, resp_dist, resp_time, lv, stim_size, speed, resp_time_coll = md.calc_response_ffi(params)
        resp_angle = np.round(resp_angle, decimals=1)
        resp_dist = np.round(resp_dist, decimals=1)
        resp_time = np.round(resp_time, decimals=3)
        lv = np.round(lv, decimals=2)
        stim_size = np.round(stim_size, decimals=1)
        speed = np.round(speed, decimals=1)
        resp_time_coll = np.round(resp_time_coll, decimals=3)
        result_values = [resp_angle, resp_dist, resp_time, lv, stim_size, speed, resp_time_coll]
        for col, value in zip(data_cols, result_values):
            data_dict[col].append(value)

    df = pd.DataFrame(data_dict)
    df['Latency (s)'] = df['Latency (s)'] - params['init_period']
    # sns.set('poster')
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(7, 32))
    plt.subplots_adjust(hspace=0.3)
    axes[0].plot(df['Reactive distance (mm)'], df['L/V (s)'], '.', ms=18, color='#cc0000')
    axes[0].set_xlabel('Reactive distance (mm)')
    axes[0].set_ylabel('L/V (s)')
    axes[0].set_ylim([0, 1.3])
    axes[0].set_xlim([0, 50])
    axes[0].set_facecolor('w')
    axes[0].set_xticks(np.arange(6) * 10)
    axes[0].set_xticklabels(np.arange(6) * 10)
    axes[0].set_yticks(np.arange(7) * 0.2)
    axes[0].set_yticklabels([0, '', 0.4, '', 0.8, '', 1.2])

    axes[1].plot(df['Latency (s)'], df['L/V (s)'], '.', ms=18, color='#cc0000')
    axes[1].set_xlabel('Latency (s)')
    axes[1].set_ylabel('L/V (s)')
    axes[1].set_xlim([0, 6])
    axes[1].set_facecolor('w')
    axes[1].set_xticks(np.arange(7))
    axes[1].set_xticklabels(np.arange(7))
    axes[1].set_yticks(np.arange(7) * 0.2)
    axes[1].set_yticklabels([0, '', 0.4, '', 0.8, '', 1.2])

    axes[2].plot(df['L/V (s)'], df['Response angle ($\degree$)'], '.', ms=18, color='#cc0000')
    axes[2].set_xlabel('L/V (s)')
    axes[2].set_ylabel('Response angle ($\degree$)')
    axes[2].set_ylim([0, 185])
    axes[2].set_xlim([0, 1.3])
    axes[2].set_facecolor('w')
    axes[2].set_xticks(np.arange(7) * 0.2)
    axes[2].set_xticklabels([0, '', 0.4, '', 0.8, '', 1.2])
    axes[2].set_yticks(np.arange(8) * 25)
    axes[2].set_yticklabels(np.arange(8) * 25)

    axes[3].plot(df['Time to collision (s)'], df['L/V (s)'], '.', ms=18, color='#cc0000')
    axes[3].set_xlabel('Time to collision (s)')
    axes[3].set_ylabel('L/V (s)')
    axes[3].set_ylim([0, 1.3])
    axes[3].set_xlim([-5, 0.1])
    axes[3].set_facecolor('w')
    axes[3].set_xticks(np.arange(-6, 1))
    axes[3].set_xticklabels(np.arange(-6, 1))
    axes[3].set_yticks(np.arange(7) * 0.2)
    axes[3].set_yticklabels([0, '', 0.4, '', 0.8, '', 1.2])

general_fontsize = 30
mpl.rcParams['font.size'] = general_fontsize
mpl.rcParams['xtick.labelsize'] = general_fontsize
mpl.rcParams['ytick.labelsize'] = general_fontsize
mpl.rcParams['axes.labelsize'] = general_fontsize
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['axes.edgecolor'] = 'k'
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.linewidth'] = 3
mpl.rcParams['axes.labelpad'] = 10
plot_response_props_column(exc_scale=40, inh_scale=9.6, vt_std=2, rho_null=2, rho_null_std=1.2, tau_inh=1, cutoff_angle=180, exc_noise=5, m=3)
plt.savefig(figure_path + 'resp_props.eps', bbox_inches='tight')