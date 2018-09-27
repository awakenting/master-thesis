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
import scipy.io as sio

from .. import model_neuronal as md


figure_path = './figures/results/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

sns.set()
sns.set_palette('colorblind')
sns_colors = sns.color_palette()
mpl.rcParams.update(custon_pgf_rcparams)

expm_fit_df = pd.read_hdf('/home/andrej/Documentt/fitting_expm_data.hdf5', key='fitting_results')

expfit_posterior = expm_fit_df['posterior_object'][0]


data = sio.loadmat('./data/external/LVsVersusSubtendedAngle.mat')
clean_dict = {'lv': np.squeeze(data['LVs']), 'resp_angle': np.squeeze(data['subtendedAngleAtResponse'])}
expm_df = pd.DataFrame(clean_dict)
expm_data = np.concatenate((clean_dict['resp_angle'], clean_dict['lv'])).reshape((1, 492))
expm_lvs = expm_data[0, 246:]
expm_thetas = expm_data[0, 0:246]

lv_bins = [(0.1, 0.28), (0.28, 0.47), (0.47, 0.65), (0.65, 0.83), (0.83, 1.01), (1.01, 1.2)]
lv_centers = np.array([0.19, 0.38, 0.56, 0.74, 0.92, 1.11])

qnt_list = []
for lv_low, lv_high in lv_bins:
    mask = (lv_low < expm_lvs) & (expm_lvs < lv_high)
    qnt_list.append(np.percentile(expm_thetas[mask], [10, 30, 50, 70, 90]))
qnt_array = np.array(qnt_list)

lv_bin_bounds = np.array([lv_bin[0] for lv_bin in lv_bins])
lv_bin_bounds = np.concatenate((lv_bin_bounds, [1.2]))

params = {'tau_m': 0.023,
          'e_l': -0.079,
          'r_m': 10 * 1e6,  # MOhm
          'v_t': -0.061,
          'init_vm_std': 0.0,
          'vt_std': 0.000,
          'rho_null': expm_fit_df['mean_rho_null'][0],
          'rho_null_std': expm_fit_df['mean_rho_null_std'][0],
          'tau_inh': 0.001,
          'rho_scale': expm_fit_df['mean_rho_scale'][0] * 1e6,
          'exc_scale': 30,
          'dt': 0.001,
          'total_time': 5,
          'init_period': 2,
          'cutoff_angle': 180,
          'noise_std_exc': expm_fit_df['mean_noise_std_exc'][0] * 1e-3,
          'noise_std_inh': 5 * 1e-3,
          'm': 3,
          'b': 0,
          'lv_min': 0.1,
          'lv_max': 1.2,
          'l_min': 10,
          'l_max': 25,
          'init_distance': 50}
nruns = 256
nreps = 100
data_cols = ['resp_angle', 'resp_dist', 'resp_time', 'lv', 'stim_size', 'speed', 'resp_time_coll']
data_dict = dict([(col_name, []) for col_name in data_cols])

model_quant_matrix = np.zeros((6, 5, nreps))
for rep_idx in range(nreps):
    data_dict = dict([(col_name, []) for col_name in data_cols])
    for i in np.arange(nruns):
        resp_angle, resp_dist, resp_time, lv, stim_size, speed, resp_time_coll = md.calc_response_ffi(params)
        result_values = [resp_angle, resp_dist, resp_time, lv, stim_size, speed, resp_time_coll]
        for col, value in zip(data_cols, result_values):
            data_dict[col].append(value)

    model_df = pd.DataFrame(data_dict)

    model_qnt_list = []
    for lv_low, lv_high in lv_bins:
        mask = (lv_low < model_df['lv']) & (model_df['lv'] < lv_high)
        model_qnt_list.append(np.percentile(model_df['resp_angle'][mask], [10, 30, 50, 70, 90]))
    cmodel_qnt_array = np.array(model_qnt_list)

    model_quant_matrix[:, :, rep_idx] = cmodel_qnt_array

mean_model_qnt = np.mean(model_quant_matrix, axis=2)

expm_color = '#00d4f9'
model_color = '#cc0000'

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.2, wspace=0.3)
ax1 = axes[0]
ax2 = axes[1]

ax1.plot(expm_df['lv'], expm_df['resp_angle'], '.', color=expm_color, ms=18)
ax1.plot(model_df['lv'], model_df['resp_angle'], '.', color=model_color, ms=18)
ax1.set_ylim([0, 180])
ax1.set_xlim([0, 1.3])
ax1.set_xlabel('L/V [s]')
ax1.set_ylabel(r'$\theta$ [\textdegree]')

hquants_expm = ax2.plot(lv_centers - 0.05, qnt_array, ls='', marker='s', color=expm_color, ms=10)
hquants_model = ax2.plot(lv_centers + 0.05, mean_model_qnt, ls='', marker='s', color=model_color, ms=10)
# hmed = plt.plot(lv_centers, qnt_array[:, 2], color='b')
hbins = ax2.vlines(lv_bin_bounds, 0, 180, linestyles='--')
ax2.set_ylim([0, 180])
ax2.set_xlim([0, 1.3])
ax2.set_xlabel('L/V [s]')
ax2.set_ylabel(r'$\theta$ [\textdegree]')

ax2.legend([hquants_expm[0], hquants_model[0]], ['experiment', 'model'], loc='upper left', bbox_to_anchor=[-0.45, -0.05],
           frameon=True)

axes = [ax1, ax2]
letters = ['A', 'B']
for ax, letter in zip(axes, letters):
    ax.text(-0.05, 1.05, letter, color='k', weight='bold', fontsize=20, transform=ax.transAxes,
            ha='center', va='center')

#plt.show()
plt.savefig(os.path.join(figure_path, 'figure_bernstein_poster_expm_fit_comparison.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(figure_path, 'figure_bernstein_poster_expm_fit_comparison.png'), dpi=300, bbox_inches='tight')