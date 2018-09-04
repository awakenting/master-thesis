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

figure_path = './figures/results/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

sns.set()
sns.set_palette('colorblind')
sns_colors = sns.color_palette()
mpl.rcParams.update(custon_pgf_rcparams)

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
plt.figure(figsize=(12, 12))
hdata = plt.plot(expm_df['lv'], expm_df['resp_angle'], ls='', marker='.', color=sns_colors[0], ms=12, alpha=0.7)
hquants = plt.plot(lv_centers, qnt_array, ls='', marker='s', color=sns_colors[2], ms=12, alpha=0.7)
hmed = plt.plot(lv_centers, qnt_array[:, 2], color=sns_colors[2])
hbins = plt.vlines(lv_bin_bounds, 0, 180, linestyles='--')

plt.xlabel('L/V [s]')
plt.ylabel(r'$\theta_{resp}$ [\textdegree]')
plt.legend([hdata[0], hquants[0], hmed[0], hbins], ['raw data', 'quantiles', 'median', 'bins'],
           loc='center left', bbox_to_anchor=[1, 0.5], numpoints=1)

plt.savefig(os.path.join(figure_path, 'figure_data_to_quant.pdf'), bbox_inches='tight')
