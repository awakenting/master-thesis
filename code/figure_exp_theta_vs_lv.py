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
    "text.usetex": True,  # use inline math for ticks
}

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
import pandas as pd

figure_path = '../figures/results/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

sns.set()
sns.set_palette('colorblind')
mpl.rcParams.update(custon_pgf_rcparams)

data = sio.loadmat('../data/external/LVsVersusSubtendedAngle.mat')
clean_dict = {'lv': np.squeeze(data['LVs']), 'resp_angle': np.squeeze(data['subtendedAngleAtResponse'])}
df = pd.DataFrame(clean_dict)

# preuss
lv_vals = np.array([0.075, 0.036, 0.02, 0.044, 0.055, 0.11, 0.03])
theta_vals = np.array([28, 24, 14, 21, 19, 22, 16])

# temizer (in ms)
temizer_lv_values = np.array([60, 60, 60, 60, 60, 60, 60,
                              120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
                              180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180,
                              240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240,
                              300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300])
temizer_theta_values = np.array([18, 18, 35, 35, 40, 65, 80,
                                 10, 15, 15, 15, 20, 20, 25, 25, 25, 25, 25, 30,
                                 15, 20, 20, 20, 20, 20, 20, 22, 25, 30, 30, 32, 40,
                                 10, 18, 18, 18, 18, 18, 20, 20, 20, 20, 25, 30, 30, 45,
                                 15, 18, 18, 18, 18, 20, 20, 20, 20, 25, 25, 35, 35, 60])

# dunn (in ms)
dunn_lv_values = np.array([510, 980, 1460, 1960, 2900])
dunn_theta_values = np.array([72, 68, 75, 68, 75])


plt.figure(figsize=(10, 8))
plot_ms = 16
plt.plot(lv_vals, theta_vals, '.', ms=plot_ms+6, label='Preuss et al. 2006')
plt.plot(temizer_lv_values/1000, temizer_theta_values, '.', ms=plot_ms+4, label='Temizer et al. 2015')
plt.plot(dunn_lv_values/1000, dunn_theta_values, '.', ms=plot_ms+6, label='Dunn et al. 2016')
plt.plot(clean_dict['lv'], clean_dict['resp_angle'], '.', ms=plot_ms, label='Bhattacharyya et al. 2017')

plt.xlabel('L/V [s]')
plt.ylabel(r'$\theta(t)$ [\textdegree]')
plt.legend()
#plt.show()
plt.savefig(os.path.join(figure_path, 'figure_expm_theta_vs_lv.pdf'), bbox_inches='tight')