import os
import numpy as np

import matplotlib as mpl

mpl.use("pgf")
general_fontsize = 16
custon_pgf_rcparams = {
    'font.family': 'serif',
    'font.serif': 'cm',
    'font.size': general_fontsize,
    'xtick.labelsize': general_fontsize,
    'ytick.labelsize': general_fontsize,
    'axes.labelsize': general_fontsize,
    'axes.titlesize': general_fontsize,
    'axes.grid': True,
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

from pypet.trajectory import Trajectory

import collective_behavior_analysis as cba

figure_path = '../../figures/results/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

sns.set()
sns.set_palette('colorblind')
sns_colors = sns.color_palette()
mpl.rcParams.update(custon_pgf_rcparams)

filename = os.path.join(os.path.expanduser('/extra/swarmstartle_results'),
                        'looming_swarm_fitted_model_fixed_rho_null_kmd_matrix.hdf5')

traj = Trajectory(filename=filename)

# Now we want to load all stored data.
traj.f_load(index=-1, load_parameters=2, load_results=1)
traj.v_auto_load = True

starttime = -5
endtime = 505
# for i in range(3):
seed_idx = 1
noise_vals = np.linspace(0.01, 0.2, 5)
vmin = np.min(0.01)
vmax = np.max(0.2)
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 3, 1, 3]})
plt.subplots_adjust(hspace=0.4)

for noise_idx, noise_val in enumerate(noise_vals[[2, 4]]):
    filter_params = ['speed0', 'noisep']

    def filter_func(speed0, noisep):
        return speed0 == 1.125 and noisep == noise_val


    idx_iterator = traj.f_find_idx(filter_params, filter_func)

    pos_data = []
    uw_data = []
    vis_method_data = []
    startle_data = []
    vm_data = []
    visangle_data = []
    # fill result arrays
    for run_idx in idx_iterator:
        traj.v_idx = run_idx
        pos_data.append(traj.f_get('results.outdata.crun.pos', fast_access=True, auto_load=True))
        uw_data.append(traj.f_get('results.outdata.crun.uw', fast_access=True, auto_load=True))
        startle_data.append(traj.f_get('results.outdata.crun.startle', fast_access=True, auto_load=True))

    pol = cba.calcPolarization(uw_data[seed_idx])
    ax1 = axes[0, noise_idx]
    ax1.plot(np.arange(len(pol)) * traj.par.output, pol, color=sns_colors[noise_idx])
    ax1.fill_between([0, 50], [0, 0], [1, 1], color='r', alpha=0.5, label='burned period')
    ax1.set_xlim([starttime, endtime])
    ax1.set_title('Noise = ' + str(noise_val))
    ax1.legend(loc='lower right')
    if noise_idx == 0:
        ax1.set_ylabel('Polarization')
    # ax1.set_ylim([0.7, 1])

    startles = np.array(startle_data[seed_idx])
    startle_rows = [np.where(startles[:, row_idx]) for row_idx in range(startles.shape[1])]
    ax2 = axes[1, noise_idx]
    for row_idx in range(startles.shape[1]):
        ax2.eventplot(np.array(startle_rows[row_idx]) * traj.par.output, orientation='horizontal', color=sns_colors[noise_idx],
                      lineoffsets=row_idx)
    if noise_idx == 0:
        ax2.set_ylabel('Fish index')
    ax2.set_xlim([starttime, endtime])
    ax2.set_title('Startle events')

    stlsum = np.sum(startles, axis=1)
    stlseries = pd.Series(stlsum)
    stlkde = stlseries.rolling(window=1000, win_type='gaussian', center=True, closed='both', min_periods=100)
    dat = stlkde.mean(std=50)
    ax3 = axes[2, noise_idx]
    ax3.fill_between(np.arange(len(stlseries)) * traj.par.output, dat.fillna(0), color=sns_colors[noise_idx], alpha=0.5)
    ax3.set_xlim([starttime, endtime])
    if noise_idx == 0:
        ax3.set_ylabel('Event frequency')

    coh = cba.calcCohesion(np.array(pos_data[seed_idx]), L=traj.par.L, BC=traj.par.BC, method='nearest')
    ax4 = axes[3, noise_idx]
    ax4.plot(np.arange(len(coh)) * traj.par.output, coh, color=sns_colors[noise_idx])
    ax4.set_xlabel('Time [s]')
    if noise_idx == 0:
        ax4.set_ylabel('Avg. NN dist.')
    ax4.set_xlim([starttime, endtime])

letters = ['A', 'B']
axes_selection = [axes[0, 0], axes[0, 1]]
for ax, letter in zip(axes_selection, letters):
    ax.text(-0.1, 1.1, letter, color='k', weight='bold', fontsize=20, transform=ax.transAxes,
            ha='center', va='center')

fig.savefig(os.path.join(figure_path, 'looming_swarm_over_time.pdf'), bbox_inches='tight')
plt.close(fig)
#plt.show()
