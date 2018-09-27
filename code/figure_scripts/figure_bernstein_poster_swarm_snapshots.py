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

from .. import analysis_collective as cba

figure_path = './figures/results/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)


sns.set()
sns.set_palette('colorblind')
sns_colors = sns.color_palette()
mpl.rcParams.update(custon_pgf_rcparams)

filename = '/home/andrej/Documents/looming_swarm_fitted_model_fixed_rho_null_kmd_matrix.hdf5'

traj = Trajectory(filename=filename)

# Now we want to load all stored data.
traj.f_load(index=-1, load_parameters=2, load_results=1, force=True)
traj.v_auto_load = True

timepoint = int(300/traj.par.output)
# for i in range(3):
seed_idx = 1
noise_vals = np.linspace(0.01, 0.2, 5)
vmin = np.min(0.01)
vmax = np.max(0.2)
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))



for noise_idx, noise_val in enumerate(noise_vals[[0, 4]]):
    fig = plt.figure(figsize=(12, 12))
    filter_params = ['speed0', 'noisep']

    def filter_func(speed0, noisep):
        return speed0 == 1.125 and noisep == noise_val


    idx_iterator = traj.f_find_idx(filter_params, filter_func)

    pos_data = []
    uw_data = []
    # fill result arrays
    for run_idx in idx_iterator:
        traj.v_idx = run_idx
        pos_data.append(traj.f_get('results.outdata.crun.pos', fast_access=True, auto_load=True))
        uw_data.append(traj.f_get('results.outdata.crun.uw', fast_access=True, auto_load=True))

    current_pos_data = pos_data[seed_idx]
    current_uw_data = uw_data[seed_idx]

    plt.plot(current_pos_data[timepoint, :, 0], current_pos_data[timepoint, :, 1], 'r.', ms=20)
    plt.quiver(current_pos_data[timepoint, :, 0], current_pos_data[timepoint, :, 1], current_uw_data[timepoint, :, 0], current_uw_data[timepoint, :, 1],
               width=0.005)

    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Noise = ' + str(noise_val))
    plt.xlim([0, traj.par.L])
    plt.ylim([0, traj.par.L])

    fig.savefig(os.path.join(figure_path, 'bernstein_poster_swarm_snapshot_noise=' + str(noise_val) + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(figure_path, 'bernstein_poster_swarm_snapshot_noise=' + str(noise_val) + '.png'), dpi=300,
                bbox_inches='tight')
    plt.close(fig)
#plt.show()
