import os
import numpy as np

import matplotlib as mpl

mpl.use("pgf")
general_fontsize = 14
custon_pgf_rcparams = {
    'font.family': 'serif',
    'font.serif': 'cm',
    'font.size': general_fontsize,
    'xtick.labelsize': general_fontsize,
    'ytick.labelsize': general_fontsize,
    'axes.labelsize': general_fontsize,
    'axes.titlesize': general_fontsize,
    'axes.grid': False,
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
import matplotlib.colors as colors
from matplotlib import gridspec
from pypet.trajectory import Trajectory

from .. import analysis_collective as cba

figure_path = './figures/results/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

filename = os.path.join(os.path.expanduser('/extra/swarmstartle_results'),
                        'looming_swarm_fitted_model_fixed_rho_null_explore.hdf5')

sns.set()
sns.set_palette('colorblind')
sns_colors = sns.color_palette()
mpl.rcParams.update(custon_pgf_rcparams)

def load_result(filename, target_vismethod, target_int_type):
    traj = Trajectory(filename=filename)

    # Now we want to load all stored data.
    traj.f_load(index=-1, load_parameters=2, load_results=1)
    traj.v_auto_load = True

    par_names = ['speed0', 'noisep', 'seed']
    filter_params = ['vis_input_method', 'int_type']
    filter_func = lambda vis_method, int_type: vis_method == target_vismethod and int_type == target_int_type

    result_specs = {'names': ['startle_freq'],
                    'funcs': [cba.calcStartlingFrequencyWithBurning],
                    'input_variables': [['results.outdata.crun.startle', 'par.total_time',
                                         'par.output']]}
    time_result_specs = {'names': ['pol', 'coh'],
                         'funcs': [cba.calcPolarization, cba.get_calcCohesion('nearest')],
                         'input_variables': [['results.outdata.crun.uw'],
                                             ['results.outdata.crun.pos', 'par.L', 'par.BC']]}

    res, time_res, ranges, unique_vals, lengths = cba.collect_filtered_results(traj, par_names,
                                                                               result_specs,
                                                                               time_result_specs,
                                                                               filter_params,
                                                                               filter_func)
    return res, time_res, ranges, unique_vals, lengths

target_int_type = 'matrix'

fig = plt.figure(figsize=(12, 10))
grid = gridspec.GridSpec(3, 3, wspace=0.4, hspace=0.1, left=0.2)

norms_list = [[colors.LogNorm(vmin=1e-3, vmax=1e1), colors.Normalize(vmin=1.5, vmax=4.5),
              colors.Normalize(vmin=0, vmax=1)],
              [colors.Normalize(vmin=5, vmax=12), colors.Normalize(vmin=1.5, vmax=4.5),
               colors.Normalize(vmin=0, vmax=1)],
              [colors.Normalize(vmin=15, vmax=20), colors.Normalize(vmin=1.5, vmax=4.5),
              colors.Normalize(vmin=0, vmax=1)]
              ]
plot_cmap = plt.cm.get_cmap('viridis', 12)
res_names = ['startle frequency', 'cohesion', 'polarization']
titles = ['Startle frequency', 'Average NN distance', 'Polarization']
vismethod_abbrevs = ['KMD', 'KMEAN', 'MAX']
letters = ['A', 'B', 'C']

for vismethod_idx, vis_method in enumerate(['knn_mean_deviate', 'knn_mean', 'max']):
    res, time_res, ranges, unique_vals, lengths = load_result(filename, vis_method, target_int_type)

    st_freqs = np.mean(res['startle_freq'], axis=-1)
    st_freqs[st_freqs == 0] = 1e-10
    pol = np.mean(np.mean(time_res['pol'], axis=-1), axis=-1)
    coh = np.mean(np.mean(time_res['coh'], axis=-1), axis=-1)

    speeds = unique_vals['speed0']
    noises = unique_vals['noisep']

    axes = []
    for res_idx, result_mat in enumerate([st_freqs, coh, pol]):
        ax = plt.Subplot(fig, grid[vismethod_idx, res_idx])
        img = ax.imshow(result_mat[:, :], interpolation='none', cmap=plot_cmap,
                        origin='lower', aspect='equal', norm=norms_list[vismethod_idx][res_idx])
        ax.set_yticks(np.arange(len(speeds)))
        ax.set_yticklabels(np.round(speeds, decimals=2))
        ax.set_xticks(np.arange(len(noises), step=2))
        ax.set_xticklabels(np.round(noises[0::2], decimals=2))
        if ax.is_first_row():
            ax.set_title(titles[res_idx])
        if ax.is_last_row():
            ax.set_xlabel('Noise on swimming direction')
        if ax.is_first_col():
            ax.set_ylabel('Mean speed [BL/s]')
        rounded_result_mat = np.round(result_mat, decimals=2)
        for speed_idx in np.arange(len(speeds)+1, step=2):
            for noise_idx in np.arange(len(noises)+1, step=2):
        # note that data coordinates are cartesian (x,y) but the according array values are in (row, column)
        # notation so that the value at (x,y) will be at array[y,x]:
                ax.text(noise_idx, speed_idx, str(rounded_result_mat[speed_idx, noise_idx]), color=sns_colors[2],
                        weight='bold', fontsize=8, family='sans-serif',
                        ha='center', va='center', alpha=1)

        if res_idx == 0:
            ax.text(-0.7, 0.5, vismethod_abbrevs[vismethod_idx], color='k', weight='bold', fontsize=14,
                    family='sans-serif', ha='center', va='center', alpha=1, transform=ax.transAxes)
            ax.text(-0.1, 1.1, letters[vismethod_idx], color='k', weight='bold', fontsize=20, transform=ax.transAxes,
                    ha='center', va='center')

        fig.add_subplot(ax)
        if ax.is_first_col():
            axes.append(ax)
        plt.colorbar(img, ax=ax, shrink=0.7)

fig.savefig(os.path.join(figure_path, 'looming_swarm_fixed_rhonull_int_type=' + target_int_type + '_comparison_new.pdf'), bbox_inches='tight')
plt.close(fig)
#plt.show()















