import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import gridspec
from matplotlib.transforms import blended_transform_factory
from itertools import product
from pypet.trajectory import Trajectory
import collective_behavior_analysis as cba


def load_result(filename, vis_input_method):
    traj = Trajectory(filename=filename)

    # Now we want to load all stored data.
    traj.f_load(index=-1, load_parameters=2, load_results=1)
    traj.v_auto_load = True

    par_names = ['speed0', 'noisep', 'seed']
    filter_params = ['vis_input_method']
    filter_func = lambda vis_method: vis_method == vis_input_method

    result_specs = {'names': ['startle_freq'],
                    'funcs': [cba.calcStartlingFrequencyWithBurning],
                    'input_variables': [['results.outdata.crun.startle', 'par.total_time',
                                         'par.output']]}
    time_result_specs = {'names': ['pol', 'coh'],
                         'funcs': [cba.calcPolarization, cba.get_calcCohesion('nearest')],
                         'input_variables': [['results.outdata.crun.uw'],
                                             ['results.outdata.crun.pos']]}

    res, time_res, ranges, unique_vals, lengths = cba.collect_filtered_results(traj, par_names,
                                                                               result_specs,
                                                                               time_result_specs,
                                                                               filter_params,
                                                                               filter_func)
    return res, time_res, ranges, unique_vals, lengths


filename = os.path.join(os.path.expanduser('/extra/swarmstartle_results'),
                        'looming_swarm_fitted_model_high_resolution.hdf5')

figure_path = '../figures/results/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

fig = plt.figure(figsize=(12, 8))
grid = gridspec.GridSpec(1, 3, wspace=0.4, hspace=0.2)

for vis_method in ['max', 'mean', 'knn_mean', 'mean_deviate', 'knn_mean_deviate']:
    res, time_res, ranges, unique_vals, lengths = load_result(filename, vis_method)

    st_freqs = np.mean(res['startle_freq'], axis=-1)
    st_freqs[st_freqs == 0] = 1e-10
    pol = np.mean(np.mean(time_res['pol'], axis=-1), axis=-1)
    coh = np.mean(np.mean(time_res['coh'], axis=-1), axis=-1)

    norms = [colors.LogNorm(vmin=1e-3, vmax=2e1), colors.Normalize(vmin=1, vmax=5),
             colors.Normalize(vmin=0, vmax=1)]
    plot_cmap = plt.cm.get_cmap('viridis', 12)
    res_names = ['startle frequency', 'cohesion', 'polarization']
    titles = ['startle frequency', 'average NN distance', 'polarization']
    speeds = unique_vals['speed0']
    noises = unique_vals['noisep']
    for res_idx, result_mat in enumerate([st_freqs, coh, pol]):
        ax = plt.Subplot(fig, grid[0, res_idx])
        img = ax.imshow(result_mat[:, :], interpolation='none', cmap=plot_cmap,
                        origin='lower', aspect='equal', norm=norms[res_idx])
        ax.set_yticks(np.arange(len(speeds)))
        ax.set_yticklabels(np.round(speeds, decimals=2))
        ax.set_xticks(np.arange(len(noises), step=2))
        ax.set_xticklabels(np.round(noises[0::2], decimals=2))
        if ax.is_first_row():
            ax.set_title(titles[res_idx], fontsize=12, fontweight='bold')
        if ax.is_last_row():
            ax.set_xlabel('Noise on swimming direction', fontsize=12)
        if ax.is_first_col():
            ax.set_ylabel('Mean speed\n[Bodylengths per seconds]', fontsize=12)
        fig.add_subplot(ax)
        plt.colorbar(img, ax=ax, shrink=0.3)
        fig.savefig(os.path.join(figure_path, 'highres_looming_swarm_full_run_all_measures_periodic_init_burn_vis_method=' + vis_method + '.pdf'), bbox_inches='tight')
        plt.close(fig)
















