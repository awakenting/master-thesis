import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import gridspec

from pypet.trajectory import Trajectory

from .. import analysis_collective as cba

figure_path = '../../figures/results/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

filename = os.path.join(os.path.expanduser('/extra/swarmstartle_results'),
                        'looming_swarm_fitted_model_high_resolution_rep_range_no_speed_noise.hdf5')


def load_result(filename, target_rep_range, target_int_type):
    traj = Trajectory(filename=filename)

    # Now we want to load all stored data.
    traj.f_load(index=-1, load_parameters=2, load_results=1)
    traj.v_auto_load = True

    par_names = ['speed0', 'noisep', 'seed']
    filter_params = ['reprange', 'int_type']
    filter_func = lambda rep_range, int_type: rep_range == target_rep_range and int_type == target_int_type

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


for target_int_type in ['voronoi_matrix', 'matrix']:

    fig = plt.figure(figsize=(16, 12))
    grid = gridspec.GridSpec(4, 3, wspace=0.1, hspace=0.4)

    for rep_range_idx, rep_range in enumerate([1.0, 1.5, 2.0, 2.5]):
        res, time_res, ranges, unique_vals, lengths = load_result(filename, rep_range, target_int_type)

        st_freqs = np.median(res['startle_freq'], axis=-1)
        st_freqs[st_freqs == 0] = 1e-10
        pol = np.mean(np.mean(time_res['pol'], axis=-1), axis=-1)
        coh = np.mean(np.mean(time_res['coh'], axis=-1), axis=-1)

        norms = [colors.Normalize(vmin=12, vmax=22), colors.Normalize(vmin=1, vmax=5),
                 colors.Normalize(vmin=0, vmax=1)]
        plot_cmap = plt.cm.get_cmap('viridis', 12)
        res_names = ['startle frequency', 'cohesion', 'polarization']
        titles = ['median st. freq.', 'NN distance', 'pol']
        speeds = unique_vals['speed0']
        noises = unique_vals['noisep']
        for res_idx, result_mat in enumerate([st_freqs, coh, pol]):
            ax = plt.Subplot(fig, grid[rep_range_idx, res_idx])
            img = ax.imshow(result_mat[:, :], interpolation='none', cmap=plot_cmap,
                            origin='lower', aspect='equal', norm=norms[res_idx])
            ax.set_yticks(np.arange(len(speeds)))
            ax.set_yticklabels(np.round(speeds, decimals=2))
            ax.set_xticks(np.arange(len(noises), step=2))
            ax.set_xticklabels(np.round(noises[0::2], decimals=2))

            rounded_noises = np.round(noises, decimals=2)
            rounded_speeds = np.round(speeds, decimals=2)
            rounded_result_mat = np.round(result_mat, decimals=1)
            for speed_idx in range(len(speeds)):
                for noise_idx in range(len(noises)):
                    # note that data coordinates are cartesian (x,y) but the according array values are in (row, column)
                    # notation so that the value at (x,y) will be at array[y,x]:
                    ax.text(noise_idx, speed_idx, str(rounded_result_mat[speed_idx, noise_idx]), color='r', weight='bold', fontsize=6,
                            ha='center', va='center', alpha=0.5)
            #if ax.is_first_row():
            ax.set_title('reprange: {}, measure: {}'.format(rep_range, titles[res_idx]), fontsize=12, fontweight='bold')
            if ax.is_last_row():
                ax.set_xlabel('Noise on swimming direction', fontsize=12)
            if ax.is_first_col():
                ax.set_ylabel('Mean speed\n[Bodylengths per seconds]', fontsize=12)
            fig.add_subplot(ax)
            plt.colorbar(img, ax=ax, shrink=0.7)
            fig.savefig(os.path.join(figure_path, 'looming_swarm_no_speed_noise_int_type=' + target_int_type + '.pdf'), bbox_inches='tight')
            plt.close(fig)

#plt.show()
















