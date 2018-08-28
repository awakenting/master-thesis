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
    'text.usetex': True,  # use inline math for ticks,
    'pgf.texsystem': 'lualatex'
}

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from pypet.trajectory import Trajectory

import collective_behavior_analysis as cba

figure_path = '../../figures/results/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

sns.set()
sns.set_palette('colorblind')
sns_colors = sns.color_palette()
mpl.rcParams.update(custon_pgf_rcparams)


def plot_startle_pos(ax, traj, uw_data, pos_data, startle_data, burn_period, pool_data=True):
    if pool_data:
        all_startle_distvecs = []
        all_nonstartle_distvecs = []
        all_startle_orientations = []
        all_startle_times = []
        for data_idx in range(len(uw_data)):
            startle_pos_results = cba.calcStartlePositionOriented(uw_data[data_idx],
                                                              pos_data[data_idx],
                                                              startle_data[data_idx],
                                                              traj.par.L,
                                                              traj.par.output,
                                                              burn_period,
                                                              None)
            all_dists, all_distvecs, startle_dists, startle_distvecs = startle_pos_results
            if not startle_distvecs is None:
                all_startle_distvecs.append(startle_distvecs)
                all_nonstartle_distvecs.append(all_distvecs)

            startle_orient_results = cba.calcStartleOrientation(uw_data[data_idx],
                                                            pos_data[data_idx],
                                                            startle_data[data_idx],
                                                            traj.par.L,
                                                            traj.par.output,
                                                            burn_period)
            startle_orientations, startle_frontness, all_orientations, all_frontness = startle_orient_results
            if not len(startle_orientations) == 0:
                all_startle_orientations.append(startle_orientations)

            startle_times = cba.get_startle_times(startle_data[data_idx], traj.par.output, burn_period)
            if not len(startle_times) == 0:
                all_startle_times.append(startle_times)

        all_startle_distvecs = np.concatenate(all_startle_distvecs)
        all_nonstartle_distvecs = np.concatenate(all_nonstartle_distvecs)
        all_startle_orientations = np.concatenate(all_startle_orientations)
        all_startle_times = np.concatenate(all_startle_times)

    if not startle_distvecs is None:
        # ax.plot(startle_distvecs[:, 0], startle_distvecs[:, 1], '.', alpha=0.5)

        sc = ax.scatter(all_nonstartle_distvecs[:, 0], all_nonstartle_distvecs[:, 1], marker='.', alpha=0.05,
                        label='nonstartle')
        sc = ax.scatter(all_startle_distvecs[:, 0], all_startle_distvecs[:, 1], marker='.', alpha=1, label='startle')
        ax.hlines(0, -25, 25, color='k', linestyles=':', lw=2)
        ax.vlines(0, -25, 25, color='k', linestyles=':', lw=2)
        ax.set_xlim([-25, 25])
        ax.set_ylim([-25, 25])
        ax.set_ylabel('Relative y-position')
        #ax.set_title('number of startles: ' + str(len(all_startle_distvecs)))
        if ax.is_first_row():
            leg = ax.legend(loc='upper right')
            for lh in leg.legendHandles:
                lh.set_alpha(1)



def plot_startle_angle(ax, traj, uw_data, pos_data, startle_data, burn_period, pool_data=True):
    if pool_data:
        all_nonstartle_angles = []
        all_startle_angles = []
        for data_idx in range(len(uw_data)):
            all_angles, startle_angles = cba.calcStartleAngle(uw_data[data_idx],
                                                          pos_data[data_idx],
                                                          startle_data[data_idx],
                                                          traj.par.L,
                                                          traj.par.output,
                                                          burn_period)
            if not len(startle_angles) == 0:
                all_startle_angles.append(startle_angles)
                all_nonstartle_angles.append(all_angles)

        all_startle_angles = np.concatenate(all_startle_angles)
        all_nonstartle_angles = np.concatenate(all_nonstartle_angles)

    hist, bin_edges = np.histogram(all_startle_angles, bins=50, density=True, label='startle')
    nonstartle_hist, bin_edges = np.histogram(all_nonstartle_angles, bins=50, density=True, label='nonstartle')
    bin_width = bin_edges[1] - bin_edges[0]

    ax.set_theta_zero_location(loc='N')
    ax.set_theta_direction('clockwise')
    ax.bar(bin_edges[:-1], hist, width=bin_width, align='edge', label='startle', alpha=0.5)
    ax.bar(bin_edges[:-1], nonstartle_hist, width=bin_width, align='edge', label='nonstartle', alpha=0.5)


def plot_cascade_sizes(ax, traj, startle_data, burn_period, time_margin, pool_data=True):
    if pool_data:
        all_cascade_sizes = []
        all_cascade_lengths = []
        all_starting_points = []
        for startle_dat in startle_data:
            cascade_sizes, cascade_lengths, starting_points = cba.calcCascadeSizes(startle_dat,
                                                                               traj.par.output,
                                                                               burn_period=burn_period,
                                                                               time_margin=time_margin)
            all_cascade_sizes.append(cascade_sizes)
            all_cascade_lengths.append(cascade_lengths)
            all_starting_points.append(starting_points)

        all_cascade_sizes = np.concatenate(all_cascade_sizes)
        all_cascade_lengths = np.concatenate(all_cascade_lengths)
        all_starting_points = np.concatenate(all_starting_points)
    ax.hist(all_cascade_sizes, bins=30, log=True)
    ax.set_title('number of cascadeds: ' + str(len(all_cascade_sizes)))


def plot_startle_dists(ax, traj, pos_data, startle_data, burn_period, pool_data=True):
    if pool_data:
        all_startle_dists = []
        all_nonstartle_dists = []
        for data_idx in range(len(pos_data)):
            startle_pos_results = cba.calcStartlePositionOriented(uw_data[data_idx],
                                                              pos_data[data_idx],
                                                              startle_data[data_idx],
                                                              traj.par.L,
                                                              traj.par.output,
                                                              burn_period,
                                                              None)
            all_dists, all_distvecs, startle_dists, startle_distvecs = startle_pos_results
            if not startle_distvecs is None:
                all_startle_dists.append(startle_dists)
                all_nonstartle_dists.append(all_dists)

        all_startle_dists = np.concatenate(all_startle_dists)
        all_nonstartle_dists = np.concatenate(all_nonstartle_dists)

    ax.hist(all_nonstartle_dists, bins=30, density=True, label='nonstartle', alpha=0.5)
    ax.hist(all_startle_dists, bins=30, density=True, label='startle', alpha=0.5)
    ax.set_ylabel('Density')
    ax.set_xlim([0, 30])
    if ax.is_first_row():
        leg = ax.legend(loc='upper right')
        for lh in leg.legendHandles:
            lh._alpha = 1


def plot_startle_frontness(ax, traj, pos_data, uw_data, startle_data, burn_period, pool_data=True):
    if pool_data:
        all_startle_front = []
        all_nonstartle_front = []
        for data_idx in range(len(pos_data)):
            startle_orient_results = cba.calcStartleOrientation(uw_data[data_idx],
                                                            pos_data[data_idx],
                                                            startle_data[data_idx],
                                                            traj.par.L,
                                                            traj.par.output,
                                                            burn_period)
            startle_orientations, startle_frontness, all_orientations, all_frontness = startle_orient_results

            if not startle_frontness is None:
                all_startle_front.append(startle_frontness)
                all_nonstartle_front.append(all_frontness)
        all_startle_front = np.concatenate(all_startle_front)
        all_nonstartle_front = np.concatenate(all_nonstartle_front)

    ax.hist(all_nonstartle_front, bins=30, density=True, label='nonstartle', alpha=0.5)
    ax.hist(all_startle_front, bins=30, density=True, label='startle', alpha=0.5)
    ax.set_ylabel('Density')
    ax.set_xlim([-25, 25])
    if ax.is_first_row():
        leg = ax.legend(loc='upper right')
        for lh in leg.legendHandles:
            lh._alpha = 1

filename = os.path.join(os.path.expanduser('/extra/swarmstartle_results'),
                        'looming_swarm_fitted_model_fixed_rho_null_kmd_matrix.hdf5')

traj = Trajectory(filename=filename)

# Now we want to load all stored data.
traj.f_load(index=-1, load_parameters=2, load_results=1)
traj.v_auto_load = True

seed_idx = 3
noise_vals = np.linspace(0.01, 0.2, 5)
speed_vals = np.linspace(0.5, 3.0, 5)
speed_val = speed_vals[1]

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 18))
plt.subplots_adjust(hspace=0.4, wspace=0.4, left=0.1)

for noise_idx, noise_val in enumerate(noise_vals[1:]):
    filter_params = ['speed0', 'noisep', 'int_type', 'vis_input_method']


    def filter_func(speed0, noisep, int_type, vis_method):
        return speed0 == speed_val and noisep == noise_val and int_type == 'matrix' and vis_method == 'knn_mean_deviate'


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

    plot_startle_pos(axes[noise_idx, 0], traj, uw_data[0:seed_idx], pos_data[0:seed_idx], startle_data[0:seed_idx],
                     burn_period=100)
    plot_startle_frontness(axes[noise_idx, 1], traj, pos_data[0:seed_idx], uw_data[0:seed_idx],
                           startle_data[0:seed_idx], burn_period=100)
    plot_startle_dists(axes[noise_idx, 2], traj, pos_data[0:seed_idx], startle_data[0:seed_idx], burn_period=100)

    if axes[noise_idx, 0].is_first_row():
        axes[noise_idx, 0].set_title('Relative position', weight='bold')
        axes[noise_idx, 1].set_title('"Frontness"', weight='bold')
        axes[noise_idx, 2].set_title('Relative distance', weight='bold')

    if axes[noise_idx, 0].is_last_row():
        axes[noise_idx, 0].set_xlabel('Relative x-position')
        axes[noise_idx, 1].set_xlabel('Frontness index')
        axes[noise_idx, 2].set_xlabel('Distance from center')

    row_label = 'Noise \n= {:.2f}'.format(noise_val)
    axes[noise_idx, 0].text(-0.4, 0.5, row_label, color='k', weight='bold', fontsize=14,
                            family='sans-serif', ha='center', va='center', alpha=1, transform=axes[noise_idx, 0].transAxes)

fig.savefig(os.path.join(figure_path, 'looming_swarm_startle_stats.pdf'), bbox_inches='tight')
plt.close(fig)
#plt.show()