import os
import numpy as np
import matplotlib.pyplot as plt
from pypet.trajectory import Trajectory


def calcStartlingFrequencyWithBurning(startles, total_time, output, time_to_burn=100):
    """
    Calculates the startling frequency while burning an initial time segment.

    :param startles: shape=(n_output_timesteps, nagents)
        An array with booleans for each agent and time point.
    :return:
    """
    first_unburned_idx = int(time_to_burn/output)
    unburned_indices = np.arange(first_unburned_idx, int(total_time/output))
    startle_freq = np.sum(np.take(startles, unburned_indices, axis=0)) / (total_time - time_to_burn)
    return startle_freq

def calcStartlingFrequency(startles, total_time):
    """
    Calculates the startling frequency.

    :param startles:
        An array with booleans for each agent and time point.
    :return:
    """
    startle_freq = np.sum(startles)/total_time
    return startle_freq

def calcStartlingRates(startles):
    """
    Calculates the startling frequency.

    :param startles:
        An array with booleans for each agent and time point.
    :return:
    """
    startle_rates = np.sum(startles, axis=-1)
    return startle_rates

def calcPolarization(direction_vecs):
    """
    Calculates the group polarization.

    We use the definition by Couzin et. al 2002 here:
    p_group(t) = 1/N * |sum from {i=1} to {N} (v_i(t)|

    where N is the number of agents and v_i is a unit direction vector.

    :param outData:
        The simulation output as defined in SwarmStartle.
    :return:
    """
    mean_ux = np.mean(direction_vecs[:, :, 0], axis=1)
    mean_uy = np.mean(direction_vecs[:, :, 1], axis=1)
    return np.sqrt(mean_ux**2 + mean_uy**2)


def calcCascadeSizes(startles):
    """Calculates the sizes of startle cascades.

    Parameters
    ----------
    startles
        An array with booleans for each agent and time point.

    Returns
    -------

    """
    ntimesteps = startles.shape[0]
    ncascades = 0
    cascade_sizes = []
    cascade_lengths = []
    single_startle = False
    ongoing_cascade = False
    current_cascade_members = []
    previous_startle_idc = []
    current_cascade_length = 0
    for t in np.arange(ntimesteps):
        startle_idc = np.where(startles[t, :])[0]
        # no startles at current time step:
        if startle_idc.size == 0:
            if ongoing_cascade:
                cascade_sizes.append(len(current_cascade_members))
                cascade_lengths.append(current_cascade_length)
                ncascades += 1
                ongoing_cascade = False
                current_cascade_members = []
            else:
                if single_startle:
                    single_startle = False
                    current_cascade_members = []
                    current_cascade_length = 0
        # at least one fish startled:
        else:
            if ongoing_cascade:
                new_members = []
                for startle_idx in startle_idc:
                    if startle_idx not in previous_startle_idc:
                        new_members.append(startle_idx)
                if not len(new_members) == 0:
                    current_cascade_members.extend(new_members)
                current_cascade_length += 1
            else:
                if single_startle:
                    new_members = []
                    for startle_idx in startle_idc:
                        if startle_idx not in previous_startle_idc:
                            new_members.append(startle_idx)
                    if not len(new_members) == 0:
                        single_startle = False
                        ongoing_cascade = True
                        current_cascade_members.extend(new_members)
                    current_cascade_length += 1
                else:
                    single_startle = True
                    current_cascade_members.extend(startle_idc)
                    current_cascade_length += 1
        previous_startle_idc = startle_idc
    return np.array(cascade_sizes), np.array(cascade_lengths)


def CalcDistVecMatrix(pos):
    """ Calculate N^2 distance matrix (d_ij)

        Returns:
        --------
        distmatrix - matrix of all pairwise distances (NxN)
        dX - matrix of all differences in x coordinate (NxN)
        dY - matrix of all differences in y coordinate (NxN)
    """
    X = np.reshape(pos[:, 0], (-1, 1))
    Y = np.reshape(pos[:, 1], (-1, 1))
    dX = np.subtract(X, X.T)
    dY = np.subtract(Y, Y.T)
    dX_period = np.copy(dX)
    dY_period = np.copy(dY)

    distmatrix = np.sqrt(dX_period ** 2 + dY_period ** 2)
    return distmatrix, dX_period, dY_period


def calcCohesion(pos, method='nearest'):
    from scipy.spatial import ConvexHull
    ntimesteps = pos.shape[0]
    coh = np.empty(ntimesteps)
    for t in np.arange(ntimesteps):
        if method == 'nearest':
            dist_mat, dx, dy = CalcDistVecMatrix(pos[t, :, :])
            np.fill_diagonal(dist_mat, np.inf)
            min_dists = np.min(dist_mat, axis=0)
            current_cohesion = np.mean(min_dists)
        elif method == 'convexhull':
            hull = ConvexHull(pos[t, :, :])
            # volume is referring to a 3D setting so in a 2D case it gives the area
            current_cohesion = hull.volume
        elif method == 'inter':
            dist_mat, dx, dy = CalcDistVecMatrix(pos[t, :, :])
            np.fill_diagonal(dist_mat, np.nan)
            mean_dists = np.nanmean(dist_mat, axis=0)
            current_cohesion = np.mean(mean_dists)
        else:
            current_cohesion = 0
        coh[t] = current_cohesion
    return coh


def get_calcCohesion(method='nearest'):
    if method == 'nearest':
        def cohesion_func(pos):
            ntimesteps = pos.shape[0]
            coh = np.empty(ntimesteps)
            for t in np.arange(ntimesteps):
                dist_mat, dx, dy = CalcDistVecMatrix(pos[t, :, :])
                np.fill_diagonal(dist_mat, np.inf)
                min_dists = np.min(dist_mat, axis=0)
                coh[t] = np.mean(min_dists)
            return coh
    elif method == 'convexhull':
        def cohesion_func(pos):
            from scipy.spatial import ConvexHull
            ntimesteps = pos.shape[0]
            coh = np.empty(ntimesteps)
            for t in np.arange(ntimesteps):
                hull = ConvexHull(pos[t, :, :])
                # volume is referring to a 3D setting so in a 2D case it gives the area
                coh[t] = hull.volume
            return coh
    elif method == 'inter':
        def cohesion_func(pos):
            ntimesteps = pos.shape[0]
            coh = np.empty(ntimesteps)
            for t in np.arange(ntimesteps):
                dist_mat, dx, dy = CalcDistVecMatrix(pos[t, :, :])
                np.fill_diagonal(dist_mat, np.nan)
                mean_dists = np.nanmean(dist_mat, axis=0)
                coh[t] = np.mean(mean_dists)
            return coh
    return cohesion_func


def getPositions(pos):
    return pos


def value_to_idx(val_range, unique_values, run_idx):
    """Return the index that belongs to the value at run index.


    Parameters
    ----------
    range
    unique_values
    run_idx

    Returns
    -------

    """
    return np.where(unique_values == val_range[run_idx])[0]


def collect_results(traj, index_param_names, result_specs, time_result_specs):
    # get param ranges and their unique values
    ranges = {}
    unique_vals = {}
    lengths = []
    for param_name in index_param_names:
        ranges[param_name] = traj.par.f_get(param_name).f_get_range()
        unique_vals[param_name] = np.unique(ranges[param_name])
        lengths.append(len(unique_vals[param_name]))

    # initiate result arrays
    results = {}
    for result_name in result_specs['names']:
        results[result_name] = np.empty(shape=lengths)

    # initiate result arrays for variables over time
    total_time = traj.par.total_time
    output_step = traj.par.output
    timesteps = int(total_time / output_step + 1)

    time_results = {}
    time_shape = lengths.copy()
    time_shape.append(timesteps)
    for result_name in time_result_specs['names']:
        time_results[result_name] = np.empty(shape=time_shape)

    # fill result arrays
    for run_idx, run_name in enumerate(traj.f_get_run_names(sort=True)):
        traj.f_set_crun(run_name)

        crun_indeces = []
        for param_name in index_param_names:
            crun_indeces.append(value_to_idx(ranges[param_name], unique_vals[param_name], run_idx))

        # fill result arrays
        for result_name, result_func, input_vars in zip(result_specs['names'],
                                                        result_specs['funcs'],
                                                        result_specs['input_variables']):
            input_values = []
            for var in input_vars:
                input_values.append(traj.f_get(var, fast_access=True, auto_load=True))

            results[result_name][tuple(crun_indeces)] = result_func(*input_values)

        # fill result arrays over time
        for result_name, result_func, input_vars in zip(time_result_specs['names'],
                                                        time_result_specs['funcs'],
                                                        time_result_specs['input_variables']):
            input_values = []
            for var in input_vars:
                input_values.append(traj.f_get(var, fast_access=True, auto_load=True))

            # assuming that e.g. crun_indeces = [1, 2, 3], the following line
            # of code produces a slicing object that is equal to
            # the notation [1, 2, 3, :], which is what we need to fill the
            # full range of the last dimension (the time dim in our case)
            time_indeces = tuple(crun_indeces + [slice(None)])
            time_results[result_name][time_indeces] = result_func(*input_values)

    # resetting trajectory to the default settings
    traj.f_restore_default()

    return results, time_results, ranges, unique_vals, lengths


def collect_filtered_results(traj, index_param_names, result_specs, time_result_specs,
                             filtered_param_names, filter_function):
    # get param ranges and their unique values
    ranges = {}
    unique_vals = {}
    lengths = []
    for param_name in index_param_names:
        ranges[param_name] = traj.par.f_get(param_name).f_get_range()
        unique_vals[param_name] = np.unique(ranges[param_name])
        lengths.append(len(unique_vals[param_name]))

    # initiate result arrays
    results = {}
    for result_name in result_specs['names']:
        results[result_name] = np.empty(shape=lengths)

    # initiate result arrays for variables over time
    total_time = traj.par.total_time
    output_step = traj.par.output
    timesteps = int(total_time / output_step + 1)

    time_results = {}
    time_shape = lengths.copy()
    time_shape.append(timesteps)
    for result_name in time_result_specs['names']:
        time_results[result_name] = np.empty(shape=time_shape)

    idx_iterator = traj.f_find_idx(filtered_param_names, filter_function)

    # fill result arrays
    for run_idx in idx_iterator:
        traj.v_idx = run_idx
        run_name = traj.v_crun

        crun_indeces = []
        for param_name in index_param_names:
            crun_indeces.append(value_to_idx(ranges[param_name], unique_vals[param_name], run_idx))

        # fill result arrays
        for result_name, result_func, input_vars in zip(result_specs['names'],
                                                        result_specs['funcs'],
                                                        result_specs['input_variables']):
            input_values = []
            for var in input_vars:
                input_values.append(traj.f_get(var, fast_access=True, auto_load=True))

            results[result_name][tuple(crun_indeces)] = result_func(*input_values)

        # fill result arrays over time
        for result_name, result_func, input_vars in zip(time_result_specs['names'],
                                                        time_result_specs['funcs'],
                                                        time_result_specs['input_variables']):
            input_values = []
            for var in input_vars:
                input_values.append(traj.f_get(var, fast_access=True, auto_load=True))

            # assuming that e.g. crun_indeces = [1, 2, 3], the following line
            # of code produces a slicing object that is equal to
            # the notation [1, 2, 3, :], which is what we need to fill the
            # full range of the last dimension (the time dim in our case)
            time_indeces = tuple(crun_indeces + [slice(None)])
            time_results[result_name][time_indeces] = result_func(*input_values)

    # resetting trajectory to the default settings
    traj.f_restore_default()

    return results, time_results, ranges, unique_vals, lengths


def plot_result_heatmaps(result_mat, unique_vals, title='no title', save_fig=False,
                         figure_path='./results/figures/', fig_name='test'):
    startle_noises = unique_vals['noise_startle']
    nstartle_noises = len(startle_noises)
    speeds = unique_vals['speed0']
    noises = unique_vals['noisep']

    fig, axes = plt.subplots(nrows=1, ncols=nstartle_noises, figsize=(16, 8))
    # we take the mean over the different seed values (i.e. random number initializations):
    overall_max = np.max(result_mat)

    for ax, cstartle_noise, cstartle_idx in zip(axes, startle_noises, np.arange(nstartle_noises)):
        img = ax.imshow(result_mat[:, :, cstartle_idx], interpolation='none',
                        origin='lower', aspect='auto', vmax=overall_max, vmin=0)
        ax.set_yticks(np.arange(len(speeds)))
        ax.set_yticklabels(np.round(speeds, decimals=1))
        ax.set_xticks(np.arange(len(noises)))
        ax.set_xticklabels(np.round(noises, decimals=1))
        ax.set_ylabel('Mean speed [Bodylengths per seconds]')
        ax.set_xlabel('Noise on swimming direction')
        ax.set_title('startle noise = {:2.2f}'.format(cstartle_noise))
        plt.colorbar(img, ax=ax, shrink=0.8)

    plt.subplots_adjust(wspace=0.5, hspace=0.4, right=0.95, left=0.05)
    plt.suptitle(title)

    if save_fig:
        fig.savefig(os.path.join(figure_path, fig_name), dpi=120)
        plt.close(fig)
    else:
        plt.show()


def plot_general_result_heatmaps(result_mat, unique_vals, index_param_names, xlabel,
                                 ylabel, title='no title', save_fig=False,
                                 figure_path='./results/figures/', fig_name='test'):
    par1_vals = unique_vals[index_param_names[0]]
    par2_vals = unique_vals[index_param_names[1]]
    par3_vals = unique_vals[index_param_names[2]]
    npar3 = len(par3_vals)

    fig, axes = plt.subplots(nrows=1, ncols=npar3, figsize=(20, 8))
    # we take the mean over the different seed values (i.e. random number initializations):
    overall_max = np.max(result_mat)

    for ax, cpar_val, cpar_idx in zip(axes, par3_vals, np.arange(npar3)):
        img = ax.imshow(result_mat[:, :, cpar_idx], interpolation='none',
                        origin='lower', aspect='equal', vmax=overall_max, vmin=0)
        ax.set_yticks(np.arange(len(par1_vals)))
        ax.set_yticklabels(np.round(par1_vals, decimals=2))
        ax.set_xticks(np.arange(len(par2_vals)))
        ax.set_xticklabels(np.round(par2_vals, decimals=2))
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_title('{} = {:2.2f}'.format(index_param_names[2], cpar_val))
        plt.colorbar(img, ax=ax, shrink=0.8)

    plt.subplots_adjust(wspace=0.5, hspace=0.4, right=0.95, left=0.05)
    plt.suptitle(title)

    if save_fig:
        fig.savefig(os.path.join(figure_path, fig_name), dpi=120)
        plt.close(fig)
    else:
        plt.show()

if __name__ == '__main__':
    #filename = os.path.join('hdf5', 'schrecksim_test.hdf5')
    filename = os.path.join(os.path.expanduser('~/Documents/swarmstartle_results/hdf5'), 'schrecksim_base_params_v2_voronoi.hdf5')
    traj = Trajectory(filename=filename)

    # Now we want to load all stored data.
    traj.f_load(index=-1, load_parameters=2, load_results=1)
    traj.v_auto_load = True

    figure_path = './results/figures/'
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    par_names = ['speed0', 'noisep', 'gamma_startle', 'seed']
    filter_params = ['dist_pow', 'noisev']
    filter_func = lambda dist_pow, noisev: dist_pow == 3 and noisev == 0.0

    result_specs = {'names': ['startle_freq'],
                    'funcs': [calcStartlingFrequency],
                    'input_variables': [['results.outdata.crun.startle', 'par.total_time']]}
    time_result_specs = {'names': ['pol'],
                         'funcs': [calcPolarization],
                         'input_variables': [['results.outdata.crun.uw']]}

    res, time_res, ranges, unique_vals, lengths = collect_filtered_results(traj, par_names,
                                                                           result_specs,
                                                                           time_result_specs,
                                                                           filter_params,
                                                                           filter_func)
    nseeds = len(unique_vals['seed'])
    print(res['startle_freq'].shape)

    startle_freqs = res['startle_freq']
    title = 'Startle frequency (mean of {:} runs)'.format(nseeds)
    ylabel = 'Mean speed [Bodylengths per seconds]'
    xlabel = 'Noise on swimming direction'
    index_par_names = ['speed0', 'noisep', 'gamma_startle']
    startle_freqs_mean = np.mean(startle_freqs, axis=-1)
    plot_general_result_heatmaps(startle_freqs_mean, unique_vals, index_par_names, xlabel, ylabel, title=title)

    title = 'Startle frequency (variance within {:} runs)'.format(nseeds)
    startle_freqs_var = np.var(startle_freqs, axis=-1)
    plot_general_result_heatmaps(startle_freqs_var, unique_vals, index_par_names, xlabel, ylabel, title=title)

    """
    par_names = ['speed0', 'noisep', 'gamma_startle', 'dist_pow', 'noisev', 'seed']
    
    result_specs = {'names': ['startle_freq'],
                    'funcs': [calcStartlingFrequency],
                    'input_variables': [['results.outdata.crun.startle', 'par.total_time']]}
    time_result_specs = {'names': ['pol'],
                         'funcs': [calcPolarization],
                         'input_variables': [['results.outdata.crun.uw']]}
    
    res, time_res, ranges, unique_vals, lengths = collect_results(traj, par_names, result_specs, time_result_specs)
    nseeds = len(unique_vals['seed'])
    
    startle_freqs = res['startle_freq']
    title = 'Startle frequency (mean of {:} runs)'.format(nseeds)
    startle_freqs_mean = np.mean(startle_freqs, axis=-1)
    plot_result_heatmaps(startle_freqs_mean, unique_vals, title=title)
    
    title = 'Startle frequency (variance within {:} runs)'.format(nseeds)
    startle_freqs_var = np.var(startle_freqs, axis=-1)
    plot_result_heatmaps(startle_freqs_var, unique_vals, title=title)
    
    
    pol = time_res['pol']
    pol_mean = np.mean(pol, axis=-1)
    pol_var = np.var(pol, axis=-1)
    
    title_mean = 'Mean polarization over time (mean of {:} runs)'.format(nseeds)
    pol_mean_mean = np.mean(pol_mean, axis=-1)
    plot_result_heatmaps(pol_mean_mean, unique_vals, title=title_mean)
    
    title_var = 'Variance of polarization over time (mean of {:} runs)'.format(nseeds)
    pol_var_mean = np.mean(pol_var, axis=-1)
    plot_result_heatmaps(pol_var_mean, unique_vals, title=title_var)
    """