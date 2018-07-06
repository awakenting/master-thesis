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

def calcMedianStartlingFrequency(startles, total_time, output, time_to_burn=100):
    """
    Calculates the startling frequency while burning an initial time segment.

    :param startles: shape=(n_output_timesteps, nagents)
        An array with booleans for each agent and time point.
    :return:
    """
    first_unburned_idx = int(time_to_burn/output)
    unburned_indices = np.arange(first_unburned_idx, int(total_time/output))
    agent_startle_freq = np.sum(np.take(startles, unburned_indices, axis=0), axis=0) / (total_time - time_to_burn)
    median_startle_freq = np.median(agent_startle_freq)
    return median_startle_freq

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


def CalcDistVecMatrix(pos, L, BC):
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
    if BC == 0:
        dX_period[dX > +0.5 * L] -= L
        dY_period[dY > +0.5 * L] -= L
        dX_period[dX < -0.5 * L] += L
        dY_period[dY < -0.5 * L] += L
    distmatrix = np.sqrt(dX_period ** 2 + dY_period ** 2)
    return distmatrix, dX_period, dY_period


def PeriodicDist(x, y, L=10.0, dim=2):
    """ Returns the distance vector of two position vectors x,y
        by tanking periodic boundary conditions into account.

        Input parameters: L - system size, dim - number of dimensions
    """
    distvec = (y - x)
    distvec_periodic = np.copy(distvec)
    distvec_periodic[distvec < -0.5 * L] += L
    distvec_periodic[distvec > 0.5 * L] -= L

    return distvec_periodic


def calc_periodic_mass_center(pos, arena_size):
    periodic_center = np.zeros(2)
    norm_term = 2*np.pi/arena_size
    for dim_idx in range(2):
        circ_coords = np.zeros((pos.shape[0], 2))
        circ_coords[:, 0] = np.cos((pos[:, dim_idx] * norm_term)) / norm_term
        circ_coords[:, 1] = np.sin((pos[:, dim_idx] * norm_term)) / norm_term
        mean_circ_coords = np.mean(circ_coords, axis=0)
        mean_angle = np.arctan2(-mean_circ_coords[1], -mean_circ_coords[0]) + np.pi
        periodic_center[dim_idx] = mean_angle/norm_term
    return periodic_center


def calcCohesion(pos, L, BC, method='nearest'):
    from scipy.spatial import ConvexHull
    ntimesteps = pos.shape[0]
    coh = np.empty(ntimesteps)
    for t in np.arange(ntimesteps):
        if method == 'nearest':
            dist_mat, dx, dy = CalcDistVecMatrix(pos[t, :, :], L, BC)
            np.fill_diagonal(dist_mat, np.inf)
            min_dists = np.min(dist_mat, axis=0)
            current_cohesion = np.mean(min_dists)
        elif method == 'convexhull':
            hull = ConvexHull(pos[t, :, :])
            # volume is referring to a 3D setting so in a 2D case it gives the area
            current_cohesion = hull.volume
        elif method == 'convexhull_periodic':
            cmass_center = calc_periodic_mass_center(pos[t, :, :], L)
            center_distvecs = PeriodicDist(cmass_center, pos[t, :, :], L)
            hull = ConvexHull(center_distvecs)
            # volume is referring to a 3D setting so in a 2D case it gives the area
            current_cohesion = hull.volume
        elif method == 'inter':
            dist_mat, dx, dy = CalcDistVecMatrix(pos[t, :, :], L, BC)
            np.fill_diagonal(dist_mat, np.nan)
            mean_dists = np.nanmean(dist_mat, axis=0)
            current_cohesion = np.mean(mean_dists)
        else:
            current_cohesion = 0
        coh[t] = current_cohesion
    return coh


def get_calcCohesion(method='nearest'):
    if method == 'nearest':
        def cohesion_func(pos, L, BC):
            ntimesteps = pos.shape[0]
            coh = np.empty(ntimesteps)
            for t in np.arange(ntimesteps):
                dist_mat, dx, dy = CalcDistVecMatrix(pos[t, :, :], L, BC)
                np.fill_diagonal(dist_mat, np.inf)
                min_dists = np.min(dist_mat, axis=0)
                coh[t] = np.mean(min_dists)
            return coh
    elif method == 'convexhull':
        def cohesion_func(pos, L, BC):
            from scipy.spatial import ConvexHull
            ntimesteps = pos.shape[0]
            coh = np.empty(ntimesteps)
            for t in np.arange(ntimesteps):
                hull = ConvexHull(pos[t, :, :])
                # volume is referring to a 3D setting so in a 2D case it gives the area
                coh[t] = hull.volume
            return coh
    elif method == 'inter':
        def cohesion_func(pos, L, BC):
            ntimesteps = pos.shape[0]
            coh = np.empty(ntimesteps)
            for t in np.arange(ntimesteps):
                dist_mat, dx, dy = CalcDistVecMatrix(pos[t, :, :], L, BC)
                np.fill_diagonal(dist_mat, np.nan)
                mean_dists = np.nanmean(dist_mat, axis=0)
                coh[t] = np.mean(mean_dists)
            return coh
    return cohesion_func


def calcStartlePositionOriented(uw, pos, startles, L, output_step, burn_period=50, cohesion_measures=None):
    burn_period_steps = int(burn_period / output_step)
    ntimesteps = pos.shape[0]
    all_dists = np.array([])
    all_distvecs = None
    startle_dists = np.array([])
    startle_distvecs = None
    for t in np.arange(burn_period_steps, ntimesteps):
        cmass_center = calc_periodic_mass_center(pos[t, :, :], L)
        center_distvecs = PeriodicDist(cmass_center, pos[t, :, :], L)
        if cohesion_measures is not None:
            center_distvecs = center_distvecs / cohesion_measures[t]

        mean_uw = np.mean(uw[t, :, :], axis=0)
        mean_uw_unit = mean_uw / np.linalg.norm(mean_uw)
        relative_ypos = center_distvecs[:, 0] * mean_uw_unit[0] + center_distvecs[:, 1] * mean_uw_unit[1]

        orth_mean_uw_unit = np.array([mean_uw_unit[1], - mean_uw_unit[0]])
        relative_xpos = center_distvecs[:, 0] * orth_mean_uw_unit[0] + center_distvecs[:, 1] * orth_mean_uw_unit[1]

        relative_pos = np.concatenate((relative_xpos[:, np.newaxis], relative_ypos[:, np.newaxis]), axis=1)

        center_dists = np.sqrt(center_distvecs[:, 0] ** 2 + center_distvecs[:, 1] ** 2)

        startle_idc = np.where(startles[t, :])[0]
        nonstartle_mask = np.ones(len(startles[t, :]), dtype=np.bool)
        nonstartle_mask[startle_idc] = 0
        if startle_idc.size == 0:
            continue
        else:
            startle_dists = np.concatenate((startle_dists, center_dists[startle_idc]))
            if startle_distvecs is None:
                startle_distvecs = relative_pos[startle_idc]
            else:
                startle_distvecs = np.concatenate((startle_distvecs, relative_pos[startle_idc]))

        all_dists = np.concatenate((all_dists, center_dists[nonstartle_mask]))
        if all_distvecs is None:
            all_distvecs = relative_pos[nonstartle_mask]
        else:
            all_distvecs = np.concatenate((all_distvecs, relative_pos[nonstartle_mask]))
    return all_dists, all_distvecs, startle_dists, startle_distvecs


def calcStartleAngle(uw, pos, startles, L, output_step, burn_period=50):
    burn_period_steps = int(burn_period / output_step)
    ntimesteps = pos.shape[0]
    all_angles = np.array([])
    startle_angles = np.array([])
    for t in np.arange(burn_period_steps, ntimesteps):
        cmass_center = calc_periodic_mass_center(pos[t, :, :], L)
        center_distvecs = PeriodicDist(cmass_center, pos[t, :, :], L)

        mean_uw = np.mean(uw[t, :, :], axis=0)
        mean_uw_unit = mean_uw / np.linalg.norm(mean_uw)

        mean_uw_angle = np.arctan2(mean_uw_unit[1], mean_uw_unit[0]) + np.pi
        angles = mean_uw_angle - (np.arctan2(center_distvecs[:, 1], center_distvecs[:, 0]) + np.pi)

        startle_idc = np.where(startles[t, :])[0]
        nonstartle_mask = np.ones(len(startles[t, :]), dtype=np.bool)
        nonstartle_mask[startle_idc] = 0
        if startle_idc.size == 0:
            continue
        else:
            startle_angles = np.concatenate((startle_angles, angles[startle_idc]))

        all_angles = np.concatenate((all_angles, angles[nonstartle_mask]))
    return all_angles, startle_angles


def calcStartleOrientation(uw, pos, startles, L, output_step, burn_period=50):
    burn_period_steps = int(burn_period / output_step)
    ntimesteps = pos.shape[0]
    all_orientations = np.array([])
    all_frontness = np.array([])
    startle_orientations = np.array([])
    startle_frontness = np.array([])
    for t in np.arange(burn_period_steps, ntimesteps):
        startle_idc = np.where(startles[t, :])[0]
        nonstartle_mask = np.ones(len(startles[t, :]), dtype=np.bool)
        nonstartle_mask[startle_idc] = 0
        if startle_idc.size == 0:
            continue
        else:
            cmass_center = calc_periodic_mass_center(pos[t, :, :], L)
            center_distvecs = PeriodicDist(cmass_center, pos[t, :, :], L)

            mean_uw = np.mean(uw[t, :, :], axis=0)
            mean_uw_unit = mean_uw / np.linalg.norm(mean_uw)
            frontness_vals = center_distvecs[:, 0] * mean_uw_unit[0] + center_distvecs[:, 1] * mean_uw_unit[1]
            orientations = uw[t, :, 0] * mean_uw_unit[0] + uw[t, :, 1] * mean_uw_unit[1]

            uw_angles = np.arctan2(uw[t, :, 1], uw[t, :, 0])
            mean_uw_angle = np.arctan2(mean_uw_unit[1], mean_uw_unit[0])
            orientation_angles = np.pi - np.abs(np.abs(uw_angles - mean_uw_angle) - np.pi)
            orientation_angles = orientation_angles / np.pi * 180

            all_orientations = np.concatenate((all_orientations, orientation_angles[nonstartle_mask]))
            all_frontness = np.concatenate((all_frontness, frontness_vals[nonstartle_mask]))

            startle_orientations = np.concatenate((startle_orientations, orientation_angles[startle_idc]))
            startle_frontness = np.concatenate((startle_frontness, frontness_vals[startle_idc]))
    return startle_orientations, startle_frontness, all_orientations, all_frontness


def get_startle_times(startles, output_step, burn_period=50):
    burn_period_steps = int(burn_period / output_step)
    ntimesteps = startles.shape[0]
    startle_times = np.array([])
    for t in np.arange(burn_period_steps, ntimesteps):
        startle_idc = np.where(startles[t, :])[0]
        nonstartle_mask = np.ones(len(startles[t, :]), dtype=np.bool)
        nonstartle_mask[startle_idc] = 0
        if startle_idc.size == 0:
            continue
        else:
            current_startle_times = np.ones(len(startle_idc)) * (t / output_step)
            startle_times = np.concatenate((startle_times, current_startle_times))
    return startle_times


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
