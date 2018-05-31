import time
import matplotlib

matplotlib.use('TKAgg')

import SwarmStartleLooming as sw
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(200)
# Initialize Parameters
N = 10
L = 50
total_time = 50.0
dt = 0.001

speed0 = 1.5
alpha = 1
noisep = 0.01
noisev = 0.0
BC = 0 #'along_wall'
IC = 10 # initial condition

repstrength = 1.0
algstrength = 0.5
attstrength = 0.3

reprange = 1.0
algrange = 5.0
attrange = 25.0

repsteep = -20

output = 0.05

int_type = 'matrix'
startle = True

amplitude_startle = 50.0
duration_startle = 0.070

# initialize system parameters
paraSystem = sw.BaseParams(N=N, L=L, time=total_time, dt=dt, BC=BC, IC=IC, output=output, int_type=int_type, startle=startle)
# initialize prey parameters
paraFish = sw.AgentParams(paraSystem, speed0=speed0, alpha=alpha,
                          repstrength=repstrength, reprange=reprange, repsteepness=repsteep,
                          algstrength=algstrength, algrange=algrange,
                          attstrength=attstrength, attrange=attrange,
                          noisep=noisep, noisev=noisev,
                          amplitude_startle=amplitude_startle,
                          duration_startle=duration_startle,
                          print_startles=True,
                          r_m=10*1e6,
                          tau_m=0.023,
                          e_l=-0.079,
                          v_t=-0.061,
                          vt_std=0.000,
                          tau_rho=0.001,
                          rho_null=3.6,
                          rho_null_std=0.7,
                          rho_scale=8.16 * 1e6,
                          exc_scale=30,
                          noise_std_exc=0.0027,
                          noise_std_inh=0.000,
                          vis_input_m=3,
                          vis_input_b=0,
                          vis_input_method='max',
                          vis_input_k=3
                          )

#outData, agentData = sw.RunAnimate(paraSystem, paraFish)
outData, agentData = sw.RunAnimateWithStartlePotential(paraSystem, paraFish, startleAgent=1)

#starttime = time.time()

#outData, agentData = sw.SingleRun(paraSystem, paraFish)

startles = np.array(outData['startle'])


#endtime = time.time()
#print('Total time needed: ' + str(int((endtime - starttime))) + ' seconds or '
#      + str(int((endtime - starttime) / 60)) + ' min')


def calcCascadeSizes(startles, output_step, time_margin=0.2):
    """Calculates the sizes of startle cascades.

    Parameters
    ----------
    startles
        An array with booleans for each agent and time point.

    Returns
    -------

    """
    time_margin_steps = int(time_margin/output_step)
    ntimesteps = startles.shape[0]
    ncascades = 0
    cascade_sizes = []
    cascade_lengths = []
    single_startle = False
    ongoing_cascade = False
    current_cascade_members = []
    previous_startle_idc = []
    last_startle_time_idx = 0
    current_cascade_length = 0
    starting_points = []
    for t in np.arange(ntimesteps):
        startle_idc = np.where(startles[t, :])[0]
        time_since_last_startle = t - last_startle_time_idx
        # no startles at current time step:
        if startle_idc.size == 0:
            if ongoing_cascade and time_since_last_startle > time_margin_steps:
                cascade_sizes.append(len(current_cascade_members))
                cascade_lengths.append(current_cascade_length)
                ncascades += 1
                ongoing_cascade = False
                current_cascade_members = []
                current_cascade_length = 0
            else:
                if single_startle and time_since_last_startle > time_margin_steps:
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
                        starting_points.append(t-1)
                        current_cascade_members.extend(new_members)
                    current_cascade_length += 1
                else:
                    single_startle = True
                    current_cascade_members.extend(startle_idc)
                    current_cascade_length += 1
            last_startle_time_idx = t
            previous_startle_idc = startle_idc
    return np.array(cascade_sizes), np.array(cascade_lengths), np.array(starting_points)


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


def calcCohesion(pos, method='nearest'):
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
        elif method == 'inter':
            dist_mat, dx, dy = CalcDistVecMatrix(pos[t, :, :], L, BC)
            np.fill_diagonal(dist_mat, np.nan)
            mean_dists = np.nanmean(dist_mat, axis=0)
            current_cohesion = np.mean(mean_dists)
        else:
            current_cohesion = 0
        coh[t] = current_cohesion
    return coh


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


def calcStartlePosition(pos, startles):
    """

    :param pos:
    :param startles:
    :return:
    """
    ntimesteps = startles.shape[0]
    for t in np.arange(ntimesteps):
        startle_idc = np.where(startles[t, :])[0]
        if startle_idc.size == 0:
            continue
        else:
            cmass_center = calc_periodic_mass_center(pos, L)
            center_distvecs = PeriodicDist(cmass_center, pos, L)
            center_dists = np.sqrt(center_distvecs[:, 0] ** 2 + center_distvecs[:, 1] ** 2)


    return
vis_angles = np.array(outData['vis_angles'])
plt.figure()
plt.plot(vis_angles[:, 0:5])
plt.xlabel('time')
plt.ylabel('visual angle ($\degree$)')
plt.show()

vel_data = np.array(outData['vel'])
plt.figure()
plt.plot(vel_data[:, 0:5, 0])
plt.xlabel('time')
plt.ylabel('velocity in x direction')
plt.show()

plt.figure()
plt.plot(vel_data[:, 0:5, 1])
plt.xlabel('time')
plt.ylabel('velocity in y direction')
plt.show()


plt.figure()
szs, lens, starts = calcCascadeSizes(startles, output_step=output, time_margin=0.2)
plt.imshow(startles, interpolation='none', aspect='auto')
plt.hlines(starts, 0, 40, colors='r')
plt.title('sizes: ' + str(szs) + ', lengths: ' + str(lens))
plt.show()


conv_coh = calcCohesion(np.array(outData['pos']), method='convexhull')
nn_coh = calcCohesion(np.array(outData['pos']), method='nearest')
ii_coh = calcCohesion(np.array(outData['pos']), method='inter')
plt.figure()
#plt.plot(conv_coh, label='convex hull')
plt.plot(nn_coh, label='nearest neighbour')
#plt.plot(ii_coh, label='inter-individual')
plt.vlines(starts, 0, 400, colors='r')
plt.legend()
plt.show()


