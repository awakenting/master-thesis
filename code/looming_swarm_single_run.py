import time
import matplotlib

matplotlib.use('TKAgg')

import SwarmStartleLooming as sw
import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(200)
# Initialize Parameters
N = 40
L = 50
total_time = 100.0
dt = 0.001

speed0 = 0.5
alpha = 0.8
noisep = 0.01
noisev = 0.05
BC = 0 #'along_wall'
IC = 10 # initial condition

repstrength = 1.0
algstrength = 0.5
attstrength = 0.3

reprange = 1.0
algrange = 5.0
attrange = 25.0

repsteep = -20

output = 0.5

int_type = 'matrix'
startle = True

amplitude_startle = 5.0
duration_startle = 1.0
threshold_startle = 1.0
noise_startle = 0.02
gamma_startle = 10.0
increment_startle = 0.2
only_positive_startle = False
vis_input_const = 1
dist_pow = 1
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
                          threshold_startle=threshold_startle,
                          noise_startle=noise_startle,
                          gamma_startle=gamma_startle,
                          increment_startle=increment_startle,
                          only_positive_startle=only_positive_startle,
                          vis_input_const=vis_input_const,
                          dist_pow=dist_pow,
                          r_m=10*1e6,
                          tau_m=0.023,
                          e_l=-0.079,
                          v_t=-0.061,
                          vt_std=0.003,
                          tau_rho=0.001,
                          rho_scale=9.6 * 1e6,
                          exc_scale=30,
                          noise_std_exc=0.010,
                          noise_std_inh=0.005,
                          bodylength=1,
                          )

#outData, agentData = sw.RunAnimate(paraSystem, paraFish)
outData, agentData = sw.RunAnimateWithStartlePotential(paraSystem, paraFish)

#starttime = time.time()

#outData, agentData = sw.SingleRun(paraSystem, paraFish)

startles = np.array(outData['startle'])


#endtime = time.time()
#print('Total time needed: ' + str(int((endtime - starttime))) + ' seconds or '
#      + str(int((endtime - starttime) / 60)) + ' min')


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
    starting_points = []
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
                        starting_points.append(t-1)
                        current_cascade_members.extend(new_members)
                    current_cascade_length += 1
                else:
                    single_startle = True
                    current_cascade_members.extend(startle_idc)
                    current_cascade_length += 1
        previous_startle_idc = startle_idc
    return np.array(cascade_sizes), np.array(cascade_lengths), np.array(starting_points)


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

vis_angles = np.array(outData['vis_angles'])
plt.figure()
plt.plot(vis_angles[:, 0:5])
plt.xlabel('time')
plt.ylabel('visual angle ($\degree$)')
plt.show()

plt.figure()
szs, lens, starts = calcCascadeSizes(startles)
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


