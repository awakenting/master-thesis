import numpy as np
import matplotlib.pyplot as pl
from sklearn.preprocessing import normalize


class AgentData(object):
    """ A class containing all the data structures for agents """

    def __init__(self, paraSystem):
        N = paraSystem.N
        dim = paraSystem.dim

        # Acceleration and speed
        self.alpha = np.ones(N)
        self.speed0 = np.ones(N)

        # Cartesian coordinates
        self.pos = np.zeros((N, dim))
        self.vel = np.zeros((N, dim))

        # Velocity unit vectors
        self.up = np.zeros((N, dim))
        self.uw = np.zeros((N, dim))

        # Social force variables
        self.force = np.zeros((N, dim))
        self.force_att = np.zeros((N, dim))
        self.force_rep = np.zeros((N, dim))
        self.force_alg = np.zeros((N, dim))

        self.neighbors_rep = np.zeros((N, 1))
        self.neighbors_alg = np.zeros((N, 1))
        self.neighbors_att = np.zeros((N, 1))

        self.weighted_neighbors_rep = np.zeros((N, 1))
        self.weighted_neighbors_alg = np.zeros((N, 1))
        self.weighted_neighbors_att = np.zeros((N, 1))

        # neuronal model variables
        self.allow_startling = np.zeros(N, dtype=bool)
        self.force_startle = np.zeros((N, dim))
        self.timer_startle = np.zeros(N)
        self.timer_refractory = np.zeros(N)
        self.phiDes_startle = np.zeros(N)
        self.v_m = np.ones(N)
        self.v_t = None
        self.rho = np.ones(N)
        self.noise_exc = None
        self.noise_rho = None
        self.e_l = None
        self.tau_m = None
        self.tau_rho = None
        self.rho_null = None
        self.startle = np.zeros(N, dtype=bool)
        self.neighbor_startle = np.ones((N, N))
        self.vis_angles = np.zeros(N)

        # Polar Coordinates
        self.velproj = np.ones(N)
        self.phi = np.zeros(N)
        self.theta = np.zeros(N)

        # social force strengths
        self.repstrength = np.ones(N)
        self.attstrength = np.ones(N)
        self.algstrength = np.ones(N)
        # social force ranges
        self.reprange = np.ones(N)
        self.attrange = np.ones(N)
        self.algrange = np.ones(N)
        # social force ranges
        self.repsteepness = np.ones(N)
        self.attsteepness = np.ones(N)
        self.algsteepness = np.ones(N)

        # individual noise strengths
        self.sigmap = np.zeros(N)
        self.sigmav = np.zeros(N)

        # helper methods for resetting/normalizing/updating

    def ResetForces(self):
        """ set all force arrays to zero """
        self.force.fill(0.0)
        self.force_att.fill(0.0)
        self.force_rep.fill(0.0)
        self.force_alg.fill(0.0)
        self.neighbors_rep.fill(0.0)
        self.neighbors_alg.fill(0.0)
        self.neighbors_att.fill(0.0)
        self.weighted_neighbors_rep.fill(0.0)
        self.weighted_neighbors_alg.fill(0.0)
        self.weighted_neighbors_att.fill(0.0)

    def NormalizeForces(self):
        """ normalize forces to unit vectors """
        self.force_att = normalize(self.force_att, axis=1, norm='l2')
        self.force_rep = normalize(self.force_rep, axis=1, norm='l2')
        self.force_alg = normalize(self.force_alg, axis=1, norm='l2')

    def NormalizeByCount(self):
        """ normalize social forces by the number of interaction partners """
        # repulsion
        idx_non_zero_count = np.nonzero(self.neighbors_rep)[0]
        self.force_rep[idx_non_zero_count] /= self.neighbors_rep[idx_non_zero_count]
        # alignment
        idx_non_zero_count = np.nonzero(self.neighbors_alg)[0]
        self.force_alg[idx_non_zero_count] /= self.neighbors_alg[idx_non_zero_count]
        # attraction
        idx_non_zero_count = np.nonzero(self.neighbors_att)[0]
        self.force_att[idx_non_zero_count] /= self.neighbors_att[idx_non_zero_count]

    def NormalizeByWeightedCount(self):
        """Normalize social forces by number of neighbors weighted by their
        distance.

        """
        # repulsion
        idx_non_zero_count = np.nonzero(self.weighted_neighbors_rep)[0]
        self.force_rep[idx_non_zero_count] /= self.weighted_neighbors_rep[idx_non_zero_count]
        # alignment
        idx_non_zero_count = np.nonzero(self.weighted_neighbors_alg)[0]
        self.force_alg[idx_non_zero_count] /= self.weighted_neighbors_alg[idx_non_zero_count]
        # attraction
        idx_non_zero_count = np.nonzero(self.weighted_neighbors_att)[0]
        self.force_att[idx_non_zero_count] /= self.weighted_neighbors_att[idx_non_zero_count]

    def UpdatePolar(self):
        """ calc polar angle (individual direction of motion)"""
        # self.speed=np.sum(normalize(self.vel,axis=1,norm='l2'),axis=1)
        self.phi = np.arctan2(self.vel[:, 1], self.vel[:, 0])
        self.velproj = self.vel[:, 0] * np.cos(self.phi) + self.vel[:, 1] * np.sin(self.phi)

    def UpdateCartesian(self):
        """ update Cartesian coordinates"""
        self.vel[:, 0] = self.velproj * np.cos(self.phi)
        self.vel[:, 1] = self.velproj * np.sin(self.phi)
        self.uw = normalize(self.vel, axis=1, norm='l2')
        self.up = np.copy(self.uw[:, ::-1])
        self.up[:, 0] *= -1

    def UpdateDynamicVars(self, dt, step):
        """ update vm and rho with passive parts"""
        self.v_m += dt * (-(self.v_m - self.e_l)) / self.tau_m + self.noise_exc[:, step]
        self.rho += dt * (self.rho_null - self.rho) / self.tau_rho + self.noise_rho[:, step]


class BaseParams(object):
    """ A class defining global model parameters """

    def __init__(self, dim=2, time=100, dt=0.05, output=1.0, L=50, N=10, IC=0, BC=0, startle=False, int_type='global'):
        self.dim = dim
        self.time = time
        self.dt = dt
        self.steps = int(self.time / self.dt)
        self.output = output
        self.stepout = int(self.output / self.dt)
        self.outsteps = int(self.steps / self.stepout)
        self.L = L
        self.N = N
        self.IC = IC
        self.BC = BC
        self.startle = startle
        self.int_type = int_type


class AgentParams(object):
    """ Agent Parameter class
        Required input: paraSystem
    """

    def __init__(self, paraSystem,
                 alpha=1.0, speed0=1.0,
                 wallrepdist=1, wallrepstrength=5.0,
                 attstrength=0.2, attrange=100, attsteepness=-20,
                 repstrength=1.0, reprange=1.0, repsteepness=-20,
                 algstrength=0.5, algrange=4, algsteepness=-20,
                 blindangle=0.0 * np.pi, blindanglesteepness=50,
                 kneigh=6,
                 print_startles=True,
                 noisev=0.0, noisep=0.1,
                 amplitude_startle=50.0,
                 duration_startle=0.2,
                 duration_refractory=1,
                 blocked_escape_angle=np.pi * 1.0,
                 r_m=10 * 1e6,
                 tau_m=0.023,
                 e_l=-0.079,
                 v_t=-0.061,
                 vt_std=0.003,
                 tau_rho=0.001,
                 rho_null=2,
                 rho_null_std=1,
                 rho_scale=9.6 * 1e6,
                 exc_scale=5,
                 noise_std_exc=0.005,
                 noise_std_inh=0.005,
                 vis_input_m=3,
                 vis_input_b=0,
                 vis_input_method='max',
                 vis_input_k=3
                 ):
        self.print_startles = print_startles

        self.alpha = alpha
        self.speed0 = speed0

        self.sigmav = np.sqrt(2.0 * paraSystem.dt * noisev)
        self.sigmap = np.sqrt(2.0 * paraSystem.dt * noisep)

        self.wallrepdist = wallrepdist
        self.wallrepstrength = wallrepstrength

        self.attstrength = attstrength
        self.attrange = attrange
        self.attsteepness = attsteepness

        self.algstrength = algstrength
        self.algrange = algrange
        self.algsteepness = algsteepness

        self.repstrength = repstrength
        self.reprange = reprange
        self.repsteepness = repsteepness

        self.blindangle = blindangle
        self.cosblindangle = np.cos(np.pi - 0.5 * self.blindangle)
        self.blindanglesteepness = blindanglesteepness

        # startle parameters
        if paraSystem.startle:
            self.amplitude_startle = amplitude_startle
            self.duration_startle = duration_startle
            self.duration_refractory = duration_refractory
            self.blocked_escape_angle = blocked_escape_angle
            self.r_m = r_m
            self.tau_m = tau_m
            self.e_l = e_l
            self.v_t = v_t
            self.vt_std = vt_std
            self.tau_rho = tau_rho
            self.rho_scale = rho_scale
            self.exc_scale = exc_scale
            self.rho_null = rho_null
            self.rho_null_std = rho_null_std
            self.noise_std_exc = noise_std_exc
            self.noise_std_inh = noise_std_inh
            self.vis_input_m = vis_input_m
            self.vis_input_b = vis_input_b
            self.vis_input_method = vis_input_method
            self.vis_input_k = vis_input_k

        self.kneigh = kneigh


def InitAgents(agentData, paraSystem, paraAgents):
    """ initialize agents: setting initial coordinates and parameters in agentData """
    N = paraSystem.N
    dim = paraSystem.dim
    steps = paraSystem.steps

    if paraSystem.IC == 10:
        agentData.pos = 0.5 * paraSystem.L * np.random.random((N, dim))
    else:
        agentData.pos = paraSystem.L * np.random.random((N, dim))
    # set polar coordinates
    agentData.phi = 2. * np.pi * np.random.random(N)
    agentData.velproj = np.ones(N)
    # update Cartesian coordinates
    agentData.UpdateCartesian()

    # set acceleration and speed
    agentData.alpha *= paraAgents.alpha
    agentData.speed0 *= paraAgents.speed0

    # initialize noise
    agentData.sigmav = np.ones(N) * paraAgents.sigmav
    agentData.sigmap = np.ones(N) * paraAgents.sigmap

    # set interaction parameters
    agentData.repstrength *= paraAgents.repstrength
    agentData.algstrength *= paraAgents.algstrength
    agentData.attstrength *= paraAgents.attstrength

    agentData.reprange *= paraAgents.reprange
    agentData.algrange *= paraAgents.algrange
    agentData.attrange *= paraAgents.attrange

    agentData.repsteepness *= paraAgents.repsteepness
    agentData.algsteepness *= paraAgents.algsteepness
    agentData.attsteepness *= paraAgents.attsteepness

    # reset startle variables
    if paraSystem.startle:
        agentData.allow_startling[:] = paraSystem.startle
        agentData.timer_startle = np.zeros(N)
        agentData.timer_refractory = np.zeros(N)
        sigma_exc = paraAgents.noise_std_exc * np.sqrt(paraSystem.dt)
        sigma_inh = paraAgents.noise_std_inh * np.sqrt(paraSystem.dt)
        agentData.noise_exc = np.random.normal(loc=0, scale=sigma_exc, size=(N, steps + 1))
        agentData.noise_rho = np.random.normal(loc=0, scale=sigma_inh, size=(N, steps + 1))
        agentData.v_t = np.random.normal(loc=paraAgents.v_t, scale=paraAgents.vt_std, size=(N, steps + 1))
        if not paraAgents.rho_null_std == 0:
            agentData.rho_null = np.random.lognormal(mean=paraAgents.rho_null, sigma=paraAgents.rho_null_std, size=N)/1000
        else:
            agentData.rho_null = paraAgents.rho_null/1000
        agentData.v_m *= paraAgents.e_l
        agentData.rho *= paraAgents.rho_null
        agentData.e_l = paraAgents.e_l
        agentData.tau_m = paraAgents.tau_m
        agentData.tau_rho = paraAgents.tau_rho
    return


def SaveOutData(s, agentData, outdata, paraSystem):
    """ Creates list of dictionaries for output
        each list entry corresponds to single time step

        returns list of dict()
    """

    if outdata is None:
        outdata = dict()
        outdata['pos'] = []
        outdata['vel'] = []
        outdata['phi'] = []
        outdata['vp'] = []
        outdata['uw'] = []
        outdata['force'] = []
        outdata['force_rep'] = []
        outdata['force_alg'] = []
        outdata['force_att'] = []
        outdata['timer_s'] = []
        outdata['v_m'] = []
        outdata['vis_angles'] = []
        outdata['startle'] = []
        outdata['t'] = []

    outdata['pos'].append(np.copy(agentData.pos))
    outdata['vel'].append(np.copy(agentData.vel))
    outdata['phi'].append(np.copy(agentData.phi))
    outdata['vp'].append(np.copy(agentData.velproj))
    outdata['uw'].append(np.copy(agentData.uw))
    outdata['force'].append(np.copy(agentData.force))
    outdata['force_rep'].append(np.copy(agentData.force_rep))
    outdata['force_alg'].append(np.copy(agentData.force_alg))
    outdata['force_att'].append(np.copy(agentData.force_att))
    outdata['timer_s'].append(np.copy(agentData.timer_startle))
    outdata['v_m'].append(np.copy(agentData.v_m))
    outdata['vis_angles'].append(np.copy(agentData.vis_angles))
    outdata['startle'].append(np.copy(agentData.startle))
    outdata['t'].append(s * paraSystem.dt)

    return outdata


def UpdateTotalSocialForces(agentData, paraSystem, paraAgents):
    """ Update resulting social force in the agentData class
    """
    if paraSystem.int_type == 'voronoi' or paraSystem.int_type == 'voronoi_matrix':
        agentData.NormalizeByCount()
    else:
        agentData.NormalizeByWeightedCount()

    agentData.force = (agentData.repstrength * agentData.force_rep.T
                       + agentData.attstrength * agentData.force_att.T
                       + agentData.algstrength * agentData.force_alg.T).T

    if paraSystem.startle:
        agentData.force[agentData.startle] = paraAgents.amplitude_startle * agentData.force_startle[agentData.startle]
    return


def UpdateCoordinates(agentData, paraSystem, paraAgents):
    """ update of coordinates for all agents """
    dvproj, dphi = CalculateDirectionChange(agentData, paraSystem, paraAgents)
    agentData.velproj += dvproj

    # calculate new angle / add noise only if no startle
    agentData.phi += dphi
    agentData.phi = np.fmod(agentData.phi, 2.0 * np.pi)

    # update Cartesian velocity vectors and positions
    agentData.UpdateCartesian()
    if paraSystem.BC == 0:
        agentData.pos += agentData.vel * paraSystem.dt
        agentData.pos = CalcBoundaryPeriodic(agentData.pos, paraSystem.L)
    elif paraSystem.BC == 'along_wall':
        future_pos = agentData.pos + agentData.vel * paraSystem.dt
        # update position
        newpos, boundary_crossed = CalcBoundaryRepelling(future_pos.copy(), paraSystem.L)
        agentData.pos = newpos
        # remove velocity in direction orthogonal to border
        posx = future_pos[:, 0]
        posy = future_pos[:, 1]
        old_vel = agentData.vel.copy()
        x_boundary_crossed = (posx > paraSystem.L) | (posx < 0.0)
        y_boundary_crossed = (posy > paraSystem.L) | (posy < 0.0)
        agentData.vel[x_boundary_crossed, 0] = 0
        agentData.vel[y_boundary_crossed, 1] = 0
        # if in a corner, turn around
        both_crossed = x_boundary_crossed & y_boundary_crossed
        agentData.vel[both_crossed, 0] = - old_vel[both_crossed, 0]
        agentData.UpdatePolar()
        agentData.UpdateCartesian()
    else:
        agentData.pos += agentData.vel * paraSystem.dt
        newpos, boundary_crossed = CalcBoundaryRepelling(agentData.pos, paraSystem.L)
        agentData.pos = newpos
        agentData.phi[boundary_crossed] += np.pi
        agentData.phi = np.fmod(agentData.phi, 2.0 * np.pi)
        agentData.UpdateCartesian()

    return


def PlainInit():
    """ Convenience function for default initalization of paraSystem() and paraAgents() """
    paraSystem = BaseParams()
    paraAgents = AgentParams(paraSystem)
    return paraSystem, paraAgents


def CalculateDirectionChange(agentData, paraSystem, paraAgents):
    """ calculate direction and velproj change (polar angle increment) from social forces
    """
    # project forces on polar coordinates
    forcev = np.sum(agentData.force * agentData.uw, axis=1)
    forcep = np.sum(agentData.force * agentData.up, axis=1)
    # deterministic increments
    dvproj = (agentData.alpha * (agentData.speed0 - agentData.velproj) + forcev) * paraSystem.dt
    dphi = forcep * paraSystem.dt
    # add noise
    if paraAgents.sigmav > 0.0:
        vel_noise = np.random.normal(0.0, agentData.sigmav, size=paraSystem.N)
        dvproj += vel_noise
    if paraAgents.sigmap > 0.0:
        phi_noise = np.random.normal(0.0, agentData.sigmap, size=paraSystem.N)
        dphi += phi_noise

    # account for inertia
    dphi /= (agentData.velproj + 0.01)

    return dvproj, dphi


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


# @jit('double(double,double,double)',nopython=True, nogil=True)
# @jit(nopython=True, nogil=True)
def SigThresh(x, x0=0.5, steepness=10):
    """ Sigmoid function f(x)=1/2*(tanh(a*(x-x0)+1)

        Input parameters:
        -----------------
        x:  function argument
        x0: position of the transition point (f(x0)=1/2)
        steepness:  parameter setting the steepness of the transition.
                    (positive values: transition from 0 to 1, negative values:
                    transition from 1 to 0)
    """
    return 0.5 * (np.tanh(steepness * (x - x0)) + 1)


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


def CalcVelDiffVecMatrix(vel):
    """ Calculate N^2 velocity difference matrix (dv_ij) for velocity alignment calculation

        Returns:
        --------
        dVX - matrix of all differences in vx coordinate (NxN)
        dVY - matrix of all differences in vy coordinate (NxN)
    """
    VX = np.reshape(vel[:, 0], (-1, 1))
    VY = np.reshape(vel[:, 1], (-1, 1))
    dVX = np.subtract(VX, VX.T)
    dVY = np.subtract(VY, VY.T)
    return dVX, dVY


def CalcForceVecMatrix(factormatrix, dX, dY, distmatrix):
    """ Calculate all pairwise forces
        Input:
        ------
        factormatrix - pre-factor matrix e.g. interaction strength
        distmatrix - distance matrix
        dX,dY - difference matrices for different coordinate directions

        Return:
        -------
        FX,FY - arrays with force components

    """
    dUX = np.divide(dX, distmatrix)
    dUY = np.divide(dY, distmatrix)
    FX = np.multiply(factormatrix, dUX)
    FY = np.multiply(factormatrix, dUY)

    return FX, FY


def CalcAlgVecMatrix(factormatrix, dUX, dUY):
    """ Calculate alignment forces from velocity differences
        Input:
        ------
        factormatrix - pre-factor matrix e.g. interaction strength
        dVX,dVY - difference matrices for different velocity components

        Return:
        -------
        FX,FY - arrays with force components
    """
    FX = -np.multiply(factormatrix, dUX)
    FY = -np.multiply(factormatrix, dUY)
    return FX, FY


def CalcFactorMatrix(distmatrix, force_range, force_steepness):
    """ Calculate prefactor matrices (interaction strengths)
    """
    factormatrix = SigThresh(distmatrix, force_range, force_steepness)
    return factormatrix


def CalcDistanceVec(pos_i, pos_j, L, BC, dim):
    """ Convenience function to calculate distance"""
    if BC == 0:
        distvec = PeriodicDist(pos_i, pos_j, L, dim)
    else:
        distvec = pos_j - pos_i

    return distvec


def UniqueEdgesFromTriangulation(tri):
    """ Get unique edges from Delaunay triangulation for Voronoi neighbours """

    e1 = tri.simplices[:, 0:2]
    e2 = tri.simplices[:, 1:3]
    e3 = tri.simplices[:, ::2]
    edges = np.vstack((e1, e2, e3))
    edges.sort(axis=1)
    edges_c = np.ascontiguousarray(edges).view(np.dtype((np.void, edges.dtype.itemsize * edges.shape[1])))
    _, idx = np.unique(edges_c, return_index=True)
    edges_unique = edges[idx]
    return edges_unique


def CalcWeightedNeighborsFromMatrix(factor_M):
    """Calculates weighted number of neighbors from factor matrix.

    This function first sets diagonal elements of the factor matrix
    to zero and then returns the sum over the second axis.

    Parameters
    ----------
    factor_M array, shape = (N, N)
        The factor matrix calculated by `CalcFactorMatrix()`

    Returns
    -------
    weighted_neighbors array, shape = (N, 1)
        The weighted number of neighbors calculated as the sum of
        all factors in a row.

    """
    factor_mat_no_selfconnect = np.copy(factor_M)
    np.fill_diagonal(factor_mat_no_selfconnect, 0)
    weighted_neighbors = np.sum(factor_mat_no_selfconnect, axis=1, keepdims=True)
    return weighted_neighbors


def CalcInteractionMatrix(agentData, paraSystem):
    """ Calculate Social Force matrix for repulsion,alignment,attraction
        Updates AgentData() inline and returns distmatrix
    """
    distmatrix, dX, dY = CalcDistVecMatrix(agentData.pos, paraSystem.L, paraSystem.BC)
    dVX, dVY = CalcVelDiffVecMatrix(agentData.vel)
    # repulsion
    factor_M = CalcFactorMatrix(distmatrix, agentData.reprange, agentData.repsteepness)
    FX, FY = CalcForceVecMatrix(factor_M, dX, dY, distmatrix)
    agentData.force_rep[:, 0] = np.nansum(FX, axis=1)
    agentData.force_rep[:, 1] = np.nansum(FY, axis=1)
    agentData.weighted_neighbors_rep = CalcWeightedNeighborsFromMatrix(factor_M)
    # alignment
    factor_M = CalcFactorMatrix(distmatrix, agentData.algrange, agentData.algsteepness)
    FX, FY = CalcAlgVecMatrix(factor_M, dVX, dVY)
    agentData.force_alg[:, 0] = np.nansum(FX, axis=1)
    agentData.force_alg[:, 1] = np.nansum(FY, axis=1)
    agentData.weighted_neighbors_alg = CalcWeightedNeighborsFromMatrix(factor_M)
    # attraction
    factor_M = CalcFactorMatrix(distmatrix, agentData.attrange, agentData.attsteepness)
    FX, FY = CalcForceVecMatrix(factor_M, dX, dY, distmatrix)
    agentData.force_att[:, 0] = -np.nansum(FX, axis=1)
    agentData.force_att[:, 1] = -np.nansum(FY, axis=1)
    agentData.weighted_neighbors_att = CalcWeightedNeighborsFromMatrix(factor_M)
    return distmatrix


def CalcInteractionVoronoi(agentData, paraSystem, paraAgents):
    """ Calculate effective social force using the Voronoi neighbourhood """
    from scipy.spatial import Delaunay
    tri = Delaunay(agentData.pos)
    edges = UniqueEdgesFromTriangulation(tri)
    for e in edges:
        if e[0] != e[1]:
            CalcInteractionPair(e[0], e[1], agentData, paraSystem, paraAgents)

    return edges

def CalcInteractionVoronoiMatrix(agentData, paraSystem, paraAgents):
    """ Calculate effective social force using the Voronoi neighbourhood """
    from scipy.spatial import Delaunay
    tri = Delaunay(agentData.pos)
    edges = UniqueEdgesFromTriangulation(tri)

    voronoi_mask = np.zeros((paraSystem.N, paraSystem.N))
    for e in edges:
        if e[0] != e[1]:
            voronoi_mask[e[0], e[1]] = 1
            voronoi_mask[e[1], e[0]] = 1

    distmatrix, dX, dY = CalcDistVecMatrix(agentData.pos, paraSystem.L, paraSystem.BC)
    dVX, dVY = CalcVelDiffVecMatrix(agentData.vel)
    # repulsion
    factor_M = CalcFactorMatrix(distmatrix, agentData.reprange, agentData.repsteepness)
    FX, FY = CalcForceVecMatrix(factor_M, dX, dY, distmatrix)
    voronoi_FX = FX * voronoi_mask
    voronoi_FY = FY * voronoi_mask
    voronoi_factor_M = factor_M * voronoi_mask
    agentData.force_rep[:, 0] = np.nansum(voronoi_FX, axis=1)
    agentData.force_rep[:, 1] = np.nansum(voronoi_FY, axis=1)
    agentData.neighbors_rep = np.sum(voronoi_mask, axis=1, keepdims=True)
    agentData.weighted_neighbors_rep = CalcWeightedNeighborsFromMatrix(voronoi_factor_M)
    # alignment
    factor_M = CalcFactorMatrix(distmatrix, agentData.algrange, agentData.algsteepness)
    FX, FY = CalcAlgVecMatrix(factor_M, dVX, dVY)
    voronoi_FX = FX * voronoi_mask
    voronoi_FY = FY * voronoi_mask
    voronoi_factor_M = factor_M * voronoi_mask
    agentData.force_alg[:, 0] = np.nansum(voronoi_FX, axis=1)
    agentData.force_alg[:, 1] = np.nansum(voronoi_FY, axis=1)
    agentData.neighbors_alg = np.sum(voronoi_mask, axis=1, keepdims=True)
    agentData.weighted_neighbors_alg = CalcWeightedNeighborsFromMatrix(voronoi_factor_M)
    # attraction
    factor_M = CalcFactorMatrix(distmatrix, agentData.attrange, agentData.attsteepness)
    FX, FY = CalcForceVecMatrix(factor_M, dX, dY, distmatrix)
    voronoi_FX = FX * voronoi_mask
    voronoi_FY = FY * voronoi_mask
    voronoi_factor_M = factor_M * voronoi_mask
    agentData.force_att[:, 0] = -np.nansum(voronoi_FX, axis=1)
    agentData.force_att[:, 1] = -np.nansum(voronoi_FY, axis=1)
    agentData.neighbors_att = np.sum(voronoi_mask, axis=1, keepdims=True)
    agentData.weighted_neighbors_att = CalcWeightedNeighborsFromMatrix(voronoi_factor_M)

    return edges


def CalcInteractionGlobal(agentData, paraSystem, paraAgents):
    """ Calculate effective social forces using brute force N^2 iteration """
    for i in range(paraSystem.N):
        for j in range(i + 1, paraSystem.N):
            CalcInteractionPair(i, j, agentData, paraSystem, paraAgents)
    return


def CalcSingleRepForce(i, distvec, dist, agentData):
    repfactor = SigThresh(dist, agentData.reprange[i], agentData.repsteepness[i])
    agentData.force_rep[i] -= repfactor * distvec / dist
    agentData.neighbors_rep[i] += 1
    agentData.weighted_neighbors_rep[i] += repfactor
    return


def CalcSingleAlgForce(i, dvel, dist, agentData):
    algfactor = SigThresh(dist, agentData.algrange[i], agentData.algsteepness[i])
    agentData.force_alg[i] += algfactor * dvel
    agentData.neighbors_alg[i] += 1
    agentData.weighted_neighbors_alg[i] += algfactor
    return


def CalcSingleAttForce(i, distvec, dist, agentData):
    attfactor = SigThresh(dist, agentData.attrange[i], agentData.attsteepness[i])
    agentData.force_att[i] += attfactor * distvec / dist
    agentData.neighbors_att[i] += 1
    agentData.weighted_neighbors_att[i] += attfactor
    return


def CalcInteractionPair(i, j, agentData, paraSystem, update_both=True):
    """ Calculate effective social force between two agents """

    L = paraSystem.L
    BC = paraSystem.BC
    dim = paraSystem.dim

    # calculate distance vector and scalar
    distvec = CalcDistanceVec(agentData.pos[i], agentData.pos[j], L, BC, dim)
    dist = 0.0
    for d in range(dim):
        dist += distvec[d] ** 2
    dist = np.sqrt(dist)

    # repulsion
    if agentData.repstrength[i] > 0.0:
        # calculate repulsion
        CalcSingleRepForce(i, distvec, dist, agentData)
        if update_both:
            if agentData.repstrength[j] > 0.0:
                CalcSingleRepForce(j, -distvec, dist, agentData)
    # alignment
    if agentData.algstrength[i] > 0.0:
        dvel = agentData.vel[j] - agentData.vel[i]
        # calculate alignment
        CalcSingleAlgForce(i, dvel, dist, agentData)
        if update_both:
            if agentData.algstrength[j] > 0.0:
                CalcSingleAlgForce(j, -dvel, dist, agentData)
    # attraction
    if agentData.attstrength[i] > 0.0:
        # calculate attraction
        CalcSingleAttForce(i, distvec, dist, agentData)
        if update_both:
            if agentData.attstrength[j] > 0.0:
                CalcSingleAttForce(j, -distvec, dist, agentData)
    return


def CalcBoundaryPeriodic(pos, L):
    posx = pos[:, 0]
    posy = pos[:, 1]

    posx[posx > L] = posx[posx > L] - L
    posy[posy > L] = posy[posy > L] - L
    posx[posx < 0.0] = posx[posx < 0.0] + L
    posy[posy < 0.0] = posy[posy < 0.0] + L

    pos[:, 0] = posx
    pos[:, 1] = posy

    return pos


def CalcBoundaryRepelling(pos, L):
    posx = pos[:, 0]
    posy = pos[:, 1]

    boundary_crossed = (posx > L) | (posy > L) | (posx < 0.0) | (posy < 0.0)

    posx[posx > L] = L - (posx[posx > L] - L)
    posy[posy > L] = L - (posy[posy > L] - L)
    posx[posx < 0.0] = - posx[posx < 0.0]
    posy[posy < 0.0] = - posy[posy < 0.0]

    pos[:, 0] = posx
    pos[:, 1] = posy

    return pos, boundary_crossed


def CalcStartleInfluencePair(responder, initiator, agentData, paraAgents, dt):
    """ Update internal variable due to neighbour behaviour:
        If focal individual observes a startle it receives an increment for internal LIF-dynamics,
        decreasing with distance of the startling neighbor
    """
    distvec = agentData.pos[initiator] - agentData.pos[responder]
    dist = np.linalg.norm(distvec)

    vis_angle = (np.arctan2(1 / 2, dist) * 2) / np.pi * 180
    if vis_angle > 180:
        vis_angle = 180
    stimulus = vis_angle * 3 * 1e-11
    agentData.rho += dt * (paraAgents.rho_scale * stimulus) / paraAgents.tau_rho
    agentData.v_m += dt * (paraAgents.r_m * stimulus - agentData.rho) / paraAgents.tau_m

    # calculate direction of startling
    # np.arctan2 returns values from -pi to pi but we calculate values from 0 to 2*pi, so
    # we add pi to the value from np.arctan2:
    dist_phi = np.arctan2(distvec[1], distvec[0]) + np.pi * 1.0
    open_escape_range = np.pi * 2.0 - paraAgents.blocked_escape_angle
    escape_range_mid = open_escape_range / 2.0
    agentData.phiDes_startle[responder] = (dist_phi + np.random.rand(1) * open_escape_range
                                           - escape_range_mid)


def CalcStartleInfluenceMatrix(agentData, paraAgents, paraSystem):
    """ Calculate Social Force matrix for repulsion,alignment,attraction
        Updates AgentData() inline and returns distmatrix
    """
    distmatrix, dX, dY = CalcDistVecMatrix(agentData.pos, paraSystem.L, paraSystem.BC)
    vis_angle = (np.arctan2(1 / 2, distmatrix) * 2) / np.pi * 180
    vis_angle[vis_angle > 180] = 180
    # neighbours whose calculated angle is above 180 are so close that we assume that they have no impact anymore:
    vis_angle[vis_angle == 180] = 0
    vis_angle = paraAgents.vis_input_m * vis_angle + paraAgents.vis_input_b

    if paraAgents.vis_input_method == 'max':
        vis_angle = np.max(vis_angle, axis=1)
    elif paraAgents.vis_input_method == 'mean':
        vis_angle = np.mean(vis_angle, axis=1)
    elif paraAgents.vis_input_method == 'knn_mean':
        sort_idc = np.argsort(vis_angle, axis=1)
        # argsort gives ascending order so we have to reverse the order
        sort_idc = sort_idc[:, -1::-1]
        # for integer indexing we also need the row indices:
        rows = np.arange(paraSystem.N)[:, np.newaxis]
        knn_vis_angle = vis_angle[rows, sort_idc[:, 0:paraAgents.vis_input_k]]
        vis_angle = np.mean(knn_vis_angle, axis=1)
    elif paraAgents.vis_input_method == 'mean_deviate':
        vis_angle = np.max(vis_angle, axis=1) - np.mean(vis_angle, axis=1)
    elif paraAgents.vis_input_method == 'knn_mean_deviate':
        sort_idc = np.argsort(vis_angle, axis=1)
        # argsort gives ascending order so we have to reverse the order
        sort_idc = sort_idc[:, -1::-1]
        # for integer indexing we also need the row indices:
        rows = np.arange(paraSystem.N)[:, np.newaxis]
        knn_vis_angle = vis_angle[rows, sort_idc[:, 0:paraAgents.vis_input_k]]
        knn_mean = np.mean(knn_vis_angle, axis=1)
        vis_angle = np.max(vis_angle, axis=1) - knn_mean

    agentData.vis_angles = vis_angle
    stimulus = vis_angle * 1e-11 * paraAgents.exc_scale
    agentData.rho += paraSystem.dt * (paraAgents.rho_scale * stimulus) / paraAgents.tau_rho
    agentData.v_m += paraSystem.dt * (paraAgents.r_m * stimulus - agentData.rho) / paraAgents.tau_m

    dist_phi = np.arctan2(dY, dX) + np.pi * 1.0
    dist_phi = np.mean(dist_phi, axis=1)
    open_escape_range = np.pi * 2.0 - paraAgents.blocked_escape_angle
    escape_range_mid = open_escape_range / 2.0
    agentData.phiDes_startle = (dist_phi + np.random.rand(paraSystem.N) * open_escape_range + np.pi - escape_range_mid)


def ActivationStartle(agentData, paraAgents, step):
    """ check for which individuals the startle condition is fulfilled and activate them
    """

    agentData.timer_startle[agentData.v_m > agentData.v_t[:, step]] = paraAgents.duration_startle
    agentData.timer_refractory[agentData.v_m > agentData.v_t[:, step]] = paraAgents.duration_refractory
    agentData.v_m[agentData.v_m > agentData.v_t[:, step]] = paraAgents.e_l

    # membrane potential is clipped to the resting potential for the refractory period
    agentData.v_m[agentData.timer_refractory >= 0.0] = paraAgents.e_l

    condition_new_startle = (agentData.timer_startle > 0.0) & (np.sum(agentData.force_startle) == 0) \
                            & agentData.allow_startling
    indices_new_startle = np.where(condition_new_startle)[0]

    for i in indices_new_startle:
        # new_phi_startle = agentData.phiDes_startle[i] + 0.1 * np.pi * np.random.random()
        new_phi_startle = agentData.phiDes_startle[i]
        agentData.force_startle[i, 0] = np.cos(new_phi_startle)
        agentData.force_startle[i, 1] = np.sin(new_phi_startle)
        agentData.startle[i] = 1
        if paraAgents.print_startles:
            print('Individual {} startles!!!!'.format(i))

    return


def UpdateStartle(agentData, paraSystem, paraAgents, edges=None, distmatrix=None, step=None):
    """ Update internal variable (neuronal model dynamics)
    """
    # decrease startle and refractory duration
    agentData.timer_startle[agentData.timer_startle >= 0.0] -= paraSystem.dt
    agentData.timer_refractory[agentData.timer_refractory >= 0.0] -= paraSystem.dt
    # update passive part of neuronal dynamics
    agentData.UpdateDynamicVars(paraSystem.dt, step)

    # calc social influence (inputs in neuronal dynamics)
    CalcStartleInfluenceMatrix(agentData, paraAgents, paraSystem)

    # activate individuals which crossed threshold
    ActivationStartle(agentData, paraAgents, step)
    # deactivate individuals
    agentData.startle[agentData.timer_startle <= 0.0] = 0
    agentData.startle[agentData.timer_refractory <= 0.0] = 0
    agentData.force_startle[agentData.timer_startle <= 0.0] = np.array([0., 0.])

    return


def SingleRun(paraSystem, paraAgents, agentData=None):
    """ perform a single simulation run """

    # initialize agents if not provided at run time
    if agentData is None:
        agentData = AgentData(paraSystem)
        InitAgents(agentData, paraSystem, paraAgents)

    edges = None
    outdata = SaveOutData(0, agentData, None, paraSystem)
    if (paraSystem.int_type == 'global') or (paraSystem.int_type == 'matrix'):
        edges = []
        for i in range(paraSystem.N):
            for j in range(i, paraSystem.N):
                edges.append([i, j])

    # PERFORM TIME INTEGRATION
    for s in range(1, paraSystem.steps + 1):
        # reset social forces
        agentData.ResetForces()

        # calculate social interactions
        distmatrix = None
        if paraSystem.int_type == 'global':
            CalcInteractionGlobal(agentData, paraSystem, paraAgents)
        elif paraSystem.int_type == 'matrix':
            distmatrix = CalcInteractionMatrix(agentData, paraSystem)
        elif paraSystem.int_type == 'voronoi':
            edges = CalcInteractionVoronoi(agentData, paraSystem, paraAgents)
        elif paraSystem.int_type == 'voronoi_matrix':
            edges = CalcInteractionVoronoiMatrix(agentData, paraSystem, paraAgents)

        # update startle dynamics
        if paraSystem.startle:
            UpdateStartle(agentData, paraSystem, paraAgents, edges=edges, distmatrix=distmatrix, step=s)

        # update social forces and coordinates
        UpdateTotalSocialForces(agentData, paraSystem, paraAgents)
        UpdateCoordinates(agentData, paraSystem, paraAgents)

        # simulation output
        if (s % paraSystem.stepout) == 0:
            outdata = SaveOutData(s, agentData, outdata, paraSystem)
    print('Done!')
    return outdata, agentData


def SingleRunVoronoi(paraSystem, paraAgents, agentData=None):
    """ convenience function - single run with voronoi interactions """
    if agentData is None:
        agentData = AgentData(paraSystem)
        InitAgents(agentData, paraSystem, paraAgents)

    outdata = SaveOutData(0, agentData, None, paraAgents)

    for s in range(1, paraSystem.steps + 1):
        agentData.ResetForces()
        edges = CalcInteractionVoronoi(agentData, paraSystem, paraAgents)
        distmatrix = None

        # update startle dynamics
        if paraSystem.startle:
            UpdateStartle(agentData, paraSystem, paraAgents, edges=edges, distmatrix=distmatrix, step=s)

        UpdateTotalSocialForces(agentData, paraSystem, paraAgents)
        UpdateCoordinates(agentData, paraSystem, paraAgents)

        if (s % paraSystem.stepout) == 0:
            # print('t=%02f' % (para.dt*s))
            outdata = SaveOutData(s, agentData, outdata, paraSystem)
    print('Done!')
    return outdata


def RunAnimate(paraSystem, paraAgents, agentData=None, doblit=False):
    """ run a single simulation with animation """

    # initialize agents if not provided at run time
    if agentData is None:
        agentData = AgentData(paraSystem)
        InitAgents(agentData, paraSystem, paraAgents)

    pl.close()
    fig = pl.figure(99, figsize=(10, 10))
    ax = pl.subplot()
    ax.set_aspect('equal')
    ax.set_xlim(0, paraSystem.L)
    ax.set_ylim(0, paraSystem.L)
    ax.hold(True)

    x = agentData.pos[:, 0]
    y = agentData.pos[:, 1]
    pl.show(False)
    pl.draw()
    points = ax.plot(x, y, 'ro')[0]
    pointstail = ax.plot(x, y, 'r.', alpha=0.2)[0]

    if doblit:
        # cache the background
        background = fig.canvas.copy_from_bbox(ax.bbox)

    outdata = SaveOutData(0, agentData, None, paraSystem)
    edges = None
    if (paraSystem.int_type == 'global') or (paraSystem.int_type == 'matrix'):
        edges = []
        for i in range(paraSystem.N):
            for j in range(i, paraSystem.N):
                edges.append([i, j])

    for s in range(1, paraSystem.steps + 1):
        agentData.ResetForces()

        distmatrix = None
        if paraSystem.int_type == 'global':
            CalcInteractionGlobal(agentData, paraSystem, paraAgents)
        if paraSystem.int_type == 'matrix':
            distmatrix = CalcInteractionMatrix(agentData, paraSystem)
        elif paraSystem.int_type == 'voronoi':
            edges = CalcInteractionVoronoi(agentData, paraSystem, paraAgents)

        if paraSystem.startle:
            UpdateStartle(agentData, paraSystem, paraAgents, edges=edges, distmatrix=distmatrix, step=s)
        UpdateTotalSocialForces(agentData, paraSystem, paraAgents)
        UpdateCoordinates(agentData, paraSystem, paraAgents)

        if (s % paraSystem.stepout) == 0:
            # print('t=%02f' % (paraSystem.dt*s))

            outdata = SaveOutData(s, agentData, outdata, paraSystem)
            x = agentData.pos[:, 0]
            y = agentData.pos[:, 1]
            if paraSystem.output < 1.0:
                tail_length = int(3. / paraSystem.output)
            else:
                tail_length = 5
            postail = np.array(outdata['pos'][-tail_length:-1])
            xtail = np.reshape(postail[:, :, 0], (-1, 1))
            ytail = np.reshape(postail[:, :, 1], (-1, 1))

            points.set_data(x, y)
            pointstail.set_data(xtail, ytail)

            if paraSystem.BC == -1:
                meanx = np.mean(x)
                meany = np.mean(y)
                ax.set_xlim(meanx - 0.5 * paraSystem.L, meanx + 0.5 * paraSystem.L)
                ax.set_ylim(meany - 0.5 * paraSystem.L, meany + 0.5 * paraSystem.L)

            if doblit:
                # restore background
                fig.canvas.restore_region(background)

                # redraw just the points
                ax.draw_artist(points)

                # fill in the axes rectangle
                fig.canvas.blit(ax.bbox)

            else:
                # redraw everything
                ax.set_title('t=%02f' % (paraSystem.dt * s))
                fig.canvas.draw()

    # input("Press Enter to continue...")
    # For Python 2.x use below:
    # raw_input("Press Enter to continue...")
    pl.close(fig)
    return outdata, agentData


def RunAnimateWithStartlePotential(paraSystem, paraAgents, agentData=None, doblit=False, startleAgent=0):
    """ run a single simulation with animation """

    # initialize agents if not provided at run time
    if agentData is None:
        agentData = AgentData(paraSystem)
        InitAgents(agentData, paraSystem, paraAgents)

    pl.close()
    fig, (ax1, ax2) = pl.subplots(nrows=1, ncols=2, figsize=(16, 8))
    ax1.set_aspect('equal')
    ax1.set_xlim(0, paraSystem.L)
    ax1.set_ylim(0, paraSystem.L)
    ax1.hold(True)

    x = agentData.pos[:, 0]
    y = agentData.pos[:, 1]
    pl.show(False)
    pl.draw()
    points = ax1.plot(x, y, 'ro')[0]
    pointstail = ax1.plot(x, y, 'r.', alpha=0.2)[0]

    ax2.set_xlim([0, paraSystem.time])
    # ax2.set_ylim([paraAgents.v_t + paraAgents.v_t, paraAgents.v_t * 0.5])
    ax2.set_ylim([-0.090, -0.050])
    ax2.set_title('Membrane potential of agent {:}'.format(startleAgent))
    #ax2.hlines(paraAgents.v_t, 0, paraSystem.time, linestyles='--')
    time_array = np.arange(0, int((paraSystem.time/paraSystem.dt) + 1)) * paraSystem.dt
    ax2.plot(time_array, agentData.v_t[startleAgent, :], ls='--')
    startlevar_line = pl.Line2D([], [], linewidth=2, color='k')
    ax2.add_line(startlevar_line)

    if doblit:
        # cache the background
        background = fig.canvas.copy_from_bbox(ax1.bbox)

    outdata = SaveOutData(0, agentData, None, paraSystem)
    edges = None
    if (paraSystem.int_type == 'global') or (paraSystem.int_type == 'matrix'):
        edges = []
        for i in range(paraSystem.N):
            for j in range(i, paraSystem.N):
                edges.append([i, j])

    for s in range(1, paraSystem.steps + 1):
        agentData.ResetForces()

        distmatrix = None
        if paraSystem.int_type == 'global':
            CalcInteractionGlobal(agentData, paraSystem, paraAgents)
        if paraSystem.int_type == 'matrix':
            distmatrix = CalcInteractionMatrix(agentData, paraSystem)
        elif paraSystem.int_type == 'voronoi':
            edges = CalcInteractionVoronoi(agentData, paraSystem, paraAgents)

        if paraSystem.startle:
            UpdateStartle(agentData, paraSystem, paraAgents, edges=edges, distmatrix=distmatrix, step=s)
        UpdateTotalSocialForces(agentData, paraSystem, paraAgents)

        # this is a hack to temporarily allow the prey fish to move
        # when it startles. this is achieved by calling UpdateCartesian
        # with a nonzero value for velproj so that uw and up have nonzero
        # values when UpdateCoordinates is called
        # agentData.phi[startleAgent] = agentData.phiDes_startle[startleAgent]
        # agentData.velproj[startleAgent] = 1.0
        # agentData.UpdateCartesian()
        # agentData.velproj[startleAgent] = 0

        UpdateCoordinates(agentData, paraSystem, paraAgents)

        if (s % paraSystem.stepout) == 0:
            # print('t=%02f' % (paraSystem.dt*s))

            outdata = SaveOutData(s, agentData, outdata, paraSystem)
            x = agentData.pos[:, 0]
            y = agentData.pos[:, 1]
            if paraSystem.output < 1.0:
                tail_length = int(3. / paraSystem.output)
            else:
                tail_length = 5
            postail = np.array(outdata['pos'][-tail_length:-1])
            xtail = np.reshape(postail[:, :, 0], (-1, 1))
            ytail = np.reshape(postail[:, :, 1], (-1, 1))

            points.set_data(x, y)
            pointstail.set_data(xtail, ytail)

            current_time = np.array(outdata['t'])
            current_startlevar = np.array(outdata['v_m'])
            startlevar_line.set_data(current_time, current_startlevar[:, startleAgent])

            if paraSystem.BC == -1:
                meanx = np.mean(x)
                meany = np.mean(y)
                ax1.set_xlim(meanx - 0.5 * paraSystem.L, meanx + 0.5 * paraSystem.L)
                ax1.set_ylim(meany - 0.5 * paraSystem.L, meany + 0.5 * paraSystem.L)

            if doblit:
                # restore background
                fig.canvas.restore_region(background)

                # redraw just the points
                ax1.draw_artist(points)

                # fill in the axes rectangle
                fig.canvas.blit(ax1.bbox)

            else:
                # redraw everything
                ax1.set_title('t=%02f' % (paraSystem.dt * s))
                fig.canvas.draw()

    # input("Press Enter to continue...")
    # For Python 2.x use below:
    # raw_input("Press Enter to continue...")
    pl.close(fig)
    return outdata, agentData


def SingleRunWithStartlePotential(paraSystem, paraAgents, agentData=None, startleAgent=0):
    """ perform a single simulation run """

    # initialize agents if not provided at run time
    if agentData is None:
        agentData = AgentData(paraSystem)
        InitAgents(agentData, paraSystem, paraAgents)

    outdata = SaveOutData(0, agentData, None, paraSystem)
    edges = None
    if (paraSystem.int_type == 'global') or (paraSystem.int_type == 'matrix'):
        edges = []
        for i in range(paraSystem.N):
            for j in range(i, paraSystem.N):
                edges.append([i, j])

    # PERFORM TIME INTEGRATION
    for s in range(1, paraSystem.steps + 1):
        # reset social forces
        agentData.ResetForces()

        # calculate social interactions
        distmatrix = None
        if paraSystem.int_type == 'global':
            CalcInteractionGlobal(agentData, paraSystem, paraAgents)
        elif paraSystem.int_type == 'matrix':
            distmatrix = CalcInteractionMatrix(agentData, paraSystem)
        elif paraSystem.int_type == 'voronoi':
            edges = CalcInteractionVoronoi(agentData, paraSystem, paraAgents)

        # update startle dynamics
        if paraSystem.startle:
            UpdateStartle(agentData, paraSystem, paraAgents, edges=edges, distmatrix=distmatrix, step=s)

        # update social forces and coordinates
        UpdateTotalSocialForces(agentData, paraSystem, paraAgents)

        # this is a hack to temporarily allow the prey fish to move
        # when it startles. this is achieved by calling UpdateCartesian
        # with a nonzero value for velproj so that uw and up have nonzero
        # values when UpdateCoordinates is called
        agentData.phi[startleAgent] = agentData.phiDes_startle[startleAgent]
        agentData.velproj[startleAgent] = 1.0
        agentData.UpdateCartesian()
        agentData.velproj[startleAgent] = 0

        UpdateCoordinates(agentData, paraSystem, paraAgents)

        # simulation output
        if (s % paraSystem.stepout) == 0:
            outdata = SaveOutData(s, agentData, outdata, paraSystem)
    print('Done!')
    return outdata, agentData
