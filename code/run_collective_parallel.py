import os
import time
import numpy as np
import swarm_looming_startle as sw
from pypet import Environment
from pypet import pypetconstants
from pypet.utils.explore import cartesian_product

def store_outdata(traj, outdata):
    keys_to_store = ['pos', 'uw', 'startle', 'vis_angles', 'v_m']
    for key in keys_to_store:
        outdata_array = np.array(outdata[key])
        traj.f_add_result('outdata.$.' + key, outdata_array, comment='outdata')

def run_sim(traj):
    np.random.seed(traj.seed)

    # initialize system parameters
    paraSystem = sw.BaseParams(N=traj.N, L=traj.L, time=traj.total_time, dt=traj.dt,
                               BC=traj.BC, IC=traj.IC, output=traj.output,int_type=traj.int_type,
                               startle=traj.startle)
    # initialize prey parameters
    paraFish = sw.AgentParams(paraSystem, speed0=traj.speed0, alpha=traj.alpha,
                              repstrength=traj.repstrength, reprange=traj.reprange,
                              algstrength=traj.algstrength, algrange=traj.algrange,
                              attstrength=traj.attstrength, attrange=traj.attrange,
                              noisep=traj.noisep, noisev=traj.noisev,
                              amplitude_startle=traj.amplitude_startle,
                              duration_startle=traj.duration_startle,
                              duration_refractory=traj.duration_refractory,
                              r_m=traj.r_m,
                              tau_m=traj.tau_m,
                              e_l=traj.e_l,
                              v_t=traj.v_t,
                              vt_std=traj.vt_std,
                              tau_rho=traj.tau_rho,
                              rho_scale=traj.rho_scale,
                              exc_scale=traj.exc_scale,
                              noise_std_exc=traj.noise_std_exc,
                              noise_std_inh=traj.noise_std_inh,
                              print_startles=traj.print_startles,
                              vis_input_m=traj.vis_input_m,
                              vis_input_b=traj.vis_input_b,
                              vis_input_method=traj.vis_input_method,
                              vis_input_k=traj.vis_input_k
                              )

    outData, agentData = sw.SingleRun(paraSystem, paraFish)
    store_outdata(traj, outData)


# Create an environment that handles running
filename = os.path.join(os.path.expanduser('~/Documents/swarmstartle_results/hdf5'), 'looming_swarm_fitted_model_fixed_rho_null_explore.hdf5')
env = Environment(trajectory='looming_swarm',
                  filename=filename,
                  overwrite_file=True,
                  file_title='looming_swarm_simulation',
                  comment='The first exploration',
                  #git_repository='../SwarmStartle/',
                  #git_message='automatic commit by pypet:',
                  #git_fail=False,
                  multiproc=True,
                  ncores=6,
                  use_pool=True,  # Our runs are inexpensive we can get rid of overhead
                  # by using a pool
                  freeze_input=True,  # We can avoid some
                  # overhead by freezing the input to the pool
                  wrap_mode=pypetconstants.WRAP_MODE_QUEUE,
                  graceful_exit=True,  # We want to exit in a data friendly way
                  # that safes all results after hitting CTRL+C, try it ;-)
                  large_overview_tables=True  # To see a nice overview of all
                  # computed `z` values in the resulting HDF5 file.
                  # Per default disabled for more compact HDF5 files.
                  )

# The environment has created a trajectory container for us
traj = env.trajectory

# Add both parameters
traj.f_add_parameter('N', 40, comment='Number of fish')
traj.f_add_parameter('L', 50, comment='Length of the quadratic field')
traj.f_add_parameter('total_time', 500, comment='time of the simulation in seconds (per defintion)')
traj.f_add_parameter('dt', 0.001, comment='The size of the time step')
traj.f_add_parameter('seed', 999, comment='the seed value for numpy.random')
traj.f_add_parameter('print_startles', False, comment='whether startling events should be printed')

traj.f_add_parameter('alpha', 1.0, comment='factor that determines how fast the speed will relax to speed0')
traj.f_add_parameter('speed0', 1.0, comment='the default speed of the fish in terms of bodylength/second')
traj.f_add_parameter('noisep', 0.1, comment='the noise on the swimming direction')
traj.f_add_parameter('noisev', 0.0, comment='the noise on the swimming speed')
traj.f_add_parameter('BC', 0, comment='boundary condition, 1 means repelling boundaries and 0 means periodic boundaries')
traj.f_add_parameter('IC', 10, comment='initial condition, 10 means that intial position are restricted to a smaller area')
traj.f_add_parameter('repstrength', 1.0, comment='repulsion strength')
traj.f_add_parameter('algstrength', 0.5, comment='alignment strength')
traj.f_add_parameter('attstrength', 0.3, comment='attraction strength')
traj.f_add_parameter('reprange', 1.0, comment='repulsion range')
traj.f_add_parameter('algrange', 5.0, comment='alignment range')
traj.f_add_parameter('attrange', 25.0, comment='attraction range')
traj.f_add_parameter('output', 0.05, comment='the interval in which output data is stored')
traj.f_add_parameter('int_type', 'matrix', comment='the interaction type')

traj.f_add_parameter('startle', True, comment='whether fish should be able to startle or not')
traj.f_add_parameter('amplitude_startle', 50.0, comment='amplitude of the startling response')
traj.f_add_parameter('duration_startle', 0.070, comment='the duration of the startling response')
traj.f_add_parameter('duration_refractory', 1.0, comment='the refractory period for startling')

traj.f_add_parameter('r_m', 10*1e6, comment='membrane resistance in Ohm')
traj.f_add_parameter('tau_m', 0.023, comment='membrane time constant in seconds')
traj.f_add_parameter('e_l', -0.079, comment='membrane resting potential in Volt')
traj.f_add_parameter('v_t', -0.061, comment='firing threshold in Volt')
traj.f_add_parameter('vt_std', 0.000, comment='standard deviation of normal distribution around v_t in Volt')
traj.f_add_parameter('tau_rho', 0.001, comment='time constant of inhibitory population activity in seconds')
traj.f_add_parameter('rho_scale', 8.16*1e6, comment='scaling factor of visual input for inhibitory population')
traj.f_add_parameter('exc_scale', 30, comment='general scaling factor of visual input')
traj.f_add_parameter('noise_std_exc', 0.0027, comment='standard deviation of noise for M-cell')
traj.f_add_parameter('noise_std_inh', 0.000, comment='standard deviation of noise for inhibitory population')
traj.f_add_parameter('rho_null', 20.6, comment='default activity level of inhibitory population')
traj.f_add_parameter('rho_null_std', 0.0, comment='default activity level of inhibitory population')
traj.f_add_parameter('vis_input_m', 3, comment='slope of the linear transformation of the visual angle')
traj.f_add_parameter('vis_input_b', 0, comment='offset of the linear transformation of the visual angle')
traj.f_add_parameter('vis_input_method', 'knn_mean_deviate', comment='how to combine the visual input from all neighbors')
traj.f_add_parameter('vis_input_k', 3, comment='number of neighbors to consider if vis_input_method is "knn_mean"')

# Explore the parameters with a cartesian product
traj.f_explore(cartesian_product({'seed': np.arange(200, 203).tolist(),
                                  'speed0': np.linspace(0.5, 3.0, 5).tolist(),
                                  'noisep': np.linspace(0.01, 0.2, 5).tolist(),
                                  'int_type': ['matrix', 'voronoi_matrix'],
                                  'vis_input_method': ['max', 'knn_mean', 'knn_mean_deviate']
                                  }))

# Run the simulation
starttime = time.time()

env.run(run_sim)

endtime = time.time()
print('Total time needed: ' + str(int((endtime - starttime))) + ' seconds or '
      + str(int((endtime - starttime) / 60)) + ' min '
      + 'or ' + str(int((endtime - starttime) / 3600)) + ' hours')

env.disable_logging()