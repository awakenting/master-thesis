
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from numba import jit
import models as md

# # Feedforward inhibition model

# In[4]:


#@jit
def jit_ffi_model(tau_m, e_l, r_m, stimulus, noise_exc, noise_inh, v_t, dt, total_time, init_vm_std, vt_std, rho_null,
                  tau_inh, rho_scale, init_period):
    ntime_steps = int((total_time+init_period)/dt)
    time_points = np.arange(ntime_steps)*dt
    v_m = np.zeros(ntime_steps)
    v_m[0] = np.random.normal(loc=e_l, scale=init_vm_std)
    v_t = np.random.normal(loc=v_t, scale=vt_std, size=ntime_steps)
    rho_inh = np.zeros(ntime_steps)
    t_spks = []
    idx_spks = []
    # integration:
    t = 1
    while t < ntime_steps:
        # calculate activation of inhibitory population
        rho_inh[t] = rho_inh[t-1] + dt*(rho_null - rho_inh[t-1] + rho_scale*stimulus[t])/tau_inh + noise_inh[t]
        # calculate lif dynamics with inhibitory input
        v_m[t] = v_m[t - 1] + dt*(- (v_m[t - 1] - e_l) + r_m*stimulus[t] - rho_inh[t-1])/tau_m + noise_exc[t]
        if v_m[t] > v_t[t]:
            v_m[t] = e_l
            if t*dt > init_period:
                t_spks.append(t*dt)
                idx_spks.append(t)
        t = t + 1
    
    return time_points, v_m, t_spks, idx_spks, rho_inh


# In[95]:


def calc_response(params):
    # sample lv values uniformly between 0.1 and 1.2
    lv = np.random.rand()*1.1 + 0.1
    # sample stimulus sizes (L) uniformly between 10 and 25
    stim_size = np.random.rand()*5 + 20
    speed = 1/(lv/stim_size)
    t, stims, tstims, dists, t_to_coll, tstim_to_coll = md.transform_stim(stim_size, speed, params['total_time'],
                                                                          params['dt'], params['m'], params['b'],
                                                                          params['init_period'], params['cutoff_angle'])

    stimulus = tstims*1e-11
    sigma_exc = params['noise_std_exc'] * np.sqrt(params['dt'])
    sigma_inh = params['noise_std_inh'] * np.sqrt(params['dt'])
    noise_exc = np.random.normal(loc=0.0, scale=sigma_exc, size=len(stimulus))
    noise_inh = np.random.normal(loc=0.0, scale=sigma_inh, size=len(stimulus))

    #np.random.seed(1)
    time, v_m, spks, spk_idc, rho_inh = jit_ffi_model(params['tau_m'], params['e_l'], params['r_m'], stimulus, noise_exc,
                                                      noise_inh, params['v_t'], params['dt'], params['total_time'],
                                                      params['init_vm_std'], params['vt_std'], params['rho_null'],
                                                      params['tau_inh'], params['rho_scale'], params['init_period'])
    if not len(spks)==0:
        first_spike = spks[0]
        first_spike_idx = spk_idc[0]
    else:
        first_spike = 0
        first_spike_idx = 0
        
    if not first_spike_idx >= len(t_to_coll):
        resp_in_t_to_coll = t_to_coll[first_spike_idx]
    else:
        resp_in_t_to_coll = 0
        
    return stims[first_spike_idx], dists[first_spike_idx], first_spike, lv, stim_size, speed, resp_in_t_to_coll


total_time = 5
dt = 0.001

params = {'tau_m': 0.023,
          'e_l': -0.079,
          'r_m': 10*1e6,
          'v_t': -0.061,
          'init_vm_std': 0,
          'vt_std': 0,
          'rho_null': 0,
          'tau_inh': 0.001,
          'rho_scale': 7.6*1e6,
          'dt': dt,
          'total_time': total_time,
          'init_period': 2,
          'noise_std_exc': 0,
          'noise_std_inh': 0,
          'n_timepoints': int(total_time/dt),
          'cutoff_angle': 180,
          'm': 5,
          'b': 0}

nruns = 100
rstims = np.zeros(nruns)
rdists = np.zeros(nruns)
reaction_times = np.zeros(nruns)
speeds = np.zeros(nruns)
for i in np.arange(nruns):
    rstims[i], rdists[i], reaction_times[i], lv, stim_size, speeds[i], resp_in_t_to_coll = calc_response(params)

sns.set()
sns.distplot(rstims)
print('Mean visual angle: ' + str(np.mean(rstims)))
plt.figure()
sns.distplot(rdists)
print('Mean distance: ' + str(np.mean(rdists)))
plt.figure()
sns.distplot(reaction_times)
plt.figure()
sns.distplot(speeds)
print('Mean reaction time: ' + str(np.mean(reaction_times)))
print('resp in ttc: ' + str(resp_in_t_to_coll))
plt.show()
