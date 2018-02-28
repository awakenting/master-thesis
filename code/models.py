import numpy as np
from numba import jit

def generate_stimulus(stim_size=1, speed=1, length=10, dt=0.1, init_distance=50, init_period=0):
    total_length = length + init_period
    stim_timesteps = np.arange(int(length/dt))*dt
    all_timesteps = np.arange(int(total_length/dt))*dt
    
    init_period_distances = np.ones(int(init_period/dt))*init_distance
    distances = np.concatenate((init_period_distances, init_distance - stim_timesteps*speed))
    
    angles = np.arctan2(stim_size/2, distances)*2
    angles_degrees = angles/np.pi*180
    return all_timesteps, angles_degrees, distances

def f_theta_linear(theta, m, b):
    return theta*m + b

def transform_stim(stim_size, speed, length, dt, m=1.5, b=0, init_period=0):
    t, stims, dists = generate_stimulus(stim_size=stim_size, speed=speed, length=length, dt=dt, init_period=init_period)
    collision_idx = np.argmin(np.abs(dists))
    t_collision = t[collision_idx]
    stim_collision = stims[collision_idx]
    before_collision_mask = t<t_collision
    t_to_collision = t[before_collision_mask] - t_collision
    stim_to_collision = stims[before_collision_mask]
    transformed_stim_to_collision = f_theta_linear(stim_to_collision, m=m, b=b)
    transformed_stims = f_theta_linear(stims, m=m, b=b)
    return t, stims, transformed_stims, dists, t_to_collision, transformed_stim_to_collision

def lif_dynamics(tau_m, e_l, r_m, stimulus, noise, v_t, dt, total_time, init_vm_std, vt_std):
    ntime_steps = int(total_time/dt)
    time_points = np.arange(ntime_steps)*dt
    v_m = np.zeros(ntime_steps)
    v_m[0] = np.random.normal(loc=e_l, scale=init_vm_std)
    v_t = np.random.normal(loc=v_t, scale=vt_std, size=ntime_steps)
    t_spks = []
    idx_spks = []
    # integration:
    t = 1
    while t < ntime_steps:
        v_m[t] = v_m[t - 1] + dt*(- (v_m[t - 1] - e_l) + r_m*stimulus[t])/tau_m + noise[t]
        if v_m[t] > v_t[t]:
            v_m[t] = e_l
            t_spks.append(t*dt)
            idx_spks.append(t)
        t = t + 1

    return time_points, v_m, t_spks, idx_spks

@jit
def jit_lif_dynamics(tau_m, e_l, r_m, stimulus, noise, v_t, dt, total_time, init_vm_std, vt_std, init_period):
    ntime_steps = int((total_time + init_period) / dt)
    time_points = np.arange(ntime_steps)*dt
    v_m = np.zeros(ntime_steps)
    v_m[0] = np.random.normal(loc=e_l, scale=init_vm_std)
    v_t = np.random.normal(loc=v_t, scale=vt_std, size=ntime_steps)
    t_spks = []
    idx_spks = []
    # integration:
    t = 1
    while t < ntime_steps:
        v_m[t] = v_m[t - 1] + dt*(- (v_m[t - 1] - e_l) + r_m*stimulus[t])/tau_m + noise[t]
        if v_m[t] > v_t[t]:
            v_m[t] = e_l
            if t*dt > init_period:
                t_spks.append(t*dt)
                idx_spks.append(t)
        t = t + 1

    return time_points, v_m, t_spks, idx_spks

def calc_response(params):
    lv = np.random.rand()*1.1 + 0.1
    stim_size = np.random.rand()*15 + 10
    speed = 1/(lv/stim_size)
    t, stims, tstims, dists, t_to_coll, tstim_to_coll = transform_stim(stim_size, speed, params['total_time'],
                                                                       params['dt'], params['m'], params['b'],
                                                                       params['init_period'])

    stimulus = tstims*1e-11
    sigma = params['noise_std'] * np.sqrt(params['dt'])
    noise_vals = np.random.normal(loc=0.0, scale=sigma, size=len(stimulus))

    #np.random.seed(1)
    time, v_m, spks, spk_idc = jit_lif_dynamics(params['tau_m'], params['e_l'], params['r_m'], stimulus, noise_vals,
                                                params['v_t'], params['dt'], params['total_time'], params['init_vm_std'],
                                                params['vt_std'], params['init_period'])
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

def lv_map(lv):
    if lv > 0.1 and lv < 0.28:
        return 0.19
    elif lv > 0.28 and lv < 0.47:
        return 0.38
    elif lv > 0.47 and lv < 0.65:
        return 0.56
    elif lv > 0.65 and lv < 0.83:
        return 0.74
    elif lv > 0.83 and lv < 1.01:
        return 0.92
    else:
        return 1.11
