import numpy as np
from numba import jit


def f_theta_linear(theta, m, b):
    return theta*m + b


def generate_stimulus(stim_size=1, speed=1, length=10, dt=0.1, init_distance=50, init_period=0, cutoff_angle=None):
    total_length = length + init_period
    stim_timesteps = np.arange(np.ceil(length/dt))*dt
    all_timesteps = np.arange(np.ceil(total_length/dt))*dt
    
    init_period_distances = np.ones(np.ceil(init_period/dt).astype(int))*init_distance
    distances = np.concatenate((init_period_distances, init_distance - stim_timesteps*speed))
    
    angles = np.arctan2(stim_size/2, distances)*2
    angles_degrees = angles/np.pi*180
    if not cutoff_angle is None:
        cutoff_mask = angles_degrees > cutoff_angle
        angles_degrees[cutoff_mask] = cutoff_angle
    return all_timesteps, angles_degrees, distances


def transform_stim(stim_size, speed, length, dt, m=1.5, b=0, init_period=0, cutoff_angle=None, init_distance=50):
    t, stims, dists = generate_stimulus(stim_size=stim_size, speed=speed, length=length, dt=dt, init_period=init_period,
                                        cutoff_angle=cutoff_angle, init_distance=init_distance)
    transformed_stims = f_theta_linear(stims, m=m, b=b)

    collision_idx = np.argmin(np.abs(dists))
    t_collision = t[collision_idx]
    before_collision_mask = t < t_collision

    t_to_collision = t[before_collision_mask] - t_collision
    transformed_stim_to_collision = transformed_stims[before_collision_mask]

    return t, stims, transformed_stims, dists, t_to_collision, transformed_stim_to_collision


def lif_dynamics(tau_m, e_l, r_m, stimulus, noise, v_t, dt, total_time, init_vm_std, vt_std, init_period):
    ntime_steps = int((total_time + init_period)/dt)
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
            if t * dt > init_period:
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


@jit
def jit_ffi_model(tau_m, e_l, r_m, stimulus, noise_exc, noise_inh, v_t, dt, total_time, init_vm_std, vt_std, rho_null,
                  tau_inh, rho_scale, init_period):
    ntime_steps = int((total_time + init_period) / dt)
    time_points = np.arange(ntime_steps) * dt
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
        rho_inh[t] = rho_inh[t - 1] + dt * (rho_null - rho_inh[t - 1] + rho_scale * stimulus[t]) / tau_inh + noise_inh[t]
        # calculate lif dynamics with inhibitory input
        v_m[t] = v_m[t - 1] + dt * (- (v_m[t - 1] - e_l) + r_m * stimulus[t] - rho_inh[t - 1]) / tau_m + noise_exc[t]
        if v_m[t] > v_t[t]:
            v_m[t] = e_l
            if t * dt > init_period:
                t_spks.append(t * dt)
                idx_spks.append(t)
        t = t + 1

    return time_points, v_m, t_spks, idx_spks, rho_inh


def calc_response_lif(params):
    # sample lv values uniformly between lv_min and lv_max
    lv = np.random.rand() * (params['lv_max'] - params['lv_min']) + params['lv_min']
    # sample stimulus sizes (L) uniformly between l_min and l_max
    # stim_size = np.random.rand()*(params['l_max']-params['l_min']) + params['l_min']
    stim_size = np.random.randint(params['l_min'], params['l_max'])
    speed = 1 / (lv / stim_size)
    t, stims, tstims, dists, t_to_coll, tstim_to_coll = transform_stim(stim_size, speed, params['total_time'],
                                                                       params['dt'], params['m'], params['b'],
                                                                       params['init_period'], params['cutoff_angle'],
                                                                       params['init_distance'])

    stimulus = tstims*1e-11
    sigma = params['noise_std'] * np.sqrt(params['dt'])
    noise_vals = np.random.normal(loc=0.0, scale=sigma, size=len(stimulus))

    time, v_m, spks, spk_idc = jit_lif_dynamics(params['tau_m'], params['e_l'], params['r_m'], stimulus, noise_vals,
                                                params['v_t'], params['dt'], params['total_time'], params['init_vm_std'],
                                                params['vt_std'], params['init_period'])
    if not len(spks) == 0:
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


def calc_response_ffi(params):
    # sample lv values uniformly between lv_min and lv_max
    lv = np.random.rand() * (params['lv_max'] - params['lv_min']) + params['lv_min']
    # sample stimulus sizes (L) uniformly between l_min and l_max
    # stim_size = np.random.rand()*(params['l_max']-params['l_min']) + params['l_min']
    stim_size = np.random.randint(params['l_min'], params['l_max'])
    speed = 1 / (lv / stim_size)
    t, stims, tstims, dists, t_to_coll, tstim_to_coll = transform_stim(stim_size, speed, params['total_time'],
                                                                       params['dt'], params['m'], params['b'],
                                                                       params['init_period'], params['cutoff_angle'],
                                                                       params['init_distance'])

    stimulus = tstims * 1e-11 * params['exc_scale']
    sigma_exc = params['noise_std_exc'] * np.sqrt(params['dt'])
    sigma_inh = params['noise_std_inh'] * np.sqrt(params['dt'])
    noise_exc = np.random.normal(loc=0.0, scale=sigma_exc, size=len(stimulus))
    noise_inh = np.random.normal(loc=0.0, scale=sigma_inh, size=len(stimulus))

    rho_null = np.random.lognormal(mean=params['rho_null'], sigma=params['rho_null_std'])
    #rho_null = np.random.exponential(scale=params['rho_null'])
    rho_null = rho_null / 1000
    time, v_m, spks, spk_idc, rho_inh = jit_ffi_model(params['tau_m'], params['e_l'], params['r_m'], stimulus,
                                                      noise_exc, noise_inh, params['v_t'], params['dt'],
                                                      params['total_time'], params['init_vm_std'], params['vt_std'],
                                                      rho_null, params['tau_inh'], params['rho_scale'],
                                                      params['init_period'])
    if not len(spks) == 0:
        first_spike = spks[0]
        first_spike_idx = spk_idc[0]
    else:
        first_spike = -1
        first_spike_idx = -1

    if not first_spike_idx >= len(t_to_coll):
        resp_in_t_to_coll = t_to_coll[first_spike_idx]
    else:
        resp_in_t_to_coll = 0
    
    return stims[first_spike_idx], dists[first_spike_idx], first_spike, lv, stim_size, speed, resp_in_t_to_coll


def lv_map(lv):
    if 0.1 < lv < 0.28:
        return 0.19
    elif 0.28 < lv < 0.47:
        return 0.38
    elif 0.47 < lv < 0.65:
        return 0.56
    elif 0.65 < lv < 0.83:
        return 0.74
    elif 0.83 < lv < 1.01:
        return 0.92
    else:
        return 1.11
