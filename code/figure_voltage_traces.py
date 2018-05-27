import os
import numpy as np

import matplotlib as mpl

mpl.use("pgf")
general_fontsize = 20
custon_pgf_rcparams = {
    'font.family': 'serif',
    'font.serif': 'cm',
    'font.size': general_fontsize,
    'xtick.labelsize': general_fontsize,
    'ytick.labelsize': general_fontsize,
    'axes.labelsize': general_fontsize,
    'axes.titlesize': general_fontsize,
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
    'text.usetex': True  # use inline math for ticks
}

import matplotlib.pyplot as plt
import seaborn as sns
import models as md

figure_path = '../figures/results/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

sns.set()
sns.set_palette('colorblind')
sns_colors = sns.color_palette()
mpl.rcParams.update(custon_pgf_rcparams)

default_params = {'tau_m': 0.023,
                  'e_l': -0.079,
                  'r_m': 10*1e6,
                  'v_t': -0.061,
                  'init_vm_std': 0.000,
                  'vt_std': 0.000,
                  'rho_null': 0,
                  'rho_null_std': 0,
                  'tau_inh': 0.005,
                  'rho_scale': 9.0*1e6,
                  'exc_scale': 30,
                  'dt': 0.0001,
                  'total_time': 5,
                  'init_period': 0.2,
                  'noise_std_exc': 1*1e-3,
                  'noise_std_inh': 0*1e-3,
                  'cutoff_angle': 180,
                  'm': 1,
                  'b': 0,
                  'lv_min': 0.1,
                  'lv_max': 1.2,
                  'l_min': 10,
                  'l_max': 25,
                  'init_distance': 50}


# generate looming stimulus
LV_vals = np.array([0.19, 0.38, 0.56, 0.74, 0.93, 1.11])
stim_size = 10
speeds = 1/(LV_vals/stim_size)
speed = speeds[5]
cutoff_angle = 180

fig, axes = plt.subplots(4, 2, figsize=(12, 16))
plt.subplots_adjust(hspace=0.5, wspace=0.3, bottom=0.2, left=0.12, right=0.94)

column_titles = ['Slow stimulus', 'Fast stimulus']

for stim_idx, (stim_size, speed_idx) in enumerate(zip([10, 25], [5, 0])):
    speed = speeds[speed_idx]
    t, stims, tstims, dists, t_to_coll, trans_stim_to_coll = md.transform_stim(stim_size, speed,
                                                                               default_params['total_time'],
                                                                               default_params['dt'],
                                                                               default_params['m'],
                                                                               default_params['b'],
                                                                               default_params['init_period'],
                                                                               default_params['cutoff_angle'])

    stimulus = tstims*default_params['exc_scale']*1e-11
    sigma_exc = default_params['noise_std_exc'] * np.sqrt(default_params['dt'])
    sigma_inh = default_params['noise_std_inh'] * np.sqrt(default_params['dt'])
    noise_exc = np.random.normal(loc=0.0, scale=sigma_exc, size=len(stimulus))
    noise_inh = np.random.normal(loc=0.0, scale=sigma_inh, size=len(stimulus))

    rho_null = np.random.lognormal(mean=default_params['rho_null'], sigma=default_params['rho_null_std'])
    rho_null = rho_null / 1000

    time, v_m, spks, spk_idc, rho_inh = md.jit_ffi_model(default_params['tau_m'], default_params['e_l'],
                                                         default_params['r_m'], stimulus, noise_exc, noise_inh,
                                                         default_params['v_t'], default_params['dt'],
                                                         default_params['total_time'], default_params['init_vm_std'],
                                                         default_params['vt_std'], rho_null,
                                                         default_params['tau_inh'], default_params['rho_scale'],
                                                         default_params['init_period'])

    if not len(spks)== 0:
        first_spike = spks[0]
        first_spike_idx = spk_idc[0]
    else:
        first_spike = -1
        first_spike_idx = -1

    stat_inh_time, stat_inh_v_m, stat_inh_spks, stat_inh_spk_idc, stat_inh_rho_inh = md.jit_stat_inh_model(default_params['tau_m'], default_params['e_l'],
                                                         default_params['r_m'], stimulus, noise_exc, noise_inh,
                                                         default_params['v_t'], default_params['dt'],
                                                         default_params['total_time'], default_params['init_vm_std'],
                                                         default_params['vt_std'], rho_null,
                                                         default_params['tau_inh'], default_params['rho_scale'],
                                                         default_params['init_period'])


    stat_sigma_exc = default_params['noise_std_exc']
    stat_sigma_inh = default_params['noise_std_inh']
    stat_noise_exc = np.random.normal(loc=0.0, scale=sigma_exc, size=len(stimulus))
    stat_noise_inh = np.random.normal(loc=0.0, scale=sigma_inh, size=len(stimulus))

    stat_rho_inh = rho_null + stimulus*default_params['rho_scale'] + stat_noise_inh
    stat_v_m = default_params['e_l'] + stimulus * (default_params['r_m'] - default_params['rho_scale']) - rho_null - stat_noise_inh + stat_noise_exc
    above_threshold_idc = np.where(stat_v_m > default_params['v_t'])[0]

    if not len(above_threshold_idc) == 0:
        out_of_init_period_idc = above_threshold_idc[above_threshold_idc * default_params['dt'] > default_params['init_period']]
        stat_first_spike_idx = out_of_init_period_idc[0]
        stat_first_spike = time[stat_first_spike_idx]

    plot_time_mask = slice(0, first_spike_idx)
    plot_time = time[plot_time_mask]

    stat_inh_plot_time_mask = slice(0, stat_inh_spk_idc[0])
    stat_inh_plot_time = time[stat_inh_plot_time_mask]

    stat_plot_time_mask = slice(0, stat_first_spike_idx)
    stat_plot_time = time[stat_plot_time_mask]

    plot_lw = 2.5

    axes[0, stim_idx].plot(stat_inh_plot_time, stims[stat_inh_plot_time_mask])
    axes[0, stim_idx].set_title(column_titles[stim_idx])
    axes[0, stim_idx].set_ylabel(r'$\theta$ (t) [\textdegree]')
    axes[0, stim_idx].set_yticks(np.arange(20, 90, step=10))
    axes[0, stim_idx].set_yticklabels(np.arange(20, 90, step=10))
    axes[0, stim_idx].set_ylim([10, 70])

    axes[1, stim_idx].plot(plot_time, rho_inh[plot_time_mask], lw=plot_lw)
    axes[1, stim_idx].plot(stat_inh_plot_time, stat_inh_rho_inh[stat_inh_plot_time_mask], lw=plot_lw)
    axes[1, stim_idx].plot(stat_plot_time, stat_rho_inh[stat_plot_time_mask], lw=plot_lw)
    axes[1, stim_idx].plot(plot_time, stimulus[plot_time_mask]*default_params['r_m'], lw=plot_lw, label='I(t)')
    axes[1, stim_idx].set_ylabel(r'Inputs')
    axes[1, stim_idx].legend()


    axes[2, stim_idx].plot(plot_time, stimulus[plot_time_mask]*default_params['r_m'] - rho_inh[plot_time_mask], lw=plot_lw, label=r'$\rho$(t) full model')
    axes[2, stim_idx].plot(stat_inh_plot_time, stimulus[stat_inh_plot_time_mask]*default_params['r_m'] - stat_inh_rho_inh[stat_inh_plot_time_mask], lw=plot_lw, label=r'$\rho$(t) stat. inh. model')
    axes[2, stim_idx].plot(stat_plot_time, stimulus[stat_plot_time_mask]*default_params['r_m'] - stat_rho_inh[stat_plot_time_mask], lw=plot_lw, label=r'$\rho$(t) fully stat. model')
    axes[2, stim_idx].set_ylabel(r'Effective input')
    axes[2, stim_idx].set_ylim([0, 0.02])

    axes[3, stim_idx].plot(plot_time, v_m[plot_time_mask]*1000, lw=plot_lw, label='full model')
    axes[3, stim_idx].plot(stat_inh_plot_time, stat_inh_v_m[stat_inh_plot_time_mask]*1000, lw=plot_lw, label='stat. inh. model')
    axes[3, stim_idx].plot(stat_plot_time, stat_v_m[stat_plot_time_mask]*1000, lw=plot_lw, label='fully stat. model')
    axes[3, stim_idx].set_xlabel('Time [s]')
    axes[3, stim_idx].set_ylabel('$V_m$ [mV]')
    if not len(spks) == 0:
        axes[3, stim_idx].plot(spks[0], default_params['v_t']*1000, 'r*')
        axes[3, stim_idx].plot(stat_inh_spks[0], default_params['v_t']*1000, 'r*')
        axes[3, stim_idx].plot(stat_first_spike, default_params['v_t']*1000, 'r*')

    if stim_idx == 1:
        lgd = axes[3, stim_idx].legend(loc='upper left', bbox_to_anchor=[-0.5, -0.2])


    from matplotlib.patches import ConnectionPatch
    model_spk_idc = [first_spike_idx, stat_inh_spk_idc[0], stat_first_spike_idx]
    model_spks = [first_spike, stat_inh_spks[0], stat_first_spike]
    for i, (cspk_idx, cspk) in enumerate(zip(model_spk_idc, model_spks)):
        xyA = (cspk, default_params['v_t']*1000)
        xyB = (cspk, stims[cspk_idx])
        con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA="data", coordsB="data",
                              axesA=axes[3, stim_idx], axesB=axes[0, stim_idx], linestyle=':', linewidth=2, color=sns_colors[i])
        con.set_annotation_clip(False)
        axes[3, stim_idx].add_artist(con)

        axes[0, stim_idx].hlines(stims[cspk_idx], 0, cspk, lw=2, linestyles=':', colors=sns_colors[i])

#plt.show()
plt.savefig(os.path.join(figure_path, 'figure_voltage_traces.pdf'), bbox_inches='tight')
