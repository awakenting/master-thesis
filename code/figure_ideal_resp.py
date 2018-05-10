import os
import numpy as np

import matplotlib as mpl
mpl.use("pgf")
general_fontsize = 16
custon_pgf_rcparams = {
    "font.family": 'serif',
    "font.serif": 'cm',
    'font.size': general_fontsize,
    'xtick.labelsize': general_fontsize,
    'ytick.labelsize': general_fontsize,
    'axes.labelsize': general_fontsize,
    'legend.fontsize': 12,
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
    "text.usetex": True,    # use inline math for ticks
}

import matplotlib.pyplot as plt
import seaborn as sns
import models as md

figure_path = '../figures/results/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

sns.set()
mpl.rcParams.update(custon_pgf_rcparams)


def calc_analytic_resp_props(l, lv, d_init, critical_angle):
    v = l / lv
    tan_ca = np.tan((critical_angle / 2) / 180 * np.pi)
    resp_time = d_init / v - l / (2 * tan_ca * v)
    resp_dist = d_init - v * resp_time

    t_collision = d_init / v
    resp_ttc = resp_time - t_collision
    return resp_time, resp_dist, resp_ttc

d_init = 50
critical_angle = 35
analytic_resp_props = {}
stim_size_labels = ['small', 'big']
analytic_lvs = np.arange(0.1, 1.2, step=0.01)
for l, stim_size_label in zip([10, 25], stim_size_labels):
    analytic_resp_times, analytic_resp_dists, analytic_resp_ttcs = calc_analytic_resp_props(l, analytic_lvs,
                                                                                            d_init, critical_angle)
    analytic_resp_props[stim_size_label] = {'time': analytic_resp_times,
                                            'distance': analytic_resp_dists,
                                            'ttc': analytic_resp_ttcs}

trial_length = 10
dt = 0.0001
m = 1
b = 0
critical_angle = 35

lv_list = []
stim_size_list = []
resp_t_list = []
resp_dist_list = []
resp_ttc_list = []

for i in range(1000):
    lv = np.random.rand()*1.1 + 0.1
    stim_size = np.random.rand()*15 + 10
    speed = 1/(lv/stim_size)
    m=1
    t, stims, tstims, dists, t_to_collision, transformed_stim_to_collision = md.transform_stim(stim_size, speed, trial_length, dt, m, b)

    t_resp_idx = np.argmin(np.abs(tstims - critical_angle))
    lv_list.append(lv)
    stim_size_list.append(stim_size)
    resp_t_list.append(t[t_resp_idx])
    resp_ttc_list.append(t_to_collision[t_resp_idx])
    resp_dist_list.append(dists[t_resp_idx])

lvs = np.array(lv_list)
stim_sizes = np.array(stim_size_list)
resp_times = np.array(resp_t_list)
resp_ttcs = np.array(resp_ttc_list)
resp_dists = np.array(resp_dist_list)

vmin = np.min(stim_sizes)
vmax = np.max(stim_sizes)
sm = plt.cm.ScalarMappable(cmap=mpl.cm.inferno, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))

ax1.scatter(resp_dists, lvs, c=sm.to_rgba(stim_sizes))
ax1.plot(analytic_resp_props['small']['distance'], analytic_lvs, 'r', label='analytical boundary')
ax1.plot(analytic_resp_props['big']['distance'], analytic_lvs, 'r')

ax1.set_xlabel('Response distance [mm]')
ax1.set_ylabel('L\/V [s]')
ax1.set_xlim([0, d_init])
ax1.legend()

ax2.scatter(resp_times, lvs, c=sm.to_rgba(stim_sizes))
ax2.plot(analytic_resp_props['small']['time'], analytic_lvs, 'r', label='analytical boundary')
ax2.plot(analytic_resp_props['big']['time'], analytic_lvs, 'r')

ax2.set_xlabel('Response time [s]')
ax2.set_ylabel('L\/V [s]')
ax2.legend()

ax3.scatter(resp_ttcs, lvs, c=sm.to_rgba(stim_sizes))
ax3.plot(analytic_resp_props['small']['ttc'], analytic_lvs, 'r', label='analytical boundary')
ax3.plot(analytic_resp_props['big']['ttc'], analytic_lvs, 'r')

ax3.set_xlabel('Time to collision [mm]')
ax3.set_ylabel('L\/V [s]')
ax3.legend()

axes = [ax1, ax2, ax3]
letters = ['A', 'B', 'C']
for ax, letter in zip(axes, letters):
    ax.text(-0.05, 1.05, letter, color='k', weight='bold', fontsize=20, transform=ax.transAxes,
            ha='center', va='center')

sm.set_array([])
cbar = plt.colorbar(sm, ax=ax3)
cbar.set_label('Stimulus size [mm]')
#plt.show()
plt.savefig(os.path.join(figure_path, 'figure_ideal_resp.pdf'), bbox_inches='tight')
