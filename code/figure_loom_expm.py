import os
import numpy as np

import matplotlib as mpl
mpl.use("pgf")
general_fontsize = 12
custon_pgf_rcparams = {
    "font.family": 'serif',
    "font.serif": 'cm',
    'font.size': general_fontsize,
    'xtick.labelsize': general_fontsize,
    'ytick.labelsize': general_fontsize,
    'axes.labelsize': general_fontsize,
    'legend.fontsize': 10,
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

LV_vals = np.array([0.1, 0.4, 0.8, 1.2])
stim_size_vals = [10, 25]
length = 10
dt = 0.001
m = 1
b = 0

vmin = np.min(LV_vals)
vmax = np.max(LV_vals)
sm = plt.cm.ScalarMappable(cmap=mpl.cm.viridis, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
for lv in LV_vals:
    stim_size1 = 25
    stim_size2 = 10
    speed1 = 1 / (lv / stim_size1)
    speed2 = 1 / (lv / stim_size2)
    t, stims, tstims1, dists, t_to_collision, transformed_stim_to_collision = md.transform_stim(stim_size1, speed1,
                                                                                                length, dt, m, b)
    t, stims, tstims2, dists, t_to_collision, transformed_stim_to_collision = md.transform_stim(stim_size2, speed2,
                                                                                                length, dt, m, b)

    ax1.fill_between(t, tstims1, tstims2, color=sm.to_rgba(lv), alpha=0.3, lw=2, linestyle='-')

    for stim_size in stim_size_vals:
        speed = 1/(lv/stim_size)
        t, stims, tstims, dists, t_to_collision, transformed_stim_to_collision = md.transform_stim(stim_size, speed, length, dt, m, b)
        if stim_size == stim_size_vals[0]:
            ax1.plot(t, tstims, c=sm.to_rgba(lv), label='l/v = ' + str(lv))
        else:
            ax1.plot(t, tstims, c=sm.to_rgba(lv))
ax1.plot([-0.05, -0.05], [tstims1[0], tstims2[0]], 'tab:red', lw=2, label='possible initial\nangles')

ax1.set_xlim([-0.1, 6])
ax1.set_ylim([0, 100])
ax1.set_xlabel("time [s]")
ax1.set_ylabel(r"$\theta(t)$ [\textdegree]")
ax1.legend()



lv = 1.2
angle_dist = 30
sns_colors = sns.color_palette()
plot_colors = [sm.to_rgba(lv), 'tab:red']
for stim_size_idx, stim_size in enumerate(stim_size_vals):
    speed = 1/(lv/stim_size)
    t, stims, tstims, dists, t_to_collision, transformed_stim_to_collision = md.transform_stim(stim_size, speed, length, dt, m, b)
    ax2.plot(t, tstims, color=plot_colors[stim_size_idx], label='l/v = ' + str(lv) + ', size = ' + str(stim_size))
    if stim_size == stim_size_vals[0]:
        target_tstims = tstims

    if stim_size == stim_size_vals[-1]:
        t_equal_angle_idx = np.argmin(np.abs(target_tstims - tstims[0]))
        t_equal_angle = t[t_equal_angle_idx]

        t_second_arrow_idx = np.argmin(np.abs(tstims - (tstims[0] + angle_dist)))
        t_third_arrow_idx = np.argmin(np.abs(tstims - (tstims[0] + 2*angle_dist)))

        ax2.plot([0.1, t_equal_angle - 0.2], [tstims[0], tstims[0]], color=plot_colors[1], ls='--')
        ax2.plot(t_equal_angle - 0.2, tstims[0], color=plot_colors[1], marker='>')

        ax2.plot([t[t_second_arrow_idx] + 0.1, t[t_second_arrow_idx] + t_equal_angle - 0.2],
                 [tstims[t_second_arrow_idx], tstims[t_second_arrow_idx]], color=plot_colors[1], ls='--')
        ax2.plot(t[t_second_arrow_idx] + t_equal_angle - 0.2, tstims[t_second_arrow_idx], color=plot_colors[1],
                 marker='>')

        ax2.plot([t[t_third_arrow_idx] + 0.1, t[t_third_arrow_idx] + t_equal_angle - 0.2],
                 [tstims[t_third_arrow_idx],tstims[t_third_arrow_idx]], color=plot_colors[1], ls='--')
        ax2.plot(t[t_third_arrow_idx] + t_equal_angle - 0.2, tstims[t_third_arrow_idx], color=plot_colors[1],
                 marker='>')

        ax2.plot(t + t_equal_angle, tstims, color=plot_colors[1], ls='--', label='l/v = ' + str(lv) + ', size = ' + str(stim_size) + ' shifted')


ax1.text(-0.05, 1.05, 'A', color='k', weight='bold', fontsize=20, transform=ax1.transAxes, ha='center', va='center')
ax2.text(-0.05, 1.05, 'B', color='k', weight='bold', fontsize=20, transform=ax2.transAxes, ha='center', va='center')

ax2.set_xlim([-0.1, 6])
ax2.set_ylim([0, 100])
ax2.set_xlabel(r"time [s]")
ax2.set_ylabel(r"$\theta(t)$ [\textdegree]")
ax2.legend()
#plt.show()
plt.savefig(os.path.join(figure_path, 'figure_theta_lv_test.pdf'), bbox_inches='tight')
