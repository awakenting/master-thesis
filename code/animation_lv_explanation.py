import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as manimation


target_lv = 0.5
init_angle_degree = 10
init_angle = init_angle_degree/180 * np.pi
l1 = 10
l2 = 25
v1 = l1/target_lv
v2 = l2/target_lv
init_dist1 = l1/(2*np.tan(init_angle/2))
init_dist2 = l2/(2*np.tan(init_angle/2))
print(init_dist1, init_dist2)
print(v1, v2)

length = 5
dt = 0.0001
t = np.arange(np.ceil(length/dt))*dt
dists1 = init_dist1 - v1*t
t_coll1_idx = np.argmin(np.abs(dists1))
dists2 = init_dist2 - v2*t
sample_idc = np.arange(0, t_coll1_idx, step=100)

vmin = np.min(0)
vmax = np.max(t[t_coll1_idx])
sm = plt.cm.ScalarMappable(cmap=mpl.cm.inferno, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
time_colors = sm.to_rgba(t[sample_idc])


FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=30, metadata=metadata)

fig = plt.figure(figsize=(12, 12))
ax = plt.subplot()
ax.set_aspect('equal')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show(False)
plt.draw()

stim1 = ax.plot([-dists1[0], -dists1[0]], [-l1/2, l1/2], color='tab:red', lw=3, label='stimulus 1')[0]
stim2 = ax.plot([-dists2[0], -dists2[0]], [-l2/2, l2/2], color='tab:blue', lw=3, label='stimulus 2')[0]


angle_line = ax.plot([-dists2[0], 0], [l2/2, 0], 'k')[0]

ax.plot(0, 0, '.', ms=20, label='fish')
ax.hlines(0, -150, 0, linestyles='--')
ax.set_xlabel('Distance to fish')
ax.set_ylabel('')
ax.legend()


with writer.saving(fig, "lv_explanation_test.mp4", 100):
    for tpoint in sample_idc:
        stim1.set_data([-dists1[tpoint], -dists1[tpoint]], [-l1/2, l1/2])
        stim2.set_data([-dists2[tpoint], -dists2[tpoint]], [-l2 / 2, l2 / 2])
        angle_line.set_data([-dists2[tpoint], 0], [l2/2, 0])
        writer.grab_frame()
    
plt.close(fig)