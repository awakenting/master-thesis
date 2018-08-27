import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import seaborn as sns
import time

from pypet.trajectory import Trajectory
import collective_behavior_analysis as cba

figure_path = '../../figures/results/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

sns.set()
sns.set_palette('colorblind')
sns_colors = sns.color_palette()
#mpl.rcParams.update(custon_pgf_rcparams)

def plotSavedAnimation(positionList, startle_list, L, doblit=False, sleepTime=0.01, stoptime_step=100):
    """Plot animation with position data from RunAnimation or SingleRun"""
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    fig = plt.figure(99, figsize=(10, 10))
    ax = plt.subplot()
    ax.set_aspect('equal')
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)

    x = positionList[0][:, 0]
    y = positionList[0][:, 1]
    plt.show(False)
    plt.draw()
    points = ax.plot(x, y, 'ro')[0]
    pointstail = ax.plot(x, y, 'r.', alpha=0.2)[0]

    firings = ax.plot(0, 0, 'k*')[0]
    firingsx = []
    firingsy = []

    if doblit:
        # cache the background
        background = fig.canvas.copy_from_bbox(ax.bbox)



    with writer.saving(fig, "firing_map_test_new.mp4", 100):
        for step in range(1, stoptime_step):
            x = positionList[step][:, 0]
            y = positionList[step][:, 1]
            tail_length = 5
            firstTailIndex = np.max([0, step - tail_length])
            postail = np.array(positionList[firstTailIndex: step])
            xtail = np.reshape(postail[:, :, 0], (-1, 1))
            ytail = np.reshape(postail[:, :, 1], (-1, 1))

            startle_mask = np.array(startle_list[step], dtype=bool)
            if np.any(startle_mask):
                firingsx.extend(positionList[step][startle_mask, 0])
                firingsy.extend(positionList[step][startle_mask, 1])
                if len(firingsx) > 5:
                    firingsx = firingsx[-5:]
                    firingsy = firingsy[-5:]

            points.set_data(x, y)
            pointstail.set_data(xtail, ytail)

            firings.set_data(firingsx, firingsy)

            # if doblit:
            #     # restore background
            #     fig.canvas.restore_region(background)
            #
            #     # redraw just the points
            #     ax.draw_artist(points)
            #
            #     # fill in the axes rectangle
            #     fig.canvas.blit(ax.bbox)
            #
            # else:
            #     # redraw everything
            #     ax.set_title('step=%02f' % (step))
            #     fig.canvas.draw()

            writer.grab_frame()

            time.sleep(sleepTime)
    plt.close(fig)


filename = os.path.join(os.path.expanduser('/extra/swarmstartle_results'),
                        'looming_swarm_fitted_model_fixed_rho_null_kmd_matrix.hdf5')

traj = Trajectory(filename=filename)

# Now we want to load all stored data.
traj.f_load(index=-1, load_parameters=2, load_results=1)
traj.v_auto_load = True

starttime = -5
endtime = 505
# for i in range(3):
seed_idx = 1
noise_vals = np.linspace(0.01, 0.2, 5)

#for noise_idx, noise_val in enumerate(noise_vals[[2, 4]]):
noise_idx = 3
noise_val = noise_vals[noise_idx]
filter_params = ['speed0', 'noisep']

def filter_func(speed0, noisep):
    return speed0 == 1.125 and noisep == noise_val


idx_iterator = traj.f_find_idx(filter_params, filter_func)

pos_data = []
uw_data = []
vis_method_data = []
startle_data = []
vm_data = []
visangle_data = []
# fill result arrays
for run_idx in idx_iterator:
    traj.v_idx = run_idx
    pos_data.append(traj.f_get('results.outdata.crun.pos', fast_access=True, auto_load=True))
    startle_data.append(traj.f_get('results.outdata.crun.startle', fast_access=True, auto_load=True))

startles = np.array(startle_data[seed_idx])

#pos, startles = load_pos_data(metric_filename, 1.0, 0.07, 10, 3, 0)

pos = np.array(pos_data[seed_idx])
stoptime = 100
stop_idx = int(stoptime/traj.par.output)

plotSavedAnimation(pos, startles, L=50, sleepTime=0.01, stoptime_step=stop_idx)

