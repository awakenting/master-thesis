
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.io as sio


data = sio.loadmat('LVsVersusSubtendedAngle.mat')
clean_dict = {'lv': np.squeeze(data['LVs']), 'resp_angle': np.squeeze(data['subtendedAngleAtResponse'])}
df = pd.DataFrame(clean_dict)
df.describe()



import delfi
from delfi.simulator import GaussMixture
from delfi.simulator.BaseSimulator import BaseSimulator
from delfi.inference import SNPE
import delfi.distribution as dd
from delfi.summarystats import Identity
from delfi.generator import Default
from delfi.generator import MPGenerator
from delfi.utils import viz

import models as md


class FFI(BaseSimulator):
    def __init__(self, dim=1, seed=None, fixed_params=None):
        """Feedforward inhibition model simulator
        Integrates input until a given threshold is reached at which we
        define the response time.
        Parameters
        ----------
        dim : int
            Number of dimensions of parameters
        noise_cov : list
            Covariance of noise on observations
        seed : int or None
            If set, randomness is seeded
        """
        super().__init__(dim_param=dim, seed=seed)
        self.fixed_params = fixed_params

    @copy_ancestor_docstring
    def gen_single(self, param):
        # See BaseSimulator for docstring
        param = np.asarray(param).reshape(-1)
        assert param.ndim == 1
        assert param.shape[0] == self.dim_param

        model_params = self.fixed_params.copy()
        model_params['rho_null'] = param[0]
        model_params['noise_std_exc'] = param[1]*1e-3
        model_params['vt_std'] = param[2]*1e-3
        model_params['exc_scale'] = param[3]
        model_params['rho_scale'] = param[4]*1e6
        model_params['rho_null_std'] = param[5]
        nruns = 246
        rstims = np.zeros((1, nruns))
        lvs = np.zeros((1, nruns))
        for i in np.arange(nruns):
            rstims[0, i], rdist, reaction_time, lvs[0, i], stim_size, speed, rtime_to_coll = md.calc_response_ffi(params)

        data = np.concatenate((rstims, lvs), axis=1)
        return {'data': data}


total_time = 5
dt = 0.001

params = {'tau_m': 0.023,
          'e_l': -0.079,
          'r_m': 10*1e6,
          'v_t': -0.061,
          'init_vm_std': 0.001,
          'vt_std': 0.004,
          'rho_null': 2,
          'rho_null_std': 1,
          'tau_inh': 0.001,
          'rho_scale': 9.6*1e6,
          'exc_scale': 30,
          'dt': dt,
          'total_time': total_time,
          'init_period': 2,
          'noise_std_exc': 5*1e-3,
          'noise_std_inh': 5*1e-3,
          'n_timepoints': int(total_time/dt),
          'cutoff_angle': 180,
          'm': 3,
          'b': 0,
          'lv_min': 0.1,
          'lv_max': 1.2,
          'l_min': 10,
          'l_max': 25,
          'init_distance': 50}

ffim = FFI(dim=6, fixed_params=params)

# params are: m, noise_std, vt_std
p = dd.Uniform(lower=[1, 0, 0, 1, 9.6, 0], upper=[5, 20, 5, 5, 9.8, 5])
s = Identity()
#g = MPGenerator(models=[ffim]*5, prior=p, summary=s)
g = Default(model=ffim, prior=p, summary=s)



gparams, stats = g.gen(10)
plt.plot(stats[0, 0:246], stats[0, 246:], '.')
plt.xlabel(r'response angle')
plt.ylabel(r'lv')
plt.title('m = ' + str(gparams[1]))



expm_data = np.concatenate((clean_dict['resp_angle'], clean_dict['lv'])).reshape((1, 492))



inf_snpe = SNPE(generator=g, n_components=1, n_hiddens=[50]*3, obs=expm_data)
logs, tds, posteriors = inf_snpe.run(n_train=[500, 200], n_rounds=2, epochs=1000)
posterior = posteriors[-1]


# In[86]:


for log_idx, log in enumerate(logs):
    plt.figure()
    plt.plot(log['loss'])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Round ' + str(log_idx+1))

lims = np.array([[0, 15],[0, 20], [0, 15], [1, 50], [5, 12]])
#for comp_idx in range(2):
viz.plot_pdf(posterior.xs[1], lims=lims, labels_params=['rho_null', 'noise_std_exc', 'vt_std', 'exc_scale', 'rho_scale'],
             figsize=(12,12), ticks=True);
#plt.title('Component ' + str(comp_idx+1))

print(posterior.mean)
print(posterior.a)

