import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.io as sio


data = sio.loadmat('LVsVersusSubtendedAngle.mat')
clean_dict = {'lv': np.squeeze(data['LVs']), 'resp_angle': np.squeeze(data['subtendedAngleAtResponse'])}
df = pd.DataFrame(clean_dict)

import delfi
from delfi.simulator import GaussMixture
from delfi.simulator.BaseSimulator import BaseSimulator
from delfi.inference import SNPE
import delfi.distribution as dd
from delfi.summarystats import Identity
from delfi.generator import Default

import models as md


class LIF(BaseSimulator):
    def __init__(self, dim=1, seed=None, fixed_params=None):
        """Leaky Integrate-and-Fire simulator
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
        model_params['m'] = param[0]
        model_params['noise_std'] = param[1]*1e-3
        model_params['vt_std'] = param[2]*1e-3
        nruns = 246
        rstims = np.zeros((1, nruns))
        lvs = np.zeros((1, nruns))
        for i in np.arange(nruns):
            rstims[0, i], rdist, reaction_time, lvs[0, i], stim_size, speed, rtime_to_coll = md.calc_response(params)

        data = np.concatenate((rstims, lvs), axis=1)
        return {'data': data}

total_time = 5
dt = 0.001
params = {'tau_m': 0.023,
          'e_l': -0.079,
          'r_m': 10*1e6,
          'v_t': -0.061,
          'init_vm_std': 0.000,
          'vt_std': 0.000,
          'dt': dt,
          'total_time': total_time,
          'init_period': 0,
          'noise_std': 5*1e-3,
          'n_timepoints': int(total_time/dt),
          'm': 5,
          'b': 0,
          'cutoff_angle': 180}

lifm = LIF(dim=3, fixed_params=params)

# params are: m, noise_std, vt_std
p = dd.Uniform(lower=[1, 0, 0], upper=[8, 10, 5])
s = Identity()
g = Default(model=lifm, prior=p, summary=s)

expm_data = np.concatenate((clean_dict['resp_angle'], clean_dict['lv'])).reshape((1, 492))

inf_snpe = SNPE(generator=g, n_components=2, n_hiddens=[50], obs=expm_data)
logs, tds, posteriors = inf_snpe.run(n_train=500, n_rounds=3)
posterior = posteriors[-1]

plt.plot(logs[0]['loss'])
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()


from delfi.utils import viz


viz.plot_pdf(posterior.xs[0], lims=[-5,15], pdf2=posterior.xs[1], labels_params=['m', 'noise_std', 'vt_std']);
plt.show()
for k in range(2):
    print(r'component {}: mixture weight = {:.2f}; mean = {:.2f}; variance = {:.2f}'.format(
        k+1, posterior.a[k], posterior.xs[k].m[0], posterior.xs[k].S[0][0]))


