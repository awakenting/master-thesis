import os
import time
import numpy as np
import pandas as pd

import delfi
from delfi.simulator.BaseSimulator import BaseSimulator
from delfi.inference import SNPE
import delfi.distribution as dd
from delfi.summarystats import Identity
from delfi.summarystats.BaseSummaryStats import BaseSummaryStats
from delfi.generator import Default
from delfi.generator import MPGenerator

import models as md

class LVQuantiles(BaseSummaryStats):
    """Reduces l/v vs response angle data into 5 quantiles for 5 l/v bins.
    """

    def __init__(self, seed=None):
        super().__init__(seed=seed)
        # should return a matrix n_samples x 30
        self.n_summary = 30
        self.lv_bins = [(0.1, 0.28), (0.28, 0.47), (0.47, 0.65), (0.65, 0.83), (0.83, 1.01), (1.01, 1.2)]

    @copy_ancestor_docstring
    def calc(self, repetition_list):
        # See BaseSummaryStats.py for docstring

        # get the number of repetitions contained
        n_reps = len(repetition_list)

        # build a matrix of n_reps x 1
        repetition_stats_matrix = np.zeros((n_reps, self.n_summary))

        # for every repetition, take the mean of the data in the dict
        for rep_idx, rep_dict in enumerate(repetition_list):
            rep_data = rep_dict['data']
            ntrials = int(rep_data.size/2)
            data_lvs = rep_data[0, ntrials:]
            data_resp_angles = rep_data[0, 0:ntrials]
            qnt_list = []
            for lv_low, lv_high in self.lv_bins:
                mask = (lv_low < data_lvs) & (data_lvs < lv_high)
                qnt_list.append(np.percentile(data_resp_angles[mask], [10, 30, 50, 70, 90]))
            summary_stats = np.concatenate(qnt_list)
            repetition_stats_matrix[rep_idx, ] = summary_stats

        return repetition_stats_matrix


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
        model_params['rho_null_std'] = param[1]
        model_params['noise_std_exc'] = param[2]*1e-3
        model_params['rho_scale'] = param[3]*1e6
        nruns = 246
        rstims = np.zeros((1, nruns))
        lvs = np.zeros((1, nruns))
        for i in np.arange(nruns):
            rstims[0, i], rdist, reaction_time, lvs[0, i], stim_size, speed, rtime_to_coll = md.calc_response_ffi(model_params)

        data = np.concatenate((rstims, lvs), axis=1)
        return {'data': data}


gendata_path = '../data/generated/'
if not os.path.exists(gendata_path):
    os.makedirs(gendata_path)

total_time = 5
dt = 0.001

fixed_params = {'tau_m': 0.023,
                'e_l': -0.079,
                'r_m': 10*1e6,
                'v_t': -0.061,
                'init_vm_std': 0.000,
                'vt_std': 0.000,
                'rho_null': 2,
                'rho_null_std': 2.8,
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

param_names = ['rho_null', 'rho_null_std', 'noise_std_exc', 'rho_scale']
lower_param_bounds = [0, 0.5, 0, 5]
upper_param_bounds = [5, 5, 5, 10]

ffim = FFI(dim=len(param_names), fixed_params=fixed_params)
p = dd.Uniform(lower=lower_param_bounds, upper=upper_param_bounds)
s = LVQuantiles()
g = MPGenerator(models=[ffim]*6, prior=p, summary=s)

gparams, stats = g.gen(10)
gen_dat = stats[0][np.newaxis, :]
gen_params = gparams[0]


def run_fit_with(g, n_hiddens, n_train, n_rounds, epochs, data, seed):
    inf_snpe = SNPE(generator=g, n_components=1, n_hiddens=n_hiddens, obs=data, verbose=False, seed=seed)
    logs, tds, posteriors = inf_snpe.run(n_train=n_train, n_rounds=n_rounds, epochs=epochs)
    posterior = posteriors[-1]
    return logs, posterior

n_hidden_values = [[1000], [2000], [4000], [330, 330, 330], [200, 200, 200, 200, 200]]
n_train_values = [[1000], [2000], [4000], [8000]]
n_rounds_values = [len(vals) for vals in n_train_values]

data_cols = ['n_hidden', 'n_train', 'n_rounds', 'mean_rho_null', 'mean_rho_null_std', 'mean_noise_std_exc',
             'mean_rho_scale','std_rho_null', 'std_rho_null_std', 'std_noise_std_exc', 'std_rho_scale', 'final_loss',
             'posterior_object', 'logs_list']
data_dict = dict([(col_name, []) for col_name in data_cols])

starttime = time.time()

for n_hidden_idx, n_hidden_val in enumerate(n_hidden_values):
    for n_train_idx, (n_train_val, n_rounds_val) in enumerate(zip(n_train_values, n_rounds_values)):
        print('current train index = ' + str(n_train_idx))
        logs, posterior = run_fit_with(g, n_hidden_val, n_train_val, n_rounds_val, epochs=100, data=gen_dat, seed=37)

        cmeans = posterior.mean
        cstds = posterior.std
        closs = int(10*np.log(logs[-1]['loss'][-1]))

        result_values = [str(n_hidden_val), str(n_train_val), len(n_train_val), cmeans[0], cmeans[1],
                         cmeans[2], cmeans[3], cstds[0], cstds[1], cstds[2], cstds[3], closs, posterior, logs]
        for col, value in zip(data_cols, result_values):
            data_dict[col].append(value)

ffi_df = pd.DataFrame(data_dict)
ffi_df.to_hdf('../data/generated/fitting_metaparams_exploration.hdf5', key='fitting_results', mode='w')

gen_params_dict = dict([(param_name, param_val) for param_name, param_val in zip(param_names, gen_params)])
gen_params_df = pd.DataFrame(gen_params_dict, index=[0])
gen_params_df.to_hdf('../data/generated/fitting_metaparams_exploration.hdf5', key='ground_truth_params', mode='a')

for wk in g.workers:
    wk.terminate()

endtime = time.time()
print('Total time needed: ' + str(int((endtime - starttime))) + ' seconds or '
      + str(int((endtime - starttime) / 60)) + ' min '
      + 'or ' + str(int((endtime - starttime) / 3600)) + ' hours')
