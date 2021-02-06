import os
import sys
import subprocess
import tqdm
import importlib
import config
os.chdir(config.ROOT_DIR)

import pandas as pd
import numpy as np

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist

from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.mcmc import MCMC, NUTS

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------------------------------------------------------

sys.path.append('./ProbCox/')
import probcox as pcox #import CoxPartialLikelihood
importlib.reload(pcox)

import simulation as sim
importlib.reload(sim)

# Setup
# -----------------------------------------------------------------------------------------------------------------------------

sim_name = 'sim0'

import warnings
warnings.filterwarnings("ignore")

dtype = torch.FloatTensor

np.random.seed(43)
torch.manual_seed(43)

# Simulation Settings:
# -----------------------------------------------------------------------------------------------------------------------------

theta = np.asarray([-1, 1.5, 0, -0.5, 2, 0])[:, None]
np.savetxt('./output/simulation/' + sim_name + '/theta.txt', np.round(theta, 5))

P_binary = 3
P_continuous = 3

TVC = sim.TVC(theta=theta, P_binary=P_binary, P_continuous=P_continuous, censoring=None, dtype=dtype)
TVC.make_lambda0(scaling=35000)

# Run Inference
# -----------------------------------------------------------------------------------------------------------------------------
for iteration in tqdm.tqdm(range(1000)):
    surv, X = TVC.make_dataset(obs=1000, fraction_censored=0.5)
    pd.DataFrame(np.concatenate((surv, X), axis=1)).to_csv('./tmp/' + sim_name + '.csv', sep=';', index=False, header=False)

    total_obs = surv.shape[0]
    batch_size = 2048
    total_events = torch.sum(surv[:, -1] == 1)
    sampling_proportion = [total_obs, batch_size, total_events, None]

    # batch run
    run = True
    eta = 10.0
    while run:
        run = False
        pyro.clear_param_store()
        m = pcox.PCox(sampling_proportion=sampling_proportion)
        m.initialize(eta=eta)
        loss=[]
        for ii in (range((10000))):
            idx = np.random.choice(range(surv.shape[0]), batch_size, replace=False)
            data=[surv[idx], X[idx]]
            if torch.sum(surv[idx][:, -1]) > 0:
                loss.append(m.infer(data=data))
            if loss[-1] != loss[-1]:
                eta = eta * 0.5
                run=True
                break
    g = m.return_guide()
    out = g.quantiles([0.05, 0.5, 0.95])

    with open('./output/simulation/' + sim_name + '/probcox_theta_lower.txt', 'a') as write_out:
        write_out.write(''.join([str(ii) + '; ' for ii in out['theta'][0].detach().numpy()[:, 0].tolist()]))
        write_out.write('\n')
    with open('./output/simulation/' + sim_name + '/probcox_theta.txt', 'a') as write_out:
        write_out.write(''.join([str(ii) + '; ' for ii in out['theta'][1].detach().numpy()[:, 0].tolist()]))
        write_out.write('\n')
    with open('./output/simulation/' + sim_name + '/probcox_theta_upper.txt', 'a') as write_out:
        write_out.write(''.join([str(ii) + '; ' for ii in out['theta'][2].detach().numpy()[:, 0].tolist()]))
        write_out.write('\n')
    with open('./output/simulation/' + sim_name + '/N_obs.txt', 'a') as write_out:
        write_out.write(str(surv.shape[0]))
        write_out.write('\n')

    # full run
    run = True
    eta = 10.0
    while run:
        run = False
        pyro.clear_param_store()
        m = pcox.PCox(sampling_proportion=None)
        m.initialize(eta=eta)
        loss=[]
        for ii in (range((10000))):
            data=[surv, X]
            loss.append(m.infer(data=data))
            if loss[-1] != loss[-1]:
                eta = eta * 0.5
                run=True
                break
    g = m.return_guide()
    out = g.quantiles([0.05, 0.5, 0.95])

    with open('./output/simulation/' + sim_name + '/probcox_full_theta_lower.txt', 'a') as write_out:
        write_out.write(''.join([str(ii) + '; ' for ii in out['theta'][0].detach().numpy()[:, 0].tolist()]))
        write_out.write('\n')
    with open('./output/simulation/' + sim_name + '/probcox_full_theta.txt', 'a') as write_out:
        write_out.write(''.join([str(ii) + '; ' for ii in out['theta'][1].detach().numpy()[:, 0].tolist()]))
        write_out.write('\n')
    with open('./output/simulation/' + sim_name + '/probcox_full_theta_upper.txt', 'a') as write_out:
        write_out.write(''.join([str(ii) + '; ' for ii in out['theta'][2].detach().numpy()[:, 0].tolist()]))
        write_out.write('\n')

    # execute R script
    subprocess.check_call(['Rscript', './tmp/' + sim_name + '.R'], shell=False)

print('finished')
