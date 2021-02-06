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

import warnings
warnings.filterwarnings("ignore")

dtype = torch.FloatTensor

np.random.seed(48)
torch.manual_seed(48)

sim_name = 'sim5'
P = 50
c = 0.9
# Simulation Settings:
# -----------------------------------------------------------------------------------------------------------------------------
P_binary = int(P/2)
P_continuous = int(P/2)

theta = np.random.uniform(-2, 2, (P, 1))
np.savetxt('./output/simulation/' + sim_name + '/theta.txt', theta)

TVC = sim.TVC(theta=theta, P_binary=P_binary, P_continuous=P_continuous, censoring=None, dtype=dtype)
TVC.make_lambda0(scaling=10000000)

#t_lambda0, lambda0 = TVC.return_lambda0()
#plt.step(t_lambda0, lambda0)
#np.sum([torch.sum(TVC.sample()[0][:, -1]).numpy() for ii in range(1000)])

# Run Inference
# -----------------------------------------------------------------------------------------------------------------------------
for iteration in range(1000):
    surv, X = TVC.make_dataset(obs=500, fraction_censored=c)
    pd.DataFrame(np.concatenate((surv, X), axis=1)).to_csv('./tmp/' + sim_name + '.csv', sep=';', index=False, header=False)

    total_obs = surv.shape[0]
    batch_size = 2048
    total_events = torch.sum(surv[:, -1] == 1)
    sampling_proportion = [total_obs, batch_size, total_events, None]

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

    # execute R script
    subprocess.check_call(['Rscript', './tmp/' + sim_name + '.R'], shell=False)

'''
f, ax = plt.subplots(figsize=(6, 6))
ax.plot(theta, pyro.get_param_store()['AutoMultivariateNormal.loc'].detach().numpy(), ls='', marker='.')
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
plt.show()
plt.close()
'''
print('finsihed')
