import os
import sys
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

np.random.seed(43)
torch.manual_seed(43)


# Simulation Settings:
# -----------------------------------------------------------------------------------------------------------------------------

theta = np.asarray([-1, 1.5, 0, -0.5, 2, 0])[:, None]

P_binary = 3
P_continuous = 3

TVC = sim.TVC(theta=theta, P_binary=P_binary, P_continuous=P_continuous, censoring=None, dtype=dtype)
TVC.make_lambda0(scaling=35000)

#t_lambda0, lambda0 = TVC.return_lambda0()
#plt.step(t_lambda0, lambda0)
#np.sum([torch.sum(TVC.sample()[0][:, -1]).numpy() for ii in range(1000)])

surv, X = TVC.make_dataset(obs=1000, fraction_censored=0.5)
total_obs = surv.shape[0]
batch_size = 512
total_events = torch.sum(surv[:, -1] == 1)
sampling_proportion = [total_obs, batch_size, total_events, None]

# Run Inference
# -----------------------------------------------------------------------------------------------------------------------------
run = True
eta = 10.0
while run:
    run = False
    pyro.clear_param_store()
    m = pcox.PCox(sampling_proportion=sampling_proportion)
    m.initialize(eta=eta)
    loss=[]
    for ii in tqdm.tqdm(range((5000))):
        idx = np.random.choice(range(surv.shape[0]), batch_size, replace=False)
        data=[surv[idx], X[idx]]
        loss.append(m.infer(data=data))
        if loss[-1] != loss[-1]:
            eta = eta * 0.5
            run=True
            break

# Evaluate
# -----------------------------------------------------------------------------------------------------------------------------
theta_est = pyro.get_param_store()['AutoMultivariateNormal.loc'].detach().numpy()

f, ax = plt.subplots(figsize=(10, 10))
ax.plot(theta, theta_est, ls='', marker='.')
ax.set_xlim([-1.5, 2.5])
ax.set_ylim([-1.5, 2.5])
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
plt.show()
plt.close()
