
# Module
# =======================================================================================================================

import tqdm
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist

import warnings
warnings.filterwarnings("ignore")

import probcox as pcox

dtype = torch.FloatTensor

np.random.seed(34)
torch.manual_seed(234)

# Simulation Settings
# =======================================================================================================================

# group 1
I = 250 # Number of individuals
P_binary = 3
P_continuous = 3
P = P_binary + P_continuous
theta_m = np.asarray([-0.7, 0.8, 0.6, -0.9, 1.1, -0.4])[:, None]
scale = 11 # Scaling factor for Baseline Hazard

# Class for simulation
TVC = pcox.TVC(theta=theta_m, P_binary=P_binary, P_continuous=P_continuous, dtype=dtype)

# Sample baseline hazard - scale is set to define censorship/events
TVC.make_lambda0(scale=scale)

# Sample Data
np.random.seed(5)
torch.manual_seed(4)
surv = torch.zeros((0, 3))
X = torch.zeros((0, 7))
ii = 0
for __ in (range(I)):
    a, b = TVC.sample(frailty=None)
    b = torch.cat((b, torch.ones((b.shape[0], 1))), axis=1)
    surv = torch.cat((surv, a))
    X = torch.cat((X, b))
    ii += 1

# group 2
I = 250 # Number of individuals
P_binary = 3
P_continuous = 3
P = P_binary + P_continuous
theta_f = np.asarray([-0.3, 1.2, 0.1, -0.5, 0.7, -0.4])[:, None]
scale = 7 # Scaling factor for Baseline Hazard

# Class for simulation
TVC = pcox.TVC(theta=theta_f, P_binary=P_binary, P_continuous=P_continuous, dtype=dtype)

# Sample baseline hazard - scale is set to define censorship/events
TVC.make_lambda0(scale=scale)

# Sample Data
ii = 0
for __ in (range(I)):
    a, b = TVC.sample(frailty=None)
    b = torch.cat((b, torch.zeros((b.shape[0], 1))), axis=1)
    surv = torch.cat((surv, a))
    X = torch.cat((X, b))
    ii += 1


# Inference Setup
# =======================================================================================================================
# Custom linear predictor - Here: simple linear combination
def predictor(data):
    theta =  pyro.sample("theta", dist.StudentT(1, loc=0, scale=0.001).expand([data[1].shape[1], 1])).type(dtype)
    pred = torch.mm(data[1], theta)
    return(pred)

def evaluate(surv, X, batchsize, sampling_proportion, iter_, predictor=predictor):
    sampling_proportion[1] = batchsize
    eta=5 # paramter for optimization
    run = True # repeat initalization if NAN encounterd while training - gauge correct optimization settings
    while run:
        run = False
        pyro.clear_param_store()
        m = pcox.PCox(sampling_proportion=sampling_proportion, predictor=predictor)
        m.initialize(eta=eta, num_particles=1, rank=7)
        loss=[0]
        for ii in tqdm.tqdm(range((iter_))):
            idx = np.unique(np.concatenate((np.random.choice(np.where(surv[:, -1]==1)[0], 2, replace=False), np.random.choice(range(surv.shape[0]), batchsize-2, replace=False)))) # random sample of data - force at least two events (no evaluation otherwise)
            data=[surv[idx], X[idx]] # subsampled data
            loss.append(m.infer(data=data))
            # divergence check
            if loss[-1] != loss[-1]:
                eta = eta * 0.1
                run=True
                break
    g = m.return_guide()
    out = g.quantiles([0.025, 0.5, 0.975])
    return(out)


total_obs = surv.shape[0]
total_events = torch.sum(surv[:, -1] == 1).numpy().tolist()
pyro.clear_param_store()
out = evaluate(batchsize=512, iter_=25000, surv=surv, X=X, sampling_proportion=[total_obs, None, total_events, None])

# plot the results
theta_est = out['theta'][1].detach().numpy()
theta_est_lower = out['theta'][0].detach().numpy()
theta_est_upper = out['theta'][2].detach().numpy()

fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
ax.errorbar(theta_m[:6, 0], theta_est[:6, 0], yerr=(theta_est[:6, 0] - theta_est_lower[:6, 0], theta_est_upper[:6, 0]- theta_est[:6, 0]),  ls='', c=".3", capsize=2, capthick=0.95, elinewidth=0.95)
ax.plot(theta_m[:6, 0], theta_est[:6, 0], ls='', c=".3", marker='x', ms=2)
ax.set(xlim=(-2, 2), ylim=(-2, 2))
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c="red", linewidth=0.75)
ax.set_yticks([-2, 0, 2])
ax.set_ylim([-2, 2])
ax.set_xticks([-2, 0, 2])
ax.set_xlim([-2, 2])
ax.set_xlabel('theta')
ax.set_ylabel('theta_hat')
plt.show()
plt.close()


fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
ax.errorbar(theta_f[:6, 0], theta_est[:6, 0], yerr=(theta_est[:6, 0] - theta_est_lower[:6, 0], theta_est_upper[:6, 0]- theta_est[:6, 0]),  ls='', c=".3", capsize=2, capthick=0.95, elinewidth=0.95)
ax.plot(theta_f[:6, 0], theta_est[:6, 0], ls='', c=".3", marker='x', ms=2)
ax.set(xlim=(-2, 2), ylim=(-2, 2))
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c="red", linewidth=0.75)
ax.set_yticks([-2, 0, 2])
ax.set_ylim([-2, 2])
ax.set_xticks([-2, 0, 2])
ax.set_xlim([-2, 2])
ax.set_xlabel('theta')
ax.set_ylabel('theta_hat')
plt.show()
plt.close()
