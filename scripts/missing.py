# Modules
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

np.random.seed(124)
torch.manual_seed(432)


# Simulation Settings
# =======================================================================================================================
I = 1000 # Number of individuals
P_binary = 3
P_continuous = 3
P = P_binary + P_continuous
theta = np.asarray([-0.9, 1.2, 0.4, 0.5, -1.1, 0.9])[:, None]
scale = 5 # Scaling factor for Baseline Hazard

# Class for simulation
TVC = pcox.TVC(theta=theta, P_binary=P_binary, P_continuous=P_continuous, dtype=dtype)

# Sample baseline hazard - scale is set to define censorship/events
TVC.make_lambda0(scale=scale)

# Sample Data
np.random.seed(5)
torch.manual_seed(4)
surv = torch.zeros((0, 3))
X = torch.zeros((0, P))
Z = torch.zeros((0, I))
ii = 0
for __ in (range(I)):
    a, b = TVC.sample(frailty=None)
    Z_ = torch.zeros((a.shape[0], I))
    Z_[:, ii] = 1
    surv = torch.cat((surv, a))
    X = torch.cat((X, b))
    Z = torch.cat((Z, Z_))
    ii += 1

total_obs = surv.shape[0]
total_events = torch.sum(surv[:, -1] == 1).numpy().tolist()


# Inference Setup - all
# =======================================================================================================================
def evaluate(surv, X, batchsize, sampling_proportion, iter_, predictor=None):
    sampling_proportion[1] = batchsize
    eta=5 # paramter for optimization
    run = True # repeat initalization if NAN encounterd while training - gauge correct optimization settings
    while run:
        run = False
        pyro.clear_param_store()
        m = pcox.PCox(sampling_proportion=sampling_proportion, predictor=predictor)
        m.initialize(eta=eta, num_particles=5)
        loss=[0]
        for ii in tqdm.tqdm(range((iter_))):
            idx = np.unique(np.concatenate((np.random.choice(np.where(surv[:, -1]==1)[0], 1, replace=False), np.random.choice(range(surv.shape[0]), batchsize-1, replace=False)))) # random sample of data - force at least two events (no evaluation otherwise)
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

pyro.clear_param_store()
out = evaluate(batchsize=256, iter_=10000, surv=surv, X=X, sampling_proportion=[total_obs, None, total_events, None])

# plot the results
theta_est = out['theta'][1].detach().numpy()
theta_est_lower = out['theta'][0].detach().numpy()
theta_est_upper = out['theta'][2].detach().numpy()

fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
ax.errorbar(theta[:, 0], theta_est[:, 0], yerr=(theta_est[:, 0] - theta_est_lower[:, 0], theta_est_upper[:, 0]- theta_est[:, 0]),  ls='', c=".3", capsize=2, capthick=0.95, elinewidth=0.95)
ax.plot(theta[:, 0], theta_est[:, 0], ls='', c=".3", marker='x', ms=2)
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

# Inference Setup - complete case
# =======================================================================================================================

idx_missing = torch.rand((X.shape[0], 3)) < 0.4
X[:, 3:][idx_missing] = float('nan')

total_obs = surv[torch.sum(torch.isnan(X), axis=1) == 0].shape[0]
total_events = torch.sum(surv[torch.sum(torch.isnan(X), axis=1) == 0][:, -1] == 1).numpy().tolist()
pyro.clear_param_store()
out = evaluate(batchsize=256, iter_=20000, surv=surv[torch.sum(torch.isnan(X), axis=1) == 0], X=X[torch.sum(torch.isnan(X), axis=1) == 0], sampling_proportion=[total_obs, None, total_events, None])

# plot the results
theta_est = out['theta'][1].detach().numpy()
theta_est_lower = out['theta'][0].detach().numpy()
theta_est_upper = out['theta'][2].detach().numpy()

fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
ax.errorbar(theta[:, 0], theta_est[:, 0], yerr=(theta_est[:, 0] - theta_est_lower[:, 0], theta_est_upper[:, 0]- theta_est[:, 0]),  ls='', c=".3", capsize=2, capthick=0.95, elinewidth=0.95)
ax.plot(theta[:, 0], theta_est[:, 0], ls='', c=".3", marker='x', ms=2)
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



# Inference Setup - imputed
# =======================================================================================================================

def predictor(data, dtype=dtype):
    # imputation
    quant_mu1 = pyro.sample("quant_mu1", dist.Normal(0, 1).expand([1])).type(dtype)
    quant_sigma1 = pyro.sample("quant_sigma1", dist.Uniform(0.5, 1.5).expand([1])).type(dtype)
    quant_impute1 = pyro.sample("quant_impute1", dist.Normal(quant_mu1, quant_sigma1).expand([256]).mask(False)).type(dtype)

    quant_mu2 = pyro.sample("quant_mu2", dist.Normal(0, 1).expand([1])).type(dtype)
    quant_sigma2 = pyro.sample("quant_sigma2", dist.Uniform(0.5, 1.5).expand([1])).type(dtype)
    quant_impute2 = pyro.sample("quant_impute2", dist.Normal(quant_mu2, quant_sigma2).expand([256]).mask(False)).type(dtype)

    quant_mu3 = pyro.sample("quant_mu3", dist.Normal(0, 1).expand([1])).type(dtype)
    quant_sigma3 = pyro.sample("quant_sigma3", dist.Uniform(0.5, 1.5).expand([1])).type(dtype)
    quant_impute3 = pyro.sample("quant_impute3", dist.Normal(quant_mu3, quant_sigma3).expand([256]).mask(False)).type(dtype)

    #qual_p = pyro.sample("qual_p", dist.Beta(1, 1).expand([3])).type(dtype)
    #qual_impute = pyro.sample("qual_impute", dist.Uniform(0, 1).expand([256, 3])).type(dtype)
    #qual_impute = (qual_impute <= qual_p).type(dtype)

    X_ = data[1].clone()
    X_[:, -1][data[2][:, -1]] = quant_impute1[data[2][:, -1]]
    X_[:, -2][data[2][:, -2]] = quant_impute2[data[2][:, -2]]
    X_[:, -3][data[2][:, -3]] = quant_impute3[data[2][:, -3]]
    #X_[:, 3:][data[2][:, 3:]] = qual_impute[data[2][:, 3:]]

    pyro.sample("var1", dist.Normal(quant_mu1, quant_sigma1), obs=X_[:, -1][~data[2][:, -1]])
    pyro.sample("var2", dist.Normal(quant_mu2, quant_sigma2), obs=X_[:, -2][~data[2][:, -2]])
    pyro.sample("var3", dist.Normal(quant_mu3, quant_sigma3), obs=X_[:, -3][~data[2][:, -3]])
    #pyro.sample("var2", dist.Bernoulli(qual_p), obs=X_[:, 3:])

    # inference
    theta =  pyro.sample("theta", dist.Normal(loc=0, scale=1).expand([data[1].shape[1], 1])).type(dtype)
    pred = torch.mm(X_, theta)
    return(pred)

def evaluate(surv, X, batchsize, sampling_proportion, iter_, predictor=predictor):
    sampling_proportion[1] = batchsize
    eta=5 # paramter for optimization
    run = True # repeat initalization if NAN encounterd while training - gauge correct optimization settings
    while run:
        run = False
        pyro.clear_param_store()
        m = pcox.PCox(sampling_proportion=sampling_proportion, predictor=predictor)
        m.initialize(eta=eta, num_particles=3)
        loss=[0]
        for ii in tqdm.tqdm(range((iter_))):
            idx = np.unique(np.concatenate((np.random.choice(np.where(surv[:, -1]==1)[0], 1, replace=False), np.random.choice(range(surv.shape[0]), batchsize, replace=False))))[:batchsize] # random sample of data - force at least two events (no evaluation otherwise)
            data=[surv[idx], X[idx], idx_missing[idx]] # subsampled data
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
out = evaluate(batchsize=256, iter_=20000, surv=surv, X=X, sampling_proportion=[total_obs, None, total_events, None])

# plot the results
theta_est = out['theta'][1].detach().numpy()
theta_est_lower = out['theta'][0].detach().numpy()
theta_est_upper = out['theta'][2].detach().numpy()

fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
ax.errorbar(theta[:, 0], theta_est[:, 0], yerr=(theta_est[:, 0] - theta_est_lower[:, 0], theta_est_upper[:, 0]- theta_est[:, 0]),  ls='', c=".3", capsize=2, capthick=0.95, elinewidth=0.95)
ax.plot(theta[:, 0], theta_est[:, 0], ls='', c=".3", marker='x', ms=2)
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



qual_p1 = pyro.sample("qual_p1", dist.Beta(1, 1).expand([1])).type(dtype)
qual_impute1 = pyro.sample("qual_impute1", dist.Uniform(0, 1).expand([256])).type(dtype)
qual_impute1 = (qual_impute1 <= qual_p1).type(dtype)

qual_p2 = pyro.sample("qual_p2", dist.Beta(1, 1).expand([1])).type(dtype)
qual_impute2 = pyro.sample("qual_impute2", dist.Uniform(0, 1).expand([256])).type(dtype)
qual_impute2 = (qual_impute2 <= qual_p2).type(dtype)

qual_p3 = pyro.sample("qual_p3", dist.Beta(1, 1).expand([1])).type(dtype)
qual_impute3 = pyro.sample("qual_impute3", dist.Uniform(0, 1).expand([256])).type(dtype)
qual_impute3 = (qual_impute1 <= qual_p3).type(dtype)
