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
from pyro.infer import Predictive

import warnings
warnings.filterwarnings("ignore")

import probcox as pcox

dtype = torch.FloatTensor

np.random.seed(7834)
torch.manual_seed(6347)


# Simulation Settings
# =======================================================================================================================
I = 1000 # Number of individuals
P_binary = 3
P_continuous = 3
P = P_binary + P_continuous
theta = np.asarray([-0.9, 1.2, 0.4, 0.5, -1.1, 0.9])[:, None]
scale = 7 # Scaling factor for Baseline Hazard

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



# No missing data
# =======================================================================================================================
def predictor(data):
    theta =  pyro.sample("theta", dist.StudentT(1, loc=0, scale=0.001).expand([1, data[1].shape[1]])).type(dtype)
    pred = torch.mm(data[1], theta.T)
    return(pred)

def guide(data, rank=6):
    cov_diag = pyro.param("cov_diag", torch.full((data[1].shape[1],), 0.001), constraint=constraints.positive)
    cov_factor = pyro.param("cov_factor", torch.randn((data[1].shape[1], rank)) * 0.001)
    loc = pyro.param('loc', torch.zeros(data[1].shape[1]))
    pyro.sample("theta", dist.LowRankMultivariateNormal(loc, cov_factor, cov_diag).expand((1,)))

def evaluate(surv, X, batchsize, sampling_proportion, iter_, predictor=predictor, guide=guide):
    sampling_proportion[1] = batchsize
    eta=1 # paramter for optimization
    run = True # repeat initalization if NAN encounterd while training - gauge correct optimization settings
    while run:
        run = False
        pyro.clear_param_store()
        m = pcox.PCox(sampling_proportion=sampling_proportion, predictor=predictor, guide=guide)
        m.initialize(eta=eta, num_particles=4)
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
    mm = m.return_model()
    #out = g.quantiles([0.025, 0.5, 0.975])
    plt.plot(loss)
    return(g, mm)

total_obs = surv.shape[0]
total_events = torch.sum(surv[:, -1] == 1).numpy().tolist()
pyro.clear_param_store()
g, mm = evaluate(batchsize=512, iter_=50000, surv=surv, X=X, sampling_proportion=[total_obs, None, total_events, None])

predictive = Predictive(model=mm, guide=g, num_samples=1000, return_sites=('theta', 'obs'))
samples = predictive([surv, X])['theta']

out = np.percentile(np.squeeze(samples.detach().numpy()), [5, 50, 95], axis=0)
theta_est_lower = out[0, :][:, None]
theta_est = out[1, :][:, None]
theta_est_upper = out[2, :][:, None]

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

# Set missingness
# =======================================================================================================================
idx_missing = torch.rand((X.shape[0], 3)) < 0.25
X[:, 3:][idx_missing] = float('nan')


# Complete case
# =======================================================================================================================
def evaluate(surv, X, batchsize, sampling_proportion, iter_, predictor=predictor):
    sampling_proportion[1] = batchsize
    eta=5 # paramter for optimization
    run = True # repeat initalization if NAN encounterd while training - gauge correct optimization settings
    while run:
        run = False
        pyro.clear_param_store()
        m = pcox.PCox(sampling_proportion=sampling_proportion, predictor=predictor)
        m.initialize(eta=eta, num_particles=1, rank=6)
        loss=[0]
        for ii in tqdm.tqdm(range((iter_))):
            idx = np.unique(np.concatenate((np.random.choice(np.where(surv[:, -1]==1)[0], 1, replace=False), np.random.choice(range(surv.shape[0]), batchsize, replace=False))))[:batchsize] # random sample of data - force at least two events (no evaluation otherwise)
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

total_obs = surv[torch.sum(idx_missing, axis=1) == 0].shape[0]
total_events = torch.sum(surv[torch.sum(idx_missing, axis=1) == 0][:, -1] == 1).numpy().tolist()
pyro.clear_param_store()
out = evaluate(batchsize=512, iter_=50000, surv=surv[torch.sum(idx_missing, axis=1) == 0], X=X[torch.sum(idx_missing, axis=1) == 0], sampling_proportion=[total_obs, None, total_events, None])

# plot the results
theta_est = out['theta'][1].detach().numpy().T
theta_est_lower = out['theta'][0].detach().numpy().T
theta_est_upper = out['theta'][2].detach().numpy().T

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


# Imputed
# =======================================================================================================================

def predictor(data, dtype=dtype):
    # imputation
    quant_mu = pyro.sample("quant_mu", dist.Normal(0, 1).expand([3])).type(dtype)
    quant_sigma = pyro.sample("quant_sigma", dist.Uniform(0.5, 1.5).expand([3])).type(dtype)
    quant_impute = pyro.sample("quant_impute", dist.Normal(quant_mu, quant_sigma).expand([512, 3]).mask(False)).type(dtype)

    #qual_p = pyro.sample("qual_p", dist.Beta(1, 1).expand([3])).type(dtype)
    #qual_impute = pyro.sample("qual_impute", dist.Uniform(0, 1).expand([256, 3])).type(dtype)
    #qual_impute = (qual_impute <= qual_p).type(dtype)

    X_ = data[1].clone()
    X_[:, 3:][data[2]] = quant_impute[data[2]]

    #X_[:, 3:][data[2][:, 3:]] = qual_impute[data[2][:, 3:]]

    pyro.sample("quant_unobserved", dist.Normal(quant_mu, quant_sigma), obs=X_[:, 3:])

    #pyro.sample("var2", dist.Bernoulli(qual_p), obs=X_[:, 3:])

    # inference
    #theta =  pyro.sample("theta", dist.Normal(loc=0, scale=1).expand([data[1].shape[1], 1])).type(dtype)
    theta =  pyro.sample("theta", dist.StudentT(1, loc=0, scale=0.001).expand([1, data[1].shape[1]])).type(dtype)
    pred = torch.mm(X_, theta.T)
    return(pred)

def guide(data, rank=6):
    quant_q1 = pyro.param("quant_q1", torch.tensor([0.0]))
    quant_q2 = pyro.param("quant_q2", torch.tensor([1.0]))
    quant_q3 = pyro.param("quant_q3", torch.tensor([0.5]), constraint=constraints.positive)
    quant_q4 = pyro.param("quant_q4", torch.tensor([1.5]), constraint=constraints.positive)

    quant_mu = pyro.sample("quant_mu", dist.Normal(quant_q1, quant_q2).expand([3])).type(dtype)
    quant_sigma = pyro.sample("quant_sigma", dist.Uniform(quant_q3, quant_q4).expand([3])).type(dtype)
    quant_impute = pyro.sample("quant_impute", dist.Normal(quant_mu, quant_sigma).expand([512, 3]).mask(False)).type(dtype)

    cov_diag = pyro.param("cov_diag", torch.full((data[1].shape[1],), 0.001), constraint=constraints.positive)
    cov_factor = pyro.param("cov_factor", torch.randn((data[1].shape[1], rank)) * 0.001)
    loc = pyro.param('loc', torch.zeros(data[1].shape[1]))
    pyro.sample("theta", dist.LowRankMultivariateNormal(loc, cov_factor, cov_diag).expand((1,)))

def evaluate(surv, X, batchsize, sampling_proportion, iter_, predictor=predictor, guide=guide):
    sampling_proportion[1] = batchsize
    eta=1 # paramter for optimization
    run = True # repeat initalization if NAN encounterd while training - gauge correct optimization settings
    while run:
        run = False
        pyro.clear_param_store()
        m = pcox.PCox(sampling_proportion=sampling_proportion, predictor=predictor, guide=guide)
        m.initialize(eta=eta, num_particles=4)
        loss=[0]
        for ii in tqdm.tqdm(range((iter_))):
            idx = np.unique(np.concatenate((np.random.choice(np.where(surv[:, -1]==1)[0], 1, replace=False), np.random.choice(range(surv.shape[0]), batchsize, replace=False))))[:512] # random sample of data - force at least two events (no evaluation otherwise)
            data=[surv[idx], X[idx], idx_missing[idx]] # subsampled data
            loss.append(m.infer(data=data))
            # divergence check
            if loss[-1] != loss[-1]:
                eta = eta * 0.1
                run=True
                break

    g = m.return_guide()
    mm = m.return_model()
    #out = g.quantiles([0.025, 0.5, 0.975])
    plt.plot(loss)
    return(g, mm)


total_obs = surv.shape[0]
total_events = torch.sum(surv[:, -1] == 1).numpy().tolist()
pyro.clear_param_store()
#out = evaluate(batchsize=512, iter_=10, surv=surv, X=X, sampling_proportion=[total_obs, None, total_events, None])
g, mm = evaluate(batchsize=512, iter_=20000, surv=surv, X=X, sampling_proportion=[total_obs, None, total_events, None])

predictive = Predictive(model=mm, guide=g, num_samples=1000, return_sites=('theta', 'obs'))
samples = predictive([surv[:512], X[:512], idx_missing[:512]])['theta']

out = np.percentile(np.squeeze(samples.detach().numpy()), [5, 50, 95], axis=0)
theta_est_lower = out[0, :][:, None]
theta_est = out[1, :][:, None]
theta_est_upper = out[2, :][:, None]

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
