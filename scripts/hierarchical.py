
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
from pyro.infer import Predictive
import warnings
warnings.filterwarnings("ignore")

import probcox as pcox

dtype = torch.FloatTensor

np.random.seed(34)
torch.manual_seed(234)

# Simulation Settings
# =======================================================================================================================

# Male
I = 2000
P_binary = 3
P_continuous = 3
P = P_binary + P_continuous
theta_m = np.asarray([-0.4, 0.9, 0, 1.2, -0.2, 0.5])[:, None]
scale = 3 # Scaling factor for Baseline Hazard

# Class for simulation
TVC = pcox.TVC(theta=theta_m, P_binary=P_binary, P_continuous=P_continuous, dtype=dtype)

# Sample baseline hazard - scale is set to define censorship/events
TVC.make_lambda0(scale=scale)

# Sample Data
np.random.seed(5)
torch.manual_seed(4)
surv_m = torch.zeros((0, 3))
X_m = torch.zeros((0, 6))
ii = 0
for __ in (range(I)):
    a, b = TVC.sample(frailty=None)
    surv_m = torch.cat((surv_m, a))
    X_m = torch.cat((X_m, b))
    ii += 1

torch.sum(surv_m[:, -1]==1)
plt.hist(surv_m[surv_m[:, -1]==1, 1].detach().numpy())

# Female
I = 2000
P_binary = 3
P_continuous = 3
P = P_binary + P_continuous
theta_f = np.asarray([-0.7, 0.6, 0.4, 1, 0, -0.3])[:, None]
scale = 5 # Scaling factor for Baseline Hazard

# Class for simulation
TVC = pcox.TVC(theta=theta_f, P_binary=P_binary, P_continuous=P_continuous, dtype=dtype)

# Sample baseline hazard - scale is set to define censorship/events
TVC.make_lambda0(scale=scale)

# Sample Data
np.random.seed(9)
torch.manual_seed(42)
surv_f = torch.zeros((0, 3))
X_f = torch.zeros((0, 6))
ii = 0
for __ in (range(I)):
    a, b = TVC.sample(frailty=None)
    surv_f = torch.cat((surv_f, a))
    X_f = torch.cat((X_f, b))
    ii += 1
torch.sum(surv_f[:, -1]==1)
plt.hist(surv_f[surv_f[:, -1]==1, 1].detach().numpy())


# Inference Male only
# =======================================================================================================================
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
        m.initialize(eta=eta, num_particles=1, rank=6)
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

total_obs = surv_m.shape[0]
total_events = torch.sum(surv_m[:, -1] == 1).numpy().tolist()
pyro.clear_param_store()
out = evaluate(batchsize=512, iter_=10000, surv=surv_m, X=X_m, sampling_proportion=[total_obs, None, total_events, None])

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



# Inference Female only
# =======================================================================================================================
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

total_obs = surv_f.shape[0]
total_events = torch.sum(surv_f[:, -1] == 1).numpy().tolist()
pyro.clear_param_store()
out = evaluate(batchsize=512, iter_=10000, surv=surv_f, X=X_f, sampling_proportion=[total_obs, None, total_events, None])

# plot the results
theta_est = out['theta'][1].detach().numpy()
theta_est_lower = out['theta'][0].detach().numpy()
theta_est_upper = out['theta'][2].detach().numpy()

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



# Inference Joint
# =======================================================================================================================

def predictor(data, dtype=dtype):
    theta =  pyro.sample("theta", dist.StudentT(1, loc=0, scale=0.001).expand([1, data[0][1].shape[1]])).type(dtype)
    theta_m =  pyro.sample("theta_m", dist.StudentT(1, loc=theta, scale=0.001)).type(dtype)
    theta_f =  pyro.sample("theta_f", dist.StudentT(1, loc=theta, scale=0.001)).type(dtype)
    #pyro.sample("theta", dist.StudentT(1, loc=0, scale=0.001).expand([1, data[0][1].shape[1], 2]).to_event(1)).type(dtype)
    pred = [torch.mm(data[0][1], theta_f.T), torch.mm(data[1][1], theta_m.T)]
    return(pred)

def guide(data, rank=6):
    cov_diag1 = pyro.param("cov_diag1", torch.full((data[0][1].shape[1],), 0.01), constraint=constraints.positive)
    cov_factor1 = pyro.param("cov_factor1", torch.randn((data[0][1].shape[1], rank)) * 0.01)
    loc1 = pyro.param('loc1', torch.zeros(data[0][1].shape[1]))
    pyro.sample("theta", dist.LowRankMultivariateNormal(loc1, cov_factor1, cov_diag1).expand((1,)))

    cov_diag2 = pyro.param("cov_diag2", torch.full((data[0][1].shape[1],), 0.01), constraint=constraints.positive)
    cov_factor2 = pyro.param("cov_factor2", torch.randn((data[0][1].shape[1], rank)) * 0.01)
    loc2 = pyro.param('loc2', torch.zeros(data[0][1].shape[1]))
    pyro.sample("theta_f", dist.LowRankMultivariateNormal(loc2, cov_factor2, cov_diag2).expand((1,)))

    cov_diag3 = pyro.param("cov_diag3", torch.full((data[0][1].shape[1],), 0.01), constraint=constraints.positive)
    cov_factor3 = pyro.param("cov_factor3", torch.randn((data[0][1].shape[1], rank)) * 0.01)
    loc3 = pyro.param('loc3', torch.zeros(data[0][1].shape[1]))
    pyro.sample("theta_m", dist.LowRankMultivariateNormal(loc3, cov_factor3, cov_diag3).expand((1,)))

def evaluate(surv, X, batchsize, sampling_proportion, iter_, predictor=predictor, guide=guide):
    eta=5# paramter for optimization
    run = True # repeat initalization if NAN encounterd while training - gauge correct optimization settings
    while run:
        run = False
        pyro.clear_param_store()
        m = pcox.PCox(sampling_proportion=sampling_proportion, predictor=predictor, guide=guide, levels=2)
        m.initialize(eta=eta, num_particles=1)
        loss=[0]
        for ii in tqdm.tqdm(range((iter_))):
            idx_f = np.unique(np.concatenate((np.random.choice(np.where(surv[0][:, -1]==1)[0], 1, replace=False), np.random.choice(range(surv[0].shape[0]), batchsize, replace=False))))[:512] # random sample of data - force at least two events (no evaluation otherwise)
            idx_m = np.unique(np.concatenate((np.random.choice(np.where(surv[1][:, -1]==1)[0], 1, replace=False), np.random.choice(range(surv[1].shape[0]), batchsize, replace=False))))[:512]
            data=[[surv[0][idx_f], X[0][idx_f]], [surv[1][idx_m], X[1][idx_m]]] # subsampled data
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

sampling_proportion_f=[surv_f.shape[0], 512, torch.sum(surv_f[:, -1] == 1).numpy().tolist(), None]
sampling_proportion_m=[surv_m.shape[0], 512, torch.sum(surv_m[:, -1] == 1).numpy().tolist(), None]

pyro.clear_param_store()
#out = evaluate(batchsize=512, iter_=10, surv=surv, X=X, sampling_proportion=[total_obs, None, total_events, None])
g, mm = evaluate(batchsize=512, iter_=50000, surv=[surv_f, surv_m], X=[X_f, X_m], sampling_proportion = [sampling_proportion_f, sampling_proportion_m])

predictive = Predictive(model=mm, guide=g, num_samples=1000, return_sites=('theta', 'theta_f', 'theta_m', 'obs'))
samples = predictive([[surv_f, X_f],[surv_m, X_m]])


out = np.percentile(np.squeeze(samples['theta_f'].detach().numpy()), [5, 50, 95], axis=0)
theta_est_lower = out[0, :][:, None]
theta_est = out[1, :][:, None]
theta_est_upper = out[2, :][:, None]

fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
ax.errorbar(theta_f[:, 0], theta_est[:, 0], yerr=(theta_est[:, 0] - theta_est_lower[:, 0], theta_est_upper[:, 0]- theta_est[:, 0]),  ls='', c=".3", capsize=2, capthick=0.95, elinewidth=0.95)
ax.plot(theta_f[:, 0], theta_est[:, 0], ls='', c=".3", marker='x', ms=2)
ax.set(xlim=(-2, 2), ylim=(-2, 2))
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c="red", linewidth=0.75)
ax.set_yticks([-2, 0, 2])
ax.set_ylim([-2, 2])
ax.set_xticks([-2, 0, 2])
ax.set_xlim([-2, 2])
ax.set_xlabel('theta_f')
ax.set_ylabel('theta_hat')
plt.show()
plt.close()


out = np.percentile(np.squeeze(samples['theta_m'].detach().numpy()), [5, 50, 95], axis=0)
theta_est_lower = out[0, :][:, None]
theta_est = out[1, :][:, None]
theta_est_upper = out[2, :][:, None]

fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
ax.errorbar(theta_m[:, 0], theta_est[:, 0], yerr=(theta_est[:, 0] - theta_est_lower[:, 0], theta_est_upper[:, 0]- theta_est[:, 0]),  ls='', c=".3", capsize=2, capthick=0.95, elinewidth=0.95)
ax.plot(theta_m[:, 0], theta_est[:, 0], ls='', c=".3", marker='x', ms=2)
ax.set(xlim=(-2, 2), ylim=(-2, 2))
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c="red", linewidth=0.75)
ax.set_yticks([-2, 0, 2])
ax.set_ylim([-2, 2])
ax.set_xticks([-2, 0, 2])
ax.set_xlim([-2, 2])
ax.set_xlabel('theta_m')
ax.set_ylabel('theta_hat')
plt.show()
plt.close()
