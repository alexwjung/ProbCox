import os
import sys
import tqdm
import importlib
import config
os.chdir(config.ROOT_DIR)

import torch

import numpy as np
import matplotlib.pyplot as plt

# Setup
# -----------------------------------------------------------------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

dtype = torch.FloatTensor

np.random.seed(43)
torch.manual_seed(43)

# Custom Functions
# -----------------------------------------------------------------------------------------------------------------------------

def forward_fill(x):
    for ii in range(1, x.shape[0]):
        if np.sum(x[ii, :]) == 0:
            x[ii, :] = x[ii-1, :]
    return(x)

def logisitc(x):
    return(1/(1+np.exp(-x)))

def logit(x):
    return(np.log(x/(1-x)))

# Baseline Hazard
# -----------------------------------------------------------------------------------------------------------------------------

def baseline_hazard():
    '''
    Sample baseline hazard
    '''
    tsum = 0
    t = [0]
    lambda0 = [10e-10]
    while tsum < 30000:
        t.append(t[-1] + np.round(np.random.gamma(4, 500)).astype(int))
        if t[-1] <= 17000:
            lambda0.append(lambda0[-1] + np.random.gamma(2, 1))
        elif np.logical_and(t[-1] > 15000, t[-1] <= 25000):
            lambda0.append(lambda0[-1] + np.random.gamma(1, 10))
        else:
            lambda0.append(lambda0[-1] - np.random.gamma(1, 5))
        if lambda0[-1] <= 0:
            lambda0[-1] = 10e10
        tsum = t[-1]
    t = np.asarray(t)
    lambda0 = np.asarray(lambda0)[:, None]
    lambda0 = lambda0 / (np.max(lambda0)*100000)
    return(t, lambda0)

for ii in range(1):
    t_lambda0, lambda0 = baseline_hazard()
    plt.step(t_lambda0, lambda0)
plt.show()
plt.close()


lambda0 = logit(lambda0)


# Covariates
# -----------------------------------------------------------------------------------------------------------------------------


def TVC_survival(EOO=EOO, P_binary=P_binary, P_continious=P_continious, theta=theta, lambda0=lambda0, t_lambda0=t_lambda0, alpha=4, beta=500, dtype=dtype):
    censoring = np.round(np.random.poisson(15000) * np.random.beta(5, 1)).astype(int)
    EOO = np.minimum(censoring, EOO)
    tsum = 0
    t = [0]
    X = np.zeros((1, P_binary + P_continious))
    while tsum < EOO:
        t.append(t[-1] + np.round(np.random.gamma(alpha, beta)).astype(int))
        X = np.concatenate((X, np.concatenate((np.random.binomial(1, 0.25, (1, P_binary)), np.random.normal(0, 1, (1, P_continious))), axis=1)))
        tsum = t[-1]
    t = np.asarray(t)

    # combine with baseline hazard
    lambda0 = np.concatenate((lambda0, np.zeros((t.shape[0] + 1, 1))))
    X = np.concatenate((np.zeros((t_lambda0.shape[0], P_binary + P_continious)), X, np.zeros((1, P_binary + P_continious))))
    t = np.concatenate((t_lambda0, t, EOO))
    idx_sort = np.argsort(t)
    t = t[idx_sort]
    lambda0 = forward_fill(lambda0[idx_sort])
    X = forward_fill(X[idx_sort])

    # censor
    idx_censor = t <= EOO
    X = X[idx_censor]
    lambda0 = lambda0[idx_censor]
    t = t[idx_censor]

    # collapse
    X = np.concatenate([np.sum(X[t==ii, :], axis=0)[None, :] for ii in np.unique(t)])
    lambda0 = np.concatenate([np.sum(lambda0[t==ii, :], axis=0)[None, :] for ii in np.unique(t)])
    t = np.unique(t)

    # survial data
    H = logisitc(np.matmul(X, theta) + lambda0)

    t_diff = (t[1:] - t[:-1])
    event = False
    for ii in range(H.shape[0]-1):
        eval = np.random.uniform(0, 1, (t_diff[ii],)) <= H[ii]
        if np.any(eval):
            event = True
            break

    if event:
        t_event = np.maximum(1, np.where(eval)[0][0])
        X = X[:ii+1]
        t = t[:ii+1]
        t = np.concatenate((t, np.asarray([t[-1] + t_event])))
        time = np.concatenate((t[:-1, None], t[1:, None], np.zeros((X.shape[0], 1))), axis=1)
        time[-1, -1] = 1
    else:
        time = np.concatenate((t[:-1, None], t[1:, None], np.zeros((X.shape[0]-1, 1))), axis=1)
        X = X[:-1]
    return(torch.tensor(time).type(dtype), torch.tensor(X).type(dtype))


# Survival times
# -----------------------------------------------------------------------------------------------------------------------------



np.savetxt('/Users/alexwjung/Desktop/surv.txt', np.concatenate((surv.numpy(), X.numpy()), axis=1))

torch.min(surv[:, 1] - surv[:, 0])
surv[surv[:, 1] - surv[:, 0] <=0]

theta
theta_sim = np.zeros((20, 1))

for ii in tqdm.tqdm(range(1000)):
    surv = torch.zeros((0, 3))
    X = torch.zeros((0, P_continious + P_binary))
    for ii in (range(1000)):
        a, b = TVC_survival()
        surv = torch.cat((surv, a))
        X = torch.cat((X, b))
        torch.sum(surv[:, -1])

    total_obs = surv.shape[0]
    batch_size = 512
    total_events = torch.sum(surv[:, -1] == 1)
    sampling_proportion = [total_obs, batch_size, total_events, None]

    # -----------------------------------------------------------------------------------------------------------------------------
    run = True
    eta = 10.0
    while run:
        run = False
        pyro.clear_param_store()
        m = pcox.PCox(sampling_proportion=sampling_proportion)
        m.initialize(eta=eta)
        loss=[]
        for ii in (range((5000))):
            idx = np.random.choice(range(surv.shape[0]),s batch_size, replace=False)
            data=[surv[idx], X[idx]]
            loss.append(m.infer(data=data))
            if loss[-1] != loss[-1]:
                eta = eta * 0.5
                run=True
                break

    theta_est = pyro.get_param_store()['AutoMultivariateNormal.loc'].detach().numpy()
    theta_sim = np.concatenate((theta_sim, theta_est[:, None]), axis=1)

theta_sim.shape = theta_sim[:, 1:]
theta_sim



theta_sim.shape
theta.shape


f, ax = plt.subplots(figsize=(10, 10))
ax.plot(theta, theta_sim, ls='', marker='.')
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
plt.show()
plt.close()



     theta_sim = np.concatenate((theta_sim.numpy()[:, None], theta_est), axis=1)

# Survival times
# -----------------------------------------------------------------------------------------------------------------------------


times_steps = 15
P = 3
baseline_hazard = torch.tensor([0.001, 0.003, 0.007, 0.015, 0.03, 0.07, 0.08, 0.12, 0.15, 0.20, 0.17, 0.14, 0.13, 0.12, 0.12]).type(dtype)
baseline_hazard = logit(baseline_hazard)
theta = torch.normal(0, 0.5, (P, 1)).type(dtype)


def simulate_TDC(theta=theta, baseline_hazard=baseline_hazard, times_steps=times_steps, P=P, dtype=dtype):
    X = torch.normal(0, 1, (times_steps, P)).type(dtype)
    time = torch.cat((torch.arange(0,times_steps)[:, None], torch.arange(1,times_steps+1)[:, None], torch.zeros((times_steps, 1))), axis=1).type(dtype)
    unif_probs = torch.rand((times_steps,)).type(dtype)
    binom_probs = logistic(torch.mm(X, theta)[:, 0] + baseline_hazard)
    time[:, -1] = (binom_probs > unif_probs).type(dtype)
    idx = torch.cumsum(torch.cumsum(time[:, -1], dim=0), dim=0) <= 1
    time = time[idx]
    X = X[idx]
    return(time, X)


surv = torch.zeros((0, 3))
X = torch.zeros((0, P))
for ii in tqdm.tqdm(range(500)):
    a, b = simulate_TDC()
    surv = torch.cat((surv, a))
    X = torch.cat((X, b))


total_obs = surv.shape[0]
batch_size = 200
total_events = torch.sum(surv[:, -1] == 1)
sampling_proportion = [total_obs, batch_size, total_events, None]

# -----------------------------------------------------------------------------------------------------------------------------
pyro.clear_param_store()
m = pcox.PCox(sampling_proportion=None)
m.initialize(eta=1)
loss=[]
for ii in tqdm.tqdm(range((10000))):
    idx = np.random.choice(range(surv.shape[0]), batch_size, replace=False)
    data=[surv[idx], X[idx]]
    loss.append(m.infer(data=data))
plt.semilogy(loss)




log(1)

theta
theta_est = pyro.get_param_store()['AutoMultivariateNormal.loc'].detach()
theta_est

print(pcox.concordance(surv.detach(), torch.mm(X, (theta_est[:, None]).detach())))

print(pcox.concordance(surv.detach(), torch.mm(X, (theta).detach())))


for key, value in pyro.get_param_store().items():
    print(f"{key}:\n{value}\n")



np.log(1)


0.25 * np.exp(1.5)

logisitc(1.1204222675845161)



logisitc(logit(0.25) + 1.5)


np.exp(0.5990210269638426)



0.25/0.75 * np.exp(1.5)
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
