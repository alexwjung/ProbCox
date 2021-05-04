'''
Likelihood Approximation Simulation - additional large effect size:

- Evaluating the approximation to the likelihood through the proposed subsampling.
- Generate table from supplement

Settings:
- individuals: - I - 5000
- censorship: - C - 0.75
- covariates : - P - 20
- batch size: - B - 64, 128, 256, 512
- sigma for theta - T - 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2

'''

# Modules
# =======================================================================================================================
import os
import sys
import shutil
import subprocess
import tqdm

import numpy as np
import pandas as pd

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist

from pyro.infer import SVI, Trace_ELBO

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import probcox as pcox

dtype = torch.FloatTensor

np.random.seed(90834)
torch.manual_seed(873645)

os.chdir('/nfs/nobackup/gerstung/awj/projects/ProbCox/')

# Functions
# =======================================================================================================================

def run(surv, pred, batch, est):
    total_obs = surv.shape[0]
    total_events = torch.sum(surv[:, -1] == 1).numpy().tolist()
    sampling_proportion = [total_obs, batch, total_events, None]
    ll = []
    ll2 = []
    while len(ll) <=1000:
        idx = np.unique(np.concatenate((np.random.choice(np.where(surv[:, -1]==1)[0], 2, replace=False), np.random.choice(range(surv.shape[0]), batch-2, replace=False))))
        sampling_proportion[-1] = torch.sum(surv[idx, -1]).numpy().tolist()
        if torch.sum(surv[idx, -1]) > 1:
            e = pcox.CoxPartialLikelihood(pred=pred[idx], sampling_proportion=sampling_proportion).log_prob(surv=surv[idx]).detach().numpy()
            MPE = ((e-est)/est)
            MAPE = np.abs(MPE)
            ll.append(MPE.tolist())
            ll2.append(MAPE.tolist())
    return(np.mean(ll), np.mean(ll2))

# Simulation Settings
# =======================================================================================================================

I = [5000]
P = [20]
C = [0.75]
T = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2]
B = [64, 128, 256, 512]


# Simulation
# =======================================================================================================================

res = np.zeros((8, 4))
res2 = np.zeros((8, 4))
sim_n =[]
ii = 0
jj = 0
for t in T:

    cond = True
    scale = 10
    while cond:
        theta = np.random.normal(0, t, (20, 1))
        TVC = pcox.TVC(theta=theta, P_binary=int(10), P_continuous=int(10), dtype=dtype)
        TVC.make_lambda0(scale=scale)
        s = np.sum([torch.sum(TVC.sample()[0][:, -1]).numpy() for ii in (range(100))])/100

        if np.logical_and(s>=0.1, s<=0.9):
            cond = False
        scale = scale/5

    theta_ = torch.normal(0, t, (20, 1)).type(dtype)

    n=5000
    c = 0.75
    # make dataset
    surv, X = TVC.make_dataset(obs=n, fraction_censored=c)
    pred = torch.mm(X, theta_).type(dtype)
    minmax = str((np.round(np.min(pred.detach().numpy()), 2), np.round(np.max(pred.detach().numpy()), 2)))

    sim_n.append('I(N): ' + str(n) + '(' + str(surv.shape[0]) + ')' +', LP(min, max): ' + str(minmax))

    pred = torch.mm(X, theta_).type(dtype)
    est = pcox.CoxPartialLikelihood(pred=pred, sampling_proportion=None).log_prob(surv=surv).detach().numpy()

    # fit to batch
    for b in tqdm.tqdm(B):
        print(b)
        res[ii, jj], res2[ii, jj] = run(surv=surv, pred=pred, batch=b, est=est)
        jj += 1
    ii += 1
    jj = 0

res = np.round(res, 2)
res2 = np.round(res2, 2)


pd.DataFrame(np.concatenate((np.asarray(sim_n)[:, None], res.astype(str)), axis=1)).to_csv('./out/simulation/tables/likelihood_approx_MPE_additonal2.csv')
pd.DataFrame(np.concatenate((np.asarray(sim_n)[:, None], res2.astype(str)), axis=1)).to_csv('./out/simulation/tables/likelihood_approx_MAPE_additonal2.csv')

# combine arrays

MPE = pd.read_csv('./out/simulation/tables/likelihood_approx_MPE_additonal2.csv').iloc[:, 1:]
MAPE = pd.read_csv('./out/simulation/tables/likelihood_approx_MAPE_additonal2.csv').iloc[:, 1:]

for ii in range(MPE.shape[0]):
    for jj in range(1, MPE.shape[1]):
        MPE.iloc[ii, jj] = str(MPE.iloc[ii, jj]) + ' (' + str(MAPE.iloc[ii, jj]) + ')'
MPE.to_csv('./out/simulation/tables/likelihood_approx_additonal2_combined.csv')


print('finished')
