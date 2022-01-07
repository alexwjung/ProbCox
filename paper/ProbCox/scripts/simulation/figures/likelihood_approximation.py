'''
Plot likelihood approximation along training
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


np.random.seed(5256)
torch.manual_seed(9235)


#os.chdir('/nfs/nobackup/gerstung/awj/projects/ProbCox/')
#os.chdir('/Users/alexwjung/projects/ProbCox/paper/ProbCox/')
os.chdir('/nfs/research/gerstung/awj/projects/ProbCox/paper/ProbCox')

# Plot Settings
# =======================================================================================================================

plt.rcParams['font.size'] = 7
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
cm = 1/2.54

# Simulation Settings
# =======================================================================================================================
P_binary=3
P_continuous=3
P = P_binary + P_continuous
theta = np.random.uniform(-1.5, 1.5, (P, 1))
scale=1.5
I = 1000 # individuals
batchsize = 1024
iter_ = 5000
eta = 0.01

# Simulation Data
# =======================================================================================================================
TVC = pcox.TVC(theta=theta, P_binary=P_continuous, P_continuous=P_continuous, dtype=dtype)
TVC.make_lambda0(scale=scale)
surv = torch.zeros((0, 3))
X = torch.zeros((0, 6))
for __ in (range(I)):
    a, b = TVC.sample()
    surv = torch.cat((surv, a))
    X = torch.cat((X, b))
total_obs = surv.shape[0]
total_events = torch.sum(surv[:, -1] == 1).numpy().tolist()
sampling_proportion = [total_obs, batchsize, total_events, None]

# Inference
# =======================================================================================================================
pyro.clear_param_store()
m = pcox.PCox(sampling_proportion=sampling_proportion)
m.initialize(eta=eta)
loss=[0]
LL_full = []
LL_batch = []
LL_naive = []
for ii in tqdm.tqdm(range((iter_))):
    idx = np.random.choice(range(surv.shape[0]), batchsize, replace=False)
    data=[surv, X]
    if torch.sum(surv[idx][:, -1]) > 0:
        loss.append(m.infer(data=data))
    if loss[-1] != loss[-1]:
        eta = eta * 0.5
        run=True
        break
    g = m.return_guide()
    out = g.quantiles([0.5])
    theta_est = out['theta'][0].detach()
    with torch.no_grad():
        pred = torch.mm(X, theta_est).type(dtype)
        LL_full.append(pcox.CoxPartialLikelihood(pred=pred, sampling_proportion=None).log_prob(surv=surv).detach().numpy())
        LL_batch.append(pcox.CoxPartialLikelihood(pred=pred[idx], sampling_proportion=[total_obs, batchsize, total_events, torch.sum(surv[idx, -1]).numpy().tolist()]).log_prob(surv=surv[idx]).detach().numpy())
        LL_naive.append(pcox.CoxPartialLikelihood(pred=pred[idx], sampling_proportion=None).log_prob(surv=surv[idx]).detach().numpy() * (total_obs/batchsize))

m_est = pcox.CoxPartialLikelihood(pred=pred, sampling_proportion=None).log_prob(surv=surv).detach().numpy()
m_approx = []
m_approx_naive= []
for _ in tqdm.tqdm(range(10000)):
    idx = np.random.choice(range(surv.shape[0]), batchsize, replace=False)
    m_approx.append(pcox.CoxPartialLikelihood(pred=pred[idx], sampling_proportion=[total_obs, batchsize, total_events, torch.sum(surv[idx, -1])]).log_prob(surv=surv[idx]).detach().numpy())
    m_approx_naive.append(pcox.CoxPartialLikelihood(pred=pred[idx], sampling_proportion=None).log_prob(surv=surv[idx]).detach().numpy() * (total_obs/batchsize))

    
    
# Plots
# =======================================================================================================================

fig, ax = plt.subplots(1, 2, figsize=(13*cm, 5.5*cm), dpi=600, sharey=True)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=None)
ax[0].scatter(np.arange(5000), -np.asarray(LL_batch)[:5000], color='0.5', alpha=1, label='', s=0.2, marker='x')
ax[0].scatter(np.arange(5000), -np.asarray(LL_naive)[:5000], color='0.8', alpha=1, label='naive', s=0.2, marker='.')
ax[0].plot(np.arange(5000), -np.asarray(LL_full)[:5000], color='0.1', label='full')
ax[1].axhline(-m_est, color='#0b64e0')
ax[0].set_xlabel(r'$Steps$')
ax[0].set_ylabel(r'$-\log \mathcal{L}(D|\theta)$')

ax[0].set_yticks([750, 1500, 2250])
ax[0].set_yticklabels([750, 1500, 2250])
ax[0].set_xticks([0, 2500, 5000])
ax[0].set_xlim([0, 5000])

ax[1].hist(-np.asarray(m_approx), bins=50, alpha=1, color='0.5', density=True, orientation='horizontal', label='reweighted')
ax[1].hist(-np.asarray(m_approx_naive), bins=50, alpha=1, color='0.8', density=True, orientation='horizontal', label='naive')
ax[1].axhline(-m_est, color='0.1', label='full')
ax[1].spines['bottom'].set_visible(False)
ax[1].set_xticks([])
ax[1].legend(frameon=False, prop={'size': 6})

#plt.show()
plt.savefig('./out/simulation/figures/likelihood_approximation.eps', bbox_inches='tight', dpi=600, transparent=True)
plt.savefig('./out/simulation/figures/likelihood_approximation.png', bbox_inches='tight', dpi=600, transparent=True)
plt.savefig('./out/simulation/figures/likelihood_approximation.pdf', bbox_inches='tight', dpi=600, transparent=True)
plt.close()

