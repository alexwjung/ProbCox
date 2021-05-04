'''

High Dimensional Case Simulation - Figure:

- plot true parameters vs estimated parameters - diagional indicates ideal fit
- plot results from ProbCox vs glmnet (lambda=1se)

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


os.chdir('/nfs/nobackup/gerstung/awj/projects/ProbCox/')

# Funtion
# =======================================================================================================================
def custom_mean(X, W, col_idx):
    '''
    - average for paramters of an array selcted by an indexing matrix

    X :: array to apply mean along axis=0
    W :: indexing which elements to use for mean computatiuon
    col_idx :: indexing the columns where W is applied - otherwise standard mean without selecting elements
    '''
    m = []
    assert X.shape == W.shape
    N, M = X.shape

    for jj in range(M):
        if col_idx[jj] == True:
            m.append(np.mean(X[W[:, jj], jj]))
        else:
            m.append(np.mean(X[:, jj]))
    return(np.asarray(m))



# Plot Settings
# =======================================================================================================================

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 10
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


# Setup
# =======================================================================================================================
small_plot = True # sample 0 parameters - otherwise figure gets too large


# Plot
# =======================================================================================================================

theta = np.asarray(pd.read_csv('./out/simulation/sim_hd/theta.txt', header=None))

fig, ax = plt.subplots(1, 2, figsize=(8.27*0.90, 11.69/4), dpi=600, sharey=True)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=None)

suffix = 'rank50_b1024'
theta_est = pd.read_csv('./out/simulation/sim_hd/probcox' + str(suffix) + '_theta.txt', header=None, sep=';')
theta_est = theta_est.dropna(axis=0)
theta_est = theta_est.groupby(0).first().reset_index()
theta_est = theta_est.iloc[:, 1:-1]
theta_est = np.asarray(theta_est).astype(float)

theta_est_lower = pd.read_csv('./out/simulation/sim_hd/probcox' + str(suffix) + '_theta_lower.txt', header=None, sep=';')
theta_est_lower = theta_est_lower.dropna(axis=0)
theta_est_lower = theta_est_lower.groupby(0).first().reset_index()
theta_est_lower = theta_est_lower.iloc[:100, 1:-1]
theta_est_lower = np.asarray(theta_est_lower).astype(float)

theta_est_upper = pd.read_csv('./out/simulation/sim_hd/probcox' + str(suffix) + '_theta_upper.txt', header=None, sep=';')
theta_est_upper = theta_est_upper.dropna(axis=0)
theta_est_upper = theta_est_upper.groupby(0).first().reset_index()
theta_est_upper = theta_est_upper.iloc[:100, 1:-1]
theta_est_upper = np.asarray(theta_est_upper).astype(float)

if small_plot:
    non_zeros = 250
    idx = np.concatenate((np.arange(10), np.arange(5000, 5010), np.random.choice(np.arange(10, 5000), non_zeros, replace=False)
    , np.random.choice(np.arange(5010, 10000), non_zeros, replace=False)
    ))
    theta = theta[idx, :]
    theta_est = theta_est[:, idx]
    theta_est_lower = theta_est_lower[:, idx]
    theta_est_upper = theta_est_upper[:, idx]


for _ in range(50):
    ax[0].plot(theta + np.random.normal(0, 0.01, (theta.shape[0], 1)), theta_est[_, :], ls='', marker='.', ms=1, c='.7')
ax[0].plot(theta + np.random.normal(0, 0.01, (theta.shape[0], 1)), theta_est[0, :], ls='', marker='.', ms=1, c='0.7', label=r'$\hat{\theta}$')


W = np.sign(theta_est_lower) == np.sign(theta_est_upper) # non zero parameters estimates (based on HPD95%)
col_idx = np.logical_and(np.squeeze(theta != 0), np.sum(W, axis=0) > 5) # true non-zero parameters



ax[0].plot(theta[:10], custom_mean(theta_est, W, col_idx)[:10], ls='', marker='d', ms=2, c='#0aa9ff', label=r'$\bar{\hat{\theta}}_{binary}$')
ax[0].plot(theta[10:20], custom_mean(theta_est, W, col_idx)[10:20], ls='', marker='*', ms=3, c='#1e8725', label=r'$\bar{\hat{\theta}}_{continious}$')


ax[0].plot(theta[:20], custom_mean(theta_est_lower, W, col_idx)[:20], ls='', marker='_', ms=5, c='.2', label=r'$\bar{\hat{\theta}}_{0.975}$')
ax[0].plot(theta[:20], custom_mean(theta_est_upper, W, col_idx)[:20], ls='', marker='_', ms=5, c='.2', label=r'$\bar{\hat{\theta}}_{0.025}$')



ax[0].set(xlim=(-2, 2), ylim=(-2, 2))
ax[0].set_xlabel(r'$\theta$')
ax[0].set_ylabel(r'$\hat{\theta}$')
ax[0].set_yticks([-2, 0, 2])
ax[0].set_ylim([-2, 2])
ax[0].set_xticks([-2, 0, 2])
ax[0].set_xlim([-2, 2])
ax[0].plot(ax[0].get_xlim(), ax[0].get_ylim(), ls=':', color='black', linewidth=0.5)
ax[0].set_title(r'ProbCox', fontsize=10)


# glmnet
theta_est = pd.read_csv('./out/simulation/sim_hd/R_theta_1se.txt', header=None, sep=';')
theta_est = theta_est.dropna(axis=0)
theta_est = theta_est.groupby(0).first().reset_index()
theta_est = np.asarray(theta_est.iloc[:, 1:])

if small_plot:
    theta_est = theta_est[:, idx]

for _ in range(50):
    ax[1].plot(theta + np.random.normal(0, 0.01, (theta.shape[0], 1)), theta_est[_, :], ls='', marker='x', ms=1, c='.7')
ax[1].plot(theta + np.random.normal(0, 0.01, (theta.shape[0], 1)), theta_est[0, :], ls='', marker='x', ms=1, c='.7', label=r'$\hat{\theta}$')

W = theta_est!=0 # non zero parameters estimates (based on HPD95%)
col_idx = np.logical_and(np.squeeze(theta != 0), np.sum(W, axis=0) > 5) # true non-zero parameters

ax[1].plot(theta[:10], custom_mean(theta_est, W, col_idx)[:10], ls='', marker='d', ms=2, c='#0aa9ff')
ax[1].plot(theta[10:20], custom_mean(theta_est, W, col_idx)[10:20], ls='', marker='*', ms=3, c='#1e8725')


ax[1].set(xlim=(-2, 2), ylim=(-2, 2))
ax[1].set_xlabel(r'$\theta$')
ax[1].set_ylabel('')
ax[1].set_yticks([-2, 0, 2])
ax[1].set_ylim([-2, 2])
ax[1].set_xticks([-2, 0, 2])
ax[1].set_xlim([-2, 2])
ax[1].plot(ax[1].get_xlim(), ax[1].get_ylim(), ls=':', color='black', linewidth=0.5)
ax[0].legend(frameon=False, prop={'size': 8}, loc='lower right')
ax[1].set_title(r'glmnet', fontsize=10)

plt.savefig('./out/simulation/figures/hd.eps', bbox_inches='tight', dpi=600)
