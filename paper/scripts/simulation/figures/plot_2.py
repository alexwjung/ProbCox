'''
Plot 2 - Simulation 2 - High Dimensional data plot estimates against true parameters
'''

# Modules
# -----------------------------------------------------------------------------------------------------------------------------

import os
import sys
import glob
import subprocess
import tqdm
import importlib
os.chdir('/nfs/nobackup/gerstung/awj/projects/ProbCox/')

import pandas as pd
import numpy as np

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist

from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.mcmc import MCMC, NUTS

import matplotlib.pyplot as plt


# Custom Modules
# -----------------------------------------------------------------------------------------------------------------------------

sys.path.append('./ProbCox/')

import probcox as pcox #import CoxPartialLikelihood
importlib.reload(pcox)

import simulation as sim
importlib.reload(sim)

# Figure paramters
# -----------------------------------------------------------------------------------------------------------------------------

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 12
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


# Setup
# -----------------------------------------------------------------------------------------------------------------------------
small_plot = True

import warnings
warnings.filterwarnings("ignore")

dtype = torch.FloatTensor


# Figure
# -----------------------------------------------------------------------------------------------------------------------------


theta = np.asarray(pd.read_csv('./output/simulation/sim2/theta.txt', header=None))

fig, ax = plt.subplots(1, 2, figsize=(8.27, 11.69/4), dpi=300, sharey=True)  
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=None)

file = 'r10'
theta_est = np.asarray(pd.read_csv('./output/simulation/sim2/probcox' + file + '_theta.txt', header=None, sep=';'))[:, 1:-1]
theta_est_lower = np.asarray(pd.read_csv('./output/simulation/sim2/probcox' + file + '_theta_lower.txt', header=None, sep=';'))[:, 1:-1]
theta_est_upper = np.asarray(pd.read_csv('./output/simulation/sim2/probcox' + file + '_theta_upper.txt', header=None, sep=';'))[:, 1:-1]


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
    ax[0].plot(theta + np.random.normal(0, 0.01, (theta.shape[0], 1)), theta_est[_, :], ls='', marker='x', ms=0.7, c='.8')
ax[0].plot(theta + np.random.normal(0, 0.01, (theta.shape[0], 1)), theta_est[0, :], ls='', marker='x', ms=0.7, c='.8', label=r'$\hat{\theta}$')
ax[0].plot(theta, np.mean(theta_est, axis=0), ls='', marker='*', ms=3, c='.3', label=r'$\bar{\hat{\theta}}$')
ax[0].plot(theta, np.mean(theta_est_lower, axis=0), ls='', marker='_', ms=5, c='.2', label=r'$\bar{\hat{\theta}}_{0.975}$')
ax[0].plot(theta, np.mean(theta_est_upper, axis=0), ls='', marker='_', ms=5, c='.2', label=r'$\bar{\hat{\theta}}_{0.025}$')
ax[0].set(xlim=(-2, 2), ylim=(-2, 2))
ax[0].set_xlabel(r'$\theta$')
ax[0].set_ylabel(r'$\hat{\theta}$')
ax[0].set_yticks([-2, 0, 2])
ax[0].set_ylim([-2, 2])
ax[0].set_xticks([-2, 0, 2])
ax[0].set_xlim([-2, 2])
ax[0].plot(ax[0].get_xlim(), ax[0].get_ylim(), ls=':', color='red', linewidth=0.5)
ax[0].set_title(r'$Rank \hspace{0.05cm} 10$', fontsize=10)

file = 'r30'
theta_est = np.asarray(pd.read_csv('./output/simulation/sim2/probcox' + file + '_theta.txt', header=None, sep=';'))[:, 1:-1]
theta_est_lower = np.asarray(pd.read_csv('./output/simulation/sim2/probcox' + file + '_theta_lower.txt', header=None, sep=';'))[:, 1:-1]
theta_est_upper = np.asarray(pd.read_csv('./output/simulation/sim2/probcox' + file + '_theta_upper.txt', header=None, sep=';'))[:, 1:-1]

if small_plot:
    theta_est = theta_est[:, idx]
    theta_est_lower = theta_est_lower[:, idx]
    theta_est_upper = theta_est_upper[:, idx]

for _ in range(50):
    ax[1].plot(theta + np.random.normal(0, 0.01, (theta.shape[0], 1)), theta_est[_, :], ls='', marker='x', ms=0.7, c='.8')
ax[1].plot(theta + np.random.normal(0, 0.01, (theta.shape[0], 1)), theta_est[0, :], ls='', marker='x', ms=0.7, c='.8', label=r'$\hat{\theta}$')
ax[1].plot(theta, np.mean(theta_est, axis=0), ls='', marker='*', ms=3, c='.3', label=r'$\bar{\hat{\theta}}$')
ax[1].plot(theta, np.mean(theta_est_lower, axis=0), ls='', marker='_', ms=5, c='.2', label=r'$\bar{\hat{\theta}}_{0.975}$')
ax[1].plot(theta, np.mean(theta_est_upper, axis=0), ls='', marker='_', ms=5, c='.2', label=r'$\bar{\hat{\theta}}_{0.025}$')
ax[1].set(xlim=(-2, 2), ylim=(-2, 2))
ax[1].set_xlabel(r'$\theta$')
ax[1].set_ylabel('')
ax[1].set_yticks([-2, 0, 2])
ax[1].set_ylim([-2, 2])
ax[1].set_xticks([-2, 0, 2])
ax[1].set_xlim([-2, 2])
ax[1].plot(ax[1].get_xlim(), ax[1].get_ylim(), ls=':', color='red', linewidth=0.5)
ax[1].legend(frameon=False, prop={'size': 8}, loc='lower right')
ax[1].set_title(r'$Rank \hspace{0.05cm} 30$', fontsize=10)

#plt.show()
#plt.close()
plt.savefig('./output/simulation/figures/plot2.eps', bbox_inches='tight')

''' 
theta2 = np.asarray(pd.read_csv('./output/simulation/sim2/theta.txt', header=None))[:, 0]
theta_est2 = np.asarray(pd.read_csv('./output/simulation/sim2/probcox_theta.txt', header=None, sep=';'))[0, :-1]
ax[1].plot(theta2, theta_est2, ls='', marker='x', ms=5, c='.5')
ax[1].set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3))
ax[1].plot(ax[1].get_xlim(), ax[1].get_ylim(), ls="--", color='black', linewidth=1)
ax[1].set_xlabel(r'$\theta$')
ax[1].set_yticks([-1.3, 0, 1.3])
ax[1].set_ylim([-1.3, 1.3])
ax[1].set_xticks([-1.3, 0, 1.3])
ax[1].set_xlim([-1.3, 1.3])

plt.savefig('./output/simulation/figures/plot2.eps', bbox_inches='tight')

'''



















