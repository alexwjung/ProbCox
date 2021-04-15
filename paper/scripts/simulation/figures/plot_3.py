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

# -----------------------------------------------------------------------------------------------------------------------------

sys.path.append('./ProbCox/')

import probcox as pcox #import CoxPartialLikelihood
importlib.reload(pcox)

import simulation as sim
importlib.reload(sim)
# -----------------------------------------------------------------------------------------------------------------------------

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 12
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


# Setup
# -----------------------------------------------------------------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

dtype = torch.FloatTensor


# -----------------------------------------------------------------------------------------------------------------------------


theta = np.asarray(pd.read_csv('./output/simulation/sim3/theta.txt', header=None)).astype(float)

theta_est = np.squeeze(pd.read_csv('./output/simulation/sim3/probcox_theta.txt', header=None, sep=';'))[:2000, None].astype(float)

theta_est_lower = np.squeeze(pd.read_csv('./output/simulation/sim3/probcox_theta_lower.txt', header=None, sep=';'))[:2000, None].astype(float)

theta_est_upper = np.squeeze(pd.read_csv('./output/simulation/sim3/probcox_theta_upper.txt', header=None, sep=';'))[:2000, None].astype(float)

fig, ax = plt.subplots(1, 1, figsize=(8.27/2, 11.69/4), dpi=300)  
ax.errorbar(theta[:, 0], theta_est[:, 0], yerr=(theta_est[:, 0] - theta_est_lower[:, 0], theta_est_upper[:, 0]- theta_est[:, 0]),  ls='', c=".3", capsize=2, capthick=0.95, elinewidth=0.95)
ax.plot(theta[:, 0], theta_est[:, 0], ls='', c=".3", marker='x', ms=2)
ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c="red", linewidth=0.75)
ax.set_yticks([-1.5, 0, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_xticks([-1.5, 0, 1.5])
ax.set_xlim([-1.5, 1.5])
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\hat{\theta}$')
#plt.show()
#plt.close()
plt.savefig('./output/simulation/figures/sim3.eps', bbox_inches='tight')










