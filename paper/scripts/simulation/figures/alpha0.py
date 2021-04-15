'''
Sample baseline hazard and show overall structure 
'''

# Modules
# -----------------------------------------------------------------------------------------------------------------------------
import os
import sys
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

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Custom Modules
# -----------------------------------------------------------------------------------------------------------------------------

sys.path.append('./ProbCox/')

import probcox as pcox #import CoxPartialLikelihood
importlib.reload(pcox)

import simulation as sim
importlib.reload(sim)

# Plot Settings
# -----------------------------------------------------------------------------------------------------------------------------

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 8
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Setup
# -----------------------------------------------------------------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

dtype = torch.FloatTensor

np.random.seed(43)
torch.manual_seed(43)

# Simulation Settings:
# -----------------------------------------------------------------------------------------------------------------------------

theta = np.random.uniform(0, 0, (1, 1))
TVC = sim.TVC(theta=theta, P_binary=0, P_continuous=0, dtype=dtype)

# Plot
# -----------------------------------------------------------------------------------------------------------------------------
np.random.seed(1)
torch.manual_seed(1)
fig, ax = plt.subplots(figsize=(8.27/2, 11.69/4), dpi=300)
for _ in tqdm.tqdm(range(25)):
    TVC.make_lambda0(scale=1)
    a, b = TVC.return_lambda0()
    ax.step(a, b, color='.5', linewidth=0.35)
    ax.ticklabel_format(axis='y', style='sci')
ax.step(a, b, color='#0b64e0', linewidth=1)
ax.set_xlabel(r'$Time$')
ax.set_ylabel(r"$\alpha_0$")
ax.set_yticks([0, 0.00005, 0.0001])
ax.set_yticklabels([0, 5, 10])
ax.set_ylim([0, 0.0001])
ax.set_xticks([0, 15000, 30000])
ax.set_xlim([0, 30000])
ax.text(500, 0.00009825, r'$\times10e^{-5}$')
plt.savefig('./output/simulation/figures/alpha0.eps', bbox_inches='tight')

