'''
Figure with sample draws from baseline hazard
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

np.random.seed(43)
torch.manual_seed(43)

os.chdir('/nfs/nobackup/gerstung/awj/projects/ProbCox/')

# Plot Settings
# =======================================================================================================================

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 10
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Simulation Settings
# =======================================================================================================================

TVC = pcox.TVC(theta=None, P_binary=0, P_continuous=0, dtype=dtype)

# Plot
# =======================================================================================================================

np.random.seed(1)
torch.manual_seed(1)
fig, ax = plt.subplots(figsize=((8.27)*0.75, (11.69/4)), dpi=600)
for _ in tqdm.tqdm(range(20)):
    TVC.make_lambda0(scale=1)
    a, b = TVC.return_lambda0()
    ax.step(a, b, color='.8', linewidth=0.5)
    ax.ticklabel_format(axis='y', style='sci')
ax.step(a, b, color='.1', linewidth=1, linestyle='-')
ax.set_xlabel(r'$Time$')
ax.set_ylabel(r"$\alpha_0$")
ax.set_yticks([0, 0.00005, 0.0001])
ax.set_yticklabels([0, 5, 10])
ax.set_ylim([0, 0.0001])
ax.set_xticks([0, 15000, 30000])
ax.set_xlim([0, 30000])
ax.text(500, 0.00009760, r'$\times10e^{-5}$')
#ax.text(500, 0.00009760, r'$\times10e^{-5}$')
plt.savefig('./out/simulation/figures/baseline_hazard.eps', bbox_inches='tight', dpi=600)
