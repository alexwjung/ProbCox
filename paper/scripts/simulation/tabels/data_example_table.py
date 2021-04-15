'''
Produce small example of simulated data to show general structure
'''

# Modules
# -----------------------------------------------------------------------------------------------------------------------------

import os
import sys
import shutil
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

# Setup
# -----------------------------------------------------------------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")
dtype = torch.FloatTensor
np.random.seed(909)
torch.manual_seed(9034)

# Simulation Settings:
# -----------------------------------------------------------------------------------------------------------------------------

theta = np.asarray([-0.1, 0.8, 0, 0.8, 1.5, 0])[:, None]

P_binary = 3
P_continuous = 3

TVC = sim.TVC(theta=theta, P_binary=P_binary, P_continuous=P_continuous, dtype=dtype)
TVC.make_lambda0(scale=0.5)

# Simulate Data
# --------------------------------------------------------------------------------------------------------------
surv = torch.zeros((0, 3))
X = torch.zeros((0, 6))
for __ in (range(3)):
    a, b = TVC.sample()
    surv = torch.cat((surv, a))
    X = torch.cat((X, b))
surv

# Format Data
# --------------------------------------------------------------------------------------------------------------
dd = pd.DataFrame(np.round(surv.numpy().astype(int), 2))
X = pd.DataFrame(np.round(X.numpy(), 2))

dd = pd.concat([dd, X], axis=1)
dd.columns = ['start', 'stop', 'event' ,'X1' ,'X2' ,'X3' ,'X4' ,'X5' ,'X6']
dd.to_csv('./output/simulation/tables/example.csv', index=False)
