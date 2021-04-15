import os
import sys
import subprocess
import tqdm
import importlib
os.chdir('/nfs/nobackup/gerstung/awj/projects/ProbCox/')

import pandas as pd
import numpy as np

from multiprocessing import Pool

import torch
from torch.distributions import constraints
from torch.utils.data import Dataset, DataLoader

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

import simulation as sim
importlib.reload(sim)

# Setup
# -----------------------------------------------------------------------------------------------------------------------------

sim_name = 'sim3'

import warnings
warnings.filterwarnings("ignore")

dtype = torch.FloatTensor

np.random.seed(34)
torch.manual_seed(152)

run_id = int(sys.argv[1])
print(run_id)

# Simulation Settings:
# -----------------------------------------------------------------------------------------------------------------------------

theta = np.random.normal(0, 0.75, 30)[:, None]
theta = np.concatenate((theta[:27], np.zeros((1970, 1)), theta[-3:]))

if run_id == 0:
    np.savetxt('./output/simulation/' + sim_name + '/theta.txt', np.round(theta, 5))

P_binary = 1997
P_continuous = 3

#X = np.concatenate((np.random.binomial(1, 0.2, (1000, P_binary)), np.random.normal(0, 1, (1000, P_continuous))), axis=1)
#plt.hist(np.matmul(X, theta))

TVC = sim.TVC(theta=theta, P_binary=P_binary, P_continuous=P_continuous, dtype=dtype)
TVC.make_lambda0(scale=50)
np.sum([torch.sum(TVC.sample()[0][:, -1]).numpy() for ii in tqdm.tqdm(range(3000))])

if run_id == 0:
    a, b = TVC.return_lambda0()
    np.savetxt('./output/simulation/' + sim_name + '/lambda0.txt', np.concatenate((a[:, None], b), axis=1))

np.random.seed(run_id)
torch.manual_seed(run_id)
surv = torch.zeros((0, 3))
X = torch.zeros((0, 2000))
for __ in (range(2048)):
    a, b = TVC.sample()
    surv = torch.cat((surv, a))
    X = torch.cat((X, b))    
    
#plt.hist(surv[surv[:, -1]==1, 1])
torch.save({'surv': surv, 'X': X}, './tmp/' + str(run_id))


