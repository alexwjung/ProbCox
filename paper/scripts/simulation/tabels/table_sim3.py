
'''

Simulation 3:

Script to generate a table from the individual simulation runs

'''


# Module
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

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Custom Module
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

# Theta
# -----------------------------------------------------------------------------------------------------------------------------

theta = np.asarray(pd.read_csv('./output/simulation/sim3/theta.txt', header=None))


#ProbCox
#-----------------------------------------------------------------------------------------------------------------------------

theta = np.asarray(pd.read_csv('./output/simulation/sim3/theta.txt', header=None))

theta_est = np.squeeze(pd.read_csv('./output/simulation/sim3/probcox_theta.txt', header=None, sep=';'))[:2000, None]

theta_est_lower = np.squeeze(pd.read_csv('./output/simulation/sim3/probcox' + str(batchsize) + '_theta_lower.txt', header=None, sep=';'))[:2000, None]

theta_est_upper = np.squeeze(pd.read_csv('./output/simulation/sim3/probcox' + str(batchsize) + '_theta_upper.txt', header=None, sep=';'))[:2000, None]

res = np.concatenate((theta, theta_est, theta_est_lower, theta_est_upper), axis=1)

res = np.round(res.astype(float), 2)
pd.DataFrame(res)   
pd.DataFrame(res).to_csv('./output/simulation/tables/sim3' + batchsize + '.csv')
pd.DataFrame(np.concatenate((res[:27, :], res[-3:, :]))).to_csv('./output/simulation/tables/sim3_sup' + batchsize + '.csv')

