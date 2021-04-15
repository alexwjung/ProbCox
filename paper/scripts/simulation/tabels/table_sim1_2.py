
'''

Simulation 1:

Script to generate a table from the individual simulation runs

'''

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

# -----------------------------------------------------------------------------------------------------------------------------

sys.path.append('./ProbCox/')

import probcox as pcox #import CoxPartialLikelihood
importlib.reload(pcox)

import simulation as sim
importlib.reload(sim)

# Setup
# -----------------------------------------------------------------------------------------------------------------------------

sim_name = 'sim1_2'

import warnings
warnings.filterwarnings("ignore")

dtype = torch.FloatTensor

# Theta
# -----------------------------------------------------------------------------------------------------------------------------

theta = np.asarray(pd.read_csv('./output/simulation/sim1_2/theta.txt', header=None))


# R Cox
# -----------------------------------------------------------------------------------------------------------------------------

res = np.zeros((6, 6))
res[:, 0] = theta[:, 0]
theta_est = pd.read_csv('./output/simulation/sim1_2/R_theta.txt', header=None, sep=';')
theta_est = theta_est.dropna(axis=0)
theta_est = theta_est.groupby(0).first().reset_index()
print(theta_est.shape[0])

theta_se = pd.read_csv('./output/simulation/sim1_2/R_se.txt', header=None, sep=';')
theta_se = theta_se.dropna(axis=0)
theta_se = theta_se.groupby(0).first().reset_index()

theta_mat = theta_est.merge(theta_se, how='inner', on=0)
theta_mat = np.asarray(theta_mat.iloc[:500, 1:]).astype(float)
print(theta_mat.shape[0])

theta_est = theta_mat[:, :6]
theta_se = theta_mat[:, 6:]

res[:, 1] = np.mean(theta_est, axis=0)
res[:, 2] = np.sqrt(np.var(theta_est, axis=0))
res[:, 3] = np.sqrt(np.mean((theta_est - theta[:, 0][None, :])**2, axis=0))

theta_est_lower = theta_est - 1.96*theta_se
theta_est_upper = theta_est + 1.96*theta_se

theta_bound = np.concatenate((theta_est_lower, theta_est_upper), axis=1)

ll = []
for ii in range(6):
    ll.append(np.mean(theta_bound[:, ii+6] - theta_bound[:, ii]))
res[:, 4] = np.asarray(ll)    

ll = []
for ii in range(6):
    ll.append(np.sum(np.logical_and(theta[ii]>=theta_bound[:, ii], theta[ii]<=theta_bound[:, ii+6]))/theta_bound.shape[0])
res[:, 5] = np.asarray(ll)   

res = np.round(res, 2)
pd.DataFrame(res)
pd.DataFrame(res).to_csv('./output/simulation/tables/sim1_2_' + 'R' + '.csv')



#ProbCox
# -----------------------------------------------------------------------------------------------------------------------------

res = np.zeros((6, 6))
res[:, 0] = theta[:, 0]

batchsize  = '1024'
theta_est = pd.read_csv('./output/simulation/sim1_2/probcox' + str(batchsize) + '_theta.txt', header=None, sep=';')
theta_est = theta_est.dropna(axis=0)
theta_est = theta_est.groupby(0).first().reset_index()
theta_est = theta_est.iloc[:, :-1]
print(theta_est.shape[0])

theta_est_lower = pd.read_csv('./output/simulation/sim1_2/probcox' + str(batchsize) + '_theta_lower.txt', header=None, sep=';')
theta_est_lower = theta_est_lower.dropna(axis=0)
theta_est_lower = theta_est_lower.groupby(0).first().reset_index()
theta_est_lower = theta_est_lower.iloc[:, :-1]

theta_est_upper = pd.read_csv('./output/simulation/sim1_2/probcox' + str(batchsize) + '_theta_upper.txt', header=None, sep=';')
theta_est_upper = theta_est_upper.dropna(axis=0)
theta_est_upper = theta_est_upper.groupby(0).first().reset_index()
theta_est_upper = theta_est_upper.iloc[:, :-1]

theta_bound = theta_est_lower.merge(theta_est_upper, how='inner', on=0)
theta_bound = theta_bound.merge(theta_est, how='inner', on=0)
theta_est = np.asarray(theta_bound.iloc[:500, -6:]).astype(float)
theta_bound = theta_bound.iloc[:500, :-6]
theta_bound = np.asarray(theta_bound.iloc[:, 1:]).astype(float)


res[:, 1] = np.mean(theta_est, axis=0)
res[:, 2] = np.sqrt(np.var(theta_est, axis=0))
res[:, 3] = np.sqrt(np.mean((theta_est - theta[:, 0][None, :])**2, axis=0))

ll = []
for ii in range(6):
    ll.append(np.mean(theta_bound[:, ii+6] - theta_bound[:, ii]))
res[:, 4] = np.asarray(ll)    

ll = []
for ii in range(6):
    ll.append(np.sum(np.logical_and(theta[ii]>=theta_bound[:, ii], theta[ii]<=theta_bound[:, ii+6]))/theta_bound.shape[0])
res[:, 5] = np.asarray(ll)    

res = np.round(res, 2)
pd.DataFrame(res)   
pd.DataFrame(res).to_csv('./output/simulation/tables/sim1_2_' + batchsize + '.csv')



