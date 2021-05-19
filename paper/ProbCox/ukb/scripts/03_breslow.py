'''
Using the predicted values to obtain an estimator for the baseline hazard via the breslow estimator
'''

# Modules
# =======================================================================================================================

import os
import sys
import glob
import subprocess
import tqdm
import importlib
os.chdir('/nfs/research1/gerstung/sds/sds-ukb-cancer/projects/ProbCox/')

import pandas as pd
import numpy as np

from multiprocessing import Pool

import torch
from torch.distributions import constraints
from torch.utils.data import Dataset, DataLoader, Sampler

import pyro
import pyro.distributions as dist

from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.mcmc import MCMC, NUTS

import matplotlib.pyplot as plt

import probcox

import warnings
warnings.filterwarnings("ignore")

dtype = torch.FloatTensor

np.random.seed(87)
torch.manual_seed(34)

ROOT_DIR = '/nfs/research1/gerstung/sds/sds-ukb-cancer/'

# Custom Functions
# =======================================================================================================================
def Breslow(surv, linpred):
    surv[surv[:, -1]==1, 1] = surv[surv[:, -1]==1, 1] - 0.0000001
    event_times = surv[surv[:, -1] ==1, 1]
    event_times = event_times[np.argsort(event_times)]
    a0 = [0]
    for ii in tqdm.tqdm(range(event_times.shape[0])):
        risk_set = (surv[:, 0] < event_times[ii]) * (event_times[ii] <= surv[:, 1])
        a0.append(1/np.sum(np.exp(linpred[risk_set])))
    return(event_times, np.asarray(a0[1:]))

# Load Dataset
# =======================================================================================================================
dd = torch.load(ROOT_DIR + 'projects/ProbCox/output/prediction_train')
surv = dd['Surv']
linpred = dd['Pred']

# Breslow estimator
# =======================================================================================================================
tt, basehaz = Breslow(surv, linpred)
delta_time =[]
for jj in np.arange(0, tt.shape[0]-1):
    delta_time.append(tt[jj+1] - tt[jj])
delta_time.append(0.1)
delta_time = np.asarray(delta_time)[:, None]
delta_time = np.asarray([np.sum(delta_time[jj==tt], axis=0) for jj in np.unique(tt)])
basehaz = np.asarray([np.sum(basehaz[jj==tt], axis=0) for jj in np.unique(tt)])[:, None]
tt = np.unique(tt)
basehaz = basehaz/delta_time
A0 = np.copy(basehaz)

torch.save({'tt': tt[:-1], 'dt': delta_time[:-1], 'A0': A0[:-1]}, ROOT_DIR + 'projects/ProbCox/output/Breslow')
