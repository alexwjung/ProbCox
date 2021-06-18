
# Modules
# =======================================================================================================================
import os
import sys
import shutil
import subprocess
import tqdm
import time

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

np.random.seed(9044)
torch.manual_seed(8734)


# data
# =======================================================================================================================

dd = pd.read_csv('/nfs/nobackup/gerstung/awj/projects/ProbCox/compute/data/' + 'I16000_P3200' + '.csv')

surv = torch.from_numpy(np.asarray(dd.iloc[:, :3])).type(dtype)
X = torch.from_numpy(np.asarray(dd.iloc[:, 3:])).type(dtype)
print(X.shape)

total_obs = surv.shape[0]
total_events = torch.sum(surv[:, -1] == 1).numpy().tolist()

# ProbCox settings
# =======================================================================================================================
batchsize = 256

def predictor(data):
    theta =  pyro.sample("theta", dist.StudentT(1, loc=0, scale=0.001).expand([data[1].shape[1], 1])).type(dtype)
    pred = torch.mm(data[1], theta)
    return(pred)

sampling_proportion=[total_obs, batchsize, total_events, None]

# ProbCox
# =======================================================================================================================
def run_probcox(surv, X, sampling_proportion, predictor=predictor):
  eta=1
  run = True
  while run:
      run = False
      pyro.clear_param_store()
      m = pcox.PCox(sampling_proportion=sampling_proportion, predictor=predictor)
      m.initialize(eta=eta, rank=25)
      loss=[0]
      for ii in (range((25000))):
          # random sub-sampling
          idx = np.random.choice(range(surv.shape[0]), batchsize, replace=False)
          data=[surv[idx], X[idx]]
          loss.append(m.infer(data=data))
          # divergence check
          if loss[-1] != loss[-1]:
              eta = eta * 0.1
              run=True
              break   
  g = m.return_guide()
  out = g.quantiles([0.025, 0.5, 0.975])
  return(out)

t0 = time.time()
out = run_probcox(surv=surv, X=X, sampling_proportion=sampling_proportion)
t1 = time.time()
print(t1-t0)
