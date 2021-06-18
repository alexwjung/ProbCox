
# Modules
# =======================================================================================================================
import os
import sys
import shutil
import subprocess
import tqdm
import time
import h5py

import numpy as np
import pandas as pd

import torch
from torch.distributions import constraints
from torch.utils.data import Dataset, DataLoader, Sampler

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


# Dataloader
# =======================================================================================================================
# custom functions for collating samples loaded and how to sample them (randomly)
class RandomSampler(Sampler):
    def __init__(self, ids):
        self.ids_len = len(ids)
        self.ids = ids

    def __iter__(self):
        return iter(np.random.choice(self.ids, self.ids_len, replace=False).tolist())

    def __len__(self):
        return self.ids_len

# stack samples without adding additional dimension
def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) > 0:
        return([torch.cat([item[ii] for item in batch], 0) for ii in range(len(batch[0]))])
    else:
        return(None, None)

# dataloader object
class Pipe(Dataset):
    """"""
    def __init__(self, dir, max_obs, batch, dtype=dtype):
        # open hdf5 file
        self.f = h5py.File(dir, 'r') 
        self.dtype = dtype
        self.max_obs = max_obs
        self.batch = batch

    def __len__(self):
        # epoch size
        return(25000)

    def __disconnect__(self):
        self.f.close()

    def __getitem__(self, ii, dtype=dtype):
        np.random.seed(ii)
        idx=np.sort(np.random.choice(range(self.max_obs), self.batch, replace=False))
        time = self.f['surv'][idx, :]
        X =  self.f['X'][idx, :]
        return(time, X)

    
# Data
# =======================================================================================================================

# define data loader
file = '/nfs/nobackup/gerstung/awj/projects/ProbCox/compute/data/I16000_P3200.h5'
# defined by the full dataset
total_obs = 119071
total_events = 4034
        
loader = Pipe(dir=file, max_obs=total_obs, batch=256)
surv, X = loader.__getitem__(9)
print(X.shape)

# ProbCox settings
# =======================================================================================================================

def predictor(data):
    theta =  pyro.sample("theta", dist.StudentT(1, loc=0, scale=0.001).expand([data[1].shape[1], 1])).type(dtype)
    pred = torch.mm(data[1], theta)
    return(pred)

sampling_proportion=[total_obs, 256, total_events, None]


# ProbCox
# =======================================================================================================================
def run_probcox(sampling_proportion=sampling_proportion, predictor=predictor):
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
          surv, X = loader.__getitem__(ii)
          surv = torch.from_numpy(surv).type(dtype)
          X = torch.from_numpy(X).type(dtype)
          data=[surv, X]
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
out = run_probcox(sampling_proportion=sampling_proportion)
t1 = time.time()

print(t1-t0)

loader.__disconnect__()



