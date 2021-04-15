'''

Simulation 0:


Approximation of Likelihood through subsampling. 
- N - 5000, 10000
- C - 0.5, 0.75, 0.95, 0.99
- P - 10, 20
- B - 64, 128, 256, 512 

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

# Setup
# -----------------------------------------------------------------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

dtype = torch.FloatTensor

np.random.seed(90834)
torch.manual_seed(873645)

# Functions:
# -----------------------------------------------------------------------------------------------------------------------------

def run(surv, pred, batch, est):   
    total_obs = surv.shape[0]
    total_events = torch.sum(surv[:, -1] == 1).numpy().tolist()
    sampling_proportion = [total_obs, batch, total_events, None]
    ll = []
    ll2 = []
    while len(ll) <=1000:
        idx = np.concatenate((np.random.choice(np.where(surv[:, -1]==1)[0], 2, replace=False), np.random.choice(range(surv.shape[0]), batch-2, replace=False)))
        sampling_proportion[-1] = torch.sum(surv[idx, -1]).numpy().tolist()
        if torch.sum(surv[idx, -1]) > 1:
            e = pcox.CoxPartialLikelihood(pred=pred[idx], sampling_proportion=sampling_proportion).log_prob(surv=surv[idx]).detach().numpy()
            MPE = ((e-est)/est)
            MAPE = np.abs(MPE)
            ll.append(MPE.tolist())
            ll2.append(MAPE.tolist())
    return(np.mean(ll), np.mean(ll2))
    
# Simulation - Settings:
# -----------------------------------------------------------------------------------------------------------------------------

N = [5000, 10000]
P = [10, 20]
C = [0.5, 0.75, 0.95, 0.99]
B = [64, 128, 256, 512]


# Simulation - Run:
# -----------------------------------------------------------------------------------------------------------------------------

res = np.zeros((16, 4))
res2 = np.zeros((16, 4))
sim_n =[]
ii = 0 
jj = 0
for p in P:
    # make baselinehazard
    cond = True
    while cond:
        theta = np.random.normal(0, 0.5, (p, 1))
        #X = np.concatenate((np.random.binomial(1, 0.2, (1000, 10)), np.random.normal(0, 1, (1000, 10))), axis=1)
        #plt.hist(np.matmul(X, theta))
        TVC = sim.TVC(theta=theta, P_binary=int(p/2), P_continuous=int(p/2), dtype=dtype)
        TVC.make_lambda0(scale=4)
        s = np.sum([torch.sum(TVC.sample()[0][:, -1]).numpy() for ii in (range(100))])/100 
        
        if np.logical_and(s>=0.05, s<=0.95):
            cond = False
        theta_ = torch.normal(0, 0.5, (p, 1)).type(dtype)
        
    for n in N:
        for c in C:
            # make dataset
            surv, X = TVC.make_dataset(obs=n, fraction_censored=c)
            
            sim_n.append('N: ' + str(n) + '(' + str(surv.shape[0]) + ')' +', P: ' + str(p) + ', C: ' + str(c))
            
            pred = torch.mm(X, theta_).type(dtype)
            est = pcox.CoxPartialLikelihood(pred=pred, sampling_proportion=None).log_prob(surv=surv).detach().numpy()
            
            # fit to batch
            for b in tqdm.tqdm(B):
                res[ii, jj], res2[ii, jj] = run(surv=surv, pred=pred, batch=b, est=est)
                jj += 1
            ii += 1
            jj = 0
                    
res = np.round(res, 4)
res2 = np.round(res2, 4)
         
pd.DataFrame(np.concatenate((np.asarray(sim_n)[:, None], res.astype(str)), axis=1)).to_csv('./output/simulation/tables/likelihood_approx_MPE.csv')
pd.DataFrame(np.concatenate((np.asarray(sim_n)[:, None], res2.astype(str)), axis=1)).to_csv('./output/simulation/tables/likelihood_approx_MAPE.csv')

print('finished')