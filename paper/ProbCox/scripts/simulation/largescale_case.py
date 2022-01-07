'''


Standard Case Simulation - Case 1:


Small size simulation with N >> I >> P


individuals:  1000
covaraites:   3 binary (0.2), 3 Normal(0, 1)
theta:        -0.9, 0.2, 0, -0.4, 1.1, 0
censoring:    ~ 0.74
runs:         200 - Seed = 1, 2, ..., 200

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

from multiprocessing import Pool

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

np.random.seed(2309)
torch.manual_seed(945)

sim_name = 'sim_ls'

os.chdir('/nfs/nobackup/gerstung/awj/projects/ProbCox/paper/ProbCox')

# cluster variable
try:
    run_id = int(sys.argv[1])
except:
    run_id = 0

if run_id == 0:
    try:
        shutil.rmtree('./out/simulation/' + sim_name)
    except:
        pass
    try:
        os.mkdir('./out/simulation/' + sim_name)
    except:
        pass

# Simulation Settings
# =======================================================================================================================

I = 4000000 # Number of Individuals
P_binary = 5
P_continuous = 5
P = P_binary + P_continuous
theta = np.random.normal(0, 0.75, (10, 1))
scale = 30  # Scaling factor for Baseline Hazard

# Simulation
# =======================================================================================================================

# save theta
if run_id == 0:
    np.savetxt('./out/simulation/' + sim_name + '/theta.txt', np.round(theta, 5))
    # Rough distribution for the corresponding linear effect size
    X = np.concatenate((np.random.binomial(1, 0.2, (1000, P_binary)), np.random.normal(0, 1, (1000, P_continuous))), axis=1)
    plt.hist(np.matmul(X, theta))
    plt.show()
    plt.close()

# Class for simulation
TVC = pcox.TVC(theta=theta, P_binary=P_binary, P_continuous=P_continuous, dtype=dtype)

# Sample baseline hazard - scale is set to define censorship/events
TVC.make_lambda0(scale=scale)
if run_id == 0:
    # gauge the number to desired level of censorship
    print('\n Censorship: ', str(1 - (np.sum([torch.sum(TVC.sample()[0][:, -1]).numpy() for ii in tqdm.tqdm(range(10000))])/10000)))

# Return the underlying shape of the baseline hazard and plot
if run_id == 0:
    t_l, ll = TVC.return_lambda0()
    plt.step(t_l, ll)
    plt.show()
    plt.close()
    np.savetxt('./out/simulation/' + sim_name + '/lambda0.txt', np.concatenate((t_l[:, None], ll), axis=1))

# Sample Data
np.random.seed(run_id)
torch.manual_seed(run_id)
surv = []
X = []

for __ in tqdm.tqdm(range(I)):
    a, b = TVC.sample()
    surv.extend(a.tolist())
    X.extend(b.tolist())
    
surv = torch.tensor(surv).type(dtype)
X = torch.tensor(X).type(dtype)

if run_id == 0:
    plt.hist(surv[surv[:, -1]==1, 1])
    plt.show()
    plt.close()

total_obs = surv.shape[0]
total_events = torch.sum(surv[:, -1] == 1).numpy().tolist()

# Save information on intervall observation and number of events
if run_id != 0:
    with open('./out/simulation/' + sim_name + '/N_obs.txt', 'a') as write_out:
        write_out.write(str(run_id) + '; ' + str(surv.shape[0]) + '; ' + str(torch.sum(surv[:, -1]).detach().numpy().tolist()))
        write_out.write('\n')

idx_where = np.where(surv[:, -1]==1)[0]
total = surv.shape[0]

def sample(ii, batchsize=100, idx_where=idx_where, total=total):
    np.random.seed(ii)
    return(np.unique(np.concatenate((np.random.choice(idx_where, 1, replace=False), np.random.choice(range(total), batchsize, replace=False))))[:batchsize])
idxmat100 = []


n=0
with Pool(processes=10) as pool:
    for idx in pool.imap_unordered(sample, range(10000)):
        idxmat100.extend([idx])
        n+=1
        print(n)

idxmat100 = np.asarray(idxmat100)

# Inference Setup
# =======================================================================================================================

def predictor(data):
    theta =  pyro.sample("theta", dist.Normal(loc=0, scale=1).expand([data[1].shape[1], 1])).type(dtype)
    pred = torch.mm(data[1], theta)
    return(pred)

def evaluate(surv, X, batchsize, sampling_proportion, iter_, run_suffix, idxmat, predictor=predictor, sim_name=sim_name, run_id=run_id):
    sampling_proportion[1] = batchsize
    eta=5 # paramter for optimization
    run = True # repeat initalization if NAN encounterd while training - gauge correct optimization settings
    while run:
        run = False
        pyro.clear_param_store()
        m = pcox.PCox(sampling_proportion=sampling_proportion, predictor=predictor)
        m.initialize(eta=eta, num_particles=3)
        loss=[0]
        for ii in (range(iter_)):
            idx = idxmat[ii, :]
            data=[surv[idx], X[idx]] # subsampled data
            loss.append(m.infer(data=data))
            # divergence check
            if loss[-1] != loss[-1]:
                eta = eta * 0.1
                run=True
                break
    g = m.return_guide()
    out = g.quantiles([0.025, 0.5, 0.975])
    with open('./out/simulation/' + sim_name + '/probcox' + run_suffix + '_theta_lower.txt', 'a') as write_out:
        write_out.write(str(run_id) + '; ')
        write_out.write(''.join([str(ii) + '; ' for ii in out['theta'][0].detach().numpy()[:, 0].tolist()]))
        write_out.write('\n')
    with open('./out/simulation/' + sim_name + '/probcox' + run_suffix + '_theta.txt', 'a') as write_out:
        write_out.write(str(run_id) + '; ')
        write_out.write(''.join([str(ii) + '; ' for ii in out['theta'][1].detach().numpy()[:, 0].tolist()]))
        write_out.write('\n')
    with open('./out/simulation/' + sim_name + '/probcox' + run_suffix + '_theta_upper.txt', 'a') as write_out:
        write_out.write(str(run_id) + '; ')
        write_out.write(''.join([str(ii) + '; ' for ii in out['theta'][2].detach().numpy()[:, 0].tolist()]))
        write_out.write('\n')
        
# Run
# =======================================================================================================================
if run_id != 0:
    pyro.clear_param_store()
    out = evaluate(run_suffix='100', batchsize=100, iter_=20000, surv=surv, X=X, sampling_proportion=[total_obs, None, total_events, None], idxmat=idxmat100)
    
print('finished')

#for i in {1..10}; do bsub -env "VAR1=$i" -n 16 -M 48000 -R "rusage[mem=16000]" './largescale_case.sh'; sleep 5; done