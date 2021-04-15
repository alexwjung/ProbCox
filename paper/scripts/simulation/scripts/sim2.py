'''

Simulation 2:

High Dimensional simulation with random parameters. 
- check consitency of estimates 
- check variance of estimates 


N: 1000
P: 20 normal(0, 0.75) 
theta:  20 normal(0, 0.75) - 9980 0's
censoring: ~ 0.98


Evaluate for random subsampling:



'''


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

# -----------------------------------------------------------------------------------------------------------------------------

sys.path.append('./ProbCox/')

import probcox as pcox #import CoxPartialLikelihood
importlib.reload(pcox)

import simulation as sim
importlib.reload(sim)

# Setup
# -----------------------------------------------------------------------------------------------------------------------------

sim_name = 'sim2'

import warnings
warnings.filterwarnings("ignore")

dtype = torch.FloatTensor

np.random.seed(890)
torch.manual_seed(364)

run_id = int(sys.argv[1])
print(run_id)

if run_id == -1:
    shutil.rmtree('./output/simulation/' + sim_name)
    os.mkdir('./output/simulation/' + sim_name)
    
# Simulation Settings:
# -----------------------------------------------------------------------------------------------------------------------------
theta = np.random.normal(0, 0.75, 20)[:, None]
theta = np.concatenate((theta[:10], np.zeros((4990, 1)), theta[10:], np.zeros((4990, 1))))

if run_id == 0:
    np.savetxt('./output/simulation/' + sim_name + '/theta.txt', np.round(theta, 5))

P_binary = 5000
P_continuous = 5000

#X = np.concatenate((np.random.binomial(1, 0.2, (1000, P_binary)), np.random.normal(0, 1, (1000, P_continuous))), axis=1)
#plt.hist(np.matmul(X, theta))

TVC = sim.TVC(theta=theta, P_binary=P_binary, P_continuous=P_continuous, dtype=dtype)
TVC.make_lambda0(scale=2)
#np.sum([torch.sum(TVC.sample()[0][:, -1]).numpy() for ii in tqdm.tqdm(range(100))])

if run_id == 0:
    a, b = TVC.return_lambda0()
    np.savetxt('./output/simulation/' + sim_name + '/lambda0.txt', np.concatenate((a[:, None], b), axis=1))

# Run Inference
# -----------------------------------------------------------------------------------------------------------------------------
def predictor(data):
    theta =  pyro.sample("theta", dist.StudentT(1, loc=0, scale=0.001).expand([data[1].shape[1], 1])).type(dtype)
    pred = torch.mm(data[1], theta)
    return(pred)

def evaluate(surv, X, rank, batchsize, sampling_proportion, run_suffix, iter_, run_id=run_id, sim_name=sim_name, predictor=predictor):
    sampling_proportion[1] = batchsize
    run = True
    eta = 1.0
    while run:
        run = False
        pyro.clear_param_store()
        m = pcox.PCox(sampling_proportion=sampling_proportion, predictor=predictor)
        m.initialize(eta=eta, rank=rank, num_particles=3)
        loss=[0]
        for ii in tqdm.tqdm(range((iter_))):
            idx = np.concatenate((np.random.choice(np.where(surv[:, -1]==1)[0], 2, replace=False), np.random.choice(range(surv.shape[0]), batchsize-2, replace=False)))
            data=[surv[idx], X[idx]]
            if torch.sum(surv[idx, -1]) > 0:
                loss.append(m.infer(data=data))
            if loss[-1] != loss[-1]:
                eta = eta * 0.5
                run=True
                break 
    #plt.semilogy(loss)
    g = m.return_guide()
    out = g.quantiles([0.025, 0.5, 0.975])
    with open('./output/simulation/' + sim_name + '/probcox' + run_suffix + '_theta_lower.txt', 'a') as write_out:
        write_out.write(str(run_id) + '; ')
        write_out.write(''.join([str(ii) + '; ' for ii in out['theta'][0].detach().numpy()[:, 0].tolist()]))
        write_out.write('\n')   
    with open('./output/simulation/' + sim_name + '/probcox' + run_suffix + '_theta.txt', 'a') as write_out:
        write_out.write(str(run_id) + '; ')
        write_out.write(''.join([str(ii) + '; ' for ii in out['theta'][1].detach().numpy()[:, 0].tolist()]))
        write_out.write('\n')
    with open('./output/simulation/' + sim_name + '/probcox' + run_suffix + '_theta_upper.txt', 'a') as write_out:
        write_out.write(str(run_id) + '; ')
        write_out.write(''.join([str(ii) + '; ' for ii in out['theta'][2].detach().numpy()[:, 0].tolist()]))
        write_out.write('\n')
        
np.random.seed(run_id)
torch.manual_seed(run_id)
surv = torch.zeros((0, 3))
X = torch.zeros((0, 10000))
for __ in (range(1000)):
    a, b = TVC.sample()
    surv = torch.cat((surv, a))
    X = torch.cat((X, b))    
        
with open('./output/simulation/' + sim_name + '/N_obs.txt', 'a') as write_out:
    write_out.write(str(run_id) + '; ' + str(surv.shape[0]) + '; ' + str(torch.sum(surv[:, -1]).detach().numpy().tolist()))
    write_out.write('\n')     

total_obs = surv.shape[0]
total_events = torch.sum(surv[:, -1] == 1).numpy().tolist()

# batch model:
pyro.clear_param_store()
out = evaluate(run_suffix='r30', rank=30, batchsize=512, iter_=35000, predictor=predictor, surv=surv, X=X, sampling_proportion=[total_obs, None, total_events, None], sim_name=sim_name)

pyro.clear_param_store()
out = evaluate(run_suffix='r20', rank=20, batchsize=512, iter_=35000, predictor=predictor, surv=surv, X=X, sampling_proportion=[total_obs, None, total_events, None], sim_name=sim_name)

pyro.clear_param_store()
out = evaluate(run_suffix='r10', rank=10, batchsize=512, iter_=35000, predictor=predictor, surv=surv, X=X, sampling_proportion=[total_obs, None, total_events, None], sim_name=sim_name)


'''
theta_est = out['theta'][1]
theta_est = theta_est.detach().numpy()
f, ax = plt.subplots(figsize=(10, 10))
ax.scatter(theta[5000:], theta_est[5000:], c=".3")
ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
plt.show()
plt.close()
'''

print('finished')
