'''
Simulation 1.2:

Small simulation with selected parameters.
- check consitency of estimates
- check coverage of estimates
- High level of censoring

N: 1000
P: 3 binary (0.2), 3 Normal(0, 1)
theta: -0.9, 0.2, 0, -0.4, 1.1, 0
censoring: ~ 0.98


Evaluate for random subsampling:
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

sim_name = 'sim1_2'

import warnings
warnings.filterwarnings("ignore")

dtype = torch.FloatTensor

np.random.seed(2456)
torch.manual_seed(6784)

run_id = int(sys.argv[1])
print(run_id)


if run_id == -1:
    shutil.rmtree('./output/simulation/' + sim_name)
    os.mkdir('./output/simulation/' + sim_name)


# Simulation Settings:
# -----------------------------------------------------------------------------------------------------------------------------

theta = np.asarray([0.8, -0.5, 0, -0.7, 1, 0])[:, None]
if run_id == 0:
    np.savetxt('./output/simulation/' + sim_name + '/theta.txt', np.round(theta, 5))

P_binary = 3
P_continuous = 3

TVC = sim.TVC(theta=theta, P_binary=P_binary, P_continuous=P_continuous, dtype=dtype)
TVC.make_lambda0(scale=10)

if run_id == 0:
    a, b = TVC.return_lambda0()
    np.savetxt('./output/simulation/' + sim_name + '/lambda0.txt', np.concatenate((a[:, None], b), axis=1))

# Run Inference
# -----------------------------------------------------------------------------------------------------------------------------
def evaluate(surv, X, batchsize, sampling_proportion, run_suffix, iter_, run_id=run_id, sim_name=sim_name):
    sampling_proportion[1] = batchsize
    eta=10
    run = True
    while run:
        run = False
        pyro.clear_param_store()
        m = pcox.PCox(sampling_proportion=sampling_proportion)
        m.initialize(eta=eta, num_particles=2)
        loss=[0]
        for ii in tqdm.tqdm(range((iter_))):
            idx = np.concatenate((np.random.choice(np.where(surv[:, -1]==1)[0], 2, replace=False), np.random.choice(range(surv.shape[0]), batchsize-2, replace=False)))
            data=[surv[idx], X[idx]]
            loss.append(m.infer(data=data))
            # divergence check
            if loss[-1] != loss[-1]:
                eta = eta * 0.1
                run=True
                break
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
X = torch.zeros((0, 6))
for __ in (range(1000)):
    a, b = TVC.sample()
    surv = torch.cat((surv, a))
    X = torch.cat((X, b))

if run_id == 0:
    np.savetxt('./output/simulation/' + sim_name + '/N_timevars.txt', np.asarray([surv.shape[0]]))

pd.DataFrame(np.concatenate((surv, X), axis=1)).to_csv('./tmp/' + str(run_id) + '.csv', sep=';', index=False, header=False)
with open('./output/simulation/' + sim_name + '/N_obs.txt', 'a') as write_out:
    write_out.write(str(run_id) + '; ' + str(surv.shape[0]) + '; ' + str(torch.sum(surv[:, -1]).detach().numpy().tolist()))
    write_out.write('\n')

total_obs = surv.shape[0]
total_events = torch.sum(surv[:, -1] == 1).numpy().tolist()

# batch model:
pyro.clear_param_store()
evaluate(run_suffix='full', batchsize=total_obs, iter_=2500, surv=surv, X=X, sampling_proportion=[total_obs, None, total_events, None])

pyro.clear_param_store()
evaluate(run_suffix='1024', batchsize=1024, iter_=2500, surv=surv, X=X, sampling_proportion=[total_obs, None, total_events, None])

pyro.clear_param_store()
evaluate(run_suffix='512', batchsize=512, iter_=2500, surv=surv, X=X, sampling_proportion=[total_obs, None, total_events, None])

pyro.clear_param_store()
evaluate(run_suffix='256', batchsize=256, iter_=2500, surv=surv, X=X, sampling_proportion=[total_obs, None, total_events, None])



# execute R script
a = '''
rm(list=ls())
library(survival)
ROOT_DIR =  '/nfs/nobackup/gerstung/awj/projects/ProbCox'
'''

b = 'sim_name = ' + "'" +str(run_id)+ "'"

c = '''
sim <- read.csv(paste(ROOT_DIR, '/tmp/', sim_name, '.csv', sep='') , header=FALSE, sep=";")

sim <- as.data.frame(as.matrix(sim))
m = coxph(Surv(V1, V2, V3) ~., data=sim)

x = paste(sim_name, paste(unname(m$coefficients), collapse="; "), sep='; ')
write(x, file = paste(ROOT_DIR, '/output/simulation/sim1_2/R_theta.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")

x = paste(sim_name, paste(sqrt(diag(m$var)), collapse="; "), sep='; ')
write(x, file = paste(ROOT_DIR, '/output/simulation/sim1_2/R_se.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")
'''

with open('./tmp/' + str(run_id) + '.R', 'w') as write_out:
        write_out.write(a + b + c)

subprocess.check_call(['Rscript', './tmp/' + str(run_id) + '.R'], shell=False)

os.remove('./tmp/' + str(run_id) + '.R')
os.remove('./tmp/' + str(run_id) + '.csv')

with open('./output/simulation/' + sim_name + '/iteration.txt', 'a') as write_out:
    write_out.write(str(run_id))
    write_out.write('\n')

print('finished')
