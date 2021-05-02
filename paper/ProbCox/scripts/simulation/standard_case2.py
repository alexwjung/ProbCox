'''


Standard Case Simulation - Case 2:


Moderate sized simulation with N >> I >> P


individuals:  1000 
covaraites:   3 binary (0.2), 3 Normal(0, 1)
theta:        0.8, -0.5, 0, -0.7, 1, 0
censoring:    ~ 0.94 
runs:         100 - Seed = 1, 2, ..., 100


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

np.random.seed(2456)
torch.manual_seed(6784)

sim_name = 'sim_sc2'

os.chdir('/nfs/nobackup/gerstung/awj/projects/ProbCox/')

# cluster variable
try:
    run_id = int(sys.argv[1])
except:
    run_id = 0
    
if run_id == 0:
    try:
        shutil.rmtree('./out/' + sim_name)
    except:
        pass
    try: 
        os.mkdir('./out/' + sim_name)
    except:
        pass


# Simulation Settings
# =======================================================================================================================

I = 5000 # Number of Individuals
P_binary = 3
P_continuous = 3
P = P_binary + P_continuous
theta = np.asarray([0.8, -0.5, 0, -0.7, 1, 0])[:, None]
scale = 15  # Scaling factor for Baseline Hazard


# Simulation 
# =======================================================================================================================

# save theta
if run_id == 0:
    np.savetxt('./out/' + sim_name + '/theta.txt', np.round(theta, 5))
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
    print('\n Censorship: ', str(1 - (np.sum([torch.sum(TVC.sample()[0][:, -1]).numpy() for ii in tqdm.tqdm(range(5000))])/5000)))

# Return the underlying shape of the baseline hazard and plot
if run_id == 0:
    t_l, ll = TVC.return_lambda0()
    plt.step(t_l, ll)
    plt.show()
    plt.close()
    np.savetxt('./out/' + sim_name + '/lambda0.txt', np.concatenate((t_l[:, None], ll), axis=1))

# Sample Data 
np.random.seed(run_id)
torch.manual_seed(run_id)
surv = torch.zeros((0, 3))
X = torch.zeros((0, P))
for __ in (range(I)):
    a, b = TVC.sample()
    surv = torch.cat((surv, a))
    X = torch.cat((X, b))

if run_id == 0:
    plt.hist(surv[surv[:, -1]==1, 1])
    plt.show()
    plt.close()
    
total_obs = surv.shape[0]
total_events = torch.sum(surv[:, -1] == 1).numpy().tolist()

# Save information on intervall observation and number of events
if run_id != 0:
    with open('./out/' + sim_name + '/N_obs.txt', 'a') as write_out:
        write_out.write(str(run_id) + '; ' + str(surv.shape[0]) + '; ' + str(torch.sum(surv[:, -1]).detach().numpy().tolist()))
        write_out.write('\n')     
    
# Save data for R   
if run_id != 0:
    pd.DataFrame(np.concatenate((surv, X), axis=1)).to_csv('./tmp2/' + str(run_id) + '.csv', sep=';', index=False, header=False)


# Inference Setup 
# =======================================================================================================================
# Custom linear predictor - Here: simple linear combination
def predictor(data):
    theta =  pyro.sample("theta", dist.Normal(loc=0, scale=1).expand([data[1].shape[1], 1])).type(dtype)
    pred = torch.mm(data[1], theta)
    return(pred)

def evaluate(surv, X, batchsize, sampling_proportion, iter_, run_suffix, predictor=predictor, sim_name=sim_name, run_id=run_id):
    sampling_proportion[1] = batchsize
    eta=5 # paramter for optimization 
    run = True # repeat initalization if NAN encounterd while training - gauge correct optimization settings
    while run:
        run = False
        pyro.clear_param_store()
        m = pcox.PCox(sampling_proportion=sampling_proportion, predictor=predictor)
        m.initialize(eta=eta, num_particles=5)
        loss=[0]
        for ii in tqdm.tqdm(range((iter_))):
            idx = np.unique(np.concatenate((np.random.choice(np.where(surv[:, -1]==1)[0], 2, replace=False), np.random.choice(range(surv.shape[0]), batchsize-2, replace=False)))) # random sample of data - force at least two events (no evaluation otherwise)
            data=[surv[idx], X[idx]] # subsampled data
            loss.append(m.infer(data=data))
            data=[surv[idx], X[idx]] # subsampled data
            loss.append(m.infer(data=data))
            # divergence check
            if loss[-1] != loss[-1]:
                eta = eta * 0.1
                run=True
                break   
    g = m.return_guide()
    out = g.quantiles([0.025, 0.5, 0.975])
    with open('./out/' + sim_name + '/probcox' + run_suffix + '_theta_lower.txt', 'a') as write_out:
        write_out.write(str(run_id) + '; ')
        write_out.write(''.join([str(ii) + '; ' for ii in out['theta'][0].detach().numpy()[:, 0].tolist()]))
        write_out.write('\n')   
    with open('./out/' + sim_name + '/probcox' + run_suffix + '_theta.txt', 'a') as write_out:
        write_out.write(str(run_id) + '; ')
        write_out.write(''.join([str(ii) + '; ' for ii in out['theta'][1].detach().numpy()[:, 0].tolist()]))
        write_out.write('\n')
    with open('./out/' + sim_name + '/probcox' + run_suffix + '_theta_upper.txt', 'a') as write_out:
        write_out.write(str(run_id) + '; ')
        write_out.write(''.join([str(ii) + '; ' for ii in out['theta'][2].detach().numpy()[:, 0].tolist()]))
        write_out.write('\n')


# Run
# =======================================================================================================================
if run_id != 0:
    # batch model:
    pyro.clear_param_store()
    out = evaluate(run_suffix='full', batchsize=total_obs, iter_=25000, surv=surv, X=X, sampling_proportion=[total_obs, None, total_events, None])

    pyro.clear_param_store()
    out = evaluate(run_suffix='512', batchsize=512, iter_=25000, surv=surv, X=X, sampling_proportion=[total_obs, None, total_events, None])

    pyro.clear_param_store()
    out = evaluate(run_suffix='256', batchsize=256, iter_=25000, surv=surv, X=X, sampling_proportion=[total_obs, None, total_events, None])

    pyro.clear_param_store()
    out = evaluate(run_suffix='128', batchsize=128, iter_=25000, surv=surv, X=X, sampling_proportion=[total_obs, None, total_events, None])

    pyro.clear_param_store()
    out = evaluate(run_suffix='64', batchsize=64, iter_=25000, surv=surv, X=X, sampling_proportion=[total_obs, None, total_events, None])
    

    # execute R script
    a = ''' 
    rm(list=ls())
    library(survival)
    ROOT_DIR =  '/nfs/nobackup/gerstung/awj/projects/ProbCox'
    ''' 

    b = 'sim_name = ' + "'" +str(run_id)+ "'"

    c = '''
    sim <- read.csv(paste(ROOT_DIR, '/tmp2/', sim_name, '.csv', sep='') , header=FALSE, sep=";")

    sim <- as.data.frame(as.matrix(sim))
    m = coxph(Surv(V1, V2, V3) ~., data=sim)

    x = paste(sim_name, paste(unname(m$coefficients), collapse="; "), sep='; ')
    write(x, file = paste(ROOT_DIR, '/out/sim_sc2/R_theta.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")

    x = paste(sim_name, paste(sqrt(diag(m$var)), collapse="; "), sep='; ')
    write(x, file = paste(ROOT_DIR, '/out/sim_sc2/R_se.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")
    '''

    with open('./tmp2/' + str(run_id) + '.R', 'w') as write_out:
            write_out.write(a + b + c)

    subprocess.check_call(['Rscript', './tmp2/' + str(run_id) + '.R'], shell=False)

    os.remove('./tmp2/' + str(run_id) + '.R')
    os.remove('./tmp2/' + str(run_id) + '.csv')

print('finished')


# cluster submission
#for i in {1..100}; do bsub -env "VAR1=$i" -o /dev/null -e /dev/null -n 2 -M 2500 -R "rusage[mem=1000]" './standard_case2.sh'; sleep 30; done
#for i in {101..200}; do bsub -env "VAR1=$i" -o /dev/null -e /dev/null -n 2 -M 2500 -R "rusage[mem=1000]" './standard_case2.sh'; sleep 10; done
