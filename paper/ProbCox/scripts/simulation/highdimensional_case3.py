'''
High Dimensional Case Simulation:

Moderate sized simulation with P >> N >> I and correlated X

individuals:  1000
covaraites:   0 binary (0.35), 3000 Normal(0, 1) - correlated
theta:        ~ N(0, 0.75^2)
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

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist

from pyro.infer import SVI, Trace_ELBO

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import probcox as pcox

from simulate_correlated import TVC

dtype = torch.FloatTensor

np.random.seed(499)
torch.manual_seed(874)

sim_name = 'sim_hd3'

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

I = 1000 # Number of Individuals
P_binary = 0
P_continuous = 3000
P = P_binary + P_continuous
theta = np.random.normal(0, 0.75, 20)[:, None]

theta = np.concatenate(( np.zeros((1490, 1)), theta, np.zeros((1490, 1))))
scale = 30  # Scaling factor for Baseline Hazard

# Simulation
# =======================================================================================================================
# save theta
if run_id == 0:
    np.savetxt('./out/simulation/' + sim_name + '/theta.txt', np.round(theta, 5))

# Correlation Matrix
n = P
a = 0.75
A = np.matrix([np.random.randn(n) + np.random.randn(1)*a for i in range(n)])
A = A*np.transpose(A)
D_half = np.diag(np.diag(A)**(-0.5))
C = D_half*A*D_half
L = np.linalg.cholesky(C)
vals = list(np.array(C.ravel())[0])

if run_id == 0:
    plt.rcParams['font.size'] = 5
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    cm = 1/2.54
    
    fig, ax = plt.subplots(1, 1, figsize=(1.35*cm, 1.35*cm), dpi=600)
    ax.hist(vals, density=True, color='#79b538')
    ax.set_xlim([-1, 1])
    ax.set_xticks([-1, -0.5, 0, 0.5,  1])
    ax.set_xticklabels(['-1', '', '0', '', '1'])   
    #ax.set_xlabel('Pairwise Corr.')
    ax.set_ylabel('Density')
    plt.savefig('./out/simulation/figures/hd3_paircorr.eps', bbox_inches='tight', dpi=600, transparent=True)
    plt.savefig('./out/simulation/figures/hd3_paircorr.png', bbox_inches='tight', dpi=600, transparent=True)
    plt.savefig('./out/simulation/figures/hd3_paircorr.pdf', bbox_inches='tight', dpi=600, transparent=True)
    plt.show()
    plt.close()
    
    fig, ax = plt.subplots(1, 1, figsize=(1.35*cm, 1.35*cm), dpi=600)
    cax = ax.matshow(C, cmap='summer')
    ax.axis('off')
    cb = fig.colorbar(cax, orientation="horizontal", fraction=0.047, pad=0.025)
    cb.outline.set_visible(False)
    plt.savefig('./out/simulation/figures/hd3_corrmat.eps', bbox_inches='tight', dpi=600, transparent=True)
    plt.savefig('./out/simulation/figures/hd3_corrmat.png', bbox_inches='tight', dpi=600, transparent=True)
    plt.savefig('./out/simulation/figures/hd3_corrmat.pdf', bbox_inches='tight', dpi=600, transparent=True)
    plt.show()
    plt.close()


TVC_corr = TVC(theta=theta, P_binary=P_binary, P_continuous=P_continuous, cholesky=L, dtype=dtype)

# Sample baseline hazard - scale is set to define censorship/events
TVC_corr.make_lambda0(scale=scale)

if run_id == 0:
    # gauge the number to desired level of censorship
    print('\n Censorship: ', str(1 - (np.sum([torch.sum(TVC_corr.sample()[0][:, -1]).numpy() for ii in tqdm.tqdm(range(1000))])/1000)))

# Return the underlying shape of the baseline hazard and plot
if run_id == 0:
    t_l, ll = TVC_corr.return_lambda0()
    plt.step(t_l, ll)
    plt.show()
    plt.close()
    np.savetxt('./out/simulation/' + sim_name + '/lambda0.txt', np.concatenate((t_l[:, None], ll), axis=1))

# Sample Data
np.random.seed(run_id)
torch.manual_seed(run_id)
surv = torch.zeros((0, 3))
X = torch.zeros((0, P))
for __ in tqdm.tqdm(range(I)):
    a, b = TVC_corr.sample()
    surv = torch.cat((surv, a))
    X = torch.cat((X, b))

if run_id == 0:
    plt.hist(surv[surv[:, -1]==1, 1])
    plt.show()
    plt.close()

total_obs = surv.shape[0]
total_events = torch.sum(surv[:, -1] == 1).numpy().tolist()

#Save information on intervall observation and number of events
if run_id != 0:
    with open('./out/simulation/' + sim_name + '/N_obs.txt', 'a') as write_out:
        write_out.write(str(run_id) + '; ' + str(surv.shape[0]) + '; ' + str(torch.sum(surv[:, -1]).detach().numpy().tolist()) + '; '  + str(1-np.unique(surv[surv[:, -1] == 1, 1]).shape[0]/surv[surv[:, -1] == 1, 1].shape[0]))
        write_out.write('\n')
        
# Save data for R
if run_id != 0:
    pd.DataFrame(np.concatenate((surv, X), axis=1)).to_csv('./tmp2/' + str(run_id) + '.csv', sep=';', index=False, header=False)

# Inference Setup
# =======================================================================================================================
# Custom linear predictor - Here: simple linear combination
def predictor(data):
    theta =  pyro.sample("theta", dist.StudentT(1, loc=0, scale=0.001).expand([data[1].shape[1], 1])).type(dtype)
    pred = torch.mm(data[1], theta)
    return(pred)

def evaluate(surv, X, rank, batchsize, sampling_proportion, iter_, run_suffix, predictor=predictor, sim_name=sim_name, run_id=run_id):
    sampling_proportion[1] = batchsize
    eta=5 # paramter for optimization
    run = True # repeat initalization if NAN encounterd while training - gauge correct optimization settings
    while run:
        run = False
        pyro.clear_param_store()
        m = pcox.PCox(sampling_proportion=sampling_proportion, predictor=predictor)
        m.initialize(eta=eta, rank=rank, num_particles=5)
        loss=[0]
        for ii in tqdm.tqdm(range((iter_))):
            idx = np.unique(np.concatenate((np.random.choice(np.where(surv[:, -1]==1)[0], 1, replace=False), np.random.choice(range(surv.shape[0]), batchsize, replace=False))))[:batchsize] # random sample of data - force at least one event (no evaluation otherwise)
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
    # batch model:
    pyro.clear_param_store()
    out = evaluate(run_suffix='rank10', rank=10, batchsize=512, iter_=25000, surv=surv, X=X, sampling_proportion=[total_obs, None, total_events, None])
    pyro.clear_param_store()
    out = evaluate(run_suffix='rank30', rank=30, batchsize=512, iter_=25000, surv=surv, X=X, sampling_proportion=[total_obs, None, total_events, None])
    pyro.clear_param_store()
    out = evaluate(run_suffix='rank50', rank=50, batchsize=512, iter_=25000, surv=surv, X=X, sampling_proportion=[total_obs, None, total_events, None])
    
    # execute R script
    a = '''
    rm(list=ls())
    set.seed(13)
    library(survival)
    library(glmnet)
    require(doMC)
    registerDoMC(cores = 5)
    ROOT_DIR =  '/nfs/research/gerstung/awj/projects/ProbCox/paper/ProbCox'
    '''

    b = 'sim_name = ' + "'" + str(run_id) + "'"

    c = '''
    sim <- read.csv(paste(ROOT_DIR, '/tmp/', sim_name, '.csv', sep='') , header=FALSE, sep=";")

    sim <- as.data.frame(as.matrix(sim))
    yss = Surv(sim$V1, sim$V2, sim$V3)
    
    # Lasso 
    cv.fit <- c()
    cv.fit <- cv.glmnet(as.matrix(sim[, 4:3003]), yss, family ="cox", nfolds=5, parallel=TRUE, type.measure ="C", alpha=1)
    x = paste(sim_name, paste(unname(coef(cv.fit, s=cv.fit$lambda.min)), collapse="; "), sep='; ')
    write(x, file = paste(ROOT_DIR, '/out/simulation/sim_hd3/R_lasso_theta.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")
    x = paste(sim_name, paste(unname(coef(cv.fit, s=cv.fit$lambda.1se)), collapse="; "), sep='; ')
    write(x, file = paste(ROOT_DIR, '/out/simulation/sim_hd3/R_lasso_theta_1se.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")
    
    w1 <- 1/abs(as.numeric(coef(cv.fit, s=cv.fit$lambda.min)))
    w1[w1 == Inf] <- 1000000 
    
    w2 <- 1/abs(as.numeric(coef(cv.fit, s=cv.fit$lambda.1se)))
    w2[w2 == Inf] <- 1000000 
    
    # Adaptive Lasso
    # Alasso 
    cv.fit <- c()
    cv.fit <-cv.glmnet(as.matrix(sim[, 4:3003]), yss, family ="cox", nfolds=5, parallel=TRUE, type.measure="C", penalty.factor=w1)
    x = paste(sim_name, paste(unname(coef(cv.fit, s=cv.fit$lambda.min)), collapse="; "), sep='; ')
    write(x, file = paste(ROOT_DIR, '/out/simulation/sim_hd3/R_Alasso1_theta.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")
    x = paste(sim_name, paste(unname(coef(cv.fit, s=cv.fit$lambda.1se)), collapse="; "), sep='; ')
    write(x, file = paste(ROOT_DIR, '/out/simulation/sim_hd3/R_Alasso1_theta_1se.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")
    
    # Alasso 2
    cv.fit <- c()
    cv.fit <-cv.glmnet(as.matrix(sim[, 4:3003]), yss, family ="cox", nfolds=5, parallel=TRUE, type.measure="C", penalty.factor=w2)
    x = paste(sim_name, paste(unname(coef(cv.fit, s=cv.fit$lambda.min)), collapse="; "), sep='; ')
    write(x, file = paste(ROOT_DIR, '/out/simulation/sim_hd3/R_Alasso2_theta.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")
    x = paste(sim_name, paste(unname(coef(cv.fit, s=cv.fit$lambda.1se)), collapse="; "), sep='; ')
    write(x, file = paste(ROOT_DIR, '/out/simulation/sim_hd3/R_Alasso2_theta_1se.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")
    
    '''

    with open('./tmp/' + str(run_id) + '.R', 'w') as write_out:
            write_out.write(a + b + c)

    subprocess.check_call(['/hps/software/spack/opt/spack/linux-centos8-sandybridge/gcc-9.3.0/r-4.1.1-jkdw35fv6wrc5jm5ljuby7fp5ysnm2y2/rlib/R/bin/Rscript', '/nfs/research/gerstung/awj/projects/ProbCox/paper/ProbCox/tmp/' + str(run_id) + '.R'], shell=False)


    os.remove('./tmp/' + str(run_id) + '.R')
    os.remove('./tmp/' + str(run_id) + '.csv')

# Prior elicitation
# =======================================================================================================================

# Custom linear predictor - Here: simple linear combination
def predictor(data):
    theta =  pyro.sample("theta", dist.Laplace(loc=0, scale=0.25).expand([data[1].shape[1], 1])).type(dtype)
    pred = torch.mm(data[1], theta)
    return(pred)

def evaluate(surv, X, rank, batchsize, sampling_proportion, iter_, run_suffix, predictor=predictor, sim_name=sim_name, run_id=run_id):
    sampling_proportion[1] = batchsize
    eta=5 # paramter for optimization
    run = True # repeat initalization if NAN encounterd while training - gauge correct optimization settings
    while run:
        run = False
        pyro.clear_param_store()
        m = pcox.PCox(sampling_proportion=sampling_proportion, predictor=predictor)
        m.initialize(eta=eta, rank=rank, num_particles=5)
        loss=[0]
        for ii in tqdm.tqdm(range((iter_))):
            idx = np.unique(np.concatenate((np.random.choice(np.where(surv[:, -1]==1)[0], 1, replace=False), np.random.choice(range(surv.shape[0]), batchsize, replace=False))))[:batchsize] # random sample of data - force at least one event (no evaluation otherwise)
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

if run_id != 0:
    pyro.clear_param_store()
    out = evaluate(run_suffix='rank30_L1', rank=30, batchsize=512, iter_=25000, surv=surv, X=X, sampling_proportion=[total_obs, None, total_events, None])

    
    
    
print('finished')

#for i in 34; do bsub -env "VAR1=$i"  -n 3 -M 24000 -R "rusage[mem=8000]" './highdimensional_case3.sh'; sleep 1; done
#for i in {11..200}; do bsub -env "VAR1=$i" -o /dev/null -e /dev/null -n 2 -M 16000 -R "rusage[mem=6000]" './highdimensional_case3.sh'; sleep 1; done


