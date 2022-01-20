'''


High Dimensional Case Simulation 2:


Comparision for MCP, SCAD.


individuals:  750
covaraites:   1000 Normal(0, 1)
theta:        ~ -0.5, 0.7, 1.2, 0.65, -0.9, 1.4, 0.2, -0.4, -1.3, 0.1 (0...)
censoring:    ~ 0.75
runs:         200 - Seed = 1, 2, ..., 200

For data generation see highdimensional_case2.R

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

np.random.seed(5462)
torch.manual_seed(785)

sim_name = 'sim_hd2'

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

# Data
# =======================================================================================================================
dd = pd.read_csv('./tmp2/' + str(run_id) + '.csv')
dd = dd.iloc[:, 1:]
X = torch.from_numpy(np.asarray(dd.iloc[:, :-2]).astype(float)).type(dtype)
surv = torch.from_numpy(np.concatenate((np.zeros((750, 1)), dd.iloc[:, -2:]), axis=1).astype(float)).type(dtype)

total_obs = surv.shape[0]
total_events = torch.sum(surv[:, -1] == 1).numpy().tolist()

# Save information on intervall observation and number of events
if run_id != 0:
    with open('./out/simulation/' + sim_name + '/N_obs.txt', 'a') as write_out:
        write_out.write(str(run_id) + '; ' + str(surv.shape[0]) + '; ' + str(torch.sum(surv[:, -1]).detach().numpy().tolist()) + '; '  + str(1-np.unique(surv[surv[:, -1] == 1, 1]).shape[0]/surv[surv[:, -1] == 1, 1].shape[0]))
        write_out.write('\n')

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
    out = evaluate(run_suffix='rank5', rank=5, batchsize=256, iter_=15000, surv=surv, X=X, sampling_proportion=[total_obs, None, total_events, None])


    pyro.clear_param_store()
    out = evaluate(run_suffix='rank50', rank=50, batchsize=256, iter_=15000, surv=surv, X=X, sampling_proportion=[total_obs, None, total_events, None])
    
    # execute R script
    a = '''
    rm(list=ls())
    set.seed(13)
    library(survival)
    library(glmnet)
    require(doMC)
    registerDoMC(cores = 3)
    library(coxed)
    library(ncvreg)
    library(BVSNLP)
    ROOT_DIR =  '/nfs/nobackup/gerstung/awj/projects/ProbCox/paper/ProbCox'
    '''

    b = 'sim_name = ' + "'" + str(run_id) + "'"

    c = '''
    sim <- read.csv(paste(ROOT_DIR, '/tmp2/', sim_name, '.csv', sep='', collapse=''))

    sim <- as.data.frame(as.matrix(sim))[, 2:1003]
    yss = Surv(sim$y, sim$failed)
    

    # Lasso 
    cv.fit <- c()
    cv.fit <- cv.glmnet(as.matrix(sim[, 1:1000]), yss, family ="cox", nfolds=5, parallel=TRUE, type.measure ="C", alpha=1)
    x = paste(sim_name, paste(unname(coef(cv.fit, s=cv.fit$lambda.min)), collapse="; "), sep='; ')
    write(x, file = paste(ROOT_DIR, '/out/simulation/sim_hd2/R_lasso_theta.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")
    x = paste(sim_name, paste(unname(coef(cv.fit, s=cv.fit$lambda.1se)), collapse="; "), sep='; ')
    write(x, file = paste(ROOT_DIR, '/out/simulation/sim_hd2/R_lasso_theta_1se.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")

    w1 <- 1/abs(as.numeric(coef(cv.fit, s=cv.fit$lambda.min)))
    w1[w1 == Inf] <- 1000000 

    w2 <- 1/abs(as.numeric(coef(cv.fit, s=cv.fit$lambda.1se)))
    w2[w2 == Inf] <- 1000000 

    # Ridge 
    #cv.fit <- c()
    #cv.fit <- cv.glmnet(as.matrix(sim[, 1:1000]), yss, family ="cox", nfolds=5, parallel=TRUE, type.measure ="C", alpha=0)
    #x = paste(sim_name, paste(unname(coef(cv.fit, s=cv.fit$lambda.min)), collapse="; "), sep='; ')
    #write(x, file = paste(ROOT_DIR, '/out/simulation/sim_hd2/R_ridge_theta.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")
    #x = paste(sim_name, paste(unname(coef(cv.fit, s=cv.fit$lambda.1se)), collapse="; "), sep='; ')
    #write(x, file = paste(ROOT_DIR, '/out/simulation/sim_hd2/R_ridge_theta_1se.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")

    #w3 <- 1/abs(as.numeric(coef(cv.fit, s=cv.fit$lambda.min)))
    #w3[w3 == Inf] <- 999999999 

    #w4 <- 1/abs(as.numeric(coef(cv.fit, s=cv.fit$lambda.1se)))
    #w4[w4 == Inf] <- 999999999 


    # Adaptive Lasso
    # Alasso 
    cv.fit <- c()
    cv.fit <-cv.glmnet(as.matrix(sim[, 1:1000]), yss, family ="cox", nfolds=5, parallel=TRUE, type.measure="C", penalty.factor=w1)
    x = paste(sim_name, paste(unname(coef(cv.fit, s=cv.fit$lambda.min)), collapse="; "), sep='; ')
    write(x, file = paste(ROOT_DIR, '/out/simulation/sim_hd2/R_Alasso1_theta.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")
    x = paste(sim_name, paste(unname(coef(cv.fit, s=cv.fit$lambda.1se)), collapse="; "), sep='; ')
    write(x, file = paste(ROOT_DIR, '/out/simulation/sim_hd2/R_Alasso1_theta_1se.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")

    # Alasso 2
    cv.fit <- c()
    cv.fit <-cv.glmnet(as.matrix(sim[, 1:1000]), yss, family ="cox", nfolds=5, parallel=TRUE, type.measure="C", penalty.factor=w2)
    x = paste(sim_name, paste(unname(coef(cv.fit, s=cv.fit$lambda.min)), collapse="; "), sep='; ')
    write(x, file = paste(ROOT_DIR, '/out/simulation/sim_hd2/R_Alasso2_theta.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")
    x = paste(sim_name, paste(unname(coef(cv.fit, s=cv.fit$lambda.1se)), collapse="; "), sep='; ')
    write(x, file = paste(ROOT_DIR, '/out/simulation/sim_hd2/R_Alasso2_theta_1se.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")

    # Alasso 3
    #cv.fit <- c()
    #cv.fit <-cv.glmnet(as.matrix(sim[, 1:1000]), yss, family ="cox", nfolds=5, parallel=TRUE, type.measure="C", penalty.factor=w3)
    #x = paste(sim_name, paste(unname(coef(cv.fit, s=cv.fit$lambda.min)), collapse="; "), sep='; ')
    #write(x, file = paste(ROOT_DIR, '/out/simulation/sim_hd2/R_Alasso3_theta.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")
    #x = paste(sim_name, paste(unname(coef(cv.fit, s=cv.fit$lambda.1se)), collapse="; "), sep='; ')
    #write(x, file = paste(ROOT_DIR, '/out/simulation/sim_hd2/R_Alasso3_theta_1se.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")

    # Alasso 4
    #cv.fit <- c()
    #cv.fit <-cv.glmnet(as.matrix(sim[, 1:1000]), yss, family ="cox", nfolds=5, parallel=TRUE, type.measure="C", penalty.factor=w4)
    #x = paste(sim_name, paste(unname(coef(cv.fit, s=cv.fit$lambda.min)), collapse="; "), sep='; ')
    #write(x, file = paste(ROOT_DIR, '/out/simulation/sim_hd2/R_Alasso4_theta.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")
    #x = paste(sim_name, paste(unname(coef(cv.fit, s=cv.fit$lambda.1se)), collapse="; "), sep='; ')
    #write(x, file = paste(ROOT_DIR, '/out/simulation/sim_hd2/R_Alasso4_theta_1se.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")

    # SCAD
    m <- cv.ncvsurv(as.matrix(sim[, 1:1000]), yss, penalty=c("SCAD"), nfolds=5, se=c('bootstrap'), trace=TRUE)
    fit <- m$fit$beta[,m$min]
    x = paste(sim_name, paste(unname(m$fit$beta[,m$min]), collapse="; "), sep='; ')
    write(x, file = paste(ROOT_DIR, '/out/simulation/sim_hd2/R_SCAD_theta.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")

    # MCP
    m <- cv.ncvsurv(as.matrix(sim[, 1:1000]), yss, penalty=c("MCP"), nfolds=5, se=c('bootstrap'), trace=TRUE)
    fit <- m$fit$beta[,m$min]
    x = paste(sim_name, paste(unname(m$fit$beta[,m$min]), collapse="; "), sep='; ')
    write(x, file = paste(ROOT_DIR, '/out/simulation/sim_hd2/R_MCP_theta.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")

    # BVSNLP
    bout <- bvs(sim[, 1:1000], yss, family = "survival", nlptype = "piMOM", niter = 2000, ncpu=8)
    coef <- rep(0, 1000)
    coef[bout$HPM] <- bout$beta_hat

    x = paste(sim_name, paste(unname(coef), collapse="; "), sep='; ')
    write(x, file = paste(ROOT_DIR, '/out/simulation/sim_hd2/R_BVSNLP_theta.txt', sep=''), ncolumns = 1, append = TRUE, sep = ";")

    '''

    with open('./tmp2/' + str(run_id) + '.R', 'w') as write_out:
            write_out.write(a + b + c)

    subprocess.check_call(['Rscript', '/nfs/nobackup/gerstung/awj/projects/ProbCox/paper/ProbCox/tmp2/' + str(run_id) + '.R'], shell=False)


    os.remove('./tmp2/' + str(run_id) + '.R')
    os.remove('./tmp2/' + str(run_id) + '.csv')


print('finished')

#for i in {1..200}; do bsub -env "VAR1=$i" -o /dev/null -e /dev/null -n 5 -M 5000 -R "rusage[mem=2000]" './highdimensional_case2.sh'; sleep 10; done

#for i in 90 172; do bsub -env "VAR1=$i" -n 5 -M 6000 -R "rusage[mem=2500]" './highdimensional_case2.sh'; sleep 10; done

