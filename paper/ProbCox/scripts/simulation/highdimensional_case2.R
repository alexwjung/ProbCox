# Generate Data
rm(list=ls())
set.seed(13)
library(survival)
library(glmnet)
library(doMC)
library(coxed)
library(ncvreg)
#registerDoMC(cores = 5)

ROOT_DIR =  '/nfs/nobackup/gerstung/awj/projects/ProbCox/paper/ProbCox'
#ROOT_DIR =  '/Users/alexwjung/Downloads'

# generate paramters - 10 non -zero effects
theta <- c(-0.5, 0.7, 1.2, 0.65, -0.9, 1.4, 0.2, -0.4, -1.3, 0.1)
theta <- c(theta, replicate(990, 0))

# generate datasets
for(ii in seq(200)){
  simdata <- sim.survdata(N=750, T=10000, xvars=1000, censor=0.75, num.data.frames=1, beta=theta)$data
  write.csv(simdata, file=paste(ROOT_DIR, '/tmp2/', ii, '.csv', sep='', collapse=''))
}