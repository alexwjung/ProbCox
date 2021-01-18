import os
import sys
import tqdm
import importlib
import config
os.chdir(config.ROOT_DIR)

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

# -----------------------------------------------------------------------------------------------------------------------------
# importing csv data
colon = pd.read_csv('./data/real/colon.csv', sep=',')
surv = np.asarray(colon[['time', 'status']])
surv = np.concatenate((np.zeros((surv.shape[0], 1)), surv), axis=1)
X = np.asarray(colon[['sex', 'age', 'obstruct', 'perfor', 'adhere', 'nodes', 'differ', 'extent', 'surg', 'node4', 'etype']])
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

surv = torch.from_numpy(surv).type(torch.FloatTensor)
X = torch.from_numpy(X).type(torch.FloatTensor)

total_obs = surv.shape[0]
batch_size = 50
total_events = torch.sum(surv[:, -1] == 1)
sampling_proportion = [total_obs, batch_size, total_events, None]

# -----------------------------------------------------------------------------------------------------------------------------

pyro.clear_param_store()
m = pcox.PCox(sampling_proportion=sampling_proportion)
m.initialize(eta=0.01)
loss=[]
#idx = np.random.choice(range(surv.shape[0]), 30, replace=False)
for ii in tqdm.tqdm(range((10000))):
    idx = np.random.choice(range(surv.shape[0]), batch_size, replace=False)
    data=[surv[idx], X[idx]]
    loss.append(m.infer(data=data))
plt.semilogy(loss)
theta_est = pyro.get_param_store()['AutoMultivariateNormal.loc'].detach()
print(pcox.concordance(surv.detach(), torch.mm(X, (theta_est[:, None]).detach())))
for key, value in pyro.get_param_store().items():
    print(f"{key}:\n{value}\n")


 # -----------------------------------------------------------------------------------------------------------------------------
# %% markdown
# rm(list=ls())
# library(survival)
# library(stargazer)
#
# data("colon")
# colon
#
# colon = colon[ , 4:16]
# colon <- colon[is.finite(rowSums(colon)),]
#
# write.csv(colon, '/Users/awj/Desktop/colon.csv', sep=';')
#
# standardize <- function(x){
#   (x - mean(x))/sqrt(var(x))
# }
#
# colon[ , c(1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13)] = apply(colon[ , c(1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13)], 2, standardize)
#
# names(colon)
# m1 = coxph(Surv(time, status) ~ sex + age  + obstruct + perfor + adhere  + nodes + differ + extent + surg + node4 + etype, data=colon)
# summary(m1)
