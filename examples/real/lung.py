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

# data
lung = pd.read_csv('./data/real/lung.csv', sep=';')
surv = np.asarray(lung[['start', 'time', 'status']])
X = np.asarray(lung[['age', 'sex', 'ph.ecog', 'ph.karno', 'pat.karno', 'meal.cal', 'wt.loss']])
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

surv = torch.from_numpy(surv).type(torch.FloatTensor)
X = torch.from_numpy(X).type(torch.FloatTensor)
data=[surv, X]

total_obs = surv.shape[0]
batch_size = 40
total_events = torch.sum(surv[:, -1] == 1)
sampling_proportion = [total_obs, batch_size, total_events, None]

# -----------------------------------------------------------------------------------------------------------------------------
pyro.clear_param_store()
m = pcox.PCox(sampling_proportion=sampling_proportion)
m.initialize(eta=1.0)
loss=[]
for ii in tqdm.tqdm(range((10000))):
        #for jj in range(100):
        idx = np.random.choice(range(surv.shape[0]), batch_size, replace=False)
        data=[surv[idx], X[idx]]
        loss.append(m.infer(data=data))
plt.semilogy(loss)
theta_est = pyro.get_param_store()['AutoMultivariateNormal.loc'].detach()
print(pcox.concordance(surv.detach(), torch.mm(X, (theta_est[:, None]).detach())))
for key, value in pyro.get_param_store().items():
    print(f"{key}:\n{value}\n")


# -----------------------------------------------------------------------------------------------------------------------------

'''
R code
rm(list=ls())
library(survival)
library(stargazer)
#
data("lung")
lung
#

lung <- lung[is.finite(rowSums(lung)),]

standardize <- function(x){
(x - mean(x))/sqrt(var(x))
}
#
lung[ , c(4, 5, 6, 7, 8, 9, 10)] = apply(lung[ , c(4, 5, 6, 7, 8, 9, 10)], 2, standardize)
#
names(lung)
m1 = coxph(Surv(time, status) ~ age + sex + ph.ecog + ph.karno + pat.karno + meal.cal + wt.loss, data=lung)
summary(m1)
'''
